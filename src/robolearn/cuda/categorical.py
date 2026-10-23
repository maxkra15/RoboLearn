# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Cooperative categorical heads with persistent FP32 outputs and custom adjoints."""

import ctypes

import warp as wp
from warp_nn import nn

from ._runtime import check, library_for, pointer, stream_pointer
from .flash import _validate

_LIBRARIES = {}


def _library(device):
    if device.alias in _LIBRARIES:
        return _LIBRARIES[device.alias]
    library = library_for(device)
    if not getattr(library, "_categorical_declared", False):
        p, i, f = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        for name, arguments in {
            "forward": [p] * 5 + [i, i, p],
            "backward": [p] * 7 + [i, i, p],
            "target": [p] * 9 + [i, i, f, f, p],
        }.items():
            function = getattr(library, "rl_flash_categorical_" + name)
            function.argtypes, function.restype = arguments, i
        library._categorical_declared = True
    _LIBRARIES[device.alias] = library
    return library


class CudaCategorical(nn.Module):
    """Fuse bias, stable log-softmax and expected value without owning parameters.

    As in Warp-NN, one live forward per input shape is supported. Warm shapes
    before capture. Gradients accumulate into the caller's FP32 logits and bias;
    frozen bias parameters retain the input gradient needed by actor learning.
    """

    def __init__(self, *, requires_grad=True):
        super().__init__(requires_grad=requires_grad)
        self._library = None
        super().__post_init__()

    def __call__(self, raw, bias, support):
        if (
            raw.ndim != 2
            or raw.shape[0] < 1
            or raw.shape[1] < 2
            or bias.shape != (raw.shape[1],)
            or support.shape != bias.shape
        ):
            raise ValueError("Categorical head requires (batch,bins) logits and matching bias/support vectors")
        _validate([raw, bias, support], self.device)
        shape = raw.shape
        if shape not in self._cache:
            if self.device.is_capturing:
                raise RuntimeError("Warm the categorical head shape before capture")
            self._cache[shape] = (
                wp.empty(shape, device=self.device, requires_grad=self.requires_grad),
                wp.empty(shape[0], device=self.device, requires_grad=self.requires_grad),
                wp.empty(shape, device=self.device),
            )
        log_probs, values, scratch = self._cache[shape]
        if self._library is None:
            self._library = _library(self.device)
        library = self._library
        with wp.ScopedDevice(self.device):
            check(
                library,
                library.rl_flash_categorical_forward(
                    *(pointer(x) for x in (raw, bias, support, log_probs, values)),
                    *shape,
                    stream_pointer(self.device),
                ),
                "Native categorical forward",
            )
        tape = wp._src.context.runtime.tape
        if tape is None or log_probs.grad is None:
            return values, log_probs

        def backward():
            if raw.grad is None and bias.grad is None:
                return
            with wp.ScopedDevice(self.device):
                check(
                    library,
                    library.rl_flash_categorical_backward(
                        *(
                            pointer(x)
                            for x in (log_probs, support, log_probs.grad, values.grad, scratch, raw.grad, bias.grad)
                        ),
                        *shape,
                        stream_pointer(self.device),
                    ),
                    "Native categorical backward",
                )

        tape.record_func(backward, [x for x in (raw, bias, log_probs, values) if x.grad is not None])
        return values, log_probs


def categorical_target(
    q, log_probs, reward, terminated, discount, next_log_probs, alpha, min_v, max_v, target, maximum
):
    """Project the selected next-state distribution with cooperative shared sums."""
    device = target.device
    _validate([q, log_probs, reward, terminated, discount, next_log_probs, alpha, target, maximum], device)
    batch, bins = target.shape
    if not 2 <= bins <= 4096 or q.shape != (2, 2 * batch) or log_probs.shape != (2, 2 * batch, bins):
        raise ValueError("Categorical target requires twin concatenated current/next distributions")
    if any(x.shape != (batch,) for x in (reward, terminated, discount, next_log_probs)) or any(
        x.shape != (1,) for x in (alpha, maximum)
    ):
        raise ValueError("Categorical target requires per-transition vectors and scalar alpha/maximum")
    library = _library(device)
    with wp.ScopedDevice(device):
        check(
            library,
            library.rl_flash_categorical_target(
                *(
                    pointer(x)
                    for x in (q, log_probs, reward, terminated, discount, next_log_probs, alpha, target, maximum)
                ),
                batch,
                bins,
                min_v,
                max_v,
                stream_pointer(device),
            ),
            "Native categorical target",
        )
