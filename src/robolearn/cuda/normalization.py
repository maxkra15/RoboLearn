# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""FP32 cooperative normalization with Warp-NN parameters and custom adjoints.

Construct on the final CUDA device and warm every batch shape before capture.
Statistics and reductions use a different FP32 order from Warp-NN. Population
variance, unbiased running variance, affine formulas and gradient accumulation
are preserved. Cached outputs support one live call per shape, as in Warp-NN.
"""

import math

import warp as wp
from warp_nn import nn

from ._runtime import check, pointer, stream_pointer
from .flash import _library, _validate


def _data(parameter):
    return parameter.data if parameter is not None else None


def _gradient(array):
    return array.grad if array is not None else None


def _input(layer, value):
    if value.ndim != 2 or min(value.shape) < 1 or value.size > 2**31 - 1:
        raise ValueError("Native FlashSAC normalization requires a nonempty matrix fitting signed int")
    _validate([value], layer.device)


def _arrays(*values):
    return [value for value in values if value is not None and value.grad is not None]


class CudaBatchNorm(nn.BatchNorm):
    """Channel BatchNorm with cooperative moments and optional fused ReLU.

    State dictionary keys and parameter ownership are inherited unchanged from
    Warp-NN. Running buffers use ``(1-momentum)*old + momentum*batch_stat``;
    forward variance is biased and its running estimate has B/(B-1) correction.
    """

    def __init__(self, *args, relu=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.device.is_cuda:
            raise ValueError("CudaBatchNorm requires CUDA")
        if not math.isfinite(self.eps) or self.eps <= 0 or not 0 <= self.momentum <= 1:
            raise ValueError("BatchNorm epsilon must be positive and momentum must lie in [0, 1]")
        self.fused_relu = bool(relu)
        self._native_library = None

    def __call__(self, input):
        _input(self, input)
        if input.shape[1] != self.num_features:
            raise ValueError("Input channel count differs from CudaBatchNorm.num_features")
        training = self.training or not self.track_running_stats
        if self.training and input.shape[0] <= 1:
            raise ValueError("BatchNorm training requires more than one value per channel")
        shape = input.shape
        if shape not in self._cache:
            if self.device.is_capturing:
                raise RuntimeError("Warm this CudaBatchNorm batch shape before capture")
            batch, channels = shape
            chunks = (batch + 255) // 256
            self._cache[shape] = (
                wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=self.requires_grad),
                wp.empty(channels, dtype=wp.float32, device=self.device),
                wp.empty(channels, dtype=wp.float32, device=self.device),
                wp.empty((2 * chunks, channels), dtype=wp.float32, device=self.device),
                wp.empty((2, channels), dtype=wp.float32, device=self.device),
            )
        output, mean, variance, partial, sums = self._cache[shape]
        weight, bias = _data(self.weight), _data(self.bias)
        running_mean, running_var = _data(self.running_mean), _data(self.running_var)
        with wp.ScopedDevice(self.device):
            if self._native_library is None:
                self._native_library = _library(self.device)
            library = self._native_library
            check(
                library,
                library.rl_flash_bn_forward(
                    *(
                        pointer(value)
                        for value in (input, weight, bias, running_mean, running_var, mean, variance, partial, output)
                    ),
                    *shape,
                    self.eps,
                    self.momentum,
                    int(training),
                    int(self.fused_relu),
                    stream_pointer(self.device),
                ),
                "Native BatchNorm forward",
            )
        tape = wp._src.context.runtime.tape
        if tape is None or output.grad is None:
            return output
        selected_mean, selected_variance = (mean, variance) if training else (running_mean, running_var)

        def backward():
            with wp.ScopedDevice(self.device):
                check(
                    library,
                    library.rl_flash_bn_backward(
                        *(
                            pointer(value)
                            for value in (
                                input,
                                output.grad,
                                output,
                                weight,
                                selected_mean,
                                selected_variance,
                                partial,
                                sums,
                                input.grad,
                                _gradient(weight),
                                _gradient(bias),
                            )
                        ),
                        *shape,
                        self.eps,
                        int(training),
                        int(self.fused_relu),
                        stream_pointer(self.device),
                    ),
                    "Native BatchNorm backward",
                )

        tape.record_func(backward, _arrays(input, weight, bias, output))
        return output


class CudaRMSNorm(nn.RMSNorm):
    """Row RMSNorm with fused variance/affine forward and cooperative backward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.device.is_cuda:
            raise ValueError("CudaRMSNorm requires CUDA")
        if len(self.normalized_shape) != 1 or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("CudaRMSNorm supports one normalized channel dimension and positive epsilon")
        self._native_library = None

    def __call__(self, input):
        _input(self, input)
        if input.shape[1:] != self.normalized_shape:
            raise ValueError("Input channel count differs from CudaRMSNorm.normalized_shape")
        shape = input.shape
        if shape not in self._cache:
            if self.device.is_capturing:
                raise RuntimeError("Warm this CudaRMSNorm batch shape before capture")
            self._cache[shape] = (
                wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=self.requires_grad),
                wp.empty(shape[0], dtype=wp.float32, device=self.device),
                wp.empty(((shape[0] + 255) // 256, shape[1]), dtype=wp.float32, device=self.device),
            )
        output, variance, partial = self._cache[shape]
        weight = _data(self.weight)
        with wp.ScopedDevice(self.device):
            if self._native_library is None:
                self._native_library = _library(self.device)
            library = self._native_library
            check(
                library,
                library.rl_flash_rms_forward(
                    *(pointer(value) for value in (input, weight, variance, output)),
                    *shape,
                    self.eps,
                    stream_pointer(self.device),
                ),
                "Native RMSNorm forward",
            )
        tape = wp._src.context.runtime.tape
        if tape is None or output.grad is None:
            return output

        def backward():
            with wp.ScopedDevice(self.device):
                check(
                    library,
                    library.rl_flash_rms_backward(
                        *(
                            pointer(value)
                            for value in (input, output.grad, weight, variance, partial, input.grad, _gradient(weight))
                        ),
                        *shape,
                        self.eps,
                        stream_pointer(self.device),
                    ),
                    "Native RMSNorm backward",
                )

        tape.record_func(backward, _arrays(input, weight, output))
        return output
