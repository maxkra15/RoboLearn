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

    def __call__(self, input, *, output_fp16=None, input_gradient_fp16=None):
        """Retain FP32 values and optionally fill explicit consumer-owned FP16 buffers.

        The optional buffer receives exact FP16 rounding after affine/ReLU in
        the same kernel. It has no adjoint or normalization state; backward
        continues to use the owned FP32 output and existing FP32 statistics.
        ``input_gradient_fp16`` receives the final accumulated FP32 input
        adjoint, not just this backward contribution. Only declare it where
        this operation is the final adjoint writer before the preceding dense
        backward. Buffers must not overlap active FP32 storage or each other.
        The caller retains them and both layers through all graph replays;
        the usual single-live-call per shape contract still applies.
        """
        _input(self, input)
        if input.shape[1] != self.num_features:
            raise ValueError("Input channel count differs from CudaBatchNorm.num_features")
        for name, buffer in (("output_fp16", output_fp16), ("input_gradient_fp16", input_gradient_fp16)):
            if buffer is not None and (
                not isinstance(buffer, wp.array)
                or buffer.dtype != wp.float16
                or buffer.device != self.device
                or not buffer.is_contiguous
                or buffer.shape != input.shape
            ):
                raise ValueError(f"{name} must be a contiguous matching FP16 matrix on this CUDA device")
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
        tape = wp._src.context.runtime.tape
        if input_gradient_fp16 is not None and (tape is None or input.grad is None or output.grad is None):
            raise ValueError("input_gradient_fp16 requires a taped operation with FP32 input/output adjoints")
        for name, buffer in (("output_fp16", output_fp16), ("input_gradient_fp16", input_gradient_fp16)):
            if buffer is None:
                continue
            begin = buffer.ptr
            end = begin + buffer.size * 2
            for array in (input, output, mean, variance, partial, sums, weight, bias, running_mean, running_var):
                if array is not None:
                    for value in (array, array.grad):
                        if value is not None and begin < value.ptr + value.size * 4 and value.ptr < end:
                            raise ValueError(f"{name} must not overlap BatchNorm's active FP32 buffers or adjoints")
        if output_fp16 is not None and input_gradient_fp16 is not None:
            if (
                output_fp16.ptr < input_gradient_fp16.ptr + input_gradient_fp16.size * 2
                and input_gradient_fp16.ptr < output_fp16.ptr + output_fp16.size * 2
            ):
                raise ValueError("FP16 output and input-gradient buffers must not overlap")
        with wp.ScopedDevice(self.device):
            if self._native_library is None:
                self._native_library = _library(self.device)
            library = self._native_library
            forward = library.rl_flash_bn_forward if output_fp16 is None else library.rl_flash_bn_forward_with_fp16
            arrays = (input, weight, bias, running_mean, running_var, mean, variance, partial, output)
            if output_fp16 is not None:
                arrays += (output_fp16,)
            check(
                library,
                forward(
                    *(pointer(value) for value in arrays),
                    *shape,
                    self.eps,
                    self.momentum,
                    int(training),
                    int(self.fused_relu),
                    stream_pointer(self.device),
                ),
                "Native BatchNorm forward",
            )
        if tape is None or output.grad is None:
            return output
        selected_mean, selected_variance = (mean, variance) if training else (running_mean, running_var)

        def backward():
            with wp.ScopedDevice(self.device):
                arrays = (
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
                function = library.rl_flash_bn_backward
                if input_gradient_fp16 is not None:
                    arrays += (input_gradient_fp16,)
                    function = library.rl_flash_bn_backward_with_fp16
                check(
                    library,
                    function(
                        *(pointer(value) for value in arrays),
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
