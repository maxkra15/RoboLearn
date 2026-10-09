# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""FlashSAC normalization with Warp-NN forward and fixed-order backward sums.

Warp-NN's forward moments already use deterministic chunk reductions. These
two-dimensional layers replace its automatic broadcast adjoints, whose atomic
sums can perturb small gradients enough to change an Adam update materially.
Parameter names, running statistics and forward arithmetic remain Warp-NN's.
"""

import warp as wp
from warp_nn import nn

_CHUNK = 256


@wp.kernel(enable_backward=False)
def _channel_partials(
    input: wp.array2d(dtype=wp.float32),
    output_grad: wp.array2d(dtype=wp.float32),
    mean: wp.array(dtype=wp.float32),
    variance: wp.array(dtype=wp.float32),
    eps: float,
    rms: bool,
    dy_partials: wp.array2d(dtype=wp.float32),
    dy_x_partials: wp.array2d(dtype=wp.float32),
):
    channel, chunk = wp.tid()
    start = chunk * _CHUNK
    end = wp.min(start + _CHUNK, input.shape[0])
    dy_sum = float(0.0)
    dy_x_sum = float(0.0)
    for row in range(start, end):
        normalized = input[row, channel]
        if rms:
            normalized = normalized / wp.sqrt(variance[row] + eps)
        else:
            normalized = (normalized - mean[channel]) / wp.sqrt(variance[channel] + eps)
        dy = output_grad[row, channel]
        dy_sum = dy_sum + dy
        dy_x_sum = dy_x_sum + dy * normalized
    dy_partials[channel, chunk] = dy_sum
    dy_x_partials[channel, chunk] = dy_x_sum


@wp.kernel(enable_backward=False)
def _channel_reduce(
    dy_partials: wp.array2d(dtype=wp.float32),
    dy_x_partials: wp.array2d(dtype=wp.float32),
    count: int,
    weight_grad: wp.array(dtype=wp.float32),
    bias_grad: wp.array(dtype=wp.float32),
    has_weight: bool,
    has_bias: bool,
    sums: wp.array2d(dtype=wp.float32),
):
    channel = wp.tid()
    dy_sum = float(0.0)
    dy_x_sum = float(0.0)
    for chunk in range(dy_partials.shape[1]):
        dy_sum = dy_sum + dy_partials[channel, chunk]
        dy_x_sum = dy_x_sum + dy_x_partials[channel, chunk]
    sums[0, channel] = dy_sum / float(count)
    sums[1, channel] = dy_x_sum / float(count)
    if has_weight:
        weight_grad[channel] = weight_grad[channel] + dy_x_sum
    if has_bias:
        bias_grad[channel] = bias_grad[channel] + dy_sum


@wp.kernel(enable_backward=False)
def _batch_input_gradient(
    input: wp.array2d(dtype=wp.float32),
    output_grad: wp.array2d(dtype=wp.float32),
    mean: wp.array(dtype=wp.float32),
    variance: wp.array(dtype=wp.float32),
    weight: wp.array(dtype=wp.float32),
    sums: wp.array2d(dtype=wp.float32),
    eps: float,
    training: bool,
    input_grad: wp.array2d(dtype=wp.float32),
):
    row, channel = wp.tid()
    inverse_std = 1.0 / wp.sqrt(variance[channel] + eps)
    value = output_grad[row, channel]
    if training:
        normalized = (input[row, channel] - mean[channel]) * inverse_std
        value = value - sums[0, channel] - normalized * sums[1, channel]
    input_grad[row, channel] = input_grad[row, channel] + weight[channel] * inverse_std * value


@wp.kernel(enable_backward=False)
def _rms_row_sums(
    input: wp.array2d(dtype=wp.float32),
    output_grad: wp.array2d(dtype=wp.float32),
    weight: wp.array(dtype=wp.float32),
    sums: wp.array(dtype=wp.float32),
):
    row = wp.tid()
    total = float(0.0)
    for channel in range(input.shape[1]):
        total = total + output_grad[row, channel] * weight[channel] * input[row, channel]
    sums[row] = total / float(input.shape[1])


@wp.kernel(enable_backward=False)
def _rms_input_gradient(
    input: wp.array2d(dtype=wp.float32),
    output_grad: wp.array2d(dtype=wp.float32),
    variance: wp.array(dtype=wp.float32),
    weight: wp.array(dtype=wp.float32),
    sums: wp.array(dtype=wp.float32),
    eps: float,
    input_grad: wp.array2d(dtype=wp.float32),
):
    row, channel = wp.tid()
    inverse_std = 1.0 / wp.sqrt(variance[row] + eps)
    value = output_grad[row, channel] * weight[channel]
    value = value - input[row, channel] * inverse_std * inverse_std * sums[row]
    input_grad[row, channel] = input_grad[row, channel] + inverse_std * value


@wp.kernel(enable_backward=False)
def _log_softmax_input_gradient(
    input: wp.array2d(dtype=wp.float32),
    output_grad: wp.array2d(dtype=wp.float32),
    maximum: wp.array3d(dtype=wp.float32),
    total: wp.array3d(dtype=wp.float32),
    input_grad: wp.array2d(dtype=wp.float32),
):
    row = wp.tid()
    gradient_sum = float(0.0)
    for channel in range(input.shape[1]):
        gradient_sum = gradient_sum + output_grad[row, channel]
    m = maximum[row, 0, 0]
    denominator = total[row, 0, 0]
    for channel in range(input.shape[1]):
        probability = wp.exp(input[row, channel] - m) / denominator
        value = output_grad[row, channel] - probability * gradient_sum
        input_grad[row, channel] = input_grad[row, channel] + value


class BatchNorm(nn.BatchNorm):
    """Two-dimensional affine BatchNorm with deterministic backward reductions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backward_cache = {}

    def __call__(self, input: wp.array) -> wp.array:
        if len(input.shape) != 2:
            raise ValueError("FlashSAC BatchNorm requires a two-dimensional input.")
        tape = wp._src.context.runtime.tape
        if tape is None:
            return super().__call__(input)
        # Explicit base call avoids depending on a concrete subclass's MRO.
        wp._src.context.runtime.tape = None
        try:
            output = nn.BatchNorm.__call__(self, input)
        finally:
            wp._src.context.runtime.tape = tape
        if output.grad is None:
            return output
        shape = input.shape
        if shape not in self._backward_cache:
            chunks = (shape[0] + _CHUNK - 1) // _CHUNK
            self._backward_cache[shape] = (
                wp.empty((shape[1], chunks), dtype=wp.float32, device=self.device),
                wp.empty((shape[1], chunks), dtype=wp.float32, device=self.device),
                wp.empty((2, shape[1]), dtype=wp.float32, device=self.device),
            )
        partials, x_partials, sums = self._backward_cache[shape]
        training = self.training
        if training:
            moments = self._cache[shape][1]
            mean, variance = moments.mean, moments.var
        else:
            mean, variance = self.running_mean.data, self.running_var.data
        weight, bias = self.weight.data, self.bias.data

        def backward():
            if training or weight.grad is not None or bias.grad is not None:
                wp.launch(
                    _channel_partials,
                    dim=partials.shape,
                    inputs=[input, output.grad, mean, variance, self.eps, False],
                    outputs=[partials, x_partials],
                    device=self.device,
                )
                wp.launch(
                    _channel_reduce,
                    dim=shape[1],
                    inputs=[
                        partials,
                        x_partials,
                        shape[0],
                        weight.grad,
                        bias.grad,
                        weight.grad is not None,
                        bias.grad is not None,
                    ],
                    outputs=[sums],
                    device=self.device,
                )
            if input.grad is not None:
                wp.launch(
                    _batch_input_gradient,
                    dim=shape,
                    inputs=[input, output.grad, mean, variance, weight, sums, self.eps, training],
                    outputs=[input.grad],
                    device=self.device,
                )

        arrays = [array for array in (input, weight, bias, output) if array.grad is not None]
        tape.record_func(backward, arrays)
        return output


class RMSNorm(nn.RMSNorm):
    """Two-dimensional RMSNorm with deterministic row and channel reductions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backward_cache = {}

    def __call__(self, input: wp.array) -> wp.array:
        if len(input.shape) != 2:
            raise ValueError("FlashSAC RMSNorm requires a two-dimensional input.")
        tape = wp._src.context.runtime.tape
        if tape is None:
            return super().__call__(input)
        wp._src.context.runtime.tape = None
        try:
            output = nn.RMSNorm.__call__(self, input)
        finally:
            wp._src.context.runtime.tape = tape
        if output.grad is None:
            return output
        shape = input.shape
        if shape not in self._backward_cache:
            chunks = (shape[0] + _CHUNK - 1) // _CHUNK
            self._backward_cache[shape] = (
                wp.empty((shape[1], chunks), dtype=wp.float32, device=self.device),
                wp.empty((shape[1], chunks), dtype=wp.float32, device=self.device),
                wp.empty((2, shape[1]), dtype=wp.float32, device=self.device),
                wp.empty(shape[0], dtype=wp.float32, device=self.device),
            )
        partials, x_partials, sums, row_sums = self._backward_cache[shape]
        variance = self._cache[shape][1].var
        weight = self.weight.data

        def backward():
            if weight.grad is not None:
                wp.launch(
                    _channel_partials,
                    dim=partials.shape,
                    inputs=[input, output.grad, None, variance, self.eps, True],
                    outputs=[partials, x_partials],
                    device=self.device,
                )
                wp.launch(
                    _channel_reduce,
                    dim=shape[1],
                    inputs=[partials, x_partials, shape[0], weight.grad, None, True, False],
                    outputs=[sums],
                    device=self.device,
                )
            if input.grad is not None:
                wp.launch(
                    _rms_row_sums,
                    dim=shape[0],
                    inputs=[input, output.grad, weight],
                    outputs=[row_sums],
                    device=self.device,
                )
                wp.launch(
                    _rms_input_gradient,
                    dim=shape,
                    inputs=[input, output.grad, variance, weight, row_sums, self.eps],
                    outputs=[input.grad],
                    device=self.device,
                )

        arrays = [array for array in (input, weight, output) if array.grad is not None]
        tape.record_func(backward, arrays)
        return output


class LogSoftmax(nn.LogSoftmax):
    """Row LogSoftmax with one fixed-order backward sum per categorical row.

    The upstream forward cache supplies the maximum and exponential sum. Its
    automatic backward atomically reduces into that sum's adjoint; this callback
    instead applies ``dy - probability * sum(dy)`` without extra scratch arrays.
    """

    def __call__(self, input: wp.array) -> wp.array:
        if len(input.shape) != 2 or self.dim not in (-1, 1):
            raise ValueError("FlashSAC LogSoftmax requires rows of categorical logits.")
        tape = wp._src.context.runtime.tape
        if tape is None:
            return super().__call__(input)
        wp._src.context.runtime.tape = None
        try:
            output = nn.LogSoftmax.__call__(self, input)
        finally:
            wp._src.context.runtime.tape = tape
        if output.grad is None:
            return output
        _, _, maximum, total = self._cache[(input.shape, input.dtype)]

        def backward():
            if input.grad is not None:
                wp.launch(
                    _log_softmax_input_gradient,
                    dim=input.shape[0],
                    inputs=[input, output.grad, maximum, total],
                    outputs=[input.grad],
                    device=self.device,
                )

        arrays = [array for array in (input, output) if array.grad is not None]
        tape.record_func(backward, arrays)
        return output
