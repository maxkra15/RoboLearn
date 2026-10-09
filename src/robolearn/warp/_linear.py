# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Capture-safe Linear backward with separate matrix products and reductions."""

import warp as wp
from warp_nn import nn
from warp_nn.utils import resolve_dim

_TILE = 16
_REDUCTION_TILE = 32
_SPLIT_BATCH = 512


@wp.kernel(enable_backward=False)
def _input_gradient(
    weight: wp.array2d(dtype=wp.float32),
    output_grad: wp.array2d(dtype=wp.float32),
    input_grad: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    result = wp.tile_zeros(shape=(_TILE, _TILE), dtype=wp.float32)
    for k in range((weight.shape[0] + _REDUCTION_TILE - 1) // _REDUCTION_TILE):
        dy = wp.tile_load(output_grad, shape=(_TILE, _REDUCTION_TILE), offset=(i * _TILE, k * _REDUCTION_TILE))
        w = wp.tile_load(weight, shape=(_REDUCTION_TILE, _TILE), offset=(k * _REDUCTION_TILE, j * _TILE))
        wp.tile_matmul(dy, w, result)
    offset = (i * _TILE, j * _TILE)
    previous = wp.tile_load(input_grad, shape=(_TILE, _TILE), offset=offset)
    wp.tile_store(input_grad, previous + result, offset=offset)


@wp.kernel(enable_backward=False)
def _parameter_partials(
    input: wp.array2d(dtype=wp.float32),
    output_grad: wp.array2d(dtype=wp.float32),
    padded_out: int,
    weight_partials: wp.array2d(dtype=wp.float32),
    bias_partials: wp.array2d(dtype=wp.float32),
):
    i, j, split = wp.tid()
    result = wp.tile_zeros(shape=(_TILE, _TILE), dtype=wp.float32)
    bias = wp.tile_zeros(shape=(_TILE, 1), dtype=wp.float32)
    steps = (wp.min(_SPLIT_BATCH, input.shape[0] - split * _SPLIT_BATCH) + _REDUCTION_TILE - 1) // _REDUCTION_TILE
    for k in range(steps):
        batch_offset = split * _SPLIT_BATCH + k * _REDUCTION_TILE
        dy = wp.tile_load(output_grad, shape=(_REDUCTION_TILE, _TILE), offset=(batch_offset, i * _TILE))
        x = wp.tile_load(input, shape=(_REDUCTION_TILE, _TILE), offset=(batch_offset, j * _TILE))
        wp.tile_matmul(wp.tile_transpose(dy), x, result)
        if j == 0:
            bias = bias + wp.tile_reshape(wp.tile_sum(dy, axis=0), shape=(_TILE, 1))
    row = split * padded_out + i * _TILE
    wp.tile_store(weight_partials, result, offset=(row, j * _TILE))
    if j == 0:
        wp.tile_store(bias_partials, bias, offset=(row, 0))


@wp.kernel(enable_backward=False)
def _reduce_weight(
    partials: wp.array2d(dtype=wp.float32),
    padded_out: int,
    splits: int,
    gradient: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    value = float(0.0)
    for split in range(splits):
        value = value + partials[split * padded_out + i, j]
    gradient[i, j] = gradient[i, j] + value


@wp.kernel(enable_backward=False)
def _reduce_bias(
    partials: wp.array2d(dtype=wp.float32),
    padded_out: int,
    splits: int,
    gradient: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    value = float(0.0)
    for split in range(splits):
        value = value + partials[split * padded_out + i, 0]
    gradient[i, 0] = gradient[i, 0] + value


class TiledLinear(nn.Linear):
    """Keep Warp-NN's parameters and forward kernel; replace its taped backward.

    Each backward callback launches ordinary Warp kernels during capture. The
    graph retains those launches and persistent scratch buffers for replay.
    Parameter gradients accumulate through uniquely owned partial tiles and a
    final reduction, rather than atomics from every batch tile.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backward_cache = {}

    def __call__(self, input: wp.array) -> wp.array:
        tape = wp._src.context.runtime.tape
        if tape is None:
            return super().__call__(input)
        shape = (input.shape[0], self.out_features)
        key = (shape, input.dtype)
        if key not in self._cache:
            self._cache[key] = wp.empty(shape, dtype=input.dtype, device=self.device, requires_grad=self.requires_grad)
        output = self._cache[key]
        weight = self.weight.data
        bias = self.bias.data if self.bias is not None else None
        wp.launch_tiled(
            self._kernel,
            dim=resolve_dim(config=self._config, shape=shape, tiled=True),
            inputs=[input, weight, bias],
            outputs=[output],
            device=self.device,
            block_dim=self._config.block_dim,
            record_tape=False,
        )
        if output.grad is None:
            return output
        batch = input.shape[0]
        splits = (batch + _SPLIT_BATCH - 1) // _SPLIT_BATCH
        padded_out = (self.out_features + _TILE - 1) // _TILE * _TILE
        padded_in = (self.in_features + _TILE - 1) // _TILE * _TILE
        if batch not in self._backward_cache:
            self._backward_cache[batch] = (
                wp.empty((splits * padded_out, padded_in), dtype=wp.float32, device=self.device),
                wp.empty((splits * padded_out, 1), dtype=wp.float32, device=self.device),
            )
        weight_partials, bias_partials = self._backward_cache[batch]

        def backward():
            if input.grad is not None:
                wp.launch_tiled(
                    _input_gradient,
                    dim=((batch + _TILE - 1) // _TILE, padded_in // _TILE),
                    inputs=[weight, output.grad, input.grad],
                    device=self.device,
                    block_dim=256,
                )
            if weight.grad is not None or (bias is not None and bias.grad is not None):
                wp.launch_tiled(
                    _parameter_partials,
                    dim=(padded_out // _TILE, padded_in // _TILE, splits),
                    inputs=[input, output.grad, padded_out],
                    outputs=[weight_partials, bias_partials],
                    device=self.device,
                    block_dim=256,
                )
                if weight.grad is not None:
                    wp.launch(
                        _reduce_weight,
                        dim=weight.shape,
                        inputs=[weight_partials, padded_out, splits, weight.grad],
                        device=self.device,
                    )
                if bias is not None and bias.grad is not None:
                    wp.launch(
                        _reduce_bias,
                        dim=self.out_features,
                        inputs=[bias_partials, padded_out, splits, bias.grad],
                        device=self.device,
                    )

        arrays = [array for array in (input, weight, bias, output) if array is not None and array.grad is not None]
        tape.record_func(backward, arrays)
        return output
