# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""cuBLASLt FP32 Linear with Warp-NN parameters, caches and custom adjoints."""

from __future__ import annotations

import ctypes

import warp as wp
from warp_nn import nn

from ._runtime import NativePlan, check, library_for, pointer, stream_pointer


class CudaLinear(nn.Linear):
    """Replace dense arithmetic while retaining Warp-NN's owned parameter arrays.

    Calls use the current Warp stream and persistent per-batch output/workspace
    buffers. A complete eager forward/backward must precede outer graph capture.
    Cached activations, like Warp-NN's, cannot represent multiple live calls of
    the same shape in one tape. Keep this layer alive while its graphs are used.
    """

    def __init__(self, *args, allow_tf32: bool = False, workspace_limit: int = 32 * 1024 * 1024, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.device.is_cuda:
            raise ValueError("CudaLinear requires a CUDA device")
        if workspace_limit < 0:
            raise ValueError("workspace_limit must be nonnegative")
        self.allow_tf32 = allow_tf32
        self.workspace_limit = workspace_limit
        self._cuda_plans = {}

    def prepare(self, batch_size: int):
        """Create three GEMM plans and one persistent workspace outside capture."""
        if batch_size in self._cuda_plans:
            return self
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.device.is_capturing:
            raise RuntimeError("Call CudaLinear.prepare(batch_size) before capture")
        with wp.ScopedDevice(self.device):
            library = library_for(self.device)
            incoming, outgoing = self.in_features, self.out_features
            plans = []
            try:
                # Row-major Warp arrays are their transposed column-major views.
                # Y^T = W X^T; dX^T = W^T dY^T; dW^T = X^T dY.
                specifications = (
                    (outgoing, batch_size, incoming, 1, 0, incoming, incoming, outgoing, self.bias is not None),
                    (incoming, batch_size, outgoing, 0, 0, incoming, outgoing, incoming, False),
                    (incoming, outgoing, batch_size, 0, 1, incoming, outgoing, incoming, False),
                )
                for specification in specifications:
                    handle = library.rl_plan_create(*specification, int(self.allow_tf32), self.workspace_limit)
                    plans.append(NativePlan(library, handle, self.device))
                workspace_bytes = max(library.rl_plan_workspace(plan.handle) for plan in plans)
                if workspace_bytes == ctypes.c_size_t(-1).value:
                    check(library, -1, "Read cuBLASLt workspace size")
                workspace = wp.empty(max(workspace_bytes, 1), dtype=wp.uint8, device=self.device)
                key = ((batch_size, outgoing), wp.float32)
                if key not in self._cache:
                    # Input-only differentiation through frozen critics needs
                    # these adjoints even when layer parameters are frozen.
                    self._cache[key] = wp.empty(key[0], dtype=wp.float32, device=self.device, requires_grad=True)
                self._cuda_plans[batch_size] = (plans, workspace, workspace_bytes)
            except Exception:
                for plan in plans:
                    plan.close()
                raise
        return self

    def __call__(self, input: wp.array) -> wp.array:
        if input.dtype != wp.float32 or input.ndim != 2 or input.shape[1] != self.in_features:
            raise ValueError("CudaLinear requires float32 input shaped (batch, in_features)")
        if input.device != self.device or not input.is_contiguous:
            raise ValueError("CudaLinear requires contiguous input on the layer's CUDA device")
        batch = input.shape[0]
        self.prepare(batch)
        plans, workspace, workspace_bytes = self._cuda_plans[batch]
        forward, input_gradient, weight_gradient = plans
        library = forward.library
        output = self._cache[((batch, self.out_features), wp.float32)]
        weight = self.weight.data
        bias = self.bias.data if self.bias is not None else None
        tape = wp._src.context.runtime.tape
        if tape is not None and wp.config.verify_autograd_array_access:
            input.mark_read()
            weight.mark_read()
            if bias is not None:
                bias.mark_read()
            output.mark_write()

        def run(plan, left, right, destination, beta, bias_array=None):
            check(
                library,
                library.rl_plan_run(
                    plan.handle,
                    pointer(left),
                    pointer(right),
                    pointer(destination),
                    pointer(destination),
                    pointer(bias_array),
                    pointer(workspace),
                    workspace_bytes,
                    beta,
                    stream_pointer(self.device),
                ),
                "cuBLASLt dense operation",
            )

        with wp.ScopedDevice(self.device):
            run(forward, weight, input, output, 0.0, bias)
        if tape is None:
            return output

        def backward():
            with wp.ScopedDevice(self.device):
                if input.grad is not None:
                    run(input_gradient, weight, output.grad, input.grad, 1.0)
                if weight.grad is not None:
                    run(weight_gradient, input, output.grad, weight.grad, 1.0)
                if bias is not None and bias.grad is not None:
                    check(
                        library,
                        library.rl_bias_grad(
                            pointer(output.grad),
                            pointer(bias.grad),
                            batch,
                            self.out_features,
                            stream_pointer(self.device),
                        ),
                        "CUDA bias gradient",
                    )

        arrays = [array for array in (input, weight, bias, output) if array is not None and array.grad is not None]
        tape.record_func(backward, arrays)
        return output

    def close(self):
        """Release plans after all dependent graphs and device work have finished."""
        for plans, _, _ in self._cuda_plans.values():
            for plan in plans:
                plan.close()
        self._cuda_plans.clear()

    def to(self, device):
        device = wp.get_device(device)
        if not device.is_cuda:
            raise ValueError("CudaLinear requires a CUDA device")
        if device != self.device:
            self.close()
        return super().to(device)
