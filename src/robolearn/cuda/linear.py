# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""cuBLASLt Linear with FP32 storage and optional FP16 multiply inputs."""

from __future__ import annotations

import ctypes
import json
import time
from contextlib import contextmanager
from copy import deepcopy

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

    def __init__(
        self,
        *args,
        allow_tf32: bool = False,
        workspace_limit: int = 32 * 1024 * 1024,
        compute_dtype: str = "float32",
        autotune: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not self.device.is_cuda:
            raise ValueError("CudaLinear requires a CUDA device")
        if workspace_limit < 0:
            raise ValueError("workspace_limit must be nonnegative")
        self.allow_tf32 = allow_tf32
        self.workspace_limit = workspace_limit
        self.compute_dtype = compute_dtype
        self.autotune = bool(autotune)
        self._cuda_plans = {}
        self._tuning_metadata = {}
        self._weight_shadow = None
        self._shadow_library = None
        self._weight_shadow_managed = False

    @property
    def tuning_metadata(self):
        """Return host-only algorithm/probe records for every prepared shape/mode.

        String keys and values are JSON serializable. Candidate timings are
        scratch CUDA-event measurements, not full learner latency. With tuning
        disabled the selected algorithm is the original first valid heuristic.
        This property neither prepares operations nor accesses device values.
        """
        return deepcopy(self._tuning_metadata)

    @property
    def compute_dtype(self):
        return self._compute_dtype

    @compute_dtype.setter
    def compute_dtype(self, value):
        if value not in ("float32", "float16"):
            raise ValueError("compute_dtype must be 'float32' or 'float16'")
        self._compute_dtype = value

    @property
    def weight_shadow(self):
        """Return the prepared FP16 weight buffer, or ``None`` before preparation.

        This is derived storage, not a parameter or checkpoint entry. A caller
        managing its refresh must retain this buffer while graphs use it.
        """
        return self._weight_shadow

    def prepare_weight_shadow(self):
        """Allocate and initialize one FP16 weight shadow outside graph capture.

        All batch shapes share this address. Parameters must remain unchanged
        between a forward and its backward, including calls of other shapes.
        """
        if self._weight_shadow is not None:
            return self._weight_shadow
        if self.device.is_capturing:
            raise RuntimeError("Prepare the FP16 weight shadow before capture")
        with wp.ScopedDevice(self.device):
            library = library_for(self.device)
            shadow = wp.empty(self.weight.data.shape, dtype=wp.float16, device=self.device)
            check(
                library,
                library.rl_cast_f32_f16(
                    pointer(self.weight.data), pointer(shadow), self.weight.data.size, stream_pointer(self.device)
                ),
                "Initialize FP16 weight shadow",
            )
        self._weight_shadow = shadow
        self._shadow_library = library
        return shadow

    def set_weight_shadow_managed(self, enabled: bool = True):
        """Let a caller refresh the FP16 shadow after every master-weight mutation.

        The default recasts weights at every FP16 forward. Managed mode omits
        that launch, allowing a projection or EMA kernel to write the shadow
        together with its FP32 master. The caller must also refresh after any
        initialization, checkpoint load, or warmup-state restore. Changing the
        mode affects subsequently captured operations, not existing graphs.
        """
        if self.device.is_capturing:
            raise RuntimeError("Configure FP16 weight-shadow ownership before capture")
        if enabled:
            self.prepare_weight_shadow()
        self._weight_shadow_managed = bool(enabled)
        return self

    def refresh_weight_shadow(self):
        """Enqueue a shadow refresh; capture-safe after shadow preparation."""
        if self._weight_shadow is None:
            self.prepare_weight_shadow()
            return self
        with wp.ScopedDevice(self.device):
            check(
                self._shadow_library,
                self._shadow_library.rl_cast_f32_f16(
                    pointer(self.weight.data),
                    pointer(self._weight_shadow),
                    self.weight.data.size,
                    stream_pointer(self.device),
                ),
                "Refresh FP16 weight shadow",
            )
        return self

    @contextmanager
    def precision(self, compute_dtype: str):
        """Choose a host-side mode while retaining prepared graph buffer addresses.

        The mode used by a forward is captured by its backward callback. Warm
        every (batch size, mode) outside capture. Inference callers can scope
        the layer to ``float32`` without changing its training precision.
        """
        previous = self.compute_dtype
        self.compute_dtype = compute_dtype
        try:
            yield self
        finally:
            self.compute_dtype = previous

    def prepare(self, batch_size: int, compute_dtype: str | None = None):
        """Prepare immutable GEMM choices and optional conversion arrays outside capture.

        Opt-in autotuning probes only private native scratch buffers, for beta=0
        forward and beta=1 adjoints. Three rotated timing batches require a 3%
        median gain plus two paired wins before replacing the default heuristic.
        Existing plans are cached and never retuned by subsequent calls.
        """
        mode = self.compute_dtype if compute_dtype is None else compute_dtype
        if mode not in ("float32", "float16"):
            raise ValueError("compute_dtype must be 'float32' or 'float16'")
        cache_key = (batch_size, mode)
        if cache_key in self._cuda_plans:
            return self
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.device.is_capturing:
            raise RuntimeError("Call CudaLinear.prepare(batch_size) before capture")
        with wp.ScopedDevice(self.device):
            library = library_for(self.device)
            incoming, outgoing = self.in_features, self.out_features
            plans = []
            operations = {}
            try:
                # Row-major Warp arrays are their transposed column-major views.
                # Y^T = W X^T; dX^T = W^T dY^T; dW^T = X^T dY.
                specifications = (
                    (outgoing, batch_size, incoming, 1, 0, incoming, incoming, outgoing, self.bias is not None),
                    (incoming, batch_size, outgoing, 0, 0, incoming, outgoing, incoming, False),
                    (incoming, outgoing, batch_size, 0, 1, incoming, outgoing, incoming, False),
                )
                for index, specification in enumerate(specifications):
                    if mode == "float32":
                        handle = library.rl_plan_create(*specification, int(self.allow_tf32), self.workspace_limit)
                    else:
                        handle = library.rl_plan_create_mixed(*specification, 0, 1, self.workspace_limit)
                    plan = NativePlan(library, handle, self.device)
                    plans.append(plan)
                    beta = 0.0 if index == 0 else 1.0
                    tuning_seconds = 0.0
                    if self.autotune:
                        started = time.perf_counter()
                        check(
                            library,
                            library.rl_plan_autotune(plan.handle, beta, stream_pointer(self.device)),
                            "Autotune cuBLASLt dense operation",
                        )
                        tuning_seconds = time.perf_counter() - started
                    raw = library.rl_plan_tuning_metadata(plan.handle)
                    if raw is None:
                        check(library, -1, "Read cuBLASLt tuning metadata")
                    metadata = json.loads(raw)
                    metadata.update(
                        beta=beta,
                        tuning_seconds=tuning_seconds,
                        m=specification[0],
                        n=specification[1],
                        k=specification[2],
                        bias=specification[-1],
                        allow_tf32=self.allow_tf32 if mode == "float32" else False,
                        workspace_limit_bytes=self.workspace_limit,
                    )
                    operations[("forward", "input_gradient", "weight_gradient")[index]] = metadata
                workspace_bytes = max(library.rl_plan_workspace(plan.handle) for plan in plans)
                if workspace_bytes == ctypes.c_size_t(-1).value:
                    check(library, -1, "Read cuBLASLt workspace size")
                workspace = wp.empty(max(workspace_bytes, 1), dtype=wp.uint8, device=self.device)
                key = ((batch_size, outgoing), wp.float32)
                if key not in self._cache:
                    # Input-only differentiation through frozen critics needs
                    # these adjoints even when layer parameters are frozen.
                    self._cache[key] = wp.empty(key[0], dtype=wp.float32, device=self.device, requires_grad=True)
                converted = (
                    (
                        self.prepare_weight_shadow(),
                        wp.empty((batch_size, incoming), dtype=wp.float16, device=self.device),
                        wp.empty((batch_size, outgoing), dtype=wp.float16, device=self.device),
                    )
                    if mode == "float16"
                    else None
                )
                self._cuda_plans[cache_key] = (plans, workspace, workspace_bytes, converted)
                self._tuning_metadata[f"{batch_size}:{mode}"] = {
                    "batch_size": batch_size,
                    "compute_dtype": mode,
                    "operations": operations,
                }
            except Exception:
                for plan in plans:
                    plan.close()
                raise
        return self

    def input_fp16_buffer(self, batch_size: int) -> wp.array:
        """Return this shape's existing conversion buffer for an explicit producer.

        Prepare outside capture. A producer may write exact FP16 rounding of its
        FP32 output here, then pass this same object as ``input_fp16``. It stays
        owned by this layer and is reused for dW; the usual single-live-forward
        per shape contract applies. This adds no storage or checkpoint state.
        """
        self.prepare(batch_size, "float16")
        return self._cuda_plans[(batch_size, "float16")][3][1]

    def __call__(
        self, input: wp.array, *, compute_dtype: str | None = None, input_fp16: wp.array | None = None
    ) -> wp.array:
        if input.dtype != wp.float32 or input.ndim != 2 or input.shape[1] != self.in_features:
            raise ValueError("CudaLinear requires float32 input shaped (batch, in_features)")
        if input.device != self.device or not input.is_contiguous:
            raise ValueError("CudaLinear requires contiguous input on the layer's CUDA device")
        batch = input.shape[0]
        mode = self.compute_dtype if compute_dtype is None else compute_dtype
        self.prepare(batch, mode)
        plans, workspace, workspace_bytes, converted = self._cuda_plans[(batch, mode)]
        if input_fp16 is not None and (converted is None or input_fp16 is not converted[1]):
            raise ValueError("input_fp16 must be this layer's prepared FP16 input buffer for the current batch")
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

        def cast(source, destination):
            check(
                library,
                library.rl_cast_f32_f16(
                    pointer(source), pointer(destination), source.size, stream_pointer(self.device)
                ),
                "Convert dense inputs to FP16",
            )

        with wp.ScopedDevice(self.device):
            if converted is None:
                forward_weight, forward_input = weight, input
            else:
                forward_weight, forward_input, _ = converted
                if not self._weight_shadow_managed:
                    cast(weight, forward_weight)
                if input_fp16 is None:
                    cast(input, forward_input)
            run(forward, forward_weight, forward_input, output, 0.0, bias)
        if tape is None:
            return output

        def backward():
            with wp.ScopedDevice(self.device):
                if converted is None:
                    gradient_weight, gradient_input, gradient_output = weight, input, output.grad
                else:
                    gradient_weight, gradient_input, gradient_output = converted
                    # This cache owns the forward's converted inputs until the
                    # backward has completed; the caller must honor the usual
                    # single-live-call-per-shape Warp-NN cache contract.
                    cast(output.grad, gradient_output)
                if input.grad is not None:
                    run(input_gradient, gradient_weight, gradient_output, input.grad, 1.0)
                if weight.grad is not None:
                    run(weight_gradient, gradient_input, gradient_output, weight.grad, 1.0)
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
        if self.device.is_capturing:
            raise RuntimeError("Cannot close CudaLinear during capture")
        for plans, _, _, _ in self._cuda_plans.values():
            for plan in plans:
                plan.close()
        self._cuda_plans.clear()
        self._tuning_metadata.clear()
        self._weight_shadow = None
        self._shadow_library = None

    def to(self, device):
        device = wp.get_device(device)
        if not device.is_cuda:
            raise ValueError("CudaLinear requires a CUDA device")
        if device != self.device:
            self.close()
        return super().to(device)
