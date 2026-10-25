# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Fused native CUDA Adam retaining Warp-NN's parameter and optimizer storage."""

import ctypes
import math

import warp as wp
from warp_nn import optimizers
from warp_nn.utils import ScopedCapture

from ._runtime import NativePlan, check, library_for, pointer, stream_pointer


class CudaAdam(optimizers.Adam):
    """Use native multi-tensor updates with the existing clipping/state contract.

    Moment ordering, float32 timestep and device learning-rate arrays are owned
    by Warp-NN. Loading values into them preserves captured pointer addresses.
    Prepare before capture, and keep the optimizer alive while its graphs run.
    """

    def __init__(
        self,
        *args,
        dynamic_loss_scale: bool = False,
        initial_loss_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not self.device.is_cuda:
            raise ValueError("CudaAdam requires a CUDA device")
        self._cuda_plan = None
        if (
            not math.isfinite(initial_loss_scale)
            or not 1.1754943508222875e-38 <= initial_loss_scale <= 3.4028234663852886e38
            or not math.isfinite(growth_factor)
            or not 1.0 < growth_factor <= 3.4028234663852886e38
            or not math.isfinite(backoff_factor)
            or not 0.0 < backoff_factor < 1.0
            or isinstance(growth_interval, bool)
            or not isinstance(growth_interval, int)
            or not 0 < growth_interval <= 2147483647
        ):
            raise ValueError("Dynamic loss scale/growth/backoff/interval must be finite and valid")
        if not 1.0 < ctypes.c_float(growth_factor).value or not 0.0 < ctypes.c_float(backoff_factor).value < 1.0:
            raise ValueError("Dynamic growth/backoff factors must remain valid when stored as float32")
        self.dynamic_loss_scale = dynamic_loss_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self.loss_scale = (
            wp.array([initial_loss_scale], dtype=wp.float32, device=self.device) if dynamic_loss_scale else None
        )
        self.scale_growth_tracker = wp.zeros(1, dtype=wp.int32, device=self.device) if dynamic_loss_scale else None
        self.found_inf = wp.zeros(1, dtype=wp.int32, device=self.device) if dynamic_loss_scale else None
        self.step_skipped = wp.zeros(1, dtype=wp.int32, device=self.device) if dynamic_loss_scale else None
        self.attempted_steps = wp.zeros(1, dtype=wp.int64, device=self.device) if dynamic_loss_scale else None
        self.skipped_steps = wp.zeros(1, dtype=wp.int64, device=self.device) if dynamic_loss_scale else None

    def state_arrays(self):
        """Return persistent scaler state for in-place checkpoints and warmup restore.

        The inherited ``_timestep`` counts completed updates. All buffers here
        are independent per optimizer; no Python scalar mirror is used during
        capture. The default FP32 optimizer returns an empty mapping.
        """
        if not self.dynamic_loss_scale:
            return {}
        return {
            "loss_scale": self.loss_scale,
            "scale_growth_tracker": self.scale_growth_tracker,
            "found_inf": self.found_inf,
            "step_skipped": self.step_skipped,
            "attempted_steps": self.attempted_steps,
            "skipped_steps": self.skipped_steps,
        }

    def prepare(self):
        """Copy tensor-pointer metadata to the native plan outside capture."""
        if self._cuda_plan is not None:
            return self
        if self.device.is_capturing:
            raise RuntimeError("Call CudaAdam.prepare() before capture")
        groups = (self._parameters, self._gradients, self._m1, self._m2)
        if not self._parameters:
            raise ValueError("CudaAdam requires at least one parameter")
        for group in groups:
            for array in group:
                if array.dtype != wp.float32 or array.device != self.device or not array.is_contiguous:
                    raise ValueError("CudaAdam requires contiguous float32 arrays on its CUDA device")
        count = len(self._parameters)
        tables = [(ctypes.c_void_p * count)(*(array.ptr for array in group)) for group in groups]
        sizes = (ctypes.c_size_t * count)(*(array.size for array in self._parameters))
        with wp.ScopedDevice(self.device):
            library = library_for(self.device)
            handle = library.rl_adam_create(
                count, *tables, sizes, pointer(self._timestep), pointer(self._lr), *self._betas, self._eps
            )
            self._cuda_plan = NativePlan(library, handle, self.device, adam=True)
            if self.dynamic_loss_scale:
                try:
                    check(
                        library,
                        library.rl_adam_set_scaler(
                            handle,
                            pointer(self.loss_scale),
                            pointer(self.scale_growth_tracker),
                            pointer(self.found_inf),
                            pointer(self.step_skipped),
                            pointer(self.attempted_steps),
                            pointer(self.skipped_steps),
                            self._growth_factor,
                            self._backoff_factor,
                            self._growth_interval,
                        ),
                        "Configure CUDA dynamic loss scaling",
                    )
                except Exception:
                    self.close()
                    raise
        return self

    def step(self, *, lr: float | None = None):
        """Unscale/check, clip, and conditionally update without host synchronization.

        With dynamic scaling, seed an already allocated scalar loss gradient
        from ``loss_scale`` before backward. Nonfinite gradients skip parameters,
        moments and the completed timestep; the attempted counter and scale
        state still advance. Gradients are unscaled in place before clipping.
        """
        self.prepare()
        if lr is not None:
            self._lr.fill_(lr)

        def launch():
            plan = self._cuda_plan
            if self.dynamic_loss_scale:
                check(
                    plan.library,
                    plan.library.rl_adam_unscale(plan.handle, stream_pointer(self.device)),
                    "Unscale/check CUDA Adam gradients",
                )
            if self._max_norm is not None:
                self._launch_clip_by_total_norm()
            check(plan.library, plan.library.rl_adam_step(plan.handle, stream_pointer(self.device)), "CUDA Adam")

        with wp.ScopedDevice(self.device):
            if self._graph_step is None:
                # An outer capture records native launches directly. Otherwise
                # preserve Warp-NN's optional optimizer-only graph behavior.
                enabled = not self._disable_graph and not self.device.is_capturing
                with ScopedCapture(device=self.device, enabled=enabled) as capture:
                    launch()
                self._graph_step = capture.graph
            if self._graph_step is not None:
                wp.capture_launch(self._graph_step)

    def close(self):
        """Release the native plan after captured work and graphs are discarded."""
        self._graph_step = None
        if self._cuda_plan is not None:
            self._cuda_plan.close()
            self._cuda_plan = None
