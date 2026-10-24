# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Capture-safe whole-network projections and target EMA with persistent metadata."""

import ctypes
import math
import sys

import warp as wp

from ._runtime import check, library_for, stream_pointer


def _library(device):
    library = library_for(device)
    if getattr(library, "_flash_declared", False):
        return library
    p, n, i, f = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_float
    tables, sizes = ctypes.POINTER(p), ctypes.POINTER(n)
    signatures = {
        "projection_create": ([n, tables, tables, sizes, ctypes.POINTER(i)], p),
        "projection_create_with_shadows": ([n, tables, tables, sizes, ctypes.POINTER(i), tables], p),
        "projection_step": ([p, p], i),
        "projection_destroy": ([p], i),
        "ema_create": ([n, tables, tables, sizes, f], p),
        "ema_create_with_shadows": ([n, tables, tables, sizes, f, tables], p),
        "ema_step": ([p, p], i),
        "ema_destroy": ([p], i),
        "bn_forward": ([p] * 9 + [i, i, f, f, i, i, p], i),
        "bn_forward_with_fp16": ([p] * 10 + [i, i, f, f, i, i, p], i),
        "bn_backward": ([p] * 11 + [i, i, f, i, i, p], i),
        "bn_backward_with_fp16": ([p] * 12 + [i, i, f, i, i, p], i),
        "bn_backward_controlled": ([p] * 12 + [i, i, i, f, i, i, p], i),
        "rms_forward": ([p] * 4 + [i, i, f, p], i),
        "rms_backward": ([p] * 7 + [i, i, f, p], i),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, "rl_flash_" + name)
        function.argtypes, function.restype = arguments, result
    library._flash_declared = True
    return library


def _validate(arrays, device):
    for array in arrays:
        if array.dtype != wp.float32 or array.device != device or not array.is_contiguous:
            raise ValueError("Native FlashSAC operations require contiguous FP32 arrays on one CUDA device")


def _validate_shadows(arrays, shadows, device):
    shadows = [None] * len(arrays) if shadows is None else list(shadows)
    if len(shadows) != len(arrays):
        raise ValueError("FP16 shadow sequences must match their FP32 parameter sequences")
    for array, shadow in zip(arrays, shadows, strict=True):
        if shadow is not None and (
            shadow.dtype != wp.float16
            or shadow.device != device
            or not shadow.is_contiguous
            or shadow.shape != array.shape
        ):
            raise ValueError("Shadows must be contiguous FP16 arrays matching the FP32 shape and CUDA device")
    return shadows


class _Plan:
    """Keep pointer metadata and the referenced buffers alive through graph replay."""

    def __init__(self, arrays):
        self._arrays = list(arrays)
        if not self._arrays:
            raise ValueError("A fused operation requires at least one array")
        self.device = self._arrays[0].device
        if not self.device.is_cuda:
            raise ValueError("Native FlashSAC operations require CUDA")
        _validate(self._arrays, self.device)
        self._library = self._handle = None

    def _created(self, handle):
        if not handle:
            error = self._library.rl_last_error()
            raise RuntimeError(error.decode(errors="replace") if error else "Native FlashSAC plan creation failed")
        self._handle = handle

    def step(self):
        """Enqueue one fused operation; metadata must be prepared before capture."""
        self.prepare()
        with wp.ScopedDevice(self.device):
            check(
                self._library,
                getattr(self._library, f"rl_flash_{self._kind}_step")(self._handle, stream_pointer(self.device)),
                f"Native FlashSAC {self._kind}",
            )

    def close(self):
        """Release metadata after all dependent graphs and device work are discarded."""
        if not self._handle:
            return
        if self.device.is_capturing:
            raise RuntimeError("Cannot destroy a FlashSAC plan during capture")
        with wp.ScopedDevice(self.device):
            check(
                self._library,
                getattr(self._library, f"rl_flash_{self._kind}_destroy")(self._handle),
                f"Destroy FlashSAC {self._kind}",
            )
        self._handle = None

    def __del__(self):
        if sys.is_finalizing():
            return
        try:
            self.close()
        except Exception:
            pass


class CudaProjection(_Plan):
    """Normalize every supplied unit-linear row, BN affine vector and RMS scale.

    Each row uses ``w / max(norm(w), 1e-8)``. BN weight and bias share one
    ``sqrt(C) / sqrt(sum(w*w + b*b) + 1e-8)`` factor; RMS scales use the same
    factor without a bias. Cooperative reductions change FP32 summation order.
    Call after every scheduled optimizer attempt, including skipped AMP steps.

    Optional ``row_shadows`` mirror the supplied matrices in FP16. The same
    kernel rounds each final FP32 projected weight into its persistent shadow.
    The caller remains responsible for refreshing shadows after direct loads.
    """

    _kind = "projection"

    def __init__(self, *, rows=(), affines=(), scales=(), row_shadows=None):
        self.rows, self.affines, self.scales = list(rows), list(affines), list(scales)
        arrays = self.rows + [array for pair in self.affines for array in pair] + self.scales
        super().__init__(arrays)
        if any(array.ndim != 2 or min(array.shape) < 1 for array in self.rows):
            raise ValueError("Unit-linear weights must be nonempty matrices")
        if any(a.ndim != 1 or a.shape != b.shape or a.size == 0 for a, b in self.affines):
            raise ValueError("BN affine projections require matching nonempty vectors")
        if any(array.ndim != 1 or array.size == 0 for array in self.scales):
            raise ValueError("RMS scale projections require nonempty vectors")
        self.row_shadows = _validate_shadows(self.rows, row_shadows, self.device)

    def prepare(self):
        """Copy immutable row/vector pointer descriptors outside capture."""
        if self._handle:
            return self
        if self.device.is_capturing:
            raise RuntimeError("Prepare CudaProjection before graph capture")
        descriptors = [
            (
                array.ptr + row * array.shape[1] * 4,
                0,
                array.shape[1],
                0,
                shadow.ptr + row * array.shape[1] * 2 if shadow is not None else 0,
            )
            for array, shadow in zip(self.rows, self.row_shadows, strict=True)
            for row in range(array.shape[0])
        ]
        descriptors += [(a.ptr, b.ptr, a.size, 1, 0) for a, b in self.affines]
        descriptors += [(array.ptr, 0, array.size, 2, 0) for array in self.scales]
        count = len(descriptors)
        columns = list(zip(*descriptors, strict=True))
        first = (ctypes.c_void_p * count)(*columns[0])
        second = (ctypes.c_void_p * count)(*columns[1])
        lengths = (ctypes.c_size_t * count)(*columns[2])
        kinds = (ctypes.c_int * count)(*columns[3])
        with wp.ScopedDevice(self.device):
            self._library = _library(self.device)
            if any(shadow is not None for shadow in self.row_shadows):
                shadows = (ctypes.c_void_p * count)(*columns[4])
                self._created(
                    self._library.rl_flash_projection_create_with_shadows(count, first, second, lengths, kinds, shadows)
                )
            else:
                self._created(self._library.rl_flash_projection_create(count, first, second, lengths, kinds))
        return self


class CudaEMA(_Plan):
    """Update all target parameters as ``(1-tau)*target + tau*source`` in one launch.

    Supply parameter arrays only when matching FlashSAC target updates; running
    normalization statistics remain governed by the network forward operation.
    Optional ``target_shadows`` receive the FP16 rounding of each final target
    parameter in the same launch. Direct target loads still require a refresh.
    """

    _kind = "ema"

    def __init__(self, sources, targets, tau, *, target_shadows=None):
        self.sources, self.targets = list(sources), list(targets)
        self.tau = float(tau)
        if not self.sources or len(self.sources) != len(self.targets):
            raise ValueError("EMA requires matching nonempty source and target sequences")
        if not math.isfinite(self.tau) or not 0 <= self.tau <= 1:
            raise ValueError("EMA tau must lie in [0, 1]")
        if any(a.shape != b.shape or a.size == 0 for a, b in zip(self.sources, self.targets, strict=True)):
            raise ValueError("EMA source and target shapes must match")
        super().__init__(self.sources + self.targets)
        self.target_shadows = _validate_shadows(self.targets, target_shadows, self.device)

    def prepare(self):
        """Copy immutable tensor pointers and block descriptors outside capture."""
        if self._handle:
            return self
        if self.device.is_capturing:
            raise RuntimeError("Prepare CudaEMA before graph capture")
        count = len(self.sources)
        sources = (ctypes.c_void_p * count)(*(array.ptr for array in self.sources))
        targets = (ctypes.c_void_p * count)(*(array.ptr for array in self.targets))
        lengths = (ctypes.c_size_t * count)(*(array.size for array in self.sources))
        with wp.ScopedDevice(self.device):
            self._library = _library(self.device)
            if any(shadow is not None for shadow in self.target_shadows):
                shadows = (ctypes.c_void_p * count)(
                    *(array.ptr if array is not None else 0 for array in self.target_shadows)
                )
                self._created(
                    self._library.rl_flash_ema_create_with_shadows(count, sources, targets, lengths, self.tau, shadows)
                )
            else:
                self._created(self._library.rl_flash_ema_create(count, sources, targets, lengths, self.tau))
        return self
