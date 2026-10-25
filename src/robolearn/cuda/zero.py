# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""One capture-safe clear for an immutable collection of FP32 gradient spans."""

import ctypes
import sys

import warp as wp

from ._runtime import check, library_for, stream_pointer


def _library(device):
    library = library_for(device)
    if not getattr(library, "_zero_declared", False):
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        library.rl_zero_create.argtypes = [size, ctypes.POINTER(pointer), ctypes.POINTER(size)]
        library.rl_zero_create.restype = pointer
        library.rl_zero_step.argtypes = [pointer, pointer]
        library.rl_zero_step.restype = ctypes.c_int
        library.rl_zero_destroy.argtypes = [pointer]
        library.rl_zero_destroy.restype = ctypes.c_int
        library._zero_declared = True
    return library


class CudaZero:
    """Clear exactly the supplied adjoints, merging overlapping storage aliases.

    Preparation copies a bounded work list to CUDA; step allocates nothing and
    never reads device values. Owners and views remain alive until this plan
    and all dependent graphs are discarded. No optimizer state is inferred:
    supply the complete collection that an existing Tape.zero() would clear.
    """

    @staticmethod
    def supports(arrays, device):
        """Whether direct FP32 spans preserve each supplied array's zero_()."""
        if not device.is_cuda or wp._src.context._get_apic_capture_for_device(device) is not None:
            return False
        return all(
            isinstance(array, wp.array) and array.dtype == wp.float32 and array.device == device and array.is_contiguous
            for array in arrays
        )

    @staticmethod
    def _signature(arrays):
        # Include duplicate spans in the signature, preserving the tape's full
        # inventory while keeping merged device work free of overlapping writes.
        return tuple(sorted((int(array.ptr or 0), array.size * 4) for array in arrays))

    def __init__(self, arrays):
        self._arrays = tuple(arrays)
        if not self._arrays or not isinstance(self._arrays[0], wp.array):
            raise ValueError("CudaZero requires a nonempty sequence of Warp arrays")
        self.device = self._arrays[0].device
        if not self.supports(self._arrays, self.device):
            raise ValueError("CudaZero requires contiguous FP32 arrays on one CUDA device")
        self.signature = self._signature(self._arrays)
        ranges = []
        for address, size in self.signature:
            if not size:
                continue
            if not address:
                raise ValueError("A nonempty zero span requires allocated storage")
            end = address + size
            if ranges and address < ranges[-1][1]:
                ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
            else:
                ranges.append((address, end))
        if not ranges:
            raise ValueError("CudaZero requires at least one nonempty array")
        self._ranges = tuple(ranges)
        self.array_count = len(self._arrays)
        self.range_count = len(self._ranges)
        self.total_bytes = sum(end - start for start, end in self._ranges)
        self._library = self._handle = None

    def matches(self, arrays):
        """Check current pointer/extent inventory before reusing immutable metadata."""
        arrays = tuple(arrays)
        return self.supports(arrays, self.device) and self._signature(arrays) == self.signature

    def prepare(self):
        """Copy immutable pointer/chunk metadata outside graph capture."""
        if self._handle:
            return self
        if self.device.is_capturing:
            raise RuntimeError("Prepare CudaZero before capture")
        count = len(self._ranges)
        pointers = (ctypes.c_void_p * count)(*(start for start, _ in self._ranges))
        lengths = (ctypes.c_size_t * count)(*((end - start) // 4 for start, end in self._ranges))
        with wp.ScopedDevice(self.device):
            self._library = _library(self.device)
            self._handle = self._library.rl_zero_create(count, pointers, lengths)
            if not self._handle:
                message = self._library.rl_last_error()
                raise RuntimeError(message.decode(errors="replace") if message else "Create CudaZero failed")
        return self

    def step(self, arrays=None):
        """Enqueue the clear and preserve zero_() access bookkeeping for all views."""
        arrays = self._arrays if arrays is None else tuple(arrays)
        if not self.matches(arrays):
            raise RuntimeError("CudaZero gradient pointer/size signature changed")
        self.prepare()
        with wp.ScopedDevice(self.device):
            check(
                self._library,
                self._library.rl_zero_step(self._handle, stream_pointer(self.device)),
                "Grouped CUDA gradient clear",
            )
        for array in arrays:
            array.mark_init()

    def close(self):
        """Release metadata after dependent graphs and work have finished."""
        if not self._handle:
            return
        if self.device.is_capturing:
            raise RuntimeError("Cannot destroy CudaZero during capture")
        with wp.ScopedDevice(self.device):
            check(self._library, self._library.rl_zero_destroy(self._handle), "Destroy CudaZero")
        self._handle = None

    def __del__(self):
        if sys.is_finalizing():
            return
        try:
            self.close()
        except Exception:
            pass
