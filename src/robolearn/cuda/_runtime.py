# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Explicit NVCC compilation and ctypes ABI; importing this module needs no GPU."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

_LIBRARIES = {}
_LOCK = threading.RLock()


def build_library(*, architecture: int | str, compiler: str | None = None, cache_dir: str | Path | None = None) -> Path:
    """Compile the native source for an explicit SM architecture without using a GPU.

    Requires an NVCC toolkit and its cuBLASLt development libraries. The cache key
    includes source bytes, compiler identity/version, architecture and flags.
    No compiler or mathematical backend fallback is selected on failure.
    """
    architecture = str(architecture).removeprefix("sm_")
    if not re.fullmatch(r"[0-9]{2,3}", architecture):
        raise ValueError("architecture must be an explicit SM number, for example 89")
    requested = compiler or os.environ.get("ROBOLEARN_CUDA_NVCC", "nvcc")
    executable = shutil.which(requested)
    if executable is None:
        raise RuntimeError(f"NVCC compiler not found: {requested}; install a CUDA development toolkit")
    executable = str(Path(executable).resolve())
    version = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True).stdout
    source = Path(__file__).with_name("kernels.cu")
    flags = ["-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC", "--fmad=false", f"-arch=sm_{architecture}"]
    identity = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": executable,
        "compiler_version": version,
        "architecture": architecture,
        "flags": flags,
        "libraries": ["cublasLt"],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir or os.environ.get("ROBOLEARN_CUDA_CACHE_DIR", Path.home() / ".cache/robolearn/cuda"))
    directory = cache.expanduser().resolve() / key
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "librobolearn_cuda.so"
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_file():
            return destination
        descriptor, temporary = tempfile.mkstemp(prefix=".building-", suffix=".so", dir=directory)
        os.close(descriptor)
        try:
            command = [executable, *flags, str(source), "-lcublasLt", "-o", temporary]
            completed = subprocess.run(command, capture_output=True, text=True)
            (directory / "build.json").write_text(json.dumps({**identity, "command": command}, indent=2) + "\n")
            (directory / "build.log").write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"Native CUDA build failed; see {directory / 'build.log'}")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return destination


def _declare(library):
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    size = ctypes.c_size_t
    library.rl_last_error.argtypes = []
    library.rl_last_error.restype = ctypes.c_char_p
    library.rl_plan_create.argtypes = [integer] * 8 + [integer, integer, size]
    library.rl_plan_create.restype = pointer
    library.rl_plan_workspace.argtypes = [pointer]
    library.rl_plan_workspace.restype = size
    library.rl_plan_destroy.argtypes = [pointer]
    library.rl_plan_destroy.restype = integer
    library.rl_plan_run.argtypes = [pointer] * 7 + [size, ctypes.c_float, pointer]
    library.rl_plan_run.restype = integer
    library.rl_bias_grad.argtypes = [pointer, pointer, integer, integer, pointer]
    library.rl_bias_grad.restype = integer
    pointer_table = ctypes.POINTER(pointer)
    library.rl_adam_create.argtypes = (
        [size] + [pointer_table] * 4 + [ctypes.POINTER(size), pointer, pointer] + [ctypes.c_double] * 3
    )
    library.rl_adam_create.restype = pointer
    library.rl_adam_step.argtypes = [pointer, pointer]
    library.rl_adam_step.restype = integer
    library.rl_adam_destroy.argtypes = [pointer]
    library.rl_adam_destroy.restype = integer


def library_for(device):
    """Load the exact native build before capture, with the Warp context current."""
    if not device.is_cuda:
        raise ValueError("The native CUDA backend requires a CUDA device")
    if device.is_capturing:
        raise RuntimeError("Prepare native CUDA operations before entering graph capture")
    path = build_library(architecture=device.arch)
    with _LOCK:
        if path not in _LIBRARIES:
            library = ctypes.CDLL(str(path))
            _declare(library)
            _LIBRARIES[path] = library
        return _LIBRARIES[path]


def check(library, result, operation: str):
    if result:
        message = library.rl_last_error()
        detail = message.decode("utf-8", errors="replace") if message else "no native error message"
        raise RuntimeError(f"{operation}: {detail}")


def pointer(array):
    return ctypes.c_void_p(array.ptr) if array is not None else None


def stream_pointer(device):
    import warp as wp

    return ctypes.c_void_p(wp.get_stream(device).cuda_stream)


class NativePlan:
    """Own a native plan until the layer/optimizer and its graph are discarded."""

    def __init__(self, library, handle, device, *, adam=False):
        if not handle:
            message = library.rl_last_error()
            raise RuntimeError(message.decode("utf-8", errors="replace") if message else "Native plan creation failed")
        self.library = library
        self.handle = handle
        self.device = device
        self.adam = adam

    def close(self):
        """Release outside capture, after dependent graph work has finished."""
        if not self.handle:
            return
        if self.device.is_capturing:
            raise RuntimeError("Cannot destroy a native CUDA plan during capture")
        import warp as wp

        with wp.ScopedDevice(self.device):
            destroy = self.library.rl_adam_destroy if self.adam else self.library.rl_plan_destroy
            check(self.library, destroy(self.handle), "Destroy native CUDA plan")
        self.handle = None

    def __del__(self):
        # Python teardown can run after CUDA/Warp context teardown. The process
        # owns the remaining allocations in that case; avoid re-entering CUDA.
        if sys.is_finalizing():
            return
        try:
            self.close()
        except Exception:
            pass
