# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Optional native CUDA dense layers and Adam with Warp-owned state."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adam import CudaAdam
    from .linear import CudaLinear

__all__ = ["CudaAdam", "CudaLinear", "build_library"]


def __getattr__(name: str):
    if name == "build_library":
        from ._runtime import build_library

        return build_library
    if name == "CudaLinear":
        from .linear import CudaLinear

        return CudaLinear
    if name == "CudaAdam":
        from .adam import CudaAdam

        return CudaAdam
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
