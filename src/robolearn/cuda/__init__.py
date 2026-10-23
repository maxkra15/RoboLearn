# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Optional native CUDA learning operations with Warp-owned state."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adam import CudaAdam
    from .flash import CudaEMA, CudaProjection
    from .linear import CudaLinear
    from .normalization import CudaBatchNorm, CudaRMSNorm

__all__ = ["CudaAdam", "CudaLinear", "CudaBatchNorm", "CudaRMSNorm", "CudaProjection", "CudaEMA", "build_library"]


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
    if name in ("CudaProjection", "CudaEMA"):
        from .flash import CudaEMA, CudaProjection

        return {"CudaProjection": CudaProjection, "CudaEMA": CudaEMA}[name]
    if name in ("CudaBatchNorm", "CudaRMSNorm"):
        from .normalization import CudaBatchNorm, CudaRMSNorm

        return {"CudaBatchNorm": CudaBatchNorm, "CudaRMSNorm": CudaRMSNorm}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
