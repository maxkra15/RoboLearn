# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Optional Warp learners; FlashSAC remains an experimental FP32 backend."""

from typing import TYPE_CHECKING

from .config import PPOConfig

if TYPE_CHECKING:
    from .flashsac import WarpFlashSAC
    from .ppo import WarpPPO

__all__ = ["PPOConfig", "WarpFlashSAC", "WarpPPO"]


def __getattr__(name: str):
    if name == "WarpFlashSAC":
        from .flashsac import WarpFlashSAC

        return WarpFlashSAC
    if name == "WarpPPO":
        from .ppo import WarpPPO

        return WarpPPO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
