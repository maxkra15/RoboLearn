"""FlashSAC from Kim, Lee, and coauthors (2026), adapted for robot simulation."""

from typing import TYPE_CHECKING

from .config import FlashSACConfig

if TYPE_CHECKING:
    from .agent import FlashSAC

__all__ = ["FlashSAC", "FlashSACConfig"]


def __getattr__(name: str):
    if name == "FlashSAC":
        from .agent import FlashSAC

        return FlashSAC
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
