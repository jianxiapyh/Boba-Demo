"""Public Boba runtime imports.

Heavy CUDA/Warp modules are loaded lazily so CPU-only unit tests can import
pure helpers such as :mod:`qqtt.object_selector` without initializing Warp.
"""

from __future__ import annotations


__all__ = ("SpringMassSystemWarp", "InvPhyTrainerWarp")


def __getattr__(name: str):
    if name == "SpringMassSystemWarp":
        from .model import SpringMassSystemWarp

        return SpringMassSystemWarp
    if name == "InvPhyTrainerWarp":
        from .engine import InvPhyTrainerWarp

        return InvPhyTrainerWarp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
