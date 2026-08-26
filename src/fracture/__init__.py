"""IAAA skull-fracture detection and study-level inference."""
from __future__ import annotations
from typing import Any

__all__ = ["FracturePredictor"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Keep environment diagnostics importable when ML dependencies are absent."""
    if name == "FracturePredictor":
        from .predictor import FracturePredictor
        return FracturePredictor
    raise AttributeError(name)
