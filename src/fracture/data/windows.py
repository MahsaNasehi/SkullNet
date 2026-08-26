"""CT intensity conversion and single/2.5D image construction."""
from __future__ import annotations
import numpy as np


def to_hu(pixels: np.ndarray, slope: float | None = 1.0, intercept: float | None = 0.0) -> np.ndarray:
    return pixels.astype(np.float32) * float(1.0 if slope is None else slope) + float(0.0 if intercept is None else intercept)


def bone_window(hu: np.ndarray, level: float = 500, width: float = 2500, monochrome1: bool = False) -> np.ndarray:
    if width <= 0:
        raise ValueError("Window width must be positive")
    low, high = level - width / 2.0, level + width / 2.0
    image = ((np.clip(hu, low, high) - low) * (255.0 / (high - low))).round().astype(np.uint8)
    return 255 - image if monochrome1 else image


def make_input(images: list[np.ndarray], index: int, mode: str = "single") -> np.ndarray:
    if not images or not 0 <= index < len(images):
        raise IndexError(index)
    if mode == "single":
        return np.repeat(images[index][..., None], 3, axis=2)
    if mode != "2.5d":
        raise ValueError(f"Unsupported input mode: {mode}")
    return np.stack((images[max(0, index - 1)], images[index], images[min(len(images) - 1, index + 1)]), axis=2)

