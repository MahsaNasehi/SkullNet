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


def _neighbor_index(index: int, offset: int, length: int, boundary_mode: str) -> int:
    candidate = index + offset
    if 0 <= candidate < length:
        return candidate
    if boundary_mode == "repeat":
        return 0 if candidate < 0 else length - 1
    if boundary_mode == "mirror":
        if length == 1:
            return 0
        reflected = candidate
        while reflected < 0 or reflected >= length:
            if reflected < 0:
                reflected = -reflected
            else:
                reflected = 2 * (length - 1) - reflected
        return reflected
    raise ValueError(f"Unsupported boundary_mode: {boundary_mode}")


def make_input(
    images: list[np.ndarray],
    index: int,
    mode: str = "single",
    boundary_mode: str = "repeat",
) -> np.ndarray:
    if not images or not 0 <= index < len(images):
        raise IndexError(index)
    if mode == "single":
        return np.repeat(images[index][..., None], 3, axis=2)
    if mode != "2.5d":
        raise ValueError(f"Unsupported input mode: {mode}")
    left = _neighbor_index(index, -1, len(images), boundary_mode)
    right = _neighbor_index(index, 1, len(images), boundary_mode)
    return np.stack((images[left], images[index], images[right]), axis=2)
