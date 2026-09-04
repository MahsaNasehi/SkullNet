"""CT intensity conversion and single/2.5D image construction.

All model inputs are derived from physical Hounsfield units.  The helpers in
this module deliberately keep windowing deterministic at validation time while
allowing dataset generation to create reproducible train-only window variants.
"""
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


def window_bounds(level: float, width: float) -> tuple[float, float]:
    """Return inclusive HU display bounds for a level/width pair."""
    if width <= 0:
        raise ValueError("Window width must be positive")
    return float(level - width / 2.0), float(level + width / 2.0)


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


def context_indices(
    index: int,
    length: int,
    *,
    boundary_mode: str = "repeat",
    physical_positions: list[float | None] | None = None,
    context_distance_mm: float | None = None,
) -> tuple[int, int, int]:
    """Select left/centre/right slices, optionally by physical z distance.

    Index adjacency remains the backward-compatible default.  When a positive
    physical distance is supplied and all positions are known, the closest
    available slice to ``centre +/- distance`` is selected.  Search is limited
    to the correct side of the centre so duplicated/irregular positions cannot
    reverse the anatomical context.
    """
    if length <= 0 or not 0 <= index < length:
        raise IndexError(index)
    if context_distance_mm is None:
        return (
            _neighbor_index(index, -1, length, boundary_mode),
            index,
            _neighbor_index(index, 1, length, boundary_mode),
        )
    if context_distance_mm <= 0:
        raise ValueError("context_distance_mm must be positive")
    if physical_positions is None or len(physical_positions) != length or any(x is None for x in physical_positions):
        return context_indices(index, length, boundary_mode=boundary_mode)

    positions = np.asarray(physical_positions, dtype=float)
    centre = float(positions[index])

    def closest(indices: range, target: float, fallback_offset: int) -> int:
        candidates = list(indices)
        if not candidates:
            return _neighbor_index(index, fallback_offset, length, boundary_mode)
        return min(candidates, key=lambda candidate: abs(float(positions[candidate]) - target))

    left = closest(range(0, index), centre - context_distance_mm, -1)
    right = closest(range(index + 1, length), centre + context_distance_mm, 1)
    return left, index, right


def make_input(
    images: list[np.ndarray],
    index: int,
    mode: str = "single",
    boundary_mode: str = "repeat",
    physical_positions: list[float | None] | None = None,
    context_distance_mm: float | None = None,
) -> np.ndarray:
    if not images or not 0 <= index < len(images):
        raise IndexError(index)
    if mode == "single":
        return np.repeat(images[index][..., None], 3, axis=2)
    if mode != "2.5d":
        raise ValueError(f"Unsupported input mode: {mode}")
    left, centre, right = context_indices(
        index,
        len(images),
        boundary_mode=boundary_mode,
        physical_positions=physical_positions,
        context_distance_mm=context_distance_mm,
    )
    return np.stack((images[left], images[centre], images[right]), axis=2)


def make_hu_input(
    hu_images: list[np.ndarray],
    index: int,
    *,
    level: float,
    width: float,
    mode: str = "single",
    boundary_mode: str = "repeat",
    monochrome1: list[bool] | None = None,
    physical_positions: list[float | None] | None = None,
    context_distance_mm: float | None = None,
) -> np.ndarray:
    """Window HU arrays and construct a three-channel detector input.

    A single level/width is applied to all three neighbours in one 2.5D sample.
    This is important for window-jitter augmentation: channel differences then
    represent anatomy across z rather than inconsistent intensity mappings.
    """
    if not hu_images or not 0 <= index < len(hu_images):
        raise IndexError(index)
    flags = monochrome1 or [False] * len(hu_images)
    if len(flags) != len(hu_images):
        raise ValueError("monochrome1 flags must match hu_images")
    if mode == "single":
        image = bone_window(hu_images[index], level, width, flags[index])
        return np.repeat(image[..., None], 3, axis=2)
    if mode != "2.5d":
        raise ValueError(f"Unsupported input mode: {mode}")
    indices = context_indices(
        index,
        len(hu_images),
        boundary_mode=boundary_mode,
        physical_positions=physical_positions,
        context_distance_mm=context_distance_mm,
    )
    return np.stack(
        [bone_window(hu_images[j], level, width, flags[j]) for j in indices],
        axis=2,
    )
