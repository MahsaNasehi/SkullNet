"""Competition annotation parsing with corrected-label precedence."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass(frozen=True)
class Box:
    x: float; y: float; width: float; height: float


@dataclass(frozen=True)
class Annotation:
    boxes: tuple[Box, ...]
    source_path: Path
    provenance: str
    raw: dict[str, Any]


def _find_boxes(value: Any) -> list[Any] | None:
    if isinstance(value, dict):
        for key in ("boxes_xywh", "boxes", "bboxes", "bounding_boxes"):
            if key in value and isinstance(value[key], list): return value[key]
        for child in value.values():
            found = _find_boxes(child)
            if found is not None: return found
    return None


def parse_annotation(path: str | Path, image_shape: tuple[int, int] | None = None, clip: bool = False, provenance: str = "original") -> Annotation:
    path = Path(path)
    with path.open(encoding="utf-8") as stream: raw = json.load(stream)
    values = _find_boxes(raw)
    if values is None: raise ValueError(f"No recognized boxes array in {path}")
    boxes: list[Box] = []
    for i, item in enumerate(values):
        coords = item.get("bbox", item.get("box")) if isinstance(item, dict) else item
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError(f"Invalid XYWH box #{i} in {path}: {coords!r}")
        x, y, w, h = map(float, coords)
        if w <= 0 or h <= 0: raise ValueError(f"Non-positive box #{i} in {path}")
        if image_shape:
            rows, cols = image_shape
            if clip:
                x2, y2 = min(cols, max(0.0, x + w)), min(rows, max(0.0, y + h))
                x, y = min(cols, max(0.0, x)), min(rows, max(0.0, y)); w, h = x2 - x, y2 - y
                if w <= 0 or h <= 0: raise ValueError(f"Box #{i} lies outside image in {path}")
            elif x < 0 or y < 0 or x + w > cols or y + h > rows:
                raise ValueError(f"Out-of-bounds box #{i} in {path}")
        boxes.append(Box(x, y, w, h))
    return Annotation(tuple(boxes), path, provenance, raw)


def resolve_annotation(original_root: str | Path, corrected_root: str | Path | None, series_id: str, sop_uid: str, **kwargs: Any) -> Annotation | None:
    original = Path(original_root) / series_id / f"{sop_uid}.json"
    corrected = Path(corrected_root) / series_id / f"{sop_uid}.json" if corrected_root else None
    if corrected and corrected.exists(): return parse_annotation(corrected, provenance="corrected", **kwargs)
    if original.exists(): return parse_annotation(original, provenance="original", **kwargs)
    return None


def xywh_to_yolo(box: Box, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0: raise ValueError("Image dimensions must be positive")
    result = ((box.x + box.width / 2) / image_width, (box.y + box.height / 2) / image_height, box.width / image_width, box.height / image_height)
    if not all(0 <= value <= 1 for value in result): raise ValueError(f"YOLO coordinates outside [0,1]: {result}")
    return result
