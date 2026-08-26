"""OpenCV-only annotation rendering for headless QC."""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np


def render(image: np.ndarray, boxes: list[tuple[float, float, float, float]], title: str, output: str | Path) -> None:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for x, y, w, h in boxes: cv2.rectangle(canvas, (round(x), round(y)), (round(x+w), round(y+h)), (0, 255, 0), 2)
    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas): raise OSError(f"Failed to write {path}")
