"""Offline Ultralytics detector wrapper."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import numpy as np


class Detector:
    def __init__(
        self,
        weights: str | Path,
        confidence: float = 0.05,
        iou: float = 0.5,
        device: Any = 0,
        fp16: bool = True,
        image_size: int = 640,
    ):
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"Local detector weights not found: {path}")
        if path.stat().st_size < 100_000:
            raise ValueError(f"Detector checkpoint is empty, truncated, or implausibly small: {path}")
        # Final inference is always self-contained. Development-time checkpoint
        # acquisition is handled separately by fracture.utils.pretrained.
        os.environ["YOLO_OFFLINE"] = "1"
        from ultralytics import YOLO
        self.model = YOLO(str(path)); self.confidence, self.iou, self.device, self.fp16, self.image_size = confidence, iou, device, fp16, int(image_size)

    def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        results = self.model.predict(source=images, conf=self.confidence, iou=self.iou, imgsz=self.image_size, device=self.device, half=self.fp16, verbose=False, stream=False)
        output = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0: output.append({"boxes": [], "scores": [], "max_confidence": 0.0, "num_detections": 0}); continue
            boxes = result.boxes.xyxy.detach().cpu().numpy().tolist(); scores = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
            output.append({"boxes": boxes, "scores": scores, "max_confidence": max(scores, default=0.0), "num_detections": len(scores)})
        return output
