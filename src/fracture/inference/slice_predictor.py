"""Batched slice preprocessing and detection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from fracture.data.windows import bone_window, make_input


@dataclass(frozen=True)
class SlicePrediction:
    series_id: str; sop_uid: str; slice_index: int; physical_position: float | None
    max_confidence: float; num_detections: int; boxes: list[list[float]]; scores: list[float]


def predict_slices(
    records: list[Any],
    detector: Any,
    input_mode: str = "single",
    window_level: float = 500,
    window_width: float = 2500,
    batch_size: int = 8,
    boundary_mode: str = "repeat",
) -> list[SlicePrediction]:
    windows = [bone_window(x.hu, window_level, window_width, x.photometric_interpretation == "MONOCHROME1") for x in records]
    images = [make_input(windows, i, input_mode, boundary_mode=boundary_mode) for i in range(len(windows))]
    output: list[SlicePrediction] = []
    for start in range(0, len(images), batch_size):
        detections = detector.predict_batch(images[start:start + batch_size])
        for offset, detection in enumerate(detections):
            index = start + offset; record = records[index]
            output.append(SlicePrediction(record.series_id, record.sop_uid, index, record.physical_position, float(detection["max_confidence"]), int(detection["num_detections"]), detection["boxes"], detection["scores"]))
    return output
