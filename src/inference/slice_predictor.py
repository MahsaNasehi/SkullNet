"""Batched slice preprocessing and detection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from fracture.data.windows import make_hu_input


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
    context_distance_mm: float | None = None,
    verifier: Any = None,
) -> list[SlicePrediction]:
    """Run the detector over every slice; optionally rescore proposals with an ROI verifier.

    ``verifier`` (a ``fracture.verifier.infer.ROIVerifierInference``, or
    any object exposing the same ``rescore_slice(image, boxes, scores) ->
    (scores, max_confidence)`` interface) is applied per-slice on exactly
    the same three-channel windowed array the detector saw, so verifier
    crops use identical preprocessing (review's "preprocessing equality"
    concern) rather than a separately-recomputed image.
    """
    hu_images = [x.hu for x in records]
    monochrome1 = [x.photometric_interpretation == "MONOCHROME1" for x in records]
    physical_positions = [x.physical_position for x in records]
    images = [
        make_hu_input(
            hu_images,
            i,
            level=window_level,
            width=window_width,
            mode=input_mode,
            boundary_mode=boundary_mode,
            monochrome1=monochrome1,
            physical_positions=physical_positions,
            context_distance_mm=context_distance_mm,
        )
        for i in range(len(hu_images))
    ]
    output: list[SlicePrediction] = []
    for start in range(0, len(images), batch_size):
        detections = detector.predict_batch(images[start:start + batch_size])
        for offset, detection in enumerate(detections):
            index = start + offset; record = records[index]
            boxes, scores = detection["boxes"], detection["scores"]
            max_confidence = float(detection["max_confidence"])
            if verifier is not None and boxes:
                scores, max_confidence = verifier.rescore_slice(images[index], boxes, scores)
            output.append(SlicePrediction(record.series_id, record.sop_uid, index, record.physical_position, max_confidence, int(detection["num_detections"]), boxes, scores))
    return output
