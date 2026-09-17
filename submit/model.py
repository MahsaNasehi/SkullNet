"""IAAA 2026 Brain CT Triage — fracture detector submission (single-file).

Self-contained offline inference:
  study_dir (DICOM) -> HU window (WL=800, WW=1600) -> 2.5D YOLO detector
  -> top10_percent_mean aggregation -> Macro-F1 operating-point rescale
  -> fracture_prob

Official evaluator:
  from model import Model
  Model().predict(study_dir) -> dict[str, float]

Weights: models/best.pt
  YOLO26s-P2 Run A FULL169 fixed-59-epoch final_deployment.pt refit.
  No individual Fold checkpoint was proven to be the best final model.

Important: ICH and MLS remain zero-valued placeholders in this fracture-only
artifact. The selection metric used ground-truth ICH/MLS and is therefore not
the final submission Macro-F1 of this standalone seven-output implementation.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

WINDOW_LEVEL = 800.0
WINDOW_WIDTH = 1600.0
IMAGE_SIZE = 768
INPUT_MODE = "2.5d"
CONTEXT_DISTANCE_MM = 5.0
BATCH_SIZE = 16
CONFIDENCE = 0.01
NMS_IOU = 0.5
MAX_DETECTIONS = 300
AGGREGATION_METHOD = "top10_percent_mean"
DEFAULT_WEIGHTS_NAME = "best.pt"

# The official triage rule treats fracture_prob >= 0.5 as fracture-present.
# On the authoritative 169-Study full-study OOF cache, apparent deployment
# selection chose top10_percent_mean and this raw operating point by the
# official-style oracle-other-head three-class Macro-F1. The unbiased pooled
# cross-fitted estimate is reported separately; this all-OOF deployment point
# must not itself be described as unbiased performance. The mapping moves the
# selected raw point to the evaluator's fixed 0.5 fracture cutoff while
# retaining a continuous, monotonic score.
RAW_MACRO_F1_THRESHOLD = 0.39717610677083337
OFFICIAL_FRACTURE_THRESHOLD = 0.5

# Official intermediate schema. This package predicts fracture only;
# ICH volumes and MLS are filled with 0.0 until those heads are available.
INTERMEDIATE_KEYS = (
    "V_EDH",
    "V_SDH",
    "V_IPH",
    "V_SAH",
    "V_IVH",
    "fracture_prob",
    "MLS_mm",
)


# ---------------------------------------------------------------------------
# HU / 2.5D windowing
# ---------------------------------------------------------------------------


def to_hu(pixels: np.ndarray, slope: float | None = 1.0, intercept: float | None = 0.0) -> np.ndarray:
    return pixels.astype(np.float32) * float(1.0 if slope is None else slope) + float(
        0.0 if intercept is None else intercept
    )


def bone_window(hu: np.ndarray, level: float, width: float, monochrome1: bool = False) -> np.ndarray:
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
    raise ValueError(f"Unsupported boundary_mode: {boundary_mode}")


def context_indices(
    index: int,
    length: int,
    *,
    boundary_mode: str = "repeat",
    physical_positions: list[float | None] | None = None,
    context_distance_mm: float | None = None,
) -> tuple[int, int, int]:
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


def make_hu_input(
    hu_images: list[np.ndarray],
    index: int,
    *,
    level: float,
    width: float,
    mode: str = "2.5d",
    boundary_mode: str = "repeat",
    monochrome1: list[bool] | None = None,
    physical_positions: list[float | None] | None = None,
    context_distance_mm: float | None = None,
) -> np.ndarray:
    if not hu_images or not 0 <= index < len(hu_images):
        raise IndexError(index)
    flags = monochrome1 or [False] * len(hu_images)
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
    return np.stack([bone_window(hu_images[j], level, width, flags[j]) for j in indices], axis=2)


# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceRecord:
    path: Path
    series_id: str
    sop_uid: str
    instance_number: int | None
    image_position: tuple[float, ...] | None
    image_orientation: tuple[float, ...] | None
    physical_position: float | None
    pixel_spacing: tuple[float, ...] | None
    slice_thickness: float | None
    shape: tuple[int, int]
    hu: np.ndarray
    transfer_syntax_uid: str | None = None
    photometric_interpretation: str = "MONOCHROME2"


def _tuple(ds: Any, name: str) -> tuple[float, ...] | None:
    value = getattr(ds, name, None)
    return tuple(float(x) for x in value) if value is not None else None


def physical_position(orientation: tuple[float, ...] | None, position: tuple[float, ...] | None) -> float | None:
    if not orientation or len(orientation) != 6 or not position or len(position) != 3:
        return None
    row, col = np.asarray(orientation[:3]), np.asarray(orientation[3:])
    return float(np.dot(np.cross(row, col), np.asarray(position)))


def _pixel_array(ds: Any, path: Path) -> tuple[np.ndarray, bool]:
    try:
        return np.asarray(ds.pixel_array), False
    except Exception as first:
        try:
            import SimpleITK as sitk

            image = sitk.ReadImage(str(path))
            array = sitk.GetArrayFromImage(image)
            return np.asarray(array[0] if array.ndim == 3 else array), True
        except Exception as second:
            syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "unknown")
            uid = getattr(ds, "SOPInstanceUID", "unknown")
            raise RuntimeError(
                f"Cannot decode DICOM SOP={uid}, transfer_syntax={syntax}, path={path}: "
                f"pydicom={first}; SimpleITK={second}"
            ) from second


def read_slice(path: str | Path, series_id: str | None = None) -> SliceRecord:
    import pydicom
    from pydicom import config as pydicom_config

    path = Path(path)
    previous_validation = pydicom_config.settings.reading_validation_mode
    pydicom_config.settings.reading_validation_mode = pydicom_config.IGNORE
    try:
        ds = pydicom.dcmread(path)
        pixels, already_hu = _pixel_array(ds, path)
        if pixels.ndim != 2:
            raise ValueError(f"Expected 2D DICOM pixel data, got {pixels.shape}: {path}")
        orientation = _tuple(ds, "ImageOrientationPatient")
        position = _tuple(ds, "ImagePositionPatient")
        syntax = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "")) or None
        hu = (
            pixels.astype(np.float32)
            if already_hu
            else to_hu(pixels, getattr(ds, "RescaleSlope", 1), getattr(ds, "RescaleIntercept", 0))
        )
        return SliceRecord(
            path,
            series_id or path.parent.name,
            str(getattr(ds, "SOPInstanceUID", path.stem)),
            int(ds.InstanceNumber) if hasattr(ds, "InstanceNumber") else None,
            position,
            orientation,
            physical_position(orientation, position),
            _tuple(ds, "PixelSpacing"),
            float(ds.SliceThickness) if hasattr(ds, "SliceThickness") else None,
            tuple(pixels.shape),
            hu,
            syntax,
            str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")),
        )
    finally:
        pydicom_config.settings.reading_validation_mode = previous_validation


def sort_records(records: list[SliceRecord]) -> list[SliceRecord]:
    if records and all(x.physical_position is not None for x in records):
        return sorted(records, key=lambda x: float(x.physical_position))
    if records and all(x.instance_number is not None for x in records):
        LOGGER.warning("Falling back to InstanceNumber ordering for series %s", records[0].series_id)
        return sorted(records, key=lambda x: int(x.instance_number))
    raise ValueError("Cannot establish slice order: physical positions and InstanceNumber are incomplete")


def load_study(study_dir: str | Path, *, target_series_uid: str | None = None) -> list[SliceRecord]:
    import pydicom
    from pydicom import config as pydicom_config

    root = Path(study_dir)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and (p.suffix.lower() == ".dcm" or not p.suffix))
    if not paths:
        raise FileNotFoundError(f"No DICOM files found in {root}")
    # Full-study expansion means all slices of the selected target series,
    # including slices without metadata/annotation JSON. Never mix series.
    by_series: dict[str, list[Path]] = {}
    previous_validation = pydicom_config.settings.reading_validation_mode
    pydicom_config.settings.reading_validation_mode = pydicom_config.IGNORE
    try:
        for path in paths:
            header = pydicom.dcmread(path, stop_before_pixels=True)
            uid = str(getattr(header, "SeriesInstanceUID", ""))
            if not uid:
                raise RuntimeError(f"DICOM lacks SeriesInstanceUID: {path}")
            by_series.setdefault(uid, []).append(path)
    finally:
        pydicom_config.settings.reading_validation_mode = previous_validation
    if target_series_uid is None:
        if len(by_series) != 1:
            raise RuntimeError("Multiple DICOM series in Study; specify target_series_uid")
        target_series_uid = next(iter(by_series))
    if target_series_uid not in by_series:
        raise RuntimeError(f"Target SeriesInstanceUID not found: {target_series_uid}")
    records: list[SliceRecord] = []
    failures: list[str] = []
    for path in by_series[target_series_uid]:
        try:
            records.append(read_slice(path, root.name))
        except Exception as exc:
            failures.append(f"{path}: {exc}")
    if not records:
        raise RuntimeError("Entire study failed DICOM decoding:\n" + "\n".join(failures))
    if failures:
        raise RuntimeError("Study was only partially decoded; refusing silent omission:\n" + "\n".join(failures))
    sop_uids = [record.sop_uid for record in records]
    if len(sop_uids) != len(set(sop_uids)):
        raise RuntimeError("Duplicate SOPInstanceUID in target DICOM series")
    return sort_records(records)


# ---------------------------------------------------------------------------
# Detector + aggregation
# ---------------------------------------------------------------------------


class Detector:
    def __init__(
        self,
        weights: str | Path,
        confidence: float = CONFIDENCE,
        iou: float = NMS_IOU,
        device: Any = 0,
        fp16: bool = True,
        image_size: int = IMAGE_SIZE,
    ):
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"Local detector weights not found: {path}")
        if path.stat().st_size < 100_000:
            raise ValueError(f"Detector checkpoint is empty, truncated, or implausibly small: {path}")
        # The evaluator may have no network access and a read-only home.  These
        # variables must be set before importing Ultralytics (it reads both at
        # import time).  Ultralytics expects the literal string ``"true"`` for
        # YOLO_OFFLINE; ``"1"`` is not treated as offline.
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "Ultralytics"))
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.fp16 = fp16
        self.image_size = int(image_size)

    def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=images,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            half=self.fp16,
            max_det=MAX_DETECTIONS,
            verbose=False,
            stream=False,
        )
        output: list[dict[str, Any]] = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                output.append({"boxes": [], "scores": [], "max_confidence": 0.0, "num_detections": 0})
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
            scores = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
            output.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "max_confidence": max(scores, default=0.0),
                    "num_detections": len(scores),
                }
            )
        return output


@dataclass(frozen=True)
class SlicePrediction:
    series_id: str
    sop_uid: str
    slice_index: int
    physical_position: float | None
    max_confidence: float
    num_detections: int
    boxes: list[list[float]]
    scores: list[float]


def predict_slices(
    records: list[SliceRecord],
    detector: Detector,
    *,
    input_mode: str = INPUT_MODE,
    window_level: float = WINDOW_LEVEL,
    window_width: float = WINDOW_WIDTH,
    batch_size: int = BATCH_SIZE,
    context_distance_mm: float | None = CONTEXT_DISTANCE_MM,
) -> list[SlicePrediction]:
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
            boundary_mode="repeat",
            monochrome1=monochrome1,
            physical_positions=physical_positions,
            context_distance_mm=context_distance_mm,
        )
        for i in range(len(hu_images))
    ]
    output: list[SlicePrediction] = []
    for start in range(0, len(images), batch_size):
        detections = detector.predict_batch(images[start : start + batch_size])
        for offset, detection in enumerate(detections):
            index = start + offset
            record = records[index]
            output.append(
                SlicePrediction(
                    record.series_id,
                    record.sop_uid,
                    index,
                    record.physical_position,
                    float(detection["max_confidence"]),
                    int(detection["num_detections"]),
                    detection["boxes"],
                    detection["scores"],
                )
            )
    return output


def aggregate_top3_mean(scores: list[float]) -> float:
    array = np.asarray(scores, dtype=float)
    if array.size == 0:
        return 0.0
    ordered = np.sort(array)[::-1]
    return float(np.clip(ordered[:3].mean(), 0.0, 1.0))


def aggregate_max(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return float(np.clip(max(scores), 0.0, 1.0))


def aggregate_top10_percent_mean(scores: list[float]) -> float:
    array = np.asarray(scores, dtype=float)
    if array.size == 0:
        return 0.0
    ordered = np.sort(array)[::-1]
    count = max(1, int(np.ceil(0.10 * ordered.size)))
    return float(np.clip(ordered[:count].mean(), 0.0, 1.0))


def rescale_for_macro_f1(
    probability: float,
    raw_threshold: float = RAW_MACRO_F1_THRESHOLD,
) -> float:
    """Continuously map ``raw_threshold`` to the official cutoff of 0.5.

    The transformation is monotonic and anchored at (0, 0),
    (raw_threshold, 0.5), and (1, 1), so ranking is unchanged.
    """
    threshold = float(raw_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("raw_threshold must be strictly between 0 and 1")
    value = float(np.clip(probability, 0.0, 1.0))
    if value < threshold:
        return float(OFFICIAL_FRACTURE_THRESHOLD * value / threshold)
    return float(
        OFFICIAL_FRACTURE_THRESHOLD
        + (1.0 - OFFICIAL_FRACTURE_THRESHOLD)
        * (value - threshold)
        / (1.0 - threshold)
    )


def resolve_weights(weights: str | Path | None = None) -> Path:
    if weights is not None:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"Weights not found: {path}")
        return path.resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        here / DEFAULT_WEIGHTS_NAME,
        here / "models" / DEFAULT_WEIGHTS_NAME,
        here / "models" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {DEFAULT_WEIGHTS_NAME} next to model.py. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def pick_device(device: Any | None = None) -> Any:
    if device is not None:
        return device
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Official Model API
# ---------------------------------------------------------------------------


class Model:
    """Competition entrypoint: Model().predict(study_dir) -> intermediates."""

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        device: Any | None = None,
        batch_size: int = BATCH_SIZE,
        confidence: float = CONFIDENCE,
        aggregation_method: str = AGGREGATION_METHOD,
        optimize_macro_f1: bool = True,
        raw_macro_f1_threshold: float = RAW_MACRO_F1_THRESHOLD,
        fp16: bool | None = None,
        target_series_uid: str | None = None,
    ):
        self.weights = resolve_weights(weights)
        self.device = pick_device(device)
        self.batch_size = int(batch_size)
        self.confidence = float(confidence)
        self.aggregation_method = aggregation_method
        self.optimize_macro_f1 = bool(optimize_macro_f1)
        self.raw_macro_f1_threshold = float(raw_macro_f1_threshold)
        self.fp16 = bool(self.device != "cpu" if fp16 is None else fp16)
        self.target_series_uid = target_series_uid
        self.detector = Detector(
            self.weights,
            confidence=self.confidence,
            iou=NMS_IOU,
            device=self.device,
            fp16=self.fp16,
            image_size=IMAGE_SIZE,
        )
        self.last_slice_predictions: list[SlicePrediction] = []
        self.last_raw_fracture_prob = 0.0

    def predict_fracture_prob(self, study_dir: str) -> float:
        records = load_study(study_dir, target_series_uid=self.target_series_uid)
        predictions = predict_slices(
            records,
            self.detector,
            input_mode=INPUT_MODE,
            window_level=WINDOW_LEVEL,
            window_width=WINDOW_WIDTH,
            batch_size=self.batch_size,
            context_distance_mm=CONTEXT_DISTANCE_MM,
        )
        self.last_slice_predictions = predictions
        scores = [item.max_confidence for item in predictions]
        if self.aggregation_method == "max":
            probability = aggregate_max(scores)
        elif self.aggregation_method == "top3_mean":
            probability = aggregate_top3_mean(scores)
        elif self.aggregation_method == "top10_percent_mean":
            probability = aggregate_top10_percent_mean(scores)
        else:
            raise ValueError(f"Unsupported aggregation_method: {self.aggregation_method}")
        self.last_raw_fracture_prob = float(probability)
        if self.optimize_macro_f1:
            probability = rescale_for_macro_f1(probability, self.raw_macro_f1_threshold)
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"Invalid fracture probability: {probability}")
        return float(probability)

    def predict(self, study_dir: str) -> dict[str, float]:
        fracture_prob = self.predict_fracture_prob(study_dir)
        return {
            "V_EDH": 0.0,
            "V_SDH": 0.0,
            "V_IPH": 0.0,
            "V_SAH": 0.0,
            "V_IVH": 0.0,
            "fracture_prob": fracture_prob,
            "MLS_mm": 0.0,
        }


# ---------------------------------------------------------------------------
# Optional local CLI: write series_id,fracture_prob CSV
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fracture inference and write predictions CSV.")
    parser.add_argument("--data-dir", required=True, help="Parent directory of DICOM study folders.")
    parser.add_argument("--predictions-file-path", required=True, help="Output CSV path.")
    parser.add_argument("--weights", default=None, help="Optional override for best.pt.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model = Model(weights=args.weights, device=args.device, batch_size=args.batch_size)
    root = Path(args.data_dir)
    studies = sorted(path for path in root.iterdir() if path.is_dir())
    if args.limit is not None:
        studies = studies[: max(0, args.limit)]

    out = Path(args.predictions_file_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["series_id", *INTERMEDIATE_KEYS])
        writer.writeheader()
        for study in studies:
            result = model.predict(str(study))
            writer.writerow({"series_id": study.name, **result})
            print(
                f"{study.name}: fracture_prob={result['fracture_prob']:.6f} "
                f"(raw={model.last_raw_fracture_prob:.6f})"
            )
    print(f"Wrote {len(studies)} rows -> {out.resolve()}")


if __name__ == "__main__":
    main()
