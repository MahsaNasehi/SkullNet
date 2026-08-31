"""Generate held-out slice scores and study metrics for one fixed fold."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from fracture.data.annotations import resolve_annotation
from fracture.data.dicom import load_study
from fracture.data.metadata import load_metadata
from fracture.evaluation.series_metrics import evaluate
from fracture.inference.aggregation import aggregation_features
from fracture.inference.slice_predictor import predict_slices
from fracture.models.detector import Detector
from fracture.utils.config import load_config, require_path


def _best_f1_threshold(y_true: list[int], probability: list[float]) -> tuple[float, dict[str, float]]:
    """Report an exploratory threshold; never reuse it as an unbiased test estimate."""
    candidates = sorted({0.0, 0.5, 1.0, *map(float, probability)})
    scored = [(threshold, evaluate(y_true, probability, threshold)) for threshold in candidates]
    threshold, metrics = max(
        scored,
        key=lambda item: (item[1]["f1_at_threshold"], item[1]["sensitivity_at_threshold"], -item[0]),
    )
    return threshold, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--name", default="fold_predictions")
    parser.add_argument(
        "--detection-confidence",
        type=float,
        help="Override the detector confidence floor while retaining raw boxes for threshold-sensitivity analysis.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_path = Path(cfg["split"].get("path", "splits/folds.json"))
    folds = json.loads(split_path.read_text(encoding="utf-8"))["folds"]
    selected = next((item for item in folds if int(item["fold"]) == args.fold), None)
    if selected is None:
        raise ValueError(f"Fold {args.fold} is not present in {split_path}")

    data = cfg["data"]
    metadata = load_metadata(require_path(cfg, "data", "metadata_path"))
    series_col = data["series_id_column"]
    patient_col = data["patient_id_column"]
    label_col = data["fracture_label_column"]
    study_labels = {
        str(series): int(bool(group[label_col].max()))
        for series, group in metadata.groupby(series_col)
    }
    patient_ids = {
        str(series): str(group[patient_col].iloc[0])
        for series, group in metadata.groupby(series_col)
    }

    inference = cfg["inference"]
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    load_started = time.perf_counter()
    detector = Detector(
        args.weights,
        confidence=float(
            args.detection_confidence
            if args.detection_confidence is not None
            else inference["detection_confidence"]
        ),
        iou=float(inference["nms_iou"]),
        device=cfg["training"]["device"],
        fp16=bool(inference["fp16"]),
        image_size=int(cfg["preprocessing"]["image_size"]),
    )
    model_load_seconds = time.perf_counter() - load_started
    dicom_root = require_path(cfg, "data", "dicom_root")
    prep = cfg["preprocessing"]
    rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for series_id in map(str, selected["val_series"]):
        if series_id not in study_labels:
            raise KeyError(f"Missing study label for validation series {series_id}")
        study_started = time.perf_counter()
        records = load_study(dicom_root / series_id)
        predictions = predict_slices(
            records,
            detector,
            input_mode=prep["input_mode"],
            window_level=float(prep["window_level"]),
            window_width=float(prep["window_width"]),
            batch_size=int(inference["batch_size"]),
        )
        scores = [item.max_confidence for item in predictions]
        counts = [item.num_detections for item in predictions]
        features = aggregation_features(scores, counts, thresholds=(0.05, 0.1, 0.3, 0.5))
        study_seconds = time.perf_counter() - study_started
        rows.append({
            "series_id": series_id,
            "patient_id": patient_ids[series_id],
            "y_true": study_labels[series_id],
            "fold": args.fold,
            "slice_scores": ";".join(f"{value:.8g}" for value in scores),
            "detection_counts": ";".join(map(str, counts)),
            "num_slices": len(records),
            "study_seconds": study_seconds,
            **features,
        })
        for record, prediction in zip(records, predictions, strict=True):
            annotation = resolve_annotation(
                require_path(cfg, "data", "annotation_root"),
                cfg["data"].get("corrected_annotation_root"),
                series_id,
                record.sop_uid,
                image_shape=record.shape,
            )
            gt_boxes = [
                [box.x, box.y, box.x + box.width, box.y + box.height]
                for box in (annotation.boxes if annotation else ())
            ]
            slice_rows.append({
                "series_id": series_id,
                "patient_id": patient_ids[series_id],
                "sop_uid": record.sop_uid,
                "slice_index": prediction.slice_index,
                "physical_position": prediction.physical_position,
                "image_height": record.shape[0],
                "image_width": record.shape[1],
                "study_y_true": study_labels[series_id],
                "gt_boxes": json.dumps(gt_boxes),
                "boxes": json.dumps(prediction.boxes),
                "scores": json.dumps(prediction.scores),
                "max_confidence": prediction.max_confidence,
                "num_detections": prediction.num_detections,
            })

    y_true = [int(row["y_true"]) for row in rows]
    raw = [float(row["max_confidence"]) for row in rows]
    metrics = evaluate(y_true, raw, threshold=0.5)
    best_threshold, exploratory = _best_f1_threshold(y_true, raw)
    study_times = np.asarray([float(row["study_seconds"]) for row in rows], dtype=float)
    metrics.update({
        "fold": args.fold,
        "num_studies": len(rows),
        "num_positive_studies": int(sum(y_true)),
        "elapsed_seconds": time.perf_counter() - started,
        "model_load_seconds": model_load_seconds,
        "mean_study_seconds": float(study_times.mean()),
        "median_study_seconds": float(np.median(study_times)),
        "p95_study_seconds": float(np.percentile(study_times, 95)),
        "exploratory_best_f1_threshold": best_threshold,
        "exploratory_best_f1": exploratory["f1_at_threshold"],
        "exploratory_sensitivity": exploratory["sensitivity_at_threshold"],
        "exploratory_specificity": exploratory["specificity_at_threshold"],
    })
    try:
        import torch

        metrics["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except ImportError:
        metrics["peak_vram_bytes"] = 0

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{args.name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / f"{args.name}_slices.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(slice_rows[0]))
        writer.writeheader()
        writer.writerows(slice_rows)
    (output / f"{args.name}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
