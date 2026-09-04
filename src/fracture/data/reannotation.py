"""Build an OOF-assisted human review queue without changing any label."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from fracture.data.dicom import load_study
from fracture.data.windows import make_hu_input
from fracture.utils.config import load_config, require_path

FIELDS = [
    "series_id",
    "sop_uid",
    "slice_index",
    "fold",
    "prediction_protocol",
    "detector_checkpoint_sha256",
    "config_sha256",
    "slice_status",
    "candidate_type",
    "model_confidence",
    "gt_num_boxes",
    "max_iou",
    "reason",
    "status",
    "reviewer",
    "notes",
]
LOG_FIELDS = [
    "series_id",
    "sop_uid",
    "slice_index",
    "original_num_boxes",
    "corrected_num_boxes",
    "modification_type",
    "reason",
    "reviewer",
    "timestamp",
    "status",
    "notes",
]


def _boxes(value: Any) -> list[list[float]]:
    if value in (None, ""):
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    return [[float(coordinate) for coordinate in box] for box in parsed]


def _iou(first: list[float], second: list[float]) -> float:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def _maximum_iou(gt_boxes: list[list[float]], predicted_boxes: list[list[float]]) -> float:
    return max((_iou(gt, predicted) for gt in gt_boxes for predicted in predicted_boxes), default=0.0)


def _candidate(
    row: dict[str, Any],
    candidate_type: str,
    reason: str,
    max_iou: float,
) -> dict[str, Any]:
    return {
        key: row.get(key, "")
        for key in (
            "series_id",
            "sop_uid",
            "slice_index",
            "fold",
            "prediction_protocol",
            "detector_checkpoint_sha256",
            "config_sha256",
            "slice_status",
        )
    } | {
        "candidate_type": candidate_type,
        "model_confidence": float(row.get("max_confidence", 0) or 0),
        "gt_num_boxes": int(float(row.get("gt_num_boxes", 0) or 0)),
        "max_iou": max_iou,
        "reason": reason,
        "status": "pending",
        "reviewer": "",
        "notes": "",
    }


def candidates_from_rows(
    rows: list[dict[str, Any]],
    high_conf: float = 0.30,
    miss_conf: float = 0.05,
    tiny_area: float = 100.0,
    large_area: float = 50_000.0,
    suspicious_aspect_ratio: float = 10.0,
) -> list[dict[str, Any]]:
    """Return review candidates; predictions are proposals, never labels."""
    if not rows:
        return []
    required = {"series_id", "sop_uid", "slice_index", "fold", "prediction_protocol", "gt_num_boxes", "max_confidence"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"OOF slice CSV is missing columns: {sorted(missing)}")

    folds_by_series: dict[str, set[str]] = defaultdict(set)
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("prediction_protocol", "")) != "heldout_patient_fold":
            raise ValueError("Review candidates must come from held-out patient-fold predictions")
        series_id = str(row["series_id"])
        folds_by_series[series_id].add(str(row["fold"]))
        by_series[series_id].append(row)
    leaked = sorted(series for series, folds in folds_by_series.items() if len(folds) != 1)
    if leaked:
        raise ValueError(f"A study appears in multiple OOF folds: {leaked[:10]}")

    output: list[dict[str, Any]] = []
    for study_rows in by_series.values():
        ordered = sorted(study_rows, key=lambda item: int(item["slice_index"]))
        study_has_box = any(int(float(row.get("gt_num_boxes", 0) or 0)) > 0 for row in ordered)
        study_positive = any(int(float(row.get("study_y_true", 0) or 0)) == 1 for row in ordered)
        if study_positive and not study_has_box:
            row = max(ordered, key=lambda item: float(item.get("max_confidence", 0) or 0))
            output.append(_candidate(row, "positive_study_without_boxes", "Positive study metadata has no fracture box", 0.0))

        positive_indices = {
            int(row["slice_index"])
            for row in ordered
            if int(float(row.get("gt_num_boxes", 0) or 0)) > 0
        }
        for row in ordered:
            count = int(float(row.get("gt_num_boxes", 0) or 0))
            confidence = float(row.get("max_confidence", 0) or 0)
            gt_boxes, predicted_boxes = _boxes(row.get("gt_boxes")), _boxes(row.get("boxes"))
            max_iou = _maximum_iou(gt_boxes, predicted_boxes)
            if count == 0 and confidence >= high_conf:
                output.append(_candidate(row, "possible_missing_box", "High-confidence OOF prediction has no GT box", max_iou))
                if str(row.get("slice_status", "")) == "negative":
                    output.append(_candidate(row, "hard_negative_or_false_positive", "Verified-negative slice triggered the detector", max_iou))
            if count > 0 and confidence < miss_conf:
                output.append(_candidate(row, "missed_gt", "OOF detector missed an annotated fracture", max_iou))
            if count > 0 and predicted_boxes and max_iou < 0.10:
                output.append(_candidate(row, "localization_disagreement", "Prediction and GT have IoU below 0.10", max_iou))

            height = float(row.get("image_height", 0) or 0)
            width = float(row.get("image_width", 0) or 0)
            areas = []
            for box in gt_boxes:
                box_width, box_height = box[2] - box[0], box[3] - box[1]
                area = max(0.0, box_width) * max(0.0, box_height)
                areas.append(area)
                if box_width <= 0 or box_height <= 0 or box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                    output.append(_candidate(row, "invalid_box_geometry", "GT box is degenerate or outside the image", max_iou))
                ratio = max(box_width, box_height) / max(1e-6, min(box_width, box_height))
                if ratio >= suspicious_aspect_ratio:
                    output.append(_candidate(row, "suspicious_aspect_ratio", f"GT box aspect ratio is {ratio:.2f}", max_iou))
            if areas and min(areas) < tiny_area:
                output.append(_candidate(row, "tiny_box", f"GT box area is below {tiny_area:g} px²", max_iou))
            if areas and max(areas) > large_area:
                output.append(_candidate(row, "large_outlier_box", f"GT box area exceeds {large_area:g} px²", max_iou))
            if any(_iou(first, second) >= 0.50 for i, first in enumerate(gt_boxes) for second in gt_boxes[i + 1 :]):
                output.append(_candidate(row, "overlapping_gt_boxes", "Two GT boxes overlap with IoU >= 0.50", max_iou))

            index = int(row["slice_index"])
            if count == 0 and index - 1 in positive_indices and index + 1 in positive_indices:
                output.append(_candidate(row, "neighbor_annotation_gap", "Unboxed slice lies between two boxed slices", max_iou))

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in output:
        key = (str(row["series_id"]), str(row["sop_uid"]), str(row["candidate_type"]))
        unique[key] = row
    return sorted(
        unique.values(),
        key=lambda row: (
            str(row["candidate_type"]),
            -float(row["model_confidence"]),
            str(row["series_id"]),
            int(row["slice_index"]),
        ),
    )


def write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _visualize(candidates: list[dict[str, Any]], prediction_rows: list[dict[str, Any]], config: dict, root: Path) -> None:
    categories = {
        "possible_missing_box": "possible_missing_boxes",
        "hard_negative_or_false_positive": "possible_false_boxes",
        "missed_gt": "possible_false_boxes",
    }
    for directory in (
        "model_gt_disagreement",
        "possible_missing_boxes",
        "possible_false_boxes",
        "uncertain",
        "accepted_corrections",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    rows_by_key = {(str(row["series_id"]), str(row["sop_uid"])): row for row in prediction_rows}
    candidates_by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_series[str(candidate["series_id"])].append(candidate)
    prep = config["preprocessing"]
    dicom_root = require_path(config, "data", "dicom_root")
    for series_id, study_candidates in candidates_by_series.items():
        records = load_study(dicom_root / series_id)
        record_index = {record.sop_uid: index for index, record in enumerate(records)}
        hu_images = [record.hu for record in records]
        positions = [record.physical_position for record in records]
        flags = [record.photometric_interpretation == "MONOCHROME1" for record in records]
        for candidate in study_candidates:
            sop_uid = str(candidate["sop_uid"])
            if sop_uid not in record_index:
                continue
            index = record_index[sop_uid]
            image = make_hu_input(
                hu_images,
                index,
                level=float(prep["window_level"]),
                width=float(prep["window_width"]),
                mode="single",
                monochrome1=flags,
                physical_positions=positions,
            )
            row = rows_by_key[(series_id, sop_uid)]
            for box in _boxes(row.get("gt_boxes")):
                cv2.rectangle(image, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (0, 255, 0), 2)
            for box in _boxes(row.get("boxes")):
                cv2.rectangle(image, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (0, 128, 255), 2)
            cv2.putText(image, f"GT green | pred orange | {candidate['model_confidence']:.3f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            category = categories.get(str(candidate["candidate_type"]), "model_gt_disagreement")
            safe_sop = "".join(character if character.isalnum() or character in "._-" else "_" for character in sop_uid)
            output = root / category / f"{series_id}__{safe_sop}__{candidate['candidate_type']}.png"
            if not cv2.imwrite(str(output), image):
                raise OSError(f"OpenCV failed to write review visualization: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="OOF *_slices.csv; repeat for every fold")
    parser.add_argument("--output", default="reports/reannotation_candidates.csv")
    parser.add_argument("--log", default="reports/reannotation_log.csv")
    parser.add_argument("--config")
    parser.add_argument("--visualization-root", default="visualizations/reannotation")
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--high-confidence", type=float, default=0.30)
    parser.add_argument("--miss-confidence", type=float, default=0.05)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for filename in args.input:
        with Path(filename).open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    candidates = candidates_from_rows(rows, high_conf=args.high_confidence, miss_conf=args.miss_confidence)
    write_csv(candidates, Path(args.output), FIELDS)
    log_path = Path(args.log)
    if not log_path.exists():
        write_csv([], log_path, LOG_FIELDS)
    if not args.no_visualizations:
        if not args.config:
            raise ValueError("--config is required unless --no-visualizations is set")
        _visualize(candidates, rows, load_config(args.config), Path(args.visualization_root))
    print(f"Created {len(candidates)} human-review candidates; no annotation was modified")


if __name__ == "__main__":
    main()
