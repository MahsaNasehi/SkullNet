"""Training-fold-only HU analysis for reproducible bone-window selection.

Bounding boxes identify regions that contain fracture and adjacent cortex, not
pixel-accurate fracture masks.  Consequently this tool does not claim to learn
an "optimal" display window.  It estimates robust HU support inside positive
training boxes, reports clipping/contrast for registered candidates, and emits
a conservative data-derived starting window without touching validation data.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from fracture.utils.config import load_config, require_path

from .annotations import resolve_annotation
from .dicom import load_study


DEFAULT_CANDIDATES: tuple[tuple[float, float], ...] = (
    (500.0, 2500.0),
    (600.0, 2800.0),
    (925.0, 2050.0),
    (1125.0, 450.0),
    (950.0, 1900.0),
    (550.0, 1350.0),
    (700.0, 850.0),
    (800.0, 1600.0),
)


def _parse_candidate(value: str) -> tuple[float, float]:
    try:
        level, width = (float(part.strip()) for part in value.split(":", 1))
    except Exception as exc:
        raise argparse.ArgumentTypeError("candidate must be LEVEL:WIDTH") from exc
    if width <= 0:
        raise argparse.ArgumentTypeError("window width must be positive")
    return level, width


def _rounded_up(value: float, step: int = 100) -> float:
    return float(math.ceil(value / step) * step)


def _window_values(values: np.ndarray, level: float, width: float) -> np.ndarray:
    low, high = level - width / 2.0, level + width / 2.0
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def analyse(config: dict, fold_id: int, candidates: list[tuple[float, float]] | None = None) -> dict:
    split_path = Path(config.get("split", {}).get("path", "splits/folds.json"))
    folds = json.loads(split_path.read_text(encoding="utf-8"))["folds"]
    fold = next((item for item in folds if int(item["fold"]) == fold_id), None)
    if fold is None:
        raise ValueError(f"Fold {fold_id} is not present in {split_path}")

    train_series = set(map(str, fold["train_series"]))
    data = config["data"]
    dicom_root = require_path(config, "data", "dicom_root")
    original = require_path(config, "data", "annotation_root")
    corrected = data.get("corrected_annotation_root")
    bone_parts: list[np.ndarray] = []
    dark_parts: list[np.ndarray] = []
    per_box_bone: list[np.ndarray] = []
    box_count = positive_slice_count = 0

    for series_id in sorted(train_series):
        for record in load_study(dicom_root / series_id):
            annotation = resolve_annotation(
                original,
                corrected,
                series_id,
                record.sop_uid,
                image_shape=record.shape,
            )
            if not annotation or not annotation.boxes:
                continue
            positive_slice_count += 1
            for box in annotation.boxes:
                x0 = max(0, int(math.floor(box.x)))
                y0 = max(0, int(math.floor(box.y)))
                x1 = min(record.shape[1], int(math.ceil(box.x + box.width)))
                y1 = min(record.shape[0], int(math.ceil(box.y + box.height)))
                crop = record.hu[y0:y1, x0:x1]
                if not crop.size:
                    continue
                # A deliberately permissive threshold includes low-density
                # cortical/partial-volume pixels around the fracture line.
                bone = crop[crop >= 150.0].astype(np.float32, copy=False)
                dark = crop[(crop >= -200.0) & (crop < 150.0)].astype(np.float32, copy=False)
                if bone.size:
                    bone_parts.append(bone)
                    per_box_bone.append(bone)
                if dark.size:
                    dark_parts.append(dark)
                box_count += 1

    if not bone_parts:
        raise ValueError(f"No positive fracture-box pixels found in training side of fold {fold_id}")
    bone = np.concatenate(bone_parts)
    dark = np.concatenate(dark_parts) if dark_parts else np.asarray([], dtype=np.float32)
    bone_quantiles = np.percentile(bone, [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9])
    dark_quantiles = np.percentile(dark, [1, 5, 25, 50, 75, 95, 99]) if dark.size else np.asarray([])

    # Preserve almost all cortical values while mapping air/soft-tissue gaps to
    # the dark end.  Rounding makes the selected window stable and auditable.
    recommended_low = 0.0
    recommended_high = max(1000.0, _rounded_up(float(bone_quantiles[9])))
    recommended = {
        "window_level": (recommended_low + recommended_high) / 2.0,
        "window_width": recommended_high - recommended_low,
        "lower_hu": recommended_low,
        "upper_hu": recommended_high,
        "rationale": "[0, rounded training-box bone HU p99] preserves cortical contrast with limited high clipping",
    }

    registered = list(dict.fromkeys([*(candidates or DEFAULT_CANDIDATES), (recommended["window_level"], recommended["window_width"])]))
    candidate_rows = []
    for level, width in registered:
        low, high = level - width / 2.0, level + width / 2.0
        mapped_bone = _window_values(bone, level, width)
        mapped_dark = _window_values(dark, level, width) if dark.size else np.asarray([0.0])
        per_box_std = [float(_window_values(values, level, width).std()) for values in per_box_bone]
        candidate_rows.append(
            {
                "window_level": level,
                "window_width": width,
                "lower_hu": low,
                "upper_hu": high,
                "bone_low_clipped_fraction": float(np.mean(bone <= low)),
                "bone_high_clipped_fraction": float(np.mean(bone >= high)),
                "bone_dynamic_std": float(mapped_bone.std()),
                "median_box_bone_dynamic_std": float(np.median(per_box_std)),
                "bone_to_dark_mean_separation": float(mapped_bone.mean() - mapped_dark.mean()),
            }
        )

    return {
        "protocol": {
            "fold": fold_id,
            "selection_partition": "train_only",
            "train_studies": len(train_series),
            "positive_slices": positive_slice_count,
            "fracture_boxes": box_count,
            "annotation_version": data.get("annotation_version"),
            "bone_pixel_definition_hu": ">=150",
            "dark_pixel_definition_hu": "[-200,150)",
            "warning": "boxes are region supervision, not pixel masks; confirm the recommendation with OOF metrics",
        },
        "hu_statistics": {
            "bone_pixels": int(bone.size),
            "bone_quantile_labels": ["p0.1", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p99.9"],
            "bone_quantiles": bone_quantiles.tolist(),
            "dark_pixels": int(dark.size),
            "dark_quantile_labels": ["p1", "p5", "p25", "p50", "p75", "p95", "p99"],
            "dark_quantiles": dark_quantiles.tolist(),
        },
        "recommended_fixed_window": recommended,
        "recommended_train_jitter": {
            "level_delta": 150.0,
            "width_fraction": 0.20,
            "minimum_width": 1000.0,
            "validation_is_fixed": True,
        },
        "candidate_metrics": candidate_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--candidate", action="append", type=_parse_candidate)
    parser.add_argument("--output", default="reports/hu_window_analysis.json")
    args = parser.parse_args()
    payload = analyse(load_config(args.config), args.fold, args.candidate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
