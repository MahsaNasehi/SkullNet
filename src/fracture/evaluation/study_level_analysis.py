"""Paired Fold-0 full-study analysis for random and COCO-pretrained detectors.

This module deliberately performs no fitting or calibration. Detector predictions
are retained at a low confidence floor and all diagnostic thresholds and study
aggregators are derived from the same frozen slice predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fracture.data.dicom import load_study
from fracture.data.windows import bone_window
from fracture.evaluation.error_analysis import iou
from fracture.evaluation.predict_fold import _best_f1_threshold
from fracture.evaluation.series_metrics import evaluate
from fracture.inference.aggregation import StudyAggregator, aggregation_features
from fracture.utils.config import load_config, require_path


DETECTOR_THRESHOLDS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
MEANINGFUL_CONFIDENCE = 0.01
MATCH_IOU = 0.5


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required checkpoint is missing: {checkpoint}")
    if checkpoint.stat().st_size < 100_000:
        raise ValueError(f"Required checkpoint is empty/truncated: {checkpoint} ({checkpoint.stat().st_size} bytes)")
    return {
        "path": str(checkpoint),
        "file_size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
    }


def _prediction_files(output: Path, name: str) -> tuple[Path, Path, Path]:
    return (
        output / f"{name}.csv",
        output / f"{name}_slices.csv",
        output / f"{name}_metrics.json",
    )


def _run_fold_prediction(
    *, config: Path, fold: int, weights: Path, output: Path, name: str, reuse: bool
) -> tuple[Path, Path, Path]:
    paths = _prediction_files(output, name)
    if reuse and all(path.is_file() and path.stat().st_size for path in paths):
        return paths
    command = [
        sys.executable,
        "-m",
        "fracture.evaluation.predict_fold",
        "--config",
        str(config),
        "--fold",
        str(fold),
        "--weights",
        str(weights),
        "--output-dir",
        str(output),
        "--name",
        name,
        "--detection-confidence",
        str(DETECTOR_THRESHOLDS[0]),
    ]
    environment = os.environ.copy()
    environment["YOLO_OFFLINE"] = "1"
    subprocess.run(command, check=True, env=environment)
    return paths


def _group_slices(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["series_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["slice_index"]))
    return dict(grouped)


def _filtered_slice(row: dict[str, str], threshold: float) -> tuple[list[list[float]], list[float]]:
    boxes = json.loads(row["boxes"])
    scores = [float(value) for value in json.loads(row["scores"])]
    retained = [(box, score) for box, score in zip(boxes, scores, strict=True) if score >= threshold]
    return [item[0] for item in retained], [item[1] for item in retained]


def _study_features(rows: list[dict[str, str]], threshold: float) -> dict[str, Any]:
    slice_scores: list[float] = []
    counts: list[int] = []
    for row in rows:
        _, scores = _filtered_slice(row, threshold)
        slice_scores.append(max(scores, default=0.0))
        counts.append(len(scores))
    features: dict[str, Any] = aggregation_features(
        slice_scores, counts, thresholds=(0.05, 0.10, 0.30, 0.50)
    )
    best_index = int(np.argmax(slice_scores)) if slice_scores else 0
    features.update({
        "slice_scores": slice_scores,
        "detection_counts": counts,
        "best_slice": int(rows[best_index]["slice_index"]),
        "best_sop_uid": rows[best_index]["sop_uid"],
    })
    return features


def _paired_study_rows(
    baseline_slices: list[dict[str, str]], pretrained_slices: list[dict[str, str]], threshold: float
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    baseline = _group_slices(baseline_slices)
    pretrained = _group_slices(pretrained_slices)
    if set(baseline) != set(pretrained):
        raise ValueError("Baseline and pretrained study sets differ")
    base_features, pre_features, paired = {}, {}, []
    for series_id in sorted(baseline):
        b_rows, p_rows = baseline[series_id], pretrained[series_id]
        if [(r["sop_uid"], r["slice_index"]) for r in b_rows] != [
            (r["sop_uid"], r["slice_index"]) for r in p_rows
        ]:
            raise ValueError(f"Slice sets/order differ for {series_id}")
        b = base_features.setdefault(series_id, _study_features(b_rows, threshold))
        p = pre_features.setdefault(series_id, _study_features(p_rows, threshold))
        label = int(b_rows[0]["study_y_true"])
        patient_id = b_rows[0].get("patient_id", "")
        paired.append({
            "series_id": series_id,
            "patient_id": patient_id,
            "gt_fracture": label,
            "baseline_max_confidence": b["max_confidence"],
            "pretrained_max_confidence": p["max_confidence"],
            "baseline_second_confidence": b["second_confidence"],
            "pretrained_second_confidence": p["second_confidence"],
            "baseline_top2_mean": b["top2_mean"],
            "pretrained_top2_mean": p["top2_mean"],
            "baseline_top3_mean": b["top3_mean"],
            "pretrained_top3_mean": p["top3_mean"],
            "baseline_top5_mean": b["top5_mean"],
            "pretrained_top5_mean": p["top5_mean"],
            "baseline_num_detections": int(b["total_detections"]),
            "pretrained_num_detections": int(p["total_detections"]),
            "baseline_detected_slices": int(b["detected_slices"]),
            "pretrained_detected_slices": int(p["detected_slices"]),
            "baseline_longest_run_0_05": int(b["longest_run_ge_0_05"]),
            "pretrained_longest_run_0_05": int(p["longest_run_ge_0_05"]),
            "baseline_longest_run_0_10": int(b["longest_run_ge_0_1"]),
            "pretrained_longest_run_0_10": int(p["longest_run_ge_0_1"]),
            "baseline_best_slice": b["best_slice"],
            "pretrained_best_slice": p["best_slice"],
            "baseline_best_sop_uid": b["best_sop_uid"],
            "pretrained_best_sop_uid": p["best_sop_uid"],
            "delta_max_confidence": p["max_confidence"] - b["max_confidence"],
        })
    return paired, base_features, pre_features


def _metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    y = [int(row["gt_fracture"]) for row in rows]
    scores = [float(row[f"{prefix}_max_confidence"]) for row in rows]
    result = evaluate(y, scores, threshold=0.5)
    threshold, exploratory = _best_f1_threshold(y, scores)
    result.update({
        "diagnostic_exploratory_best_f1_threshold": threshold,
        "diagnostic_exploratory_f1": exploratory["f1_at_threshold"],
        "diagnostic_exploratory_sensitivity": exploratory["sensitivity_at_threshold"],
        "diagnostic_exploratory_specificity": exploratory["specificity_at_threshold"],
    })
    return result


def _distribution(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, name in ((1, "positive"), (0, "negative")):
        values = np.asarray([
            float(row[f"{prefix}_max_confidence"])
            for row in rows
            if int(row["gt_fracture"]) == label
        ])
        result[name] = {
            "n": len(values),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }
    result["median_separation"] = result["positive"]["median"] - result["negative"]["median"]
    result["mean_separation"] = result["positive"]["mean"] - result["negative"]["mean"]
    return result


def _ranking(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row[f"{prefix}_max_confidence"]), reverse=True)
    positives = sum(int(row["gt_fracture"]) for row in rows)
    positive_ranks = [
        rank for rank, row in enumerate(ordered, 1) if int(row["gt_fracture"]) == 1
    ]
    result: dict[str, Any] = {"positive_ranks": positive_ranks}
    for top in (5, 10, 20):
        found = sum(int(row["gt_fracture"]) for row in ordered[:top])
        result[f"positive_in_top{top}"] = found
        result[f"recall_at_top{top}"] = found / positives if positives else 0.0
    return result


def _positive_rows(
    paired: list[dict[str, Any]], baseline_slices: list[dict[str, str]], pretrained_slices: list[dict[str, str]]
) -> list[dict[str, Any]]:
    base = _group_slices(baseline_slices)
    pre = _group_slices(pretrained_slices)
    output = []
    for study in paired:
        if not int(study["gt_fracture"]):
            continue
        series_id = str(study["series_id"])
        gt_sops = {
            row["sop_uid"] for row in base[series_id] if json.loads(row["gt_boxes"])
        }
        number_boxes = sum(len(json.loads(row["gt_boxes"])) for row in base[series_id])
        item = dict(study)
        item["number_of_GT_fracture_slices"] = len(gt_sops)
        item["number_of_GT_boxes"] = number_boxes
        for name, grouped in (("baseline", base), ("pretrained", pre)):
            rows = grouped[series_id]
            per_gt = []
            meaningful_gt_slices = 0
            for row in rows:
                _, scores = _filtered_slice(row, MEANINGFUL_CONFIDENCE)
                if row["sop_uid"] in gt_sops:
                    score = max(scores, default=0.0)
                    per_gt.append((score, row["sop_uid"], int(row["slice_index"])))
                    meaningful_gt_slices += int(bool(scores))
            best_gt = max(per_gt, default=(0.0, "", -1))
            item[f"{name}_gt_slices_with_prediction_0_01"] = meaningful_gt_slices
            item[f"{name}_best_gt_fracture_confidence"] = best_gt[0]
            item[f"{name}_best_gt_fracture_sop_uid"] = best_gt[1]
            item[f"{name}_best_gt_fracture_slice"] = best_gt[2]
            item[f"{name}_max_on_gt_fracture_slice"] = item[f"{name}_best_sop_uid"] in gt_sops
            item[f"{name}_crosses_0_5"] = float(item[f"{name}_max_confidence"]) >= 0.5
        delta = float(item["delta_max_confidence"])
        item["pretrained_change"] = (
            "approximately_unchanged" if abs(delta) < 0.01 else
            "improved_by_pretrained" if delta > 0 else
            "worsened_by_pretrained"
        )
        output.append(item)
    return sorted(output, key=lambda row: float(row["pretrained_max_confidence"]), reverse=True)


def _top_negative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    negative = [row for row in rows if not int(row["gt_fracture"])]
    baseline_rank = {
        row["series_id"]: rank
        for rank, row in enumerate(sorted(negative, key=lambda x: x["baseline_max_confidence"], reverse=True), 1)
    }
    pretrained_rank = {
        row["series_id"]: rank
        for rank, row in enumerate(sorted(negative, key=lambda x: x["pretrained_max_confidence"], reverse=True), 1)
    }
    selected = [
        dict(row, baseline_negative_rank=baseline_rank[row["series_id"]], pretrained_negative_rank=pretrained_rank[row["series_id"]])
        for row in negative
        if min(baseline_rank[row["series_id"]], pretrained_rank[row["series_id"]]) <= 20
    ]
    return sorted(selected, key=lambda row: max(row["baseline_max_confidence"], row["pretrained_max_confidence"]), reverse=True)


def _box_rows(
    baseline_slices: list[dict[str, str]], pretrained_slices: list[dict[str, str]], image_size: int
) -> list[dict[str, Any]]:
    base = {(r["series_id"], r["sop_uid"]): r for r in baseline_slices}
    pre = {(r["series_id"], r["sop_uid"]): r for r in pretrained_slices}
    if set(base) != set(pre):
        raise ValueError("Slice sets differ during box-size analysis")
    output = []
    for key in sorted(base):
        b, p = base[key], pre[key]
        gt = json.loads(b["gt_boxes"])
        b_boxes, _ = _filtered_slice(b, MEANINGFUL_CONFIDENCE)
        p_boxes, _ = _filtered_slice(p, MEANINGFUL_CONFIDENCE)
        source_width, source_height = float(b["image_width"]), float(b["image_height"])
        for index, target in enumerate(gt):
            width = (target[2] - target[0]) * image_size / source_width
            height = (target[3] - target[1]) * image_size / source_height
            b_iou = max((iou(target, box) for box in b_boxes), default=0.0)
            p_iou = max((iou(target, box) for box in p_boxes), default=0.0)
            output.append({
                "series_id": key[0],
                "sop_uid": key[1],
                "slice_index": int(b["slice_index"]),
                "gt_box_index": index,
                "width_at_768": width,
                "height_at_768": height,
                "area_at_768": width * height,
                "relative_area": width * height / image_size**2,
                "baseline_max_iou_at_conf_0_01": b_iou,
                "baseline_detected_iou_0_5": b_iou >= MATCH_IOU,
                "pretrained_max_iou_at_conf_0_01": p_iou,
                "pretrained_detected_iou_0_5": p_iou >= MATCH_IOU,
            })
    return output


def _threshold_sensitivity(
    baseline_slices: list[dict[str, str]], pretrained_slices: list[dict[str, str]]
) -> list[dict[str, Any]]:
    output = []
    for threshold in DETECTOR_THRESHOLDS:
        paired, _, _ = _paired_study_rows(baseline_slices, pretrained_slices, threshold)
        for name in ("baseline", "pretrained"):
            metrics = _metrics(paired, name)
            output.append({
                "model": name,
                "detector_confidence_threshold": threshold,
                "study_probability_threshold": 0.5,
                "study_auroc": metrics["auroc"],
                "study_pr_auc": metrics["pr_auc"],
                "sensitivity_at_study_0_5": metrics["sensitivity_at_0_5"],
                "total_detections": sum(int(row[f"{name}_num_detections"]) for row in paired),
            })
    return output


def _aggregation_comparison(
    paired: list[dict[str, Any]], base_features: dict[str, dict[str, Any]], pre_features: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    labels = {str(row["series_id"]): int(row["gt_fracture"]) for row in paired}
    output = []
    for name, feature_map in (("baseline", base_features), ("pretrained", pre_features)):
        for method in ("max", "top3_mean", "consecutive"):
            aggregator = StudyAggregator(method=method, min_run=2, run_threshold=0.1)
            series = sorted(feature_map)
            scores = [
                aggregator.predict(feature_map[key]["slice_scores"], feature_map[key]["detection_counts"])
                for key in series
            ]
            metrics = evaluate([labels[key] for key in series], scores, threshold=0.5)
            output.append({
                "model": name,
                "aggregation": method,
                "study_auroc": metrics["auroc"],
                "study_pr_auc": metrics["pr_auc"],
            })
    return output


def _training_metrics(weights: Path) -> dict[str, Any] | None:
    path = weights.parent.parent / "results.csv"
    if not path.is_file():
        return None
    rows = [{key.strip(): float(value) for key, value in row.items()} for row in _read_csv(path)]
    best = max(rows, key=lambda row: row["metrics/mAP50-95(B)"])
    return {
        "results_path": str(path.resolve()),
        "epochs_completed": len(rows),
        "best_epoch": int(best["epoch"]),
        "precision": best["metrics/precision(B)"],
        "recall": best["metrics/recall(B)"],
        "mAP50": best["metrics/mAP50(B)"],
        "mAP50_95": best["metrics/mAP50-95(B)"],
    }


def _plots(output: Path, paired: list[dict[str, Any]], boxes: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for axis, name, title in zip(axes, ("baseline", "pretrained"), ("Random baseline", "COCO pretrained"), strict=True):
        positive = [r[f"{name}_max_confidence"] for r in paired if int(r["gt_fracture"])]
        negative = [r[f"{name}_max_confidence"] for r in paired if not int(r["gt_fracture"])]
        bins = np.linspace(0, max(0.5, *positive, *negative), 21)
        axis.hist(negative, bins=bins, alpha=0.65, label=f"negative n={len(negative)}")
        axis.hist(positive, bins=bins, alpha=0.75, label=f"positive n={len(positive)}")
        axis.axvline(0.5, color="black", linestyle="--", label="official 0.5")
        axis.set_title(title); axis.set_xlabel("raw max confidence"); axis.legend()
    axes[0].set_ylabel("studies")
    fig.tight_layout(); fig.savefig(output / "fold0_score_distribution.png", dpi=180); plt.close(fig)

    if boxes:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for axis, name, title in zip(axes, ("baseline", "pretrained"), ("Random baseline", "COCO pretrained"), strict=True):
            detected = [r["area_at_768"] for r in boxes if r[f"{name}_detected_iou_0_5"]]
            missed = [r["area_at_768"] for r in boxes if not r[f"{name}_detected_iou_0_5"]]
            axis.boxplot([detected or [np.nan], missed or [np.nan]], tick_labels=["detected", "missed"])
            axis.set_title(title); axis.set_ylabel("GT box area at 768² (pixels)")
        fig.tight_layout(); fig.savefig(output / "fold0_box_size_detection.png", dpi=180); plt.close(fig)


def _draw_boxes(image: np.ndarray, boxes: list[list[float]], scores: list[float], color: tuple[int, int, int], label: str):
    import cv2
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = map(lambda value: int(round(value)), box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = label if not scores else f"{label} {scores[index]:.3f}"
        cv2.putText(canvas, text, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return canvas


def _visualize(
    output: Path,
    positive: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    baseline_slices: list[dict[str, str]],
    pretrained_slices: list[dict[str, str]],
    dicom_root: Path,
) -> tuple[int, int]:
    import cv2
    base = {(r["series_id"], r["sop_uid"]): r for r in baseline_slices}
    pre = {(r["series_id"], r["sop_uid"]): r for r in pretrained_slices}
    study_cache: dict[str, dict[str, Any]] = {}

    def record(series_id: str, sop_uid: str):
        if series_id not in study_cache:
            study_cache[series_id] = {r.sop_uid: r for r in load_study(dicom_root / series_id)}
        return study_cache[series_id][sop_uid]

    positive_dir = output / "fold0_error_analysis" / "positive"
    negative_dir = output / "fold0_error_analysis" / "negative"
    positive_dir.mkdir(parents=True, exist_ok=True); negative_dir.mkdir(parents=True, exist_ok=True)
    positive_count = 0
    for item in positive:
        series_id = str(item["series_id"])
        sop_uid = str(item["pretrained_best_gt_fracture_sop_uid"] or item["baseline_best_gt_fracture_sop_uid"])
        if not sop_uid:
            continue
        b, p = base[(series_id, sop_uid)], pre[(series_id, sop_uid)]
        rec = record(series_id, sop_uid)
        image = bone_window(rec.hu, 500, 2500, rec.photometric_interpretation == "MONOCHROME1")
        gt = _draw_boxes(image, json.loads(b["gt_boxes"]), [], (0, 255, 0), "GT")
        b_boxes, b_scores = _filtered_slice(b, DETECTOR_THRESHOLDS[0])
        p_boxes, p_scores = _filtered_slice(p, DETECTOR_THRESHOLDS[0])
        b_canvas = _draw_boxes(image, b_boxes, b_scores, (255, 100, 0), "baseline")
        p_canvas = _draw_boxes(image, p_boxes, p_scores, (0, 0, 255), "pretrained")
        canvas = np.concatenate([gt, b_canvas, p_canvas], axis=1)
        cv2.putText(canvas, f"series={series_id} sop={sop_uid}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(positive_dir / f"{series_id}.png"), canvas):
            raise OSError(f"Failed to save positive visualization for {series_id}")
        positive_count += 1

    negative_count = 0
    for item in negatives:
        series_id = str(item["series_id"])
        panels = []
        for name, index in (("baseline", base), ("pretrained", pre)):
            sop_uid = str(item[f"{name}_best_sop_uid"])
            row = index[(series_id, sop_uid)]
            rec = record(series_id, sop_uid)
            image = bone_window(rec.hu, 500, 2500, rec.photometric_interpretation == "MONOCHROME1")
            boxes, scores = _filtered_slice(row, DETECTOR_THRESHOLDS[0])
            panel = _draw_boxes(image, boxes, scores, (0, 0, 255), name)
            cv2.putText(panel, f"{name} slice={row['slice_index']} SOP={sop_uid}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        if not cv2.imwrite(str(negative_dir / f"{series_id}.png"), np.concatenate(panels, axis=1)):
            raise OSError(f"Failed to save negative visualization for {series_id}")
        negative_count += 1
    return positive_count, negative_count


def _box_summary(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    detected = [r for r in rows if r[f"{model}_detected_iou_0_5"]]
    missed = [r for r in rows if not r[f"{model}_detected_iou_0_5"]]
    def describe(values: list[dict[str, Any]]) -> dict[str, Any]:
        areas = np.asarray([r["area_at_768"] for r in values], dtype=float)
        return {
            "n": len(values),
            "median_area_at_768": float(np.median(areas)) if len(areas) else None,
            "mean_area_at_768": float(np.mean(areas)) if len(areas) else None,
        }
    return {"detected": describe(detected), "missed": describe(missed)}


def _markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    clean = [[str(value) for value in row] for row in rows]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in clean),
    ])


def _report(
    output: Path,
    payload: dict[str, Any],
    positive: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    aggregation_rows: list[dict[str, Any]],
) -> None:
    b, p = payload["study_metrics"]["baseline"], payload["study_metrics"]["pretrained"]
    metric_rows = []
    for label, key in (
        ("AUROC", "auroc"), ("PR-AUC", "pr_auc"), ("Sensitivity@0.5", "sensitivity_at_0_5"),
        ("Specificity@0.5", "specificity_at_0_5"), ("Precision@0.5", "precision_at_0_5"),
        ("F1@0.5", "f1_at_0_5"), ("Brier", "brier"), ("Log-loss", "log_loss"),
        ("TP", "tp"), ("FP", "fp"), ("TN", "tn"), ("FN", "fn"),
    ):
        metric_rows.append([label, f"{b[key]:.6g}" if isinstance(b[key], float) else b[key], f"{p[key]:.6g}" if isinstance(p[key], float) else p[key]])
    positive_table = _markdown_table([
        [r["series_id"], r["patient_id"], r["number_of_GT_fracture_slices"], r["number_of_GT_boxes"],
         f"{r['baseline_max_confidence']:.4f}", f"{r['pretrained_max_confidence']:.4f}",
         r["baseline_max_on_gt_fracture_slice"], r["pretrained_max_on_gt_fracture_slice"], r["pretrained_change"]]
        for r in positive
    ], ["Series", "Patient", "GT slices", "GT boxes", "Baseline max", "COCO max", "Baseline max on GT", "COCO max on GT", "Change"])
    checkpoint_rows = [
        [name, value["path"], value["file_size_bytes"], value["sha256"]]
        for name, value in payload["checkpoints"].items()
    ]
    distribution_rows = []
    for name in ("baseline", "pretrained"):
        for label in ("positive", "negative"):
            d = payload["score_distributions"][name][label]
            distribution_rows.append([name, label, d["n"], f"{d['min']:.4f}", f"{d['median']:.4f}", f"{d['mean']:.4f}", f"{d['max']:.4f}"])
    lines = [
        "# Fold-0 Study-Level Fracture Analysis",
        "",
        "> Frozen Fold-0 validation; complete physically ordered studies; WL=500/WW=2500; 2.5D; max aggregation; no calibration. Exploratory thresholds are not unbiased test performance.",
        "",
        "## A. Checkpoints",
        "",
        _markdown_table(checkpoint_rows, ["Model", "Path", "Bytes", "SHA256"]),
        "",
        "## B. Detector metrics",
        "",
        "```json", json.dumps(payload["detector_metrics"], indent=2), "```",
        "",
        "## C. Study-level comparison",
        "",
        _markdown_table(metric_rows, ["Metric", "Random baseline", "COCO pretrained"]),
        "",
        "### DIAGNOSTIC / EXPLORATORY ONLY: best-F1 thresholds",
        "",
        f"- Baseline: threshold={b['diagnostic_exploratory_best_f1_threshold']:.6g}, F1={b['diagnostic_exploratory_f1']:.6g}, sensitivity={b['diagnostic_exploratory_sensitivity']:.6g}, specificity={b['diagnostic_exploratory_specificity']:.6g}.",
        f"- COCO: threshold={p['diagnostic_exploratory_best_f1_threshold']:.6g}, F1={p['diagnostic_exploratory_f1']:.6g}, sensitivity={p['diagnostic_exploratory_sensitivity']:.6g}, specificity={p['diagnostic_exploratory_specificity']:.6g}.",
        "",
        "## D. Every positive Fold-0 study",
        "",
        positive_table,
        "",
        "## E. Score distributions",
        "",
        _markdown_table(distribution_rows, ["Model", "Class", "N", "Min", "Median", "Mean", "Max"]),
        "",
        "See `fold0_score_distribution.png`.",
        "",
        "## F. Ranking",
        "",
        "```json", json.dumps(payload["ranking"], indent=2), "```",
        "",
        "## G. Slice-level fracture behavior",
        "",
        f"Meaningful slice prediction is defined as confidence >= {MEANINGFUL_CONFIDENCE}; spatial box detection is IoU >= {MATCH_IOU} at that confidence. Positive-study details are in `fold0_positive_studies.csv` and all positive visualizations are under `fold0_error_analysis/positive/`.",
        "",
        "## H. False negatives",
        "",
        f"At the official 0.5 study threshold: baseline FN={b['fn']}, COCO FN={p['fn']}. Inspect the complete positive table rather than relying only on this threshold.",
        "",
        "## I. False positives",
        "",
        f"At 0.5: baseline FP={b['fp']}, COCO FP={p['fp']}. The {len(negatives)}-row union of each model's top-20 negative rankings is in `fold0_top_negative_scores.csv`; images are human-review candidates only.",
        "",
        "## J. Box-size findings",
        "",
        "```json", json.dumps(payload["box_size_summary"], indent=2), "```",
        "",
        "## K. Detector-confidence sensitivity",
        "",
        _markdown_table([[r["model"], r["detector_confidence_threshold"], f"{r['study_auroc']:.4f}", f"{r['study_pr_auc']:.4f}", f"{r['sensitivity_at_study_0_5']:.4f}", r["total_detections"]] for r in threshold_rows], ["Model", "Detector threshold", "AUROC", "PR-AUC", "Sensitivity@study 0.5", "Detections"]),
        "",
        "Detector confidence threshold is not the final fracture probability threshold; the latter remains 0.5.",
        "",
        "## L. Aggregation comparison (DIAGNOSTIC / EXPLORATORY ONLY)",
        "",
        _markdown_table([[r["model"], r["aggregation"], f"{r['study_auroc']:.4f}", f"{r['study_pr_auc']:.4f}"] for r in aggregation_rows], ["Model", "Aggregation", "AUROC", "PR-AUC"]),
        "",
        "## Required conclusions",
        "",
        *[f"### Question {index}\n\n{answer}\n" for index, answer in enumerate(payload["answers"], 1)],
        "## Model decision",
        "",
        payload["decision"],
        "",
    ]
    (output / "FOLD0_STUDY_LEVEL_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def _answers(payload: dict[str, Any]) -> tuple[list[str], str]:
    b = payload["study_metrics"]["baseline"]
    p = payload["study_metrics"]["pretrained"]
    ranking_better = p["auroc"] > b["auroc"] and p["pr_auc"] > b["pr_auc"]
    actual_b = payload["box_size_summary"]["baseline"]["detected"]["n"]
    actual_p = payload["box_size_summary"]["pretrained"]["detected"]["n"]
    overlap = p["auroc"] < 0.7
    threshold_rows = payload["detector_threshold_sensitivity"]
    p_rows = [row for row in threshold_rows if row["model"] == "pretrained"]
    threshold_help = max(row["study_pr_auc"] for row in p_rows) > min(row["study_pr_auc"] for row in p_rows) + 0.02
    box = payload["box_size_summary"]["pretrained"]
    detected_area = box["detected"]["median_area_at_768"]
    missed_area = box["missed"]["median_area_at_768"]
    small_misses = detected_area is not None and missed_area is not None and missed_area < detected_area
    answers = [
        ("Yes." if ranking_better else "No or inconclusive.") + f" Baseline AUROC/PR-AUC={b['auroc']:.4f}/{b['pr_auc']:.4f}; COCO={p['auroc']:.4f}/{p['pr_auc']:.4f}.",
        f"COCO matched {actual_p} GT boxes versus {actual_b} for baseline at confidence >=0.01 and IoU >=0.5; this separates actual fracture localization from confidence-only changes.",
        ("Positive/negative overlap remains substantial, so this is not merely calibration." if overlap else "Ranking separation is useful while the 0.5 operating point is poor, consistent with under-confidence/calibration as an important component."),
        ("Lower detector thresholds materially change ranking signal, indicating weak candidates were being discarded." if threshold_help else "Lower detector thresholds do not materially improve ranking signal; missed fractures are not explained mainly by the inference cutoff."),
        ("Yes; missed-box median area is smaller than detected-box median area." if small_misses else "No clear evidence that false negatives are disproportionately smaller from this fold."),
        "The selected option below is based on study AUROC and PR-AUC first, with slice/box evidence as supporting analysis.",
    ]
    if ranking_better:
        decision = "A. COCO-pretrained V2 should proceed to 5-fold OOF"
    elif b["auroc"] > p["auroc"] and b["pr_auc"] > p["pr_auc"]:
        decision = "B. Random/current V2 should proceed to 5-fold OOF"
    else:
        decision = "C. Result is inconclusive and the corrected local-transfer Fold-0 experiment should be run before choosing"
    return answers, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fracture_25d_p2_pretrained.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--baseline-weights", required=True)
    parser.add_argument("--pretrained-weights", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()

    if args.fold != 0:
        raise ValueError("This controlled report is intentionally restricted to Fold 0")
    config = Path(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "baseline": _checkpoint_metadata(args.baseline_weights),
        "pretrained": _checkpoint_metadata(args.pretrained_weights),
    }
    if checkpoints["baseline"]["sha256"] == checkpoints["pretrained"]["sha256"]:
        raise ValueError("Baseline and pretrained checkpoints are byte-identical; refusing invalid comparison")

    baseline_files = _run_fold_prediction(
        config=config, fold=0, weights=Path(args.baseline_weights), output=output,
        name="fold0_raw_baseline", reuse=args.reuse_predictions,
    )
    pretrained_files = _run_fold_prediction(
        config=config, fold=0, weights=Path(args.pretrained_weights), output=output,
        name="fold0_raw_pretrained", reuse=args.reuse_predictions,
    )
    baseline_studies, baseline_slices = _read_csv(baseline_files[0]), _read_csv(baseline_files[1])
    pretrained_studies, pretrained_slices = _read_csv(pretrained_files[0]), _read_csv(pretrained_files[1])
    if len(baseline_studies) != 68 or len(pretrained_studies) != 68:
        raise RuntimeError(f"Fold-0 must contain 68 complete studies, got {len(baseline_studies)} and {len(pretrained_studies)}")
    if len(baseline_slices) != 1633 or len(pretrained_slices) != 1633:
        raise RuntimeError(f"Fold-0 must contain all 1,633 slices, got {len(baseline_slices)} and {len(pretrained_slices)}")

    paired, base_features, pre_features = _paired_study_rows(
        baseline_slices, pretrained_slices, DETECTOR_THRESHOLDS[0]
    )
    positive = _positive_rows(paired, baseline_slices, pretrained_slices)
    negatives = _top_negative_rows(paired)
    boxes = _box_rows(baseline_slices, pretrained_slices, int(load_config(config)["preprocessing"]["image_size"]))
    thresholds = _threshold_sensitivity(baseline_slices, pretrained_slices)
    aggregations = _aggregation_comparison(paired, base_features, pre_features)
    _write_csv(output / "fold0_study_level_comparison.csv", paired)
    _write_csv(output / "fold0_positive_studies.csv", positive)
    _write_csv(output / "fold0_top_negative_scores.csv", negatives)
    _write_csv(output / "fold0_box_size_analysis.csv", boxes)
    _write_csv(output / "fold0_detector_threshold_sensitivity.csv", thresholds)
    _write_csv(output / "fold0_aggregation_comparison.csv", aggregations)
    _plots(output, paired, boxes)
    cfg = load_config(config)
    positive_visuals, negative_visuals = _visualize(
        output, positive, negatives, baseline_slices, pretrained_slices,
        require_path(cfg, "data", "dicom_root"),
    )
    payload: dict[str, Any] = {
        "protocol": {
            "fold": 0, "studies": len(paired), "positive_studies": len(positive),
            "slices": len(baseline_slices), "aggregation": "max", "calibration": "none",
            "detector_inference_floor": DETECTOR_THRESHOLDS[0], "study_threshold": 0.5,
        },
        "checkpoints": checkpoints,
        "detector_metrics": {
            "baseline": _training_metrics(Path(args.baseline_weights)),
            "pretrained": _training_metrics(Path(args.pretrained_weights)),
        },
        "study_metrics": {"baseline": _metrics(paired, "baseline"), "pretrained": _metrics(paired, "pretrained")},
        "score_distributions": {"baseline": _distribution(paired, "baseline"), "pretrained": _distribution(paired, "pretrained")},
        "ranking": {"baseline": _ranking(paired, "baseline"), "pretrained": _ranking(paired, "pretrained")},
        "positive_studies_crossing_0_5": {
            "baseline": sum(bool(r["baseline_crosses_0_5"]) for r in positive),
            "pretrained": sum(bool(r["pretrained_crosses_0_5"]) for r in positive),
            "total": len(positive),
        },
        "negative_studies_crossing_0_5": {
            "baseline": sum(float(r["baseline_max_confidence"]) >= 0.5 for r in paired if not int(r["gt_fracture"])),
            "pretrained": sum(float(r["pretrained_max_confidence"]) >= 0.5 for r in paired if not int(r["gt_fracture"])),
        },
        "box_size_summary": {"baseline": _box_summary(boxes, "baseline"), "pretrained": _box_summary(boxes, "pretrained")},
        "detector_threshold_sensitivity": thresholds,
        "aggregation_comparison": aggregations,
        "visualizations": {"positive": positive_visuals, "negative": negative_visuals},
    }
    payload["answers"], payload["decision"] = _answers(payload)
    (output / "fold0_study_level_analysis.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _report(output, payload, positive, negatives, thresholds, aggregations)
    print(json.dumps({
        "report": str((output / "FOLD0_STUDY_LEVEL_ANALYSIS.md").resolve()),
        "decision": payload["decision"],
        "study_metrics": payload["study_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
