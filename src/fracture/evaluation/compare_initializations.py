"""Controlled Fold-0 baseline-versus-pretrained detector comparison."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from fracture.data.dicom import load_study
from fracture.data.metadata import study_intermediates
from fracture.data.windows import bone_window
from fracture.evaluation.error_analysis import iou
from fracture.evaluation.isolated_qwk import isolated_fracture_qwk
from fracture.evaluation.predict_fold import _best_f1_threshold
from fracture.evaluation.series_metrics import evaluate
from fracture.utils.config import load_config, require_path


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def _training_summary(path: str | Path) -> dict[str, Any]:
    rows = _read_csv(path)
    clean = [{key.strip(): float(value) for key, value in row.items()} for row in rows]
    metric = "metrics/mAP50-95(B)"
    best = max(clean, key=lambda row: row[metric])
    target = 0.9 * best[metric]
    convergence = next((int(row["epoch"]) for row in clean if row[metric] >= target), None)
    early = np.asarray([row[metric] for row in clean[: min(5, len(clean))]], dtype=float)
    keys = (
        "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "val/box_loss", "val/cls_loss", "val/dfl_loss",
        "metrics/precision(B)", "metrics/recall(B)",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    )
    return {
        "epochs_completed": len(clean),
        "best_epoch": int(best["epoch"]),
        "best": {key: best[key] for key in keys},
        "last_epoch": int(clean[-1]["epoch"]),
        "last": {key: clean[-1][key] for key in keys},
        "epoch_reaching_90pct_best_map50_95": convergence,
        "early_5_epoch_map50_95_mean": float(early.mean()),
        "early_5_epoch_map50_95_std": float(early.std()),
        "training_time_seconds_cumulative": clean[-1].get("time"),
    }


def _distribution(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, name in ((1, "positive"), (0, "negative")):
        values = scores[y == label]
        result[name] = {
            "n": int(len(values)),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "max": float(values.max()),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }
    result["median_separation"] = result["positive"]["median"] - result["negative"]["median"]
    result["mean_separation"] = result["positive"]["mean"] - result["negative"]["mean"]
    return result


def _study_payload(rows: list[dict[str, str]], cfg: dict) -> dict[str, Any]:
    y = np.asarray([int(row["y_true"]) for row in rows], dtype=int)
    scores = np.asarray([float(row["max_confidence"]) for row in rows], dtype=float)
    metrics = evaluate(y.tolist(), scores.tolist(), threshold=0.5)
    threshold, exploratory = _best_f1_threshold(y.tolist(), scores.tolist())
    qwk = isolated_fracture_qwk(
        [row["series_id"] for row in rows],
        scores.tolist(),
        study_intermediates(cfg["data"]),
    )
    study_seconds = np.asarray([float(row["study_seconds"]) for row in rows], dtype=float)
    return {
        "metrics": metrics,
        "score_distribution": _distribution(y, scores),
        "exploratory_best_f1_threshold": threshold,
        "exploratory_metrics": exploratory,
        "isolated_fracture_qwk": qwk,
        "runtime": {
            "mean_study_seconds": float(study_seconds.mean()),
            "median_study_seconds": float(np.median(study_seconds)),
            "p95_study_seconds": float(np.percentile(study_seconds, 95)),
        },
    }


def _index(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    result = {tuple(row[key] for key in keys): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate keys {keys}")
    return result


def _study_errors(
    baseline: list[dict[str, str]], pretrained: list[dict[str, str]], threshold: float = 0.5
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    base = _index(baseline, ("series_id",))
    pre = _index(pretrained, ("series_id",))
    if set(base) != set(pre):
        raise ValueError("Study sets differ between baseline and pretrained evaluation")
    rows: list[dict[str, Any]] = []
    for key in sorted(base):
        b, p = base[key], pre[key]
        y_true = int(b["y_true"])
        if y_true != int(p["y_true"]):
            raise ValueError(f"Label disagreement for {key[0]}")
        b_pos = float(b["max_confidence"]) >= threshold
        p_pos = float(p["max_confidence"]) >= threshold
        if y_true:
            category = (
                "found_by_both" if b_pos and p_pos else
                "found_only_by_baseline" if b_pos else
                "found_only_by_pretrained" if p_pos else
                "missed_by_both"
            )
        else:
            category = (
                "fp_in_both" if b_pos and p_pos else
                "fp_only_in_baseline" if b_pos else
                "fp_only_in_pretrained" if p_pos else
                "tn_in_both"
            )
        rows.append({
            "series_id": key[0], "y_true": y_true,
            "baseline_probability": float(b["max_confidence"]),
            "pretrained_probability": float(p["max_confidence"]),
            "official_threshold": threshold, "category": category,
        })
    return rows, dict(Counter(str(row["category"]) for row in rows))


def _max_iou(box: list[float], others: list[list[float]]) -> float:
    return max((iou(box, other) for other in others), default=0.0)


def _box_analysis(
    baseline: list[dict[str, str]], pretrained: list[dict[str, str]], image_size: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base = _index(baseline, ("series_id", "sop_uid"))
    pre = _index(pretrained, ("series_id", "sop_uid"))
    if set(base) != set(pre):
        raise ValueError("Slice sets differ between baseline and pretrained evaluation")
    gt_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    for key in sorted(base):
        b, p = base[key], pre[key]
        gt = json.loads(b["gt_boxes"])
        if gt != json.loads(p["gt_boxes"]):
            raise ValueError(f"GT disagreement for {key}")
        baseline_boxes = json.loads(b["boxes"])
        pretrained_boxes = json.loads(p["boxes"])
        width, height = float(b["image_width"]), float(b["image_height"])
        for index, target in enumerate(gt):
            resized_width = (target[2] - target[0]) * image_size / width
            resized_height = (target[3] - target[1]) * image_size / height
            b_iou, p_iou = _max_iou(target, baseline_boxes), _max_iou(target, pretrained_boxes)
            gt_rows.append({
                "series_id": key[0], "sop_uid": key[1], "slice_index": int(b["slice_index"]),
                "gt_box_index": index, "width_at_768": resized_width,
                "height_at_768": resized_height, "area_at_768": resized_width * resized_height,
                "relative_image_area": resized_width * resized_height / image_size**2,
                "baseline_max_iou": b_iou, "baseline_status": "tp" if b_iou >= 0.5 else "fn",
                "pretrained_max_iou": p_iou, "pretrained_status": "tp" if p_iou >= 0.5 else "fn",
            })
        for model_name, model_row, boxes in (("baseline", b, baseline_boxes), ("pretrained", p, pretrained_boxes)):
            scores = json.loads(model_row["scores"])
            for box, score in zip(boxes, scores, strict=True):
                max_overlap = _max_iou(box, gt)
                if max_overlap < 0.5:
                    fp_rows.append({
                        "model": model_name, "series_id": key[0], "sop_uid": key[1],
                        "slice_index": int(model_row["slice_index"]), "confidence": float(score),
                        "max_gt_iou": max_overlap, "box": json.dumps(box),
                        "review_status": "unreviewed",
                        "possible_confounders_to_check": "suture|skull_base|vascular_groove|calcification|motion|metal|beam_hardening|window_edge|postoperative",
                    })

    def summary(model: str) -> dict[str, Any]:
        statuses = [row[f"{model}_status"] for row in gt_rows]
        fp_count = sum(row["model"] == model for row in fp_rows)
        result: dict[str, Any] = {
            "gt_true_positive_boxes": statuses.count("tp"),
            "gt_false_negative_boxes": statuses.count("fn"),
            "false_positive_boxes_at_conf_0_01": fp_count,
        }
        for status in ("tp", "fn"):
            subset = [row for row in gt_rows if row[f"{model}_status"] == status]
            result[f"{status}_box_size"] = {
                "n": len(subset),
                **({
                    name: {
                        "median": float(np.median([float(row[name]) for row in subset])),
                        "mean": float(np.mean([float(row[name]) for row in subset])),
                        "min": float(np.min([float(row[name]) for row in subset])),
                        "max": float(np.max([float(row[name]) for row in subset])),
                    }
                    for name in ("width_at_768", "height_at_768", "area_at_768", "relative_image_area")
                } if subset else {}),
            }
        return result

    return gt_rows, fp_rows, {"baseline": summary("baseline"), "pretrained": summary("pretrained")}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(output: Path, baseline: list[dict[str, str]], pretrained: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for axis, name, rows in zip(axes, ("Existing V2", "COCO-pretrained V2"), (baseline, pretrained), strict=True):
        positive = [float(row["max_confidence"]) for row in rows if int(row["y_true"]) == 1]
        negative = [float(row["max_confidence"]) for row in rows if int(row["y_true"]) == 0]
        axis.hist(negative, bins=20, alpha=0.65, label=f"negative (n={len(negative)})")
        axis.hist(positive, bins=20, alpha=0.65, label=f"positive (n={len(positive)})")
        axis.axvline(0.5, color="black", linestyle="--", label="official threshold")
        axis.set_title(name); axis.set_xlabel("raw max detector confidence"); axis.legend()
    axes[0].set_ylabel("studies")
    fig.tight_layout(); fig.savefig(output / "fold0_score_distributions.png", dpi=160); plt.close(fig)


def _visual_cases(
    output: Path,
    baseline: list[dict[str, str]],
    pretrained: list[dict[str, str]],
    dicom_root: Path,
    limit: int,
) -> int:
    import cv2

    base = _index(baseline, ("series_id", "sop_uid"))
    pre = _index(pretrained, ("series_id", "sop_uid"))
    candidates = []
    for key, b in base.items():
        p = pre[key]
        gt = json.loads(b["gt_boxes"])
        b_boxes, p_boxes = json.loads(b["boxes"]), json.loads(p["boxes"])
        differing_gt = any((_max_iou(box, b_boxes) >= 0.5) != (_max_iou(box, p_boxes) >= 0.5) for box in gt)
        priority = 0 if differing_gt else 1 if gt else 2 if b_boxes or p_boxes else 3
        if priority < 3:
            candidates.append((priority, -max(float(b["max_confidence"]), float(p["max_confidence"])), key))
    candidates.sort()
    visual_dir = output / "fold0_error_visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    study_cache: dict[str, dict[str, Any]] = {}
    count = 0
    for _, _, key in candidates[:limit]:
        series_id, sop_uid = key
        if series_id not in study_cache:
            study_cache[series_id] = {record.sop_uid: record for record in load_study(dicom_root / series_id)}
        record = study_cache[series_id][sop_uid]
        b, p = base[key], pre[key]
        image = bone_window(record.hu, 500, 2500, record.photometric_interpretation == "MONOCHROME1")
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        colors = {"gt": (0, 255, 0), "baseline": (255, 100, 0), "pretrained": (0, 0, 255)}
        for label, row, box_key in (("gt", b, "gt_boxes"), ("baseline", b, "boxes"), ("pretrained", p, "boxes")):
            boxes = json.loads(row[box_key])
            scores = [] if label == "gt" else json.loads(row["scores"])
            for index, box in enumerate(boxes):
                x1, y1, x2, y2 = map(lambda value: int(round(value)), box)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), colors[label], 2)
                text = label if label == "gt" else f"{label} {scores[index]:.3f}"
                cv2.putText(canvas, text, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[label], 1, cv2.LINE_AA)
        title = f"series={series_id} slice={b['slice_index']}  GT=green baseline=blue pretrained=red"
        cv2.putText(canvas, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(visual_dir / f"{count:02d}_{series_id}_{b['slice_index']}.png"), canvas)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-studies", required=True)
    parser.add_argument("--pretrained-studies", required=True)
    parser.add_argument("--baseline-slices", required=True)
    parser.add_argument("--pretrained-slices", required=True)
    parser.add_argument("--baseline-training-results", required=True)
    parser.add_argument("--pretrained-training-results", required=True)
    parser.add_argument("--baseline-initialization")
    parser.add_argument("--pretrained-initialization", required=True)
    parser.add_argument("--output-dir", default="reports/fold0_initialization_ab")
    parser.add_argument("--visual-limit", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    baseline_studies, pretrained_studies = _read_csv(args.baseline_studies), _read_csv(args.pretrained_studies)
    baseline_slices, pretrained_slices = _read_csv(args.baseline_slices), _read_csv(args.pretrained_slices)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    study_rows, study_counts = _study_errors(baseline_studies, pretrained_studies)
    gt_rows, fp_rows, box_summary = _box_analysis(
        baseline_slices, pretrained_slices, int(cfg["preprocessing"]["image_size"])
    )
    _write_csv(output / "study_error_comparison.csv", study_rows)
    _write_csv(output / "gt_box_comparison.csv", gt_rows)
    _write_csv(output / "false_positive_review_candidates.csv", fp_rows)
    _plots(output, baseline_studies, pretrained_studies)
    visuals = _visual_cases(
        output, baseline_slices, pretrained_slices, require_path(cfg, "data", "dicom_root"), args.visual_limit
    )
    payload = {
        "comparison_protocol": {"fold": 0, "aggregation": "max", "calibration": "none", "threshold": 0.5},
        "baseline": {
            "initialization": json.loads(Path(args.baseline_initialization).read_text()) if args.baseline_initialization else None,
            "training": _training_summary(args.baseline_training_results),
            "study": _study_payload(baseline_studies, cfg),
        },
        "pretrained": {
            "initialization": json.loads(Path(args.pretrained_initialization).read_text()),
            "training": _training_summary(args.pretrained_training_results),
            "study": _study_payload(pretrained_studies, cfg),
        },
        "study_error_counts": study_counts,
        "box_analysis_at_iou_0_5": box_summary,
        "visual_cases_written": visuals,
        "anatomical_interpretation": "not_assigned; visual cases require human review",
    }
    (output / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
