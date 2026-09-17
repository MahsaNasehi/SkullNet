"""Compare held-out baseline and weak-study predictions on identical studies."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from fracture.data.metadata import study_intermediates
from fracture.data.weak_study import load_study_catalog
from fracture.evaluation.isolated_qwk import isolated_fracture_qwk
from fracture.evaluation.series_metrics import evaluate
from fracture.utils.config import load_config


def _read(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _score(row: dict[str, str], method: str) -> float:
    if method not in row:
        raise KeyError(f"Prediction CSV has no aggregation column {method!r}")
    return float(row[method])


def _metrics(
    rows: list[dict[str, str]], method: str, catalog: dict[str, Any], table: dict[str, dict[str, float]]
) -> dict[str, Any]:
    ids = [str(row["series_id"]) for row in rows]
    labels = [int(float(row["y_true"])) for row in rows]
    scores = [_score(row, method) for row in rows]
    result: dict[str, Any] = evaluate(labels, scores, threshold=0.5)
    result.update(isolated_fracture_qwk(ids, scores, table))
    for tier in ("tier1", "tier2"):
        indices = [i for i, key in enumerate(ids) if catalog[key].tier == tier and labels[i] == 0]
        result[f"{tier}_negative_count"] = len(indices)
        result[f"{tier}_negative_specificity_at_0_5"] = (
            sum(scores[i] < 0.5 for i in indices) / len(indices) if indices else None
        )
        result[f"{tier}_false_positives_at_0_5"] = sum(scores[i] >= 0.5 for i in indices)
    return result


def compare(
    baseline_rows: list[dict[str, str]],
    weak_rows: list[dict[str, str]],
    *,
    method: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    baseline_by_id = {str(row["series_id"]): row for row in baseline_rows}
    weak_by_id = {str(row["series_id"]): row for row in weak_rows}
    if len(baseline_by_id) != len(baseline_rows) or len(weak_by_id) != len(weak_rows):
        raise ValueError("Prediction files must contain each study exactly once")
    if set(baseline_by_id) != set(weak_by_id):
        raise ValueError("Baseline and weak-study prediction files do not cover identical held-out studies")
    ids = sorted(baseline_by_id)
    for key in ids:
        if baseline_by_id[key]["fold"] != weak_by_id[key]["fold"]:
            raise ValueError(f"Fold mismatch for study {key}")
        if baseline_by_id[key]["y_true"] != weak_by_id[key]["y_true"]:
            raise ValueError(f"Label mismatch for study {key}")
    catalog = load_study_catalog(config)
    table = study_intermediates(config["data"])
    baseline = _metrics([baseline_by_id[key] for key in ids], method, catalog, table)
    weak = _metrics([weak_by_id[key] for key in ids], method, catalog, table)
    return {
        "aggregation": method,
        "num_identical_heldout_studies": len(ids),
        "baseline": baseline,
        "weak_study": weak,
        "delta_weak_minus_baseline": {
            key: weak[key] - baseline[key]
            for key in (
                "auroc", "pr_auc", "sensitivity_at_threshold", "specificity_at_threshold",
                "precision_at_threshold", "f1_at_threshold", "brier", "log_loss",
                "isolated_fracture_qwk",
            )
        } | {
            "false_positives_removed": baseline["fp"] - weak["fp"],
            "true_positives_lost": baseline["tp"] - weak["tp"],
            "tier1_false_positives_removed": (
                baseline["tier1_false_positives_at_0_5"] - weak["tier1_false_positives_at_0_5"]
            ),
            "tier2_false_positives_removed": (
                baseline["tier2_false_positives_at_0_5"] - weak["tier2_false_positives_at_0_5"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--weak", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aggregation", default="top3_mean")
    parser.add_argument("--baseline-box-json")
    parser.add_argument("--weak-box-json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Comparison reports are immutable: {output}")
    report = compare(
        _read(args.baseline), _read(args.weak),
        method=args.aggregation, config=load_config(args.config),
    )
    if bool(args.baseline_box_json) != bool(args.weak_box_json):
        raise ValueError("Provide both --baseline-box-json and --weak-box-json, or neither")
    if args.baseline_box_json:
        baseline_box = json.loads(Path(args.baseline_box_json).read_text(encoding="utf-8"))
        weak_box = json.loads(Path(args.weak_box_json).read_text(encoding="utf-8"))
        keys = ("precision", "recall", "mAP50", "mAP50_95")
        report["tier1_box_metrics"] = {
            "baseline": {key: baseline_box[key] for key in keys},
            "weak_study": {key: weak_box[key] for key in keys},
            "delta_weak_minus_baseline": {key: weak_box[key] - baseline_box[key] for key in keys},
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
