"""QWK-aware threshold selection with fold-held-out reporting."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from fracture.data.metadata import study_intermediates
from fracture.evaluation.isolated_qwk import isolated_fracture_qwk
from fracture.evaluation.series_metrics import evaluate
from fracture.utils.config import load_config


def candidate_thresholds(scores: list[float]) -> list[float]:
    values = np.unique(np.clip(np.asarray(scores, dtype=float), 0.0, 1.0))
    if values.size == 0:
        raise ValueError("At least one score is required")
    midpoints = (values[:-1] + values[1:]) / 2.0
    return sorted({0.0, 0.5, 1.0, *map(float, midpoints)})


def threshold_metrics(
    series_ids: list[str],
    labels: list[int],
    scores: list[float],
    study_table: dict[str, dict[str, float]],
    threshold: float,
) -> dict[str, Any]:
    decisions = [float(value >= threshold) for value in scores]
    return {
        "decision_threshold": float(threshold),
        **evaluate(labels, decisions, threshold=0.5),
        **isolated_fracture_qwk(series_ids, decisions, study_table),
    }


def select_threshold(
    series_ids: list[str],
    labels: list[int],
    scores: list[float],
    study_table: dict[str, dict[str, float]],
    *,
    minimum_sensitivity: float | None = None,
) -> tuple[float, dict[str, Any]]:
    candidates = [
        threshold_metrics(series_ids, labels, scores, study_table, threshold)
        for threshold in candidate_thresholds(scores)
    ]
    eligible = candidates
    if minimum_sensitivity is not None:
        eligible = [
            item for item in candidates
            if float(item["sensitivity_at_threshold"]) >= float(minimum_sensitivity)
        ]
        if not eligible:
            raise ValueError(f"No threshold reaches minimum sensitivity {minimum_sensitivity:.6g}")
    best = max(
        eligible,
        key=lambda item: (
            float(item["isolated_fracture_qwk"]),
            float(item["specificity_at_threshold"]),
            float(item["sensitivity_at_threshold"]),
            float(item["decision_threshold"]),
        ),
    )
    return float(best["decision_threshold"]), best


def cross_fitted_threshold_report(
    rows: list[dict[str, str]],
    study_table: dict[str, dict[str, float]],
    *,
    score_column: str,
    minimum_sensitivity: float | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No OOF rows supplied")
    required = {"series_id", "y_true", "fold", "prediction_protocol", score_column}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"OOF CSV is missing columns: {sorted(missing)}")
    if any(row["prediction_protocol"] != "heldout_patient_fold" for row in rows):
        raise ValueError("Threshold selection accepts held-out patient-fold predictions only")
    series = [str(row["series_id"]) for row in rows]
    if len(series) != len(set(series)):
        raise ValueError("Each study must occur exactly once in OOF threshold selection")
    labels = [int(row["y_true"]) for row in rows]
    scores = [float(row[score_column]) for row in rows]
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=int)
    unique_folds = sorted(set(map(int, folds)))
    if len(unique_folds) < 2:
        raise ValueError("At least two folds are required for cross-fitted threshold selection")

    deployed = np.zeros(len(rows), dtype=float)
    fold_reports: list[dict[str, Any]] = []
    for holdout in unique_folds:
        train_indices = np.where(folds != holdout)[0]
        test_indices = np.where(folds == holdout)[0]
        threshold, selection = select_threshold(
            [series[i] for i in train_indices],
            [labels[i] for i in train_indices],
            [scores[i] for i in train_indices],
            study_table,
            minimum_sensitivity=minimum_sensitivity,
        )
        heldout_scores = [scores[i] for i in test_indices]
        deployed[test_indices] = [float(value >= threshold) for value in heldout_scores]
        heldout = threshold_metrics(
            [series[i] for i in test_indices],
            [labels[i] for i in test_indices],
            heldout_scores,
            study_table,
            threshold,
        )
        fold_reports.append({
            "holdout_fold": holdout,
            "selected_threshold": threshold,
            "selection_partition_n": int(len(train_indices)),
            "heldout_partition_n": int(len(test_indices)),
            "selection_metrics": selection,
            "heldout_metrics": heldout,
        })

    final_threshold, final_selection = select_threshold(
        series,
        labels,
        scores,
        study_table,
        minimum_sensitivity=minimum_sensitivity,
    )
    return {
        "score_column": score_column,
        "minimum_sensitivity": minimum_sensitivity,
        "protocol": "fold-held-out threshold layer over fixed base OOF predictions",
        "protocol_limit": (
            "The base detector and upstream feature selection are not nested by this tool; "
            "use inner folds for a final unbiased pipeline estimate."
        ),
        "cross_fitted": {
            **evaluate(labels, deployed.tolist(), threshold=0.5),
            **isolated_fracture_qwk(series, deployed.tolist(), study_table),
        },
        "folds": fold_reports,
        "full_oof_exploratory_threshold": final_threshold,
        "full_oof_exploratory_metrics": final_selection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--score-column", default="raw_probability")
    parser.add_argument("--minimum-sensitivity", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.oof_csv).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    report = cross_fitted_threshold_report(
        rows,
        study_intermediates(load_config(args.config)["data"]),
        score_column=args.score_column,
        minimum_sensitivity=args.minimum_sensitivity,
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Threshold report is immutable; choose a new output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
