"""Merge held-out fold predictions and fit OOF-only study components."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from fracture.data.metadata import study_intermediates
from fracture.evaluation.isolated_qwk import isolated_fracture_qwk
from fracture.evaluation.series_metrics import evaluate
from fracture.inference.aggregation import FEATURE_NAMES, StudyAggregator, aggregation_features
from fracture.inference.calibration import fit_calibrator
from fracture.utils.config import load_config


def _read_fold_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for filename in paths:
        with Path(filename).open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    required = {"series_id", "y_true", "fold", "slice_scores"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Each fold CSV needs columns {sorted(required)}")
    return rows


def _feature_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    feature_rows: list[dict] = []
    for row in rows:
        scores = [float(x) for x in row["slice_scores"].split(";") if x]
        counts = None
        if row.get("detection_counts"):
            counts = [int(x) for x in row["detection_counts"].split(";") if x != ""]
        feature_rows.append(dict(row) | aggregation_features(scores, counts))
    x = np.asarray([[float(r[name]) for name in FEATURE_NAMES] for r in feature_rows], dtype=float)
    y = np.asarray([int(r["y_true"]) for r in feature_rows], dtype=int)
    return x, y, feature_rows


def _nested_oof_probabilities(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Fit aggregator on other folds' rows; fall back to stratified CV if one fold only."""
    unique_folds = sorted(set(int(v) for v in folds))
    raw = np.zeros(len(y), dtype=float)
    if len(unique_folds) >= 2:
        for holdout in unique_folds:
            train_idx = np.where(folds != holdout)[0]
            test_idx = np.where(folds == holdout)[0]
            if len(set(y[train_idx])) < 2:
                raw[test_idx] = x[test_idx, FEATURE_NAMES.index("max_confidence")]
                continue
            model = LogisticRegression(max_iter=5000, class_weight="balanced").fit(x[train_idx], y[train_idx])
            raw[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        return raw

    # Single fold CSV: use stratified K-fold on studies to avoid fully in-sample scores.
    splits = min(5, max(2, int(y.sum()), int((1 - y).sum())))
    if len(set(y)) < 2 or len(y) < splits * 2:
        return x[:, FEATURE_NAMES.index("max_confidence")]
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(x, y):
        model = LogisticRegression(max_iter=5000, class_weight="balanced").fit(x[train_idx], y[train_idx])
        raw[test_idx] = model.predict_proba(x[test_idx])[:, 1]
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-csv", action="append", required=True)
    parser.add_argument("--config", default="configs/fracture_25d_p2.yaml")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--calibration", choices=["none", "platt", "isotonic"], default="platt")
    args = parser.parse_args()

    rows = _read_fold_rows(args.fold_csv)
    x, y, feature_rows = _feature_matrix(rows)
    folds = np.asarray([int(r["fold"]) for r in feature_rows], dtype=int)

    nested_raw = _nested_oof_probabilities(x, y, folds)
    max_scores = np.asarray([float(r["max_confidence"]) for r in feature_rows], dtype=float)
    consecutive_scores = np.asarray(
        [
            StudyAggregator("consecutive", min_run=2, run_threshold=0.1).predict(
                [float(v) for v in str(r["slice_scores"]).split(";") if v],
                [int(v) for v in str(r.get("detection_counts", "")).split(";") if v != ""] or None,
            )
            for r in feature_rows
        ],
        dtype=float,
    )

    cfg = load_config(args.config)
    study_table = study_intermediates(cfg["data"])

    candidates = {
        "max": max_scores,
        "consecutive": consecutive_scores,
        "logistic": nested_raw,
    }
    ranked = []
    for name, scores in candidates.items():
        metrics_i = evaluate(y.tolist(), scores.tolist())
        qwk_i = isolated_fracture_qwk(
            [str(r["series_id"]) for r in feature_rows],
            scores.tolist(),
            study_table,
        )
        ranked.append(
            (
                name,
                scores,
                metrics_i,
                qwk_i,
                (
                    qwk_i["isolated_fracture_qwk"],
                    metrics_i["sensitivity_at_0_5"],
                    metrics_i["pr_auc"],
                    metrics_i["auroc"],
                ),
            )
        )
    ranked.sort(key=lambda item: item[4], reverse=True)
    best_name, best_scores, best_metrics, best_qwk, _ = ranked[0]

    final_aggregator = LogisticRegression(max_iter=5000, class_weight="balanced").fit(x, y)
    deployment_raw = final_aggregator.predict_proba(x)[:, 1]
    calibrator = fit_calibrator(best_scores.tolist(), y.tolist(), args.calibration)
    calibrated = [calibrator.predict(float(p)) for p in best_scores]

    for row, nested, deploy, calibrated_p, consecutive_p in zip(
        feature_rows, nested_raw, deployment_raw, calibrated, consecutive_scores, strict=True
    ):
        row["detector_max"] = float(row["max_confidence"])
        row["consecutive_probability"] = float(consecutive_p)
        row["logistic_nested_probability"] = float(nested)
        row["deployment_fit_probability"] = float(deploy)
        row["raw_probability"] = float(row["detector_max"] if best_name == "max" else consecutive_p if best_name == "consecutive" else nested)
        row["calibrated_probability"] = float(calibrated_p)
        row["selected_aggregator"] = best_name

    calibrated_metrics = evaluate(y.tolist(), calibrated)
    calibrated_qwk = isolated_fracture_qwk(
        [str(r["series_id"]) for r in feature_rows],
        calibrated,
        study_table,
    )
    metrics = {
        **calibrated_metrics,
        **calibrated_qwk,
        "selected_aggregator": best_name,
        "calibration": args.calibration,
        "n_studies": int(len(y)),
        "n_positive": int(y.sum()),
        "candidate_comparison": {
            name: {**m, "isolated_fracture_qwk": q["isolated_fracture_qwk"]}
            for name, _, m, q, _ in ranked
        },
        "uncalibrated_selected": best_metrics,
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "oof_fracture_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    (output / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    models = Path(args.models_dir)
    models.mkdir(parents=True, exist_ok=True)
    # Always persist the logistic bundle for deployment experiments; also store the
    # selected method metadata so FracturePredictor can default correctly.
    joblib.dump(
        {
            "model": final_aggregator,
            "method": "logistic" if best_name == "logistic" else best_name,
            "feature_names": list(FEATURE_NAMES),
            "min_run": 2,
            "run_threshold": 0.1,
            "selected_on_oof": best_name,
        },
        models / "aggregator.joblib",
    )
    if args.calibration != "none" and calibrator.model is not None:
        joblib.dump({"method": args.calibration, "model": calibrator.model}, models / "calibrator.joblib")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
