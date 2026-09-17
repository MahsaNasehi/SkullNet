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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fracture.data.metadata import study_intermediates
from fracture.evaluation.isolated_qwk import isolated_fracture_qwk
from fracture.evaluation.series_metrics import evaluate
from fracture.inference.aggregation import (
    FEATURE_NAMES,
    SPATIAL_COMPACT_FEATURE_NAMES,
    SPATIAL_FEATURE_NAMES,
    StudyAggregator,
    aggregation_features,
)
from fracture.inference.calibration import fit_calibrator
from fracture.utils.config import load_config

# Compact feature subset for the "spatial_logistic" candidate
# (FRACTURE_QWK_99_REVIEW_FA.md section 10: "۴-۶ ویژگی محدود" rather than
# the full ~23-feature set), so a low-parameter spatially-aware model can
# be compared against top3_mean/consecutive/full-feature logistic on
# equal footing.
def _logistic_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", C=0.25),
    )


def _read_fold_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for filename in paths:
        with Path(filename).open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    required = {"series_id", "y_true", "fold", "slice_scores", "prediction_protocol"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Each fold CSV needs columns {sorted(required)}")
    if any(row.get("prediction_protocol") != "heldout_patient_fold" for row in rows):
        raise ValueError("OOF aggregation accepts held-out patient-fold predictions only")
    return rows


def _merge_study_model_rows(
    detector_rows: list[dict[str, str]],
    paths: list[str] | None,
) -> list[dict[str, str]]:
    if not paths:
        return detector_rows
    study_scores: dict[tuple[int, str], float] = {}
    for filename in paths:
        with Path(filename).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("prediction_protocol") != "heldout_patient_fold":
                    raise ValueError("MIL fusion accepts held-out patient-fold predictions only")
                key = (int(row["fold"]), str(row["series_id"]))
                if key in study_scores:
                    raise ValueError(f"Duplicate MIL OOF prediction for fold/study {key}")
                study_scores[key] = float(row["study_model_probability"])
    merged = []
    for source in detector_rows:
        row = dict(source)
        key = (int(row["fold"]), str(row["series_id"]))
        if key not in study_scores:
            raise ValueError(f"Missing MIL OOF prediction for fold/study {key}")
        row["study_model_probability"] = str(study_scores[key])
        merged.append(row)
    extra = set(study_scores) - {(int(row["fold"]), str(row["series_id"])) for row in detector_rows}
    if extra:
        raise ValueError(f"MIL OOF contains studies absent from detector OOF: {sorted(extra)[:10]}")
    return merged


def _merge_slice_geometry_rows(
    detector_rows: list[dict[str, str]], paths: list[str] | None
) -> list[dict[str, str]]:
    """Attach boxes and image sizes from historical held-out slice CSVs."""
    if not paths:
        return detector_rows
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    seen_slices: set[tuple[int, str, int]] = set()
    for filename in paths:
        with Path(filename).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("prediction_protocol") != "heldout_patient_fold":
                    raise ValueError("Spatial OOF features require held-out patient-fold slice predictions")
                key = (int(row["fold"]), str(row["series_id"]))
                slice_key = (*key, int(row["slice_index"]))
                if slice_key in seen_slices:
                    raise ValueError(f"Duplicate slice prediction in spatial OOF input: {slice_key}")
                seen_slices.add(slice_key)
                grouped.setdefault(key, []).append(row)

    expected = {(int(row["fold"]), str(row["series_id"])) for row in detector_rows}
    extra = set(grouped) - expected
    if extra:
        raise ValueError(f"Slice OOF contains studies absent from detector OOF: {sorted(extra)[:10]}")
    merged: list[dict[str, str]] = []
    for source in detector_rows:
        row = dict(source)
        key = (int(row["fold"]), str(row["series_id"]))
        slices = sorted(grouped.get(key, []), key=lambda item: int(item["slice_index"]))
        if not slices:
            raise ValueError(f"Missing slice OOF geometry for fold/study {key}")
        score_count = len([value for value in str(row["slice_scores"]).split(";") if value != ""])
        if len(slices) != score_count:
            raise ValueError(f"Slice/study prediction count mismatch for {key}: {len(slices)} != {score_count}")
        row["slice_boxes"] = json.dumps(
            [json.loads(item.get("boxes") or "[]") for item in slices], separators=(",", ":")
        )
        row["slice_box_scores"] = json.dumps(
            [json.loads(item.get("scores") or "[]") for item in slices], separators=(",", ":")
        )
        row["image_shapes"] = json.dumps(
            [[int(item["image_height"]), int(item["image_width"])] for item in slices],
            separators=(",", ":"),
        )
        merged.append(row)
    return merged


def _validate_oof_rows(rows: list[dict[str, str]], expected_folds: set[int]) -> set[int]:
    observed = {int(row["fold"]) for row in rows}
    duplicate_series: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        series_id = str(row["series_id"])
        if series_id in seen:
            duplicate_series.add(series_id)
        seen.add(series_id)
    if duplicate_series:
        raise ValueError(f"OOF inputs contain duplicate studies: {sorted(duplicate_series)[:10]}")
    if observed - expected_folds:
        raise ValueError(f"Unexpected fold ids: {sorted(observed - expected_folds)}")
    return observed


def _feature_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict], tuple[str, ...]]:
    feature_rows: list[dict] = []
    for row in rows:
        scores = [float(x) for x in row["slice_scores"].split(";") if x]
        counts = None
        if row.get("detection_counts"):
            counts = [int(x) for x in row["detection_counts"].split(";") if x != ""]
        positions = None
        if row.get("physical_positions"):
            positions = [float(x) if x else None for x in row["physical_positions"].split(";")]
        extras = (
            {"study_model_probability": float(row["study_model_probability"])}
            if row.get("study_model_probability") not in {None, ""}
            else None
        )
        boxes = json.loads(row["slice_boxes"]) if row.get("slice_boxes") else None
        box_scores = json.loads(row["slice_box_scores"]) if row.get("slice_box_scores") else None
        image_shapes = json.loads(row["image_shapes"]) if row.get("image_shapes") else None
        feature_rows.append(
            dict(row) | aggregation_features(
                scores,
                counts,
                positions,
                extra_features=extras,
                boxes_by_slice=boxes,
                box_scores_by_slice=box_scores,
                image_shapes=image_shapes,
            )
        )
    has_study_model = all("study_model_probability" in row for row in feature_rows)
    has_spatial = all(all(name in row for name in SPATIAL_FEATURE_NAMES) for row in feature_rows)
    feature_names = (
        FEATURE_NAMES
        + (("study_model_probability",) if has_study_model else ())
        + (SPATIAL_FEATURE_NAMES if has_spatial else ())
    )
    x = np.asarray([[float(r[name]) for name in feature_names] for r in feature_rows], dtype=float)
    y = np.asarray([int(r["y_true"]) for r in feature_rows], dtype=int)
    return x, y, feature_rows, feature_names


def _nested_oof_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> np.ndarray:
    """Fit aggregator on other folds' rows; fall back to stratified CV if one fold only."""
    unique_folds = sorted(set(int(v) for v in folds))
    raw = np.zeros(len(y), dtype=float)
    if len(unique_folds) >= 2:
        for holdout in unique_folds:
            train_idx = np.where(folds != holdout)[0]
            test_idx = np.where(folds == holdout)[0]
            if len(set(y[train_idx])) < 2:
                raw[test_idx] = x[test_idx, feature_names.index("max_confidence")]
                continue
            model = _logistic_model().fit(x[train_idx], y[train_idx])
            raw[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        return raw

    # Single fold CSV: use stratified K-fold on studies to avoid fully in-sample scores.
    minority = int(min(y.sum(), (1 - y).sum()))
    splits = min(5, minority)
    if len(set(y)) < 2 or splits < 2:
        return x[:, feature_names.index("max_confidence")]
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(x, y):
        model = _logistic_model().fit(x[train_idx], y[train_idx])
        raw[test_idx] = model.predict_proba(x[test_idx])[:, 1]
    return raw


def _cross_fitted_calibration(
    scores: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    method: str,
) -> np.ndarray:
    """Calibrate without predicting any study with a calibrator fitted on it."""
    if method == "none":
        return np.clip(scores.astype(float), 0.0, 1.0)
    calibrated = np.zeros(len(y), dtype=float)
    unique_folds = sorted(set(int(value) for value in folds))
    split_indices: list[tuple[np.ndarray, np.ndarray]] = []
    if len(unique_folds) >= 2:
        for holdout in unique_folds:
            split_indices.append((np.where(folds != holdout)[0], np.where(folds == holdout)[0]))
    else:
        minority = int(min(y.sum(), (1 - y).sum()))
        n_splits = min(5, minority)
        if len(set(y)) < 2 or n_splits < 2:
            return np.clip(scores.astype(float), 0.0, 1.0)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_indices.extend(skf.split(scores.reshape(-1, 1), y))

    for train_idx, test_idx in split_indices:
        if len(set(y[train_idx])) < 2:
            calibrated[test_idx] = scores[test_idx]
            continue
        calibrator = fit_calibrator(scores[train_idx].tolist(), y[train_idx].tolist(), method)
        calibrated[test_idx] = [calibrator.predict(float(value)) for value in scores[test_idx]]
    return np.clip(calibrated, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-csv", action="append", required=True)
    parser.add_argument(
        "--study-csv",
        action="append",
        help="Optional fold-specific StudyMIL heldout_predictions.csv to enable hybrid fusion.",
    )
    parser.add_argument(
        "--fold-slices",
        action="append",
        help="Optional held-out slice CSVs used to derive geometry-aware aggregation features.",
    )
    parser.add_argument("--config", default="configs/fracture_25d_p2.yaml")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--calibration", choices=["none", "platt", "isotonic"], default="none")
    parser.add_argument(
        "--allow-incomplete-oof",
        action="store_true",
        help="Permit diagnostic partial-fold reports. Deployment artifacts are not written.",
    )
    args = parser.parse_args()

    rows = _read_fold_rows(args.fold_csv)
    rows = _merge_slice_geometry_rows(rows, args.fold_slices)
    rows = _merge_study_model_rows(rows, args.study_csv)
    cfg = load_config(args.config)
    expected_folds = set(range(int(cfg.get("split", {}).get("num_folds", 5))))
    observed_folds = _validate_oof_rows(rows, expected_folds)
    complete_oof = observed_folds == expected_folds
    if not complete_oof and not args.allow_incomplete_oof:
        raise ValueError(
            f"Incomplete OOF input: observed folds {sorted(observed_folds)}, expected {sorted(expected_folds)}. "
            "Pass --allow-incomplete-oof only for a diagnostic report; it will not write deployment models."
        )
    x, y, feature_rows, feature_names = _feature_matrix(rows)
    folds = np.asarray([int(r["fold"]) for r in feature_rows], dtype=int)
    nested_raw = _nested_oof_probabilities(x, y, folds, feature_names)
    spatial_feature_names: tuple[str, ...] | None = None
    spatial_nested_raw: np.ndarray | None = None
    if all(name in feature_names for name in SPATIAL_FEATURE_NAMES):
        spatial_feature_names = SPATIAL_COMPACT_FEATURE_NAMES + (
            ("study_model_probability",) if "study_model_probability" in feature_names else ()
        )
        spatial_indices = [feature_names.index(name) for name in spatial_feature_names]
        spatial_nested_raw = _nested_oof_probabilities(
            x[:, spatial_indices], y, folds, spatial_feature_names
        )
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

    study_table = study_intermediates(cfg["data"])

    candidates = {
        "max": max_scores,
        "top3_mean": np.asarray([float(row["top3_mean"]) for row in feature_rows], dtype=float),
        "consecutive": consecutive_scores,
        "logistic": nested_raw,
    }
    # Compact spatial/depth-track logistic model (review section 10), only
    # comparable when every fold's held-out CSV carried per-slice
    # boxes/scores (generated by fracture.evaluation.predict_fold after
    # this change). Older OOF CSVs without boxes_per_slice/scores_per_slice
    # simply skip this candidate rather than failing.
    if spatial_nested_raw is not None:
        candidates["spatial_logistic"] = spatial_nested_raw
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
                    metrics_i["pr_auc"],
                    metrics_i["sensitivity_at_0_5"],
                    metrics_i["auroc"],
                    -metrics_i["brier"],
                    qwk_i["isolated_fracture_qwk"],
                ),
            )
        )
    ranked.sort(key=lambda item: item[4], reverse=True)
    best_name, best_scores, best_metrics, best_qwk, _ = ranked[0]

    deployment_feature_names = spatial_feature_names if best_name == "spatial_logistic" else feature_names
    deployment_indices = [feature_names.index(name) for name in deployment_feature_names]
    final_aggregator = _logistic_model().fit(x[:, deployment_indices], y)
    deployment_raw = final_aggregator.predict_proba(x[:, deployment_indices])[:, 1]
    calibrated = _cross_fitted_calibration(best_scores, y, folds, args.calibration).tolist()
    deployment_calibrator = fit_calibrator(best_scores.tolist(), y.tolist(), args.calibration)

    spatial_nested_iter = spatial_nested_raw if spatial_nested_raw is not None else [0.0] * len(feature_rows)
    for row, nested, spatial_nested, deploy, calibrated_p, consecutive_p in zip(
        feature_rows, nested_raw, spatial_nested_iter, deployment_raw, calibrated, consecutive_scores, strict=True
    ):
        row["detector_max"] = float(row["max_confidence"])
        row["consecutive_probability"] = float(consecutive_p)
        row["logistic_nested_probability"] = float(nested)
        row["spatial_logistic_nested_probability"] = float(spatial_nested)
        row["deployment_fit_probability"] = float(deploy)
        selected_raw = {
            "max": float(row["detector_max"]),
            "top3_mean": float(row["top3_mean"]),
            "consecutive": float(consecutive_p),
            "logistic": float(nested),
            "spatial_logistic": float(spatial_nested),
        }[best_name]
        row["raw_probability"] = selected_raw
        row["cross_fitted_calibrated_probability"] = float(calibrated_p)
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
        "calibration_evaluation": "cross_fitted",
        "complete_oof": complete_oof,
        "observed_folds": sorted(observed_folds),
        "expected_folds": sorted(expected_folds),
        "selection_priority": ["pr_auc", "sensitivity_at_0_5", "auroc", "brier", "isolated_fracture_qwk"],
        "n_studies": int(len(y)),
        "n_positive": int(y.sum()),
        "candidate_comparison": {
            name: {**m, "isolated_fracture_qwk": q["isolated_fracture_qwk"]}
            for name, _, m, q, _ in ranked
        },
        "uncalibrated_selected": best_metrics,
    }

    output = Path(args.output_dir)
    report_targets = [output / "oof_fracture_predictions.csv", output / "final_metrics.json"]
    existing_reports = [str(path) for path in report_targets if path.exists()]
    if existing_reports:
        raise FileExistsError(
            "OOF reports are immutable; choose a new --output-dir instead of overwriting: "
            + ", ".join(existing_reports)
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "oof_fracture_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    (output / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if not complete_oof:
        print(json.dumps(metrics, indent=2))
        print("Diagnostic incomplete-OOF report only; deployment aggregator/calibrator were not written.")
        return

    models = Path(args.models_dir)
    model_targets = [models / "aggregator.joblib"]
    if args.calibration != "none":
        model_targets.append(models / "calibrator.joblib")
    existing_models = [str(path) for path in model_targets if path.exists()]
    if existing_models:
        raise FileExistsError(
            "Deployment artifacts are immutable; choose a new --models-dir instead of overwriting: "
            + ", ".join(existing_models)
        )
    models.mkdir(parents=True, exist_ok=True)
    # Always persist the logistic bundle for deployment experiments; also store the
    # selected method metadata so FracturePredictor can default correctly.
    joblib.dump(
        {
            "model": final_aggregator,
            "method": "logistic" if best_name == "logistic" else best_name,
            "feature_names": list(deployment_feature_names),
            "min_run": 2,
            "run_threshold": 0.1,
            "selected_on_oof": best_name,
        },
        models / "aggregator.joblib",
    )
    if args.calibration != "none" and deployment_calibrator.model is not None:
        joblib.dump(
            {
                "method": args.calibration,
                "model": deployment_calibrator.model,
                "fit_source": "complete_oof_selected_raw_scores",
            },
            models / "calibrator.joblib",
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
