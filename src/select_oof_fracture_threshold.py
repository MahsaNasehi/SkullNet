"""Select a fracture threshold only from 5-fold OOF predictions.

This script never sweeps thresholds on the fixed test.  It reports:
1) a cross-fitted OOF estimate (threshold for each fold learned from the other 4),
2) one final threshold learned from all OOF studies, and
3) a secondary, one-shot application of that frozen threshold to saved test scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDS = [
    ROOT / f"outputs/yolo26s_p2_hu800_ww1600_fold{fold}/study_evaluation/study_predictions.csv"
    for fold in range(5)
]
DEFAULT_TEST = ROOT / "outputs/fixed_test_fracture_kfold_ensemble/study_predictions.csv"
DEFAULT_OUTPUT = ROOT / "outputs/oof_fracture_threshold_v1"


def decision_metrics(y_true: np.ndarray, predicted: np.ndarray,
                     ranking_scores: np.ndarray | None = None) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[False, True]).ravel()
    result: dict[str, object] = {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "accuracy": float((tp + tn) / len(y_true)),
        "binary_qwk": float(cohen_kappa_score(
            y_true.astype(int), predicted.astype(int), labels=[0, 1], weights="quadratic"
        )),
    }
    if ranking_scores is not None:
        scores = np.asarray(ranking_scores, dtype=float)
        result["study_pr_auc"] = float(average_precision_score(y_true, scores))
        result["study_roc_auc"] = float(roc_auc_score(y_true, scores))
    return result


def patient_bootstrap_decisions(table: pd.DataFrame, predicted_column: str,
                                repeats: int, seed: int) -> dict[str, object]:
    """Stratified patient-cluster CI for already-frozen binary decisions."""
    patient_truth = table.groupby("patient_id")["fracture_true"].max().astype(bool)
    positive = patient_truth[patient_truth].index.to_numpy()
    negative = patient_truth[~patient_truth].index.to_numpy()
    if not len(positive) or not len(negative):
        return {"valid_repeats": 0, "reason": "both patient classes are required"}
    groups = {patient: group for patient, group in table.groupby("patient_id", sort=False)}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"binary_qwk": [], "sensitivity": [], "specificity": []}
    for _ in range(repeats):
        sampled = np.r_[rng.choice(positive, len(positive), replace=True),
                        rng.choice(negative, len(negative), replace=True)]
        frame = pd.concat([groups[patient] for patient in sampled], ignore_index=True)
        metrics = decision_metrics(frame["fracture_true"], frame[predicted_column])
        for key in samples:
            value = metrics[key]
            if value is not None and np.isfinite(value):
                samples[key].append(float(value))
    return {
        "method": "stratified patient-cluster bootstrap of frozen decisions",
        "seed": seed, "requested_repeats": repeats,
        "percentile_95_ci": {
            key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            for key, values in samples.items()
        },
    }


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    """Return thresholds representing every distinct decision set plus 0.5."""
    unique = np.unique(np.asarray(scores, dtype=float))
    if not len(unique) or not np.isfinite(unique).all():
        raise ValueError("Scores must be non-empty and finite")
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    # nextafter(max, +inf) represents an all-negative classifier.
    return np.unique(np.r_[0.0, 0.5, midpoints, np.nextafter(unique[-1], np.inf)])


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, object], pd.DataFrame]:
    """Maximize OOF QWK with a deterministic, predeclared tie-break.

    Ties are resolved by specificity, then sensitivity, then proximity to 0.5.
    The last rule avoids an arbitrary extreme within equivalent candidates.
    """
    rows = []
    for threshold in threshold_candidates(scores):
        metrics = decision_metrics(y_true, scores >= threshold)
        rows.append({"threshold": float(threshold), **metrics})
    sweep = pd.DataFrame(rows)
    ranked = sweep.assign(distance_to_0_5=(sweep["threshold"] - 0.5).abs()).sort_values(
        ["binary_qwk", "specificity", "sensitivity", "distance_to_0_5", "threshold"],
        ascending=[False, False, False, True, False], kind="stable",
    )
    best = ranked.iloc[0]
    threshold = float(best["threshold"])
    metrics = decision_metrics(y_true, scores >= threshold, scores)
    metrics["threshold"] = threshold
    return threshold, metrics, sweep


def load_oof(paths: list[Path], manifest_path: Path) -> pd.DataFrame:
    frames = []
    for fold, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"study_id": str})
        required = {"study_id", "fracture_true", "top3_mean"}
        if not required <= set(frame.columns):
            raise ValueError(f"Missing columns in {path}: {sorted(required - set(frame.columns))}")
        frame = frame[["study_id", "fracture_true", "top3_mean"]].copy()
        frame["fold"] = fold
        frames.append(frame)
    oof = pd.concat(frames, ignore_index=True)
    if oof["study_id"].duplicated().any():
        raise ValueError("A study occurs in more than one OOF fold")

    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str, "study_id": str})
    development = manifest.loc[manifest["split"] != "test", ["patient_id", "study_id"]].drop_duplicates()
    if development["study_id"].duplicated().any():
        raise ValueError("A development study maps to multiple patients")
    if set(oof["study_id"]) != set(development["study_id"]):
        raise ValueError("OOF studies do not exactly match all non-test manifest studies")
    oof = oof.merge(development, on="study_id", validate="one_to_one")
    if oof.groupby("patient_id")["fold"].nunique().max() != 1:
        raise ValueError("Patient leakage across OOF folds")
    return oof


def cross_fitted_predictions(oof: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    parts = []
    thresholds = []
    for fold in sorted(oof["fold"].unique()):
        train = oof[oof["fold"] != fold]
        heldout = oof[oof["fold"] == fold].copy()
        threshold, train_metrics, _ = select_threshold(
            train["fracture_true"].to_numpy(bool), train["top3_mean"].to_numpy(float)
        )
        heldout["crossfit_threshold"] = threshold
        heldout["crossfit_predicted"] = heldout["top3_mean"] >= threshold
        parts.append(heldout)
        thresholds.append({"heldout_fold": int(fold), "threshold": threshold,
                           "selection_data_studies": len(train),
                           "selection_data_metrics": train_metrics})
    return pd.concat(parts, ignore_index=True).sort_values(["fold", "study_id"]), thresholds


def apply_frozen_threshold_to_test(test_path: Path, threshold: float) -> tuple[pd.DataFrame, dict[str, object]]:
    test = pd.read_csv(test_path, dtype={"patient_id": str, "study_id": str})
    fold_columns = [f"fold{fold}_top3_mean" for fold in range(5)]
    required = {"patient_id", "study_id", "fracture_true", "ensemble_top3_mean", *fold_columns}
    if not required <= set(test.columns):
        raise ValueError(f"Missing fixed-test columns: {sorted(required - set(test.columns))}")
    votes = []
    for fold, column in enumerate(fold_columns):
        vote_column = f"fold{fold}_vote_at_frozen_threshold"
        test[vote_column] = test[column].to_numpy(float) >= threshold
        votes.append(vote_column)
    test["positive_votes"] = test[votes].sum(axis=1)
    test["vote_fraction"] = test["positive_votes"] / 5.0
    test["fracture_predicted_frozen_oof_threshold"] = test["positive_votes"] >= 3
    y_true = test["fracture_true"].to_numpy(bool)
    metrics = decision_metrics(
        y_true,
        test["fracture_predicted_frozen_oof_threshold"].to_numpy(bool),
        test["vote_fraction"].to_numpy(float),
    )
    metrics.update({
        "frozen_single_model_threshold": threshold,
        "ensemble_rule": "at least 3 of 5 models have study top3_mean >= frozen OOF threshold",
        "original_mean_slice_ensemble_pr_auc": float(average_precision_score(
            y_true, test["ensemble_top3_mean"].to_numpy(float)
        )),
        "original_mean_slice_ensemble_roc_auc": float(roc_auc_score(
            y_true, test["ensemble_top3_mean"].to_numpy(float)
        )),
    })
    return test, metrics


def run(args: argparse.Namespace) -> dict[str, object]:
    if (args.output / "threshold_report.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {args.output}")
    if len(args.fold_predictions) != 5:
        raise ValueError("Exactly five OOF fold prediction files are required")

    oof = load_oof(args.fold_predictions, args.manifest)
    y = oof["fracture_true"].to_numpy(bool)
    scores = oof["top3_mean"].to_numpy(float)
    final_threshold, apparent_metrics, sweep = select_threshold(y, scores)
    fixed_0_5_metrics = decision_metrics(y, scores >= 0.5, scores)
    fixed_0_5_metrics["threshold"] = 0.5
    crossfit, fold_thresholds = cross_fitted_predictions(oof)
    crossfit_metrics = decision_metrics(
        crossfit["fracture_true"].to_numpy(bool),
        crossfit["crossfit_predicted"].to_numpy(bool),
        crossfit["top3_mean"].to_numpy(float),
    )
    test, test_metrics = apply_frozen_threshold_to_test(args.test_predictions, final_threshold)
    test_bootstrap = patient_bootstrap_decisions(
        test, "fracture_predicted_frozen_oof_threshold", args.bootstrap_repeats, args.seed
    )

    oof = oof.copy()
    oof["final_oof_threshold"] = final_threshold
    oof["predicted_at_final_oof_threshold"] = scores >= final_threshold
    report = {
        "metric_scope": "independent binary fracture QWK at study level; ICH and MLS are unused",
        "selection_rule": "maximize pooled OOF binary QWK; ties: specificity, sensitivity, proximity to 0.5",
        "score_definition": "single-fold model study top3_mean",
        "oof_studies": len(oof), "oof_patients": int(oof["patient_id"].nunique()),
        "oof_positive_studies": int(y.sum()),
        "cross_fitted_oof_metrics_unbiased_for_threshold_layer": crossfit_metrics,
        "cross_fitted_thresholds": fold_thresholds,
        "all_oof_fixed_threshold_0_5_reference": fixed_0_5_metrics,
        "final_frozen_threshold_from_all_oof": final_threshold,
        "all_oof_apparent_metrics_optimistic_for_threshold_selection": apparent_metrics,
        "fixed_test_secondary_post_hoc_analysis": test_metrics,
        "fixed_test_secondary_patient_bootstrap": test_bootstrap,
        "test_caveat": (
            "The fixed test had already been inspected at threshold 0.5 before this OOF analysis. "
            "No test threshold sweep is performed, but this is secondary/post-hoc reanalysis, not a pristine test."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.output / "oof_predictions.csv", index=False)
    crossfit.to_csv(args.output / "oof_crossfit_predictions.csv", index=False)
    sweep.to_csv(args.output / "oof_threshold_sweep.csv", index=False)
    test.to_csv(args.output / "fixed_test_secondary_predictions.csv", index=False)
    (args.output / "threshold_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-predictions", nargs=5, type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data_prepared/skull_hu800_ww1600/manifest.csv")
    parser.add_argument("--test-predictions", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
