"""CPU-only fracture-marginal OOF evaluation and cross-fitted post-processing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score, confusion_matrix, roc_auc_score

from triage_macro_f1 import triage_from_intermediates, triage_macro_f1_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLD_PREDICTIONS = [
    ROOT / ("outputs/yolo26s_p2_hu800_ww1600_fold%d/study_evaluation/study_predictions.csv" % fold)
    for fold in range(5)
]
DEFAULT_MANIFEST = ROOT / "data_prepared/skull_hu800_ww1600/manifest.csv"
INTERMEDIATE_COLUMNS = ("V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH", "MLS_mm")


def monotonic_operating_point_map(scores: Sequence[float], threshold: float) -> np.ndarray:
    """Map a raw operating point to probability 0.5 without changing rank/order."""
    values = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if threshold == 0.0:
        return 0.5 + 0.5 * values
    if threshold == 1.0:
        return 0.5 * values
    return np.where(
        values < threshold,
        0.5 * values / threshold,
        0.5 + 0.5 * (values - threshold) / (1.0 - threshold),
    )


def binary_report(y_true: Sequence[bool], probability: Sequence[float]) -> Dict[str, Any]:
    truth = np.asarray(y_true, dtype=bool)
    scores = np.asarray(probability, dtype=float)
    predicted = scores >= 0.5
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[False, True]).ravel()
    result = {
        "threshold_in_probability_space": 0.5,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "binary_qwk_diagnostic": float(cohen_kappa_score(
            truth.astype(int), predicted.astype(int), labels=[0, 1], weights="quadratic"
        )),
    }  # type: Dict[str, Any]
    if len(np.unique(truth)) == 2:
        result["pr_auc"] = float(average_precision_score(truth, scores))
        result["roc_auc"] = float(roc_auc_score(truth, scores))
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None
    return result


def oracle_triage_report(table: pd.DataFrame, probabilities: Sequence[float]) -> Dict[str, Any]:
    """Evaluate fracture while keeping the other heads at ground truth values."""
    if len(table) != len(probabilities):
        raise ValueError("One fracture probability is required per study")
    y_true = []  # type: List[int]
    y_pred = []  # type: List[int]
    for row, probability in zip(table.itertuples(index=False), probabilities):
        common = {name: float(getattr(row, name)) for name in INTERMEDIATE_COLUMNS}
        y_true.append(triage_from_intermediates(dict(common, fracture_prob=float(row.fracture_true))))
        y_pred.append(triage_from_intermediates(dict(common, fracture_prob=float(probability))))
    return {
        "metric_name": "oracle_other_heads_macro_f1",
        "warning": "Ground-truth ICH volumes and ground-truth MLS are oracle inputs; this is not final submission Macro-F1.",
        **triage_macro_f1_report(y_true, y_pred),
    }


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    values = np.unique(np.clip(np.asarray(scores, dtype=float), 0.0, 1.0))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be non-empty and finite")
    mids = (values[:-1] + values[1:]) / 2.0
    return np.unique(np.r_[0.0, 0.5, mids, 1.0])


def select_operating_point(table: pd.DataFrame, score_column: str) -> Tuple[float, Dict[str, Any]]:
    """Select threshold by oracle Macro-F1 with deterministic binary guardrails."""
    truth = table["fracture_true"].astype(bool).to_numpy()
    scores = table[score_column].to_numpy(dtype=float)
    candidates = []
    for threshold in threshold_candidates(scores):
        mapped = monotonic_operating_point_map(scores, float(threshold))
        oracle = oracle_triage_report(table, mapped)
        binary = binary_report(truth, mapped)
        candidates.append((float(threshold), oracle, binary))
    selected = max(
        candidates,
        key=lambda item: (
            item[1]["pooled_macro_f1"],
            -item[2]["fp"],
            item[2]["specificity"] if item[2]["specificity"] is not None else -1.0,
            item[2]["sensitivity"] if item[2]["sensitivity"] is not None else -1.0,
            -abs(item[0] - 0.5),
        ),
    )
    return selected[0], {
        "selection_metric": "oracle_other_heads_macro_f1",
        "selection_oracle_metrics": selected[1],
        "selection_binary_guardrails": selected[2],
    }


def load_oof(paths: Sequence[Path], manifest_path: Path, score_column: str) -> pd.DataFrame:
    if len(paths) != 5:
        raise ValueError("Exactly five held-out fold prediction CSVs are required")
    frames = []
    required = {"study_id", "fracture_true", score_column, *INTERMEDIATE_COLUMNS}
    for fold, path in enumerate(paths):
        frame = pd.read_csv(path, dtype={"study_id": str})
        missing = required - set(frame.columns)
        if missing:
            raise ValueError("%s is missing columns %s" % (path, sorted(missing)))
        frame = frame[list(required)].copy()
        frame["fold"] = fold
        frames.append(frame)
    oof = pd.concat(frames, ignore_index=True)
    if oof["study_id"].duplicated().any():
        raise ValueError("A study occurs in more than one held-out fold")
    manifest = pd.read_csv(manifest_path, dtype={"study_id": str, "patient_id": str})
    development = manifest.loc[manifest["split"] != "test", ["study_id", "patient_id"]].drop_duplicates()
    if development["study_id"].duplicated().any():
        raise ValueError("A development study maps to multiple patients")
    if set(oof["study_id"]) != set(development["study_id"]):
        raise ValueError("OOF studies do not exactly equal the non-fixed-test manifest studies")
    oof = oof.merge(development, on="study_id", validate="one_to_one")
    if int(oof.groupby("patient_id")["fold"].nunique().max()) != 1:
        raise ValueError("Patient leakage across OOF folds")
    return oof.sort_values(["fold", "study_id"]).reset_index(drop=True)


def cross_fitted_postprocess(table: pd.DataFrame, score_column: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    parts = []
    selections = []
    for fold in sorted(table["fold"].unique()):
        selection = table[table["fold"] != fold]
        heldout = table[table["fold"] == fold].copy()
        threshold, details = select_operating_point(selection, score_column)
        heldout["selected_raw_operating_point"] = threshold
        heldout["crossfit_fracture_prob"] = monotonic_operating_point_map(
            heldout[score_column].to_numpy(dtype=float), threshold
        )
        parts.append(heldout)
        selections.append({
            "heldout_fold": int(fold),
            "selected_raw_operating_point": threshold,
            "selection_studies": int(len(selection)),
            **details,
        })
    return pd.concat(parts, ignore_index=True).sort_values(["fold", "study_id"]), selections


def run(args: argparse.Namespace) -> Dict[str, Any]:
    report_path = args.output / "report.json"
    if report_path.exists():
        raise FileExistsError("Refusing to overwrite %s" % report_path)
    table = load_oof(args.fold_predictions, args.manifest, args.score_column)
    raw = table[args.score_column].to_numpy(dtype=float)
    truth = table["fracture_true"].astype(bool).to_numpy()
    crossfit, selections = cross_fitted_postprocess(table, args.score_column)
    crossfit_probability = crossfit["crossfit_fracture_prob"].to_numpy(dtype=float)

    baselines = {
        "fracture_prob_all_zero": oracle_triage_report(table, np.zeros(len(table))),
        "fracture_prob_all_one": oracle_triage_report(table, np.ones(len(table))),
        "current_raw_oof_fracture_scores": oracle_triage_report(table, raw),
    }
    baselines["fracture_prob_all_zero"]["binary_metrics"] = binary_report(truth, np.zeros(len(table)))
    baselines["fracture_prob_all_one"]["binary_metrics"] = binary_report(truth, np.ones(len(table)))
    baselines["current_raw_oof_fracture_scores"]["binary_metrics"] = binary_report(truth, raw)
    pooled = oracle_triage_report(crossfit, crossfit_probability)
    pooled["binary_metrics"] = binary_report(crossfit["fracture_true"].astype(bool), crossfit_probability)
    apparent_threshold, apparent_details = select_operating_point(table, args.score_column)
    report = {
        "evaluation_partition": "patient-grouped OOF only; fixed test is excluded",
        "primary_metric": "oracle_other_heads_macro_f1",
        "official_downstream_fracture_threshold": 0.5,
        "score_column": args.score_column,
        "baselines": baselines,
        "cross_fitted": pooled,
        "fold_selections": selections,
        "apparent_deployment_mapping": {
            "raw_operating_point": apparent_threshold,
            "warning": "Selected on all OOF; deployment parameter only, not unbiased performance.",
            **apparent_details,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    crossfit.to_csv(args.output / "cross_fitted_predictions.csv", index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-predictions", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--score-column", default="top3_mean")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fracture_macro_f1_run_a_oof")
    args = parser.parse_args()
    if not args.fold_predictions:
        args.fold_predictions = DEFAULT_FOLD_PREDICTIONS
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
