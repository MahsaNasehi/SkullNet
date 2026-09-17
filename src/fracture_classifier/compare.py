"""CPU-only, leakage-safe Study-level classifier/fusion OOF comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from aggregate_full_study_oof_cache import (
    load_cache, select_aggregator_and_threshold,
)
from fracture_classifier.data import CACHE, EXPERIMENT
from fracture_macro_f1_oof import binary_report, monotonic_operating_point_map, oracle_triage_report


ROOT = EXPERIMENT.parent.parent
AUTHORITATIVE = ROOT / "outputs/full_study_oof_ab_deployment_score_comparison_v2_169/paired_cross_fitted_predictions.csv"
ALPHAS = (0.25, 0.50, 0.75)
AGGREGATORS = ("max", "top3_mean", "top5_mean", "top10_percent_mean")
EXPECTED = {"tp": 7, "fp": 3, "fn": 17, "tn": 142}


def aggregate(scores: list[float], method: str) -> float:
    if not scores or not all(math.isfinite(x) and 0 <= x <= 1 for x in scores):
        raise ValueError("Study requires finite slice probabilities in [0,1]")
    ordered = sorted(scores, reverse=True)
    if method == "max":
        return ordered[0]
    if method in ("top3_mean", "top5_mean"):
        k = 3 if method == "top3_mean" else 5
    elif method == "top10_percent_mean":
        k = max(1, math.ceil(len(ordered) * 0.10))
    else:
        raise ValueError(f"Unknown classifier aggregation method: {method}")
    return float(np.mean(ordered[:min(k, len(ordered))]))


def fusion(detector: np.ndarray, classifier: np.ndarray, alpha: float) -> np.ndarray:
    if alpha not in ALPHAS:
        raise ValueError("Alpha must come from the frozen grid")
    result = alpha * np.asarray(detector, float) + (1.0 - alpha) * np.asarray(classifier, float)
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("Fusion probability outside [0,1]")
    return result


def _metric(table: pd.DataFrame, scores: np.ndarray) -> dict:
    triage = oracle_triage_report(table, scores)
    binary = binary_report(table["fracture_true"].astype(bool), scores)
    binary["f1"] = (2 * binary["tp"] / (2 * binary["tp"] + binary["fp"] + binary["fn"])
                    if 2 * binary["tp"] + binary["fp"] + binary["fn"] else 0.0)
    return {"triage": triage, "fracture": binary}


def _rank(table: pd.DataFrame, scores: np.ndarray) -> tuple:
    metrics = _metric(table, scores)
    triage, binary = metrics["triage"], metrics["fracture"]
    return (triage["pooled_macro_f1"], -binary["fp"], binary["specificity"],
            binary["sensitivity"], binary["pr_auc"])


def load_detector_baseline() -> pd.DataFrame:
    raw = load_cache(CACHE)
    if set(raw["score_generation_protocol"]) != {"deployment_score"}:
        raise RuntimeError("Run A detector cache is not deployment_score")
    for encoded in raw["inference_signature_json"].unique():
        signature = json.loads(encoded)
        if (int(signature.get("imgsz", -1)), float(signature.get("conf", -1)),
                float(signature.get("nms_iou", -1))) != (768, 0.01, 0.5):
            raise RuntimeError("Run A detector cache inference parameters changed")
    truth = pd.read_csv(AUTHORITATIVE, dtype={"study_id": str, "patient_id": str})
    required = ["study_id", "patient_id", "fold", "V_EDH", "V_SDH", "V_IPH", "V_SAH",
                "V_IVH", "MLS_mm", "fracture_true", "crossfit_fracture_prob_a"]
    if len(truth) != 169 or truth["study_id"].duplicated().any():
        raise RuntimeError("Authoritative truth/baseline artifact is not 169 unique Studies")
    joined = raw.merge(truth[required], on=["study_id", "patient_id", "fold"], validate="one_to_one")
    if len(joined) != 169 or joined.groupby("fold")["study_id"].nunique().tolist() != [33, 33, 32, 36, 35]:
        raise RuntimeError("Detector raw cache does not match authoritative cohort")
    # Reuse the already-authoritative held-out cross-fitted detector scores;
    # never spend GPU time re-inferring the identical detector predictions.
    table = joined.sort_values(["fold", "study_id"]).copy()
    table["detector_probability"] = table["crossfit_fracture_prob_a"].astype(float)
    if not np.isfinite(table["detector_probability"]).all() or not table["detector_probability"].between(0, 1).all():
        raise RuntimeError("Authoritative detector OOF scores are not finite probabilities")
    metrics = _metric(table, table["detector_probability"].to_numpy(float))
    binary = metrics["fracture"]
    if {key: binary[key] for key in EXPECTED} != EXPECTED:
        raise RuntimeError("Run A fracture baseline TP/FP/FN/TN mismatch")
    if not math.isclose(binary["pr_auc"], 0.5355486380673358, abs_tol=1e-9):
        raise RuntimeError("Run A Study PR-AUC baseline mismatch")
    if not math.isclose(binary["roc_auc"], 0.7298850574712643, abs_tol=1e-9):
        raise RuntimeError("Run A ROC-AUC baseline mismatch")
    if not math.isclose(metrics["triage"]["pooled_macro_f1"], 0.9696273781380164, abs_tol=1e-9):
        raise RuntimeError("Run A oracle Macro-F1 baseline mismatch")
    return table


def attach_classifier_scores(detector: pd.DataFrame, stage: str = "stage1") -> pd.DataFrame:
    if stage not in ("stage1", "stage2"):
        raise ValueError("Classifier stage must be stage1 or stage2")
    rows = []
    for fold in range(5):
        root = EXPERIMENT / ("slice_predictions" if stage == "stage1" else "slice_predictions_stage2") / f"fold_{fold}"
        marker_path = root / "COMPLETE.json"
        if not marker_path.is_file():
            raise FileNotFoundError(f"Classifier OOF inference incomplete: {marker_path}")
        marker = json.loads(marker_path.read_text())
        expected = set(map(str, marker["expected_study_ids"]))
        actual = set()
        for path in sorted((root / "studies").glob("*.json")):
            payload = json.loads(path.read_text())
            study, patient = str(payload["study_id"]), str(payload["patient_id"])
            if int(payload["fold"]) != fold or study in actual or study not in expected:
                raise RuntimeError("Unexpected/duplicate classifier Study")
            source = json.loads((CACHE / f"fold_{fold}/studies/{study}.json").read_text())
            if patient != str(source["patient_id"]) or payload["sop_uids"] != [str(x["sop_uid"]) for x in source["slices"]]:
                raise RuntimeError("Classifier/Run A exact Study-SOP coverage mismatch")
            if payload["signature"] != marker["signature"]:
                raise RuntimeError("Mixed classifier inference protocols")
            scores = list(map(float, payload["slice_probabilities"]))
            if len(scores) != len(payload["sop_uids"]):
                raise RuntimeError("Classifier slice score count mismatch")
            row = {"fold": fold, "study_id": study, "patient_id": patient}
            row.update({f"classifier_{method}": aggregate(scores, method) for method in AGGREGATORS})
            rows.append(row)
            actual.add(study)
        if actual != expected or len(actual) != [33, 33, 32, 36, 35][fold]:
            raise RuntimeError("Classifier OOF Fold coverage incomplete")
    joined = detector.merge(pd.DataFrame(rows), on=["study_id", "patient_id", "fold"], validate="one_to_one")
    if len(joined) != 169 or joined.groupby("patient_id")["fold"].nunique().max() != 1:
        raise RuntimeError("Classifier/detector pairing or patient grouping failed")
    return joined


def crossfit_classifier_fusion(table: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    parts, selections = [], []
    for fold in range(5):
        training = table[table["fold"] != fold].copy()
        heldout = table[table["fold"] == fold].copy()
        # Selection uses only other folds. Official downstream threshold remains 0.5.
        selected = max(AGGREGATORS, key=lambda method: _rank(
            training, training[f"classifier_{method}"].to_numpy(float)))
        classifier_train = training[f"classifier_{selected}"].to_numpy(float)
        classifier_val = heldout[f"classifier_{selected}"].to_numpy(float)
        detector_method, detector_threshold, _ = select_aggregator_and_threshold(training)
        detector_train = monotonic_operating_point_map(training[detector_method], detector_threshold)
        detector_val = heldout["detector_probability"].to_numpy(float)
        alpha = max(ALPHAS, key=lambda value: _rank(training, fusion(detector_train, classifier_train, value)))
        heldout["classifier_probability"] = classifier_val
        heldout["fusion_probability"] = fusion(detector_val, classifier_val, alpha)
        parts.append(heldout)
        selections.append({"heldout_fold": fold, "classifier_aggregator": selected,
                           "fusion_alpha": alpha, "detector_aggregator": detector_method,
                           "detector_raw_operating_point": detector_threshold,
                           "selection_source_folds": [x for x in range(5) if x != fold],
                           "primary_selection_metric": "oracle_other_heads_macro_f1",
                           "guardrails": ["lower FP", "specificity", "sensitivity", "PR-AUC"]})
    return pd.concat(parts, ignore_index=True).sort_values(["fold", "study_id"]), selections


def patient_bootstrap(table: pd.DataFrame, column: str, repeats: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    patient_truth = table.groupby("patient_id")["fracture_true"].max().astype(bool)
    positive = patient_truth[patient_truth].index.to_numpy()
    negative = patient_truth[~patient_truth].index.to_numpy()
    groups = {patient: group for patient, group in table.groupby("patient_id", sort=False)}
    deltas = []
    for _ in range(repeats):
        sample_ids = np.r_[rng.choice(positive, len(positive), replace=True),
                           rng.choice(negative, len(negative), replace=True)]
        sample = pd.concat([groups[patient] for patient in sample_ids], ignore_index=True)
        a = oracle_triage_report(sample, sample["detector_probability"].to_numpy(float))["pooled_macro_f1"]
        b = oracle_triage_report(sample, sample[column].to_numpy(float))["pooled_macro_f1"]
        deltas.append(b - a)
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def compare(output: Path, repeats: int, seed: int, baseline_only: bool, stage: str = "stage1") -> dict:
    detector = load_detector_baseline()
    baseline = _metric(detector, detector["detector_probability"].to_numpy(float))
    if baseline_only:
        return {"baseline_reproduced_exactly": True, "run_a_detector_only": baseline}
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite classifier OOF comparison: {output}")
    joined = attach_classifier_scores(detector, stage)
    evaluated, selections = crossfit_classifier_fusion(joined)
    systems = {name: _metric(evaluated, evaluated[column].to_numpy(float)) for name, column in (
        ("detector_only", "detector_probability"),
        ("classifier_only", "classifier_probability"),
        ("weighted_fusion", "fusion_probability"),
    )}
    base_pred = evaluated["detector_probability"].to_numpy(float) >= 0.5
    truth = evaluated["fracture_true"].astype(bool).to_numpy()
    for name, column in (("classifier_only", "classifier_probability"), ("weighted_fusion", "fusion_probability")):
        current = evaluated[column].to_numpy(float) >= 0.5
        systems[name]["positive_studies_rescued_vs_run_a"] = int((truth & ~base_pred & current).sum())
        systems[name]["new_false_positive_studies_vs_run_a"] = int((~truth & ~base_pred & current).sum())
        systems[name]["net_fn_reduction"] = 17 - systems[name]["fracture"]["fn"]
        systems[name]["net_fp_change"] = systems[name]["fracture"]["fp"] - 3
        systems[name]["paired_patient_clustered_95pct_ci_macro_f1_delta"] = patient_bootstrap(
            evaluated, column, repeats, seed)
    report = {"cohort": "169 explicit Run A OOF Studies, 155 patients", "official_threshold": 0.5,
              "metric_name": "oracle_other_heads_macro_f1_not_submission_macro_f1",
              "baseline_reproduced_exactly": True, "alpha_grid": list(ALPHAS),
              "classifier_aggregation_candidates": list(AGGREGATORS),
              "stage": stage, "selection": selections, "systems": systems,
              "bootstrap_repeats": repeats, "bootstrap_seed": seed,
              "note": "Candidate comparison on the same OOF is exploratory; no fixed test or final submission metric was used."}
    output.mkdir(parents=True)
    evaluated.to_csv(output / "paired_oof_studies.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "comparison")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
    args = parser.parse_args()
    print(json.dumps(compare(args.output, args.bootstrap_repeats, args.seed, args.baseline_only, args.stage), indent=2))


if __name__ == "__main__":
    main()
