"""CPU-only OOF FN audit and source-fold-fitted calibration/gated rescue.

The fixed Run A deployment detector transform was previously selected on all
OOF studies. New C/tau/model coefficients are cross-fitted, but the resulting
scores are conditional on that prior apparent transform; they are not an
unbiased estimate for a completely untouched protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from fracture_classifier.calibration import C_GRID, choose_c, fit_logistic, predict_logistic
from fracture_classifier.compare import (CACHE, EXPERIMENT, _metric, attach_classifier_scores,
                                         crossfit_classifier_fusion, load_detector_baseline)
from fracture_classifier.gated_fusion import TAU_GRID, choose_tau, gated_probability
from fracture_macro_f1_oof import oracle_triage_report
from submit import model as submit_model
from triage_macro_f1 import triage_from_intermediates


FN_OUTPUT = EXPERIMENT / "fn_audit"
OUTPUT = EXPERIMENT / "fusion_v2"
PAIRED = EXPERIMENT / "comparison/paired_oof_studies.csv"
MANIFEST = EXPERIMENT / "slice_label_manifest.csv"
EPS = 1e-15
THRESHOLD = 0.5
METRICS_FOR_CI = ("sensitivity", "specificity", "pr_auc", "fracture_f1",
                  "oracle_other_heads_macro_f1")
INTERMEDIATE = ("V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH", "MLS_mm")


def load_verified_oof() -> tuple[pd.DataFrame, dict]:
    """Hard-stop unless the exact historical Run A baseline is reproduced."""
    baseline = load_detector_baseline()  # exact TP/AP/AUC/Macro-F1 assertions live here
    metric = _metric(baseline, baseline["detector_probability"].to_numpy(float))
    b = metric["fracture"]
    if ([b[key] for key in ("tp", "fp", "fn", "tn")] != [7, 3, 17, 142]
            or not math.isclose(b["pr_auc"], 0.5355486380673358, abs_tol=1e-12)
            or not math.isclose(b["roc_auc"], 0.7298850574712643, abs_tol=1e-12)
            or not math.isclose(metric["triage"]["pooled_macro_f1"], 0.9696273781380164, abs_tol=1e-12)):
        raise RuntimeError("Authoritative Run A detector baseline did not reproduce EXACTLY")
    joined = attach_classifier_scores(baseline)
    historical, selections = crossfit_classifier_fusion(joined)
    saved = pd.read_csv(PAIRED, dtype={"study_id": str, "patient_id": str})
    columns = ["fold", "study_id", "patient_id"]
    paired = historical.merge(saved[columns + ["detector_probability", "classifier_probability",
                                         "fusion_probability"]], on=columns, validate="one_to_one",
                              suffixes=("", "_saved"))
    if len(paired) != 169 or paired["study_id"].duplicated().any():
        raise RuntimeError("Historical OOF classifier comparison cohort changed")
    for column in ("detector_probability", "classifier_probability", "fusion_probability"):
        if not np.allclose(paired[column].to_numpy(float), paired[column + "_saved"].to_numpy(float),
                           atol=1e-12, rtol=0):
            raise RuntimeError(f"Historical OOF {column} does not reproduce")
        paired = paired.drop(columns=[column + "_saved"])
    if (paired.groupby("patient_id")["fold"].nunique().max() != 1
            or paired.groupby("fold")["study_id"].nunique().tolist() != [33, 33, 32, 36, 35]
            or paired["patient_id"].nunique() != 155
            or int(paired["fracture_true"].astype(bool).sum()) != 24):
        raise RuntimeError("Patient leakage or incomplete OOF cohort")
    manifest = pd.read_csv(CACHE.parents[1] / "data_prepared/skull_hu800_ww1600/manifest.csv",
                           dtype={"study_id": str, "patient_id": str})
    fixed = manifest[manifest["split"] == "test"]
    if set(paired["study_id"]) & set(fixed["study_id"]) or set(paired["patient_id"]) & set(fixed["patient_id"]):
        raise RuntimeError("Fixed-test Study/patient entered OOF analysis")
    if (submit_model.AGGREGATION_METHOD != "top10_percent_mean"
            or submit_model.RAW_MACRO_F1_THRESHOLD != 0.39717610677083337
            or submit_model.OFFICIAL_FRACTURE_THRESHOLD != THRESHOLD):
        raise RuntimeError("No unique established Run A deployment detector transform")
    paired["classifier_top5"] = paired["classifier_top5_mean"].astype(float)
    paired["detector_fixed"] = paired["top10_percent_mean"].map(
        lambda value: submit_model.rescale_for_macro_f1(
            float(value), submit_model.RAW_MACRO_F1_THRESHOLD))
    paired["f0_fixed"] = (0.75 * paired["detector_fixed"] + 0.25 * paired["classifier_top5"]).clip(0, 1)
    paired = paired.sort_values(["fold", "study_id"]).reset_index(drop=True)
    return paired, {"run_a_detector_only": metric, "historical_selections": selections,
                    "baseline_reproduced_exactly": True,
                    "detector_transform": {
                        "aggregator": submit_model.AGGREGATION_METHOD,
                        "raw_operating_point": submit_model.RAW_MACRO_F1_THRESHOLD,
                        "prior_selection_scope": "all-OOF apparent deployment selection; fixed before this v2 experiment",
                        "crossfit_caveat": "New C/tau are source-fold-only, but this fixed detector mapping had earlier all-OOF label selection."}}


def category(detector: float, classifier: float) -> str:
    if not 0 <= detector < THRESHOLD:
        raise ValueError("Diagnostic FN category requires detector-negative probability")
    if detector >= 0.30:
        return "near_threshold"
    return "detector_low_classifier_high" if classifier >= THRESHOLD else "both_low"


def _slice_info(row) -> dict:
    fold, study = int(row.fold), str(row.study_id)
    raw = json.loads((CACHE / f"fold_{fold}/studies/{study}.json").read_text())
    cls = json.loads((EXPERIMENT / f"slice_predictions/fold_{fold}/studies/{study}.json").read_text())
    sops = [str(item["sop_uid"]) for item in raw["slices"]]
    if (raw["series_id"] != study or str(raw["patient_id"]) != str(row.patient_id)
            or cls["sop_uids"] != sops or cls["study_id"] != study):
        raise RuntimeError("Detector/classifier raw slice cache is not aligned")
    d = np.asarray([float(item["max_confidence"]) for item in raw["slices"]])
    c = np.asarray(cls["slice_probabilities"], dtype=float)
    if len(d) != len(c) or len(d) != int(row.included_target_series_slices):
        raise RuntimeError("Full-study OOF slice count mismatch")
    n10 = max(1, math.ceil(len(d) * 0.10))
    agg = lambda scores, count: float(np.mean(np.sort(scores)[::-1][:min(count, len(scores))]))
    if (not math.isclose(agg(d, n10), float(row.top10_percent_mean), abs_tol=1e-9)
            or not math.isclose(agg(c, 5), float(row.classifier_top5), abs_tol=1e-9)):
        raise RuntimeError("Raw slice aggregation differs from verified OOF Study table")
    return {"slice_count": len(d), "detector_slice_max": float(d.max()),
            "detector_slice_top3_mean": agg(d, 3),
            "detector_slice_top5_mean": agg(d, 5),
            "detector_slice_top10_percent_mean": agg(d, n10),
            "classifier_slice_max": float(c.max()),
            "classifier_slice_top3_mean": agg(c, 3),
            "classifier_slice_top5_mean": agg(c, 5),
            "classifier_slice_top10_percent_mean": agg(c, n10),
            "top_detector_slice_ids": ";".join(sops[i] for i in np.argsort(-d)[:3]),
            "top_classifier_slice_ids": ";".join(sops[i] for i in np.argsort(-c)[:3])}


def _triage_for(row, fracture: float) -> int:
    return triage_from_intermediates({**{key: float(getattr(row, key)) for key in INTERMEDIATE},
                                      "fracture_prob": float(fracture)})


def fn_and_distribution_audit(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    truth = table["fracture_true"].astype(bool)
    fn = table[truth & (table["detector_probability"] < THRESHOLD)].copy()
    if len(fn) != 17:
        raise RuntimeError("Run A FN count is not 17")
    fn_rows, impact_rows, review_rows = [], [], []
    labels = pd.read_csv(MANIFEST, dtype={"study_id": str, "sop_uid": str})
    gt_positive_sops = labels[(labels["is_reviewed"].astype(bool)) & (labels["fracture_label"] == 1)].groupby(
        "study_id")["sop_uid"].apply(lambda values: ";".join(values.astype(str))).to_dict()
    for row in fn.itertuples(index=False):
        detector, classifier, fusion = float(row.detector_probability), float(row.classifier_top5), float(row.fusion_probability)
        diagnostic = category(detector, classifier)
        info = _slice_info(row)
        common = {"study_id": str(row.study_id), "patient_id": str(row.patient_id), "fold": int(row.fold),
                  "ground_truth": 1, "detector_probability": detector,
                  "classifier_top5_probability": classifier, "existing_fusion_probability": fusion,
                  "detector_prediction": 0, "classifier_prediction": int(classifier >= THRESHOLD),
                  "existing_fusion_prediction": int(fusion >= THRESHOLD),
                  "diagnostic_category": diagnostic, **info}
        fn_rows.append(common)
        negative_triage, positive_triage = _triage_for(row, 0.0), _triage_for(row, 1.0)
        transition = "no change" if negative_triage == positive_triage else f"{negative_triage} -> {positive_triage}"
        if transition not in {"no change", "0 -> 1", "0 -> 2", "1 -> 2"}:
            raise RuntimeError(f"Unexpected oracle triage transition: {transition}")
        impact_rows.append({"study_id": str(row.study_id), "patient_id": str(row.patient_id),
                            "fold": int(row.fold), "triage_if_fracture_negative": negative_triage,
                            "triage_if_fracture_positive": positive_triage, "transition": transition,
                            "changes_triage": negative_triage != positive_triage})
        review_rows.append({"study_id": str(row.study_id), "fold": int(row.fold),
                            "detector_probability": detector, "classifier_probability": classifier,
                            "fusion_probability": fusion, "diagnostic_category": diagnostic,
                            "priority": {"detector_low_classifier_high": 1, "near_threshold": 2, "both_low": 3}[diagnostic],
                            "top_detector_slice_ids": info["top_detector_slice_ids"],
                            "top_classifier_slice_ids": info["top_classifier_slice_ids"],
                            "gt_positive_slice_ids": gt_positive_sops.get(str(row.study_id), "")})
    # Additional positive disagreement is diagnostic, never a selection feature.
    for row in table[truth & (table["detector_probability"] >= THRESHOLD)].itertuples(index=False):
        if abs(float(row.detector_probability) - float(row.classifier_top5)) < 0.30:
            continue
        info = _slice_info(row)
        review_rows.append({"study_id": str(row.study_id), "fold": int(row.fold),
                            "detector_probability": float(row.detector_probability),
                            "classifier_probability": float(row.classifier_top5),
                            "fusion_probability": float(row.fusion_probability),
                            "diagnostic_category": "positive_detector_classifier_disagreement",
                            "priority": 4, "top_detector_slice_ids": info["top_detector_slice_ids"],
                            "top_classifier_slice_ids": info["top_classifier_slice_ids"],
                            "gt_positive_slice_ids": gt_positive_sops.get(str(row.study_id), "")})
    distributions = []
    for system, column in (("detector", "detector_probability"),
                           ("classifier", "classifier_top5"),
                           ("existing_fusion", "fusion_probability")):
        for label, subset in (("positive", table[truth]), ("negative", table[~truth])):
            values = subset[column].to_numpy(float)
            quantiles = np.percentile(values, [0, 10, 25, 50, 75, 90, 100])
            distributions.append({"system": system, "class": label, "n": len(values),
                                  **dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"),
                                             map(float, quantiles)))})
    fn_table = pd.DataFrame(fn_rows).sort_values("detector_probability").reset_index(drop=True)
    impacts = pd.DataFrame(impact_rows).sort_values("study_id")
    reviews = pd.DataFrame(review_rows).sort_values(["priority", "study_id"])
    summary = {"detector_false_negatives": len(fn_table),
               "category_counts": fn_table["diagnostic_category"].value_counts().to_dict(),
               "score_extrema": {},
               "detector_fn_scores_sorted_ascending": fn_table[["study_id", "detector_probability"]].to_dict("records"),
               "decision_impact": {"changes_triage": int(impacts["changes_triage"].sum()),
                                   "no_change": int((~impacts["changes_triage"]).sum()),
                                   "transition_counts": impacts["transition"].value_counts().to_dict()}}
    for name, column in (("detector", "detector_probability"),
                         ("classifier", "classifier_top5"),
                         ("existing_fusion", "fusion_probability")):
        summary["score_extrema"][name] = {"min_positive": float(table.loc[truth, column].min()),
                                          "max_negative": float(table.loc[~truth, column].max())}
    return fn_table, pd.DataFrame(distributions), impacts, reviews, summary


def plot_distributions(table: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = table["fracture_true"].astype(bool)
    for name, column in (("detector", "detector_probability"),
                         ("classifier", "classifier_top5"),
                         ("fusion", "fusion_probability")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(table.loc[~truth, column], bins=np.linspace(0, 1, 31), alpha=0.6, label="negative")
        ax.hist(table.loc[truth, column], bins=np.linspace(0, 1, 31), alpha=0.6, label="positive")
        ax.axvline(THRESHOLD, color="black", linestyle="--", label="official 0.5")
        ax.set(xlabel="Study probability", ylabel="Studies", title=f"Run A OOF {name} score distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"score_histogram_{name}.png", dpi=150)
        plt.close(fig)


def crossfit_v2(table: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    parts, selections = [], []
    for heldout_fold in range(5):
        source = table[table["fold"] != heldout_fold].copy()
        heldout = table[table["fold"] == heldout_fold].copy()
        if (len(source) + len(heldout) != 169 or heldout.empty
                or set(source["patient_id"]) & set(heldout["patient_id"])):
            raise RuntimeError("Outer fold patient/Study leakage")
        c, inner, c_candidates = choose_c(source)
        tau, tau_candidates = choose_tau(inner)
        model = fit_logistic(source, c)
        heldout["f1_meta"] = predict_logistic(model, heldout)
        heldout["f2_gated"] = gated_probability(heldout["detector_fixed"], heldout["f1_meta"], tau)
        parts.append(heldout)
        selections.append({"heldout_fold": heldout_fold,
                           "source_folds": [fold for fold in range(5) if fold != heldout_fold],
                           "source_studies": len(source), "heldout_studies": len(heldout),
                           "selected_C": c, "selected_tau_low": tau,
                           "C_selection_metric": "minimum inner-source-fold pooled log loss",
                           "tau_selection_metric": "inner-source-fold oracle-other-heads Macro-F1, then FP/specificity/sensitivity/PR-AUC",
                           "C_candidates": c_candidates, "tau_candidates": tau_candidates,
                           "beta0": float(model.intercept_[0]),
                           "beta_detector_logit": float(model.coef_[0, 0]),
                           "beta_classifier_logit": float(model.coef_[0, 1])})
    result = pd.concat(parts, ignore_index=True).sort_values(["fold", "study_id"]).reset_index(drop=True)
    if len(result) != 169 or result["study_id"].duplicated().any():
        raise RuntimeError("Incomplete outer cross-fitted v2 predictions")
    return result, selections


def system_report(table: pd.DataFrame, column: str) -> dict:
    truth = table["fracture_true"].astype(bool).to_numpy()
    scores = table[column].to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise RuntimeError(f"Invalid Study scores for {column}")
    binary = _metric(table, scores)["fracture"]
    binary["balanced_accuracy"] = (binary["sensitivity"] + binary["specificity"]) / 2
    binary["brier_score"] = float(brier_score_loss(truth, scores))
    binary["log_loss"] = float(log_loss(truth, np.clip(scores, EPS, 1 - EPS), labels=[0, 1]))
    triage = oracle_triage_report(table, scores)
    base_pred = table["detector_probability"].to_numpy(float) >= THRESHOLD
    now = scores >= THRESHOLD
    return {"binary_fracture": binary, "oracle_other_heads_triage": triage,
            "fn_rescued_vs_run_a": int(np.sum(truth & ~base_pred & now)),
            "new_fp_vs_run_a": int(np.sum(~truth & ~base_pred & now)),
            "fp_removed_vs_run_a": int(np.sum(~truth & base_pred & ~now))}


def _oracle_labels(table: pd.DataFrame, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth_labels, prediction_labels = [], []
    for row, value in zip(table.itertuples(index=False), probability):
        truth_labels.append(_triage_for(row, float(row.fracture_true)))
        prediction_labels.append(_triage_for(row, float(value)))
    return np.asarray(truth_labels, int), np.asarray(prediction_labels, int)


def _macro_f1_fast(truth: np.ndarray, prediction: np.ndarray) -> float:
    matrix = np.bincount(3 * truth + prediction, minlength=9).reshape(3, 3)
    tp = np.diag(matrix)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.mean(np.divide(2 * tp, denominator, out=np.zeros(3, float), where=denominator != 0)))


def _bootstrap_values(truth: np.ndarray, probability: np.ndarray,
                      triage_truth: np.ndarray, triage_prediction: np.ndarray,
                      indices: np.ndarray) -> dict:
    y = truth[indices]
    score = probability[indices]
    decision = score >= THRESHOLD
    tp = np.sum(y & decision)
    fn = np.sum(y & ~decision)
    tn = np.sum(~y & ~decision)
    fp = np.sum(~y & decision)
    return {"sensitivity": float(tp / (tp + fn)),
            "specificity": float(tn / (tn + fp)),
            "pr_auc": float(average_precision_score(y, score)),
            "fracture_f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "oracle_other_heads_macro_f1": _macro_f1_fast(triage_truth[indices], triage_prediction[indices])}


def patient_bootstrap(table: pd.DataFrame, systems: dict[str, str], repeats: int, seed: int) -> dict:
    """Paired stratified patient-cluster bootstrap; boxes/slices are never sampled."""
    if repeats <= 0:
        raise ValueError("Bootstrap repeats must be positive")
    rng = np.random.default_rng(seed)
    truth = table["fracture_true"].astype(bool).to_numpy()
    patient_truth = table.groupby("patient_id")["fracture_true"].max().astype(bool)
    positive = patient_truth[patient_truth].index.to_numpy()
    negative = patient_truth[~patient_truth].index.to_numpy()
    groups = {patient: group.index.to_numpy(dtype=int) for patient, group in table.groupby("patient_id", sort=False)}
    all_columns = {"run_a_detector": "detector_probability", **systems}
    score = {name: table[column].to_numpy(float) for name, column in all_columns.items()}
    triage = {name: _oracle_labels(table, values) for name, values in score.items()}
    delta = {name: {metric: [] for metric in METRICS_FOR_CI} for name in systems}
    for _ in range(repeats):
        sampled = np.r_[rng.choice(positive, len(positive), replace=True),
                        rng.choice(negative, len(negative), replace=True)]
        indices = np.concatenate([groups[patient] for patient in sampled])
        base = _bootstrap_values(truth, score["run_a_detector"], *triage["run_a_detector"], indices)
        for name in systems:
            candidate = _bootstrap_values(truth, score[name], *triage[name], indices)
            for metric in METRICS_FOR_CI:
                delta[name][metric].append(candidate[metric] - base[metric])
    return {name: {metric: {"delta_95_ci": [float(np.percentile(values, 2.5)),
                                               float(np.percentile(values, 97.5))]}
                   for metric, values in entries.items()} for name, entries in delta.items()}


def verdict(report: dict, selections: list[dict], bootstrap: dict) -> dict:
    candidate = report["F2_gated_rescue"]
    binary = candidate["binary_fracture"]
    triage = candidate["oracle_other_heads_triage"]
    baseline = report["run_a_detector"]["oracle_other_heads_triage"]
    class_drops = {label: triage["classwise"][label]["f1"] - baseline["classwise"][label]["f1"]
                   for label in ("0", "1", "2")}
    c_values = {item["selected_C"] for item in selections}
    tau_values = [item["selected_tau_low"] for item in selections]
    unstable = ((min(c_values) == min(C_GRID) and max(c_values) == max(C_GRID))
                or max(tau_values) - min(tau_values) > 0.20)
    ci_lower = bootstrap["F2_gated_rescue"]["oracle_other_heads_macro_f1"]["delta_95_ci"][0]
    criteria = {"tp_at_least_10": binary["tp"] >= 10,
                "fn_at_most_14": binary["fn"] <= 14,
                "fp_at_most_5": binary["fp"] <= 5,
                # User's 0.417/0.966 cutoffs are three-decimal representations
                # of 10/24 and 140/145; compare at their stated precision.
                "sensitivity_at_least_0_417": round(binary["sensitivity"], 3) >= 0.417,
                "specificity_at_least_0_966": round(binary["specificity"], 3) >= 0.966,
                "pr_auc_at_least_0_56": binary["pr_auc"] >= 0.56,
                "oracle_macro_f1_above_baseline": triage["pooled_macro_f1"] > baseline["pooled_macro_f1"],
                "no_class_f1_drop_over_0_01": min(class_drops.values()) >= -0.01,
                "macro_ci_excludes_material_negative_minus_0_01": ci_lower >= -0.01,
                "parameters_not_extremely_unstable": not unstable}
    return {"verdict": "GO" if all(criteria.values()) else "REJECT",
            "criteria": criteria, "classwise_f1_delta": class_drops,
            "macro_f1_delta": triage["pooled_macro_f1"] - baseline["pooled_macro_f1"],
            "parameter_instability_definition": "extreme if C selects both grid endpoints or tau span >0.20",
            "material_negative_macro_f1_definition": "paired CI lower bound below -0.01",
            "preferred_pr_auc_at_least_0_58": binary["pr_auc"] >= 0.58,
            "preferred_macro_f1_delta_at_least_0_002":
                triage["pooled_macro_f1"] - baseline["pooled_macro_f1"] >= 0.002}


def save_outputs(table: pd.DataFrame, fn_table: pd.DataFrame, dist: pd.DataFrame,
                 impacts: pd.DataFrame, reviews: pd.DataFrame, fn_summary: dict,
                 report: dict, fn_dir: Path, output: Path) -> None:
    if fn_dir.exists() or output.exists():
        raise FileExistsError("Refusing to overwrite existing FN audit or fusion v2 report")
    fn_dir.mkdir(parents=True)
    output.mkdir(parents=True)
    fn_table.to_csv(fn_dir / "fn_audit.csv", index=False)
    dist.to_csv(fn_dir / "score_distribution.csv", index=False)
    impacts.to_csv(fn_dir / "decision_impact_audit.csv", index=False)
    reviews.to_csv(fn_dir / "fn_visual_review_manifest.csv", index=False)
    (fn_dir / "fn_summary.json").write_text(json.dumps(fn_summary, indent=2) + "\n")
    plot_distributions(table, fn_dir)
    table[["study_id", "patient_id", "fold", "fracture_true", "detector_probability",
           "classifier_top5", "fusion_probability", "detector_fixed", "f0_fixed",
           "f1_meta", "f2_gated"]].to_csv(output / "cross_fitted_study_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def run(repeats: int = 5000, seed: int = 20260916,
        fn_dir: Path = FN_OUTPUT, output: Path = OUTPUT) -> dict:
    if fn_dir.exists() or output.exists():
        raise FileExistsError("Existing v2 output must not be overwritten")
    table, provenance = load_verified_oof()
    print("Exact Run A OOF baseline reproduced; cohort and fixed detector transform verified.", flush=True)
    fn_table, distributions, impacts, reviews, fn_summary = fn_and_distribution_audit(table)
    print("Seventeen OOF false negatives and score distributions audited.", flush=True)
    evaluated, selections = crossfit_v2(table)
    print("Five outer-fold source-only calibrations and gates completed.", flush=True)
    systems = {name: system_report(evaluated, column) for name, column in (
        ("run_a_detector", "detector_probability"),
        ("existing_weighted_fusion", "fusion_probability"),
        ("F0_fixed_transform_weighted", "f0_fixed"),
        ("F1_calibrated_logistic", "f1_meta"),
        ("F2_gated_rescue", "f2_gated"))}
    columns = {"existing_weighted_fusion": "fusion_probability",
               "F0_fixed_transform_weighted": "f0_fixed",
               "F1_calibrated_logistic": "f1_meta", "F2_gated_rescue": "f2_gated"}
    bootstrap = patient_bootstrap(evaluated, columns, repeats, seed)
    print(f"Paired patient-clustered bootstrap completed: {repeats} repeats.", flush=True)
    baseline = systems["run_a_detector"]
    for name in columns:
        for metric in METRICS_FOR_CI:
            key = "f1" if metric == "fracture_f1" else metric
            if metric == "oracle_other_heads_macro_f1":
                point = (systems[name]["oracle_other_heads_triage"]["pooled_macro_f1"]
                         - baseline["oracle_other_heads_triage"]["pooled_macro_f1"])
            else:
                point = systems[name]["binary_fracture"][key] - baseline["binary_fracture"][key]
            bootstrap[name][metric]["point_delta"] = point
    decision = verdict(systems, selections, bootstrap)
    report = {"analysis_scope": "existing_169_study_patient_heldout_oof_predictions_only",
              "warning": "New meta C/tau/coefs are source-fold-only cross-fitted, but fixed detector top10/raw 0.397176... deployment mapping was previously selected on all OOF labels. Conditional v2 results are not strictly untouched-protocol unbiased; no fixed test or FULL169 prediction was used.",
              "cohort": {"studies": 169, "patients": 155, "positive": 24,
                         "negative": 145, "per_fold_studies": [33, 33, 32, 36, 35],
                         "fixed_test_overlap": 0},
              "official_threshold": THRESHOLD, "baseline_provenance": provenance,
              "systems": systems, "outer_fold_selections": selections,
              "bootstrap": {"method": "paired stratified patient-cluster bootstrap",
                            "repeats": repeats, "seed": seed, "deltas_vs_run_a": bootstrap},
              "fn_summary": fn_summary, "go_reject": decision,
              "next_experiment": ("deployment-protocol finalization" if decision["verdict"] == "GO"
                                  else "P3+P4+P5 frozen multi-scale classifier; not more epochs on current P5 classifier")}
    save_outputs(evaluated, fn_table, distributions, impacts, reviews, fn_summary,
                 report, fn_dir, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--fn-output", type=Path, default=FN_OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.bootstrap_repeats, args.seed, args.fn_output, args.output)
    print(json.dumps({"baseline_reproduced_exactly": report["baseline_provenance"]["baseline_reproduced_exactly"],
                      "go_reject": report["go_reject"], "systems": report["systems"],
                      "fn_output": str(args.fn_output), "comparison_output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
