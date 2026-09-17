"""CPU-only matched Run A/P5/multi-scale held-out OOF comparison; no selection."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fracture_classifier.compare import aggregate, attach_classifier_scores, load_detector_baseline
from fracture_classifier.compare_fusion_v2 import METRICS_FOR_CI, _bootstrap_values, _oracle_labels, system_report
from fracture_classifier.data import CACHE, EXPERIMENT, ROOT, detector_checkpoints, sha256
from fracture_classifier.multiscale_inference import FOLD_COUNTS, OOF_OUTPUT


P5_BEST_SLICE_AP = (0.16135244870897936, 0.10082844866574697,
                    0.0815086212203611, 0.24723180240050285,
                    0.08455013880810518)
AGGREGATORS = ("max", "top3_mean", "top5_mean", "top10_percent_mean")
HISTORICAL_P5 = EXPERIMENT / "comparison/paired_oof_studies.csv"


def load_multiscale_studies(detector: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = pd.read_csv(EXPERIMENT / "slice_label_manifest.csv",
                         dtype={"study_id": str, "patient_id": str, "sop_uid": str})
    if labels.groupby("patient_id").fold.nunique().max() != 1:
        raise RuntimeError("Cross-fold patient leakage in label manifest")
    for fold in range(5):
        fold_dir = OOF_OUTPUT / "slice_predictions" / f"fold_{fold}"
        marker_path = fold_dir / "COMPLETE.json"
        if not marker_path.is_file():
            raise FileNotFoundError(f"Multi-scale held-out Fold incomplete: {marker_path}")
        marker = json.loads(marker_path.read_text())
        expected = set(map(str, marker["expected_study_ids"]))
        source_marker = json.loads((CACHE / f"fold_{fold}/COMPLETE.json").read_text())
        signature = marker.get("signature", {})
        classifier_path = ROOT / f"outputs/run_a_fracture_multiscale_fold{fold}/best_classifier.pt"
        if (not classifier_path.is_file()
                or signature.get("fold") != fold
                or signature.get("stage") != "frozen_multiscale_gap"
                or signature.get("classifier_sha256") != sha256(classifier_path)
                or signature.get("source_detector_sha256") != sha256(detector_checkpoints()[fold])
                or signature.get("slice_label_manifest_sha256") != sha256(EXPERIMENT / "slice_label_manifest.csv")
                or signature.get("source_full_study_cache") != str(CACHE)
                or (signature.get("imgsz"), signature.get("context_distance_mm"),
                    signature.get("window_level"), signature.get("window_width")) !=
                    (768, 5.0, 800.0, 1600.0)):
            raise RuntimeError("Multi-scale Fold cache/checkpoint/preprocessing provenance mismatch")
        assigned = labels[labels.fold == fold][["study_id", "patient_id"]].drop_duplicates()
        if (len(expected) != FOLD_COUNTS[fold] or expected != set(map(str, source_marker["expected_study_ids"]))
                or set(assigned.study_id) != expected or assigned.study_id.duplicated().any()
                or int(marker["studies"]) != FOLD_COUNTS[fold]):
            raise RuntimeError("Explicit held-out multi-scale Study coverage changed")
        expected_patient = dict(zip(assigned.study_id, assigned.patient_id))
        files = sorted((fold_dir / "studies").glob("*.json"))
        if {path.stem for path in files} != expected:
            raise RuntimeError("Missing/extra multi-scale Study cache")
        for path in files:
            payload = json.loads(path.read_text())
            study, patient = str(payload["study_id"]), str(payload["patient_id"])
            source = json.loads((CACHE / f"fold_{fold}/studies/{study}.json").read_text())
            sops = [str(item["sop_uid"]) for item in source["slices"]]
            if (int(payload["fold"]) != fold or patient != expected_patient[study]
                    or patient != str(source["patient_id"]) or payload["sop_uids"] != sops
                    or payload["signature"] != marker["signature"]):
                raise RuntimeError("Multi-scale/Run A exact Study-patient-SOP pairing failed")
            scores = list(map(float, payload["slice_probabilities"]))
            if len(scores) != len(sops):
                raise RuntimeError("Multi-scale full-study slice count mismatch")
            row = {"fold": fold, "study_id": study, "patient_id": patient}
            row.update({f"multi_{method}": aggregate(scores, method) for method in AGGREGATORS})
            rows.append(row)
    merged = detector.merge(pd.DataFrame(rows), on=["fold", "study_id", "patient_id"],
                            validate="one_to_one")
    if len(merged) != 169 or merged.study_id.duplicated().any():
        raise RuntimeError("New multi-scale OOF cohort is not exact 169 Studies")
    return merged


def load_p5_matched(detector: pd.DataFrame) -> pd.DataFrame:
    joined = attach_classifier_scores(detector)
    for fold in range(5):
        marker = json.loads((EXPERIMENT / f"slice_predictions/fold_{fold}/COMPLETE.json").read_text())
        p5_checkpoint = ROOT / f"outputs/run_a_fracture_classifier_fold{fold}/best_classifier.pt"
        if (marker.get("signature", {}).get("classifier_sha256") != sha256(p5_checkpoint)
                or marker.get("signature", {}).get("source_detector_sha256") !=
                sha256(detector_checkpoints()[fold])):
            raise RuntimeError("Historical P5 cache/checkpoint provenance mismatch")
    historical = pd.read_csv(HISTORICAL_P5, dtype={"study_id": str, "patient_id": str})
    if len(historical) != 169 or historical.study_id.duplicated().any():
        raise RuntimeError("Historical P5 OOF artifact is not 169 unique Studies")
    matched = joined.merge(historical[["fold", "study_id", "patient_id",
                                       "classifier_probability", "fusion_probability"]],
                           on=["fold", "study_id", "patient_id"], validate="one_to_one")
    if (len(matched) != 169
            or not np.allclose(matched["classifier_top5_mean"], matched["classifier_probability"],
                               rtol=0, atol=1e-12)
            or not np.allclose(0.75 * matched["detector_probability"] +
                               0.25 * matched["classifier_top5_mean"], matched["fusion_probability"],
                               rtol=0, atol=1e-12)):
        raise RuntimeError("Existing P5 top5/0.75 fusion does not reproduce exactly")
    return matched


def slice_validation_results() -> list[dict]:
    results = []
    for fold in range(5):
        path = ROOT / f"outputs/run_a_fracture_multiscale_fold{fold}/classifier_metrics.json"
        complete = ROOT / f"outputs/run_a_fracture_multiscale_fold{fold}/COMPLETE.json"
        if not path.is_file() or not complete.is_file():
            raise FileNotFoundError(f"Multi-scale Fold {fold} training is not complete")
        result = json.loads(path.read_text())
        status = json.loads(complete.read_text())
        if (int(result["fold"]) != fold or int(status["fold"]) != fold
                or status.get("epochs_completed") != 10 or len(result["history"]) != 10):
            raise RuntimeError("Multi-scale Fold training history/completion mismatch")
        best_epoch = int(result["best_epoch"])
        selected = result["history"][best_epoch - 1]
        if (not 1 <= best_epoch <= 10 or selected["pr_auc"] != result["best_validation_slice_pr_auc"]):
            raise RuntimeError("Best multi-scale checkpoint selection changed")
        checkpoint = torch.load(ROOT / f"outputs/run_a_fracture_multiscale_fold{fold}/best_classifier.pt",
                                map_location="cpu", weights_only=False)
        if (checkpoint.get("stage") != 1 or checkpoint.get("epoch_one_based") != best_epoch
                or checkpoint.get("validation_slice_metrics", {}).get("pr_auc") != selected["pr_auc"]
                or checkpoint.get("source_detector_sha256") != sha256(detector_checkpoints()[fold])):
            raise RuntimeError("Best multi-scale checkpoint does not match validation selection")
        results.append({"fold": fold, "best_epoch": best_epoch,
                        "existing_p5_best_slice_pr_auc": P5_BEST_SLICE_AP[fold],
                        "delta_slice_pr_auc": selected["pr_auc"] - P5_BEST_SLICE_AP[fold],
                        **{key: selected[key] for key in ("pr_auc", "roc_auc", "precision_at_0_5",
                                                           "recall_at_0_5", "f1_at_0_5")}})
    return results


def paired_bootstrap(table: pd.DataFrame, repeats: int, seed: int) -> dict:
    """Same resampled patients and Studies for both paired contrasts."""
    if repeats <= 0:
        raise ValueError("Bootstrap repeats must be positive")
    rng = np.random.default_rng(seed)
    truth = table["fracture_true"].astype(bool).to_numpy()
    patient_truth = table.groupby("patient_id")["fracture_true"].max().astype(bool)
    positive = patient_truth[patient_truth].index.to_numpy()
    negative = patient_truth[~patient_truth].index.to_numpy()
    groups = {patient: group.index.to_numpy(dtype=int)
              for patient, group in table.groupby("patient_id", sort=False)}
    columns = {"run_a_detector": "detector_probability", "existing_p5_fusion": "p5_fusion",
               "new_multiscale_fusion": "multi_fusion"}
    probabilities = {name: table[column].to_numpy(float) for name, column in columns.items()}
    triage = {name: _oracle_labels(table, values) for name, values in probabilities.items()}
    contrasts = {name: {metric: [] for metric in METRICS_FOR_CI}
                 for name in ("vs_run_a", "vs_existing_p5_fusion")}
    for _ in range(repeats):
        sampled = np.r_[rng.choice(positive, len(positive), replace=True),
                        rng.choice(negative, len(negative), replace=True)]
        indices = np.concatenate([groups[patient] for patient in sampled])
        measured = {name: _bootstrap_values(truth, probabilities[name], *triage[name], indices)
                    for name in columns}
        for contrast, baseline in (("vs_run_a", "run_a_detector"),
                                   ("vs_existing_p5_fusion", "existing_p5_fusion")):
            for metric in METRICS_FOR_CI:
                contrasts[contrast][metric].append(measured["new_multiscale_fusion"][metric]
                                                   - measured[baseline][metric])
    return {contrast: {metric: [float(np.percentile(values, 2.5)),
                                float(np.percentile(values, 97.5))]
                       for metric, values in entries.items()}
            for contrast, entries in contrasts.items()}


def compare(output: Path, repeats: int = 5000, seed: int = 20260916) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite multi-scale OOF comparison: {output}")
    detector = load_detector_baseline()  # exact historical Run A metrics asserted here
    p5 = load_p5_matched(detector)
    table = load_multiscale_studies(p5).sort_values(["fold", "study_id"]).reset_index(drop=True)
    slices = slice_validation_results()
    table["p5_top5"] = table["classifier_top5_mean"].astype(float)
    table["multi_top5"] = table["multi_top5_mean"].astype(float)
    table["p5_fusion"] = 0.75 * table["detector_probability"] + 0.25 * table["p5_top5"]
    table["multi_fusion"] = 0.75 * table["detector_probability"] + 0.25 * table["multi_top5"]
    systems = {name: system_report(table, column) for name, column in (
        ("A_run_a_detector", "detector_probability"),
        ("B_existing_p5_classifier", "p5_top5"),
        ("C_new_multiscale_classifier", "multi_top5"),
        ("D_existing_p5_fusion", "p5_fusion"),
        ("E_new_multiscale_fusion", "multi_fusion"))}
    historical = systems["D_existing_p5_fusion"]
    if ([historical["binary_fracture"][key] for key in ("tp", "fp", "fn", "tn")] != [8, 4, 16, 141]
            or not math.isclose(historical["oracle_other_heads_triage"]["pooled_macro_f1"],
                                0.9696273781380164, abs_tol=1e-12)):
        raise RuntimeError("Historical P5 fusion baseline failed exact reproduction")
    diagnostic = {method: system_report(table, f"multi_{method}")
                  for method in AGGREGATORS if method != "top5_mean"}
    ci = paired_bootstrap(table, repeats, seed)
    mean_ap = float(np.mean([row["pr_auc"] for row in slices]))
    improved = sum(row["delta_slice_pr_auc"] > 0 for row in slices)
    candidate = systems["E_new_multiscale_fusion"]
    binary = candidate["binary_fracture"]
    triage = candidate["oracle_other_heads_triage"]
    base_triage = systems["A_run_a_detector"]["oracle_other_heads_triage"]
    class_delta = {label: triage["classwise"][label]["f1"] - base_triage["classwise"][label]["f1"]
                   for label in ("0", "1", "2")}
    criteria = {"mean_slice_pr_auc_at_least_0_17": mean_ap >= 0.17,
                "at_least_4_folds_improve": improved >= 4,
                "tp_at_least_10": binary["tp"] >= 10,
                "fn_at_most_14": binary["fn"] <= 14,
                "fp_at_most_5": binary["fp"] <= 5,
                "sensitivity_at_least_0_417": round(binary["sensitivity"], 3) >= 0.417,
                "specificity_at_least_0_966": round(binary["specificity"], 3) >= 0.966,
                "pr_auc_at_least_0_56": binary["pr_auc"] >= 0.56,
                "oracle_macro_f1_above_run_a": triage["pooled_macro_f1"] > base_triage["pooled_macro_f1"],
                "no_class_f1_drop_over_0_01": min(class_delta.values()) >= -0.01}
    report = {"analysis_scope": "169 explicit patient-held-out Run A OOF Studies only",
              "warning": "Oracle-other-head triage uses GT ICH/MLS; not final submission Macro-F1. Candidate choice on same OOF is exploratory.",
              "cohort": {"studies": 169, "patients": 155, "positive": 24, "negative": 145,
                         "per_fold_studies": list(FOLD_COUNTS)},
              "primary_classifier_aggregation": "top5_mean",
              "diagnostic_aggregations_not_selected": list(diagnostic),
              "fusion": "0.75*Run A historical held-out detector_prob + 0.25*classifier_top5_prob",
              "official_threshold": 0.5, "slice_validation_per_fold": slices,
              "mean_p5_slice_pr_auc": float(np.mean(P5_BEST_SLICE_AP)),
              "mean_multiscale_slice_pr_auc": mean_ap, "folds_improved": improved,
              "systems": systems, "diagnostic_multiscale_aggregation": diagnostic,
              "bootstrap": {"method": "paired stratified patient-cluster bootstrap",
                            "repeats": repeats, "seed": seed, "delta_95_ci": ci},
              "predefined_go": {"verdict": "GO" if all(criteria.values()) else "REJECT",
                                "criteria": criteria, "classwise_f1_delta_vs_run_a": class_delta,
                                "prefer_pr_auc_at_least_0_58": binary["pr_auc"] >= 0.58}}
    output.mkdir(parents=True, exist_ok=False)
    table[["study_id", "patient_id", "fold", "fracture_true", "detector_probability",
           "p5_top5", "multi_top5", "p5_fusion", "multi_fusion"] +
          [f"multi_{method}" for method in AGGREGATORS if method != "top5_mean"]].to_csv(
              output / "paired_oof_studies.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OOF_OUTPUT / "comparison")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    print(json.dumps(compare(args.output, args.bootstrap_repeats, args.seed), indent=2))


if __name__ == "__main__":
    main()
