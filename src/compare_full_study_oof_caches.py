"""CPU-only paired Run A/B comparison of completed full-study OOF caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from aggregate_full_study_oof_cache import attach_truth, crossfit_cache_table, load_cache
from fracture_macro_f1_oof import binary_report, oracle_triage_report
from oof_cohort import derive_explicit_validation_cohort, study_sets


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data_prepared/skull_hu800_ww1600"
EQUIVALENT_FULL_STUDY_INCLUSION_PROTOCOLS = {
    "target_series_header_guard_v1_all_valid_dicoms",
    "explicit_validation_studies_v2_all_valid_target_series_dicoms",
}


def validate_complete_cache(cache_dir: Path, expected_by_fold: Dict[int, set]) -> None:
    seen = set()
    for fold in range(5):
        marker = cache_dir / ("fold_%d" % fold) / "COMPLETE.json"
        if not marker.is_file():
            raise FileNotFoundError("Missing completion marker %s" % marker)
        payload = json.loads(marker.read_text(encoding="utf-8"))
        marker_expected = set(str(value) for value in payload.get("expected_study_ids", []))
        actual = set()
        for path in (cache_dir / ("fold_%d" % fold) / "studies").glob("*.json"):
            content = json.loads(path.read_text(encoding="utf-8"))
            study = str(content.get("series_id", ""))
            if int(content.get("fold", -1)) != fold or not study:
                raise RuntimeError("Invalid Study/fold identity in %s" % path)
            if study in actual or study in seen:
                raise RuntimeError("Cached Study %s occurs more than once" % study)
            actual.add(study)
            seen.add(study)
        extra = actual - expected_by_fold[fold]
        missing = expected_by_fold[fold] - actual
        if extra:
            raise RuntimeError("Unexpected cached Studies in Fold %d: %s" % (fold, sorted(extra)))
        if missing:
            raise RuntimeError("Expected cached Studies missing in Fold %d: %s" % (fold, sorted(missing)))
        if marker_expected != expected_by_fold[fold] or int(payload.get("studies", -1)) != len(actual):
            raise RuntimeError("Completion marker cohort mismatch for fold %d" % fold)


def normalized_protocol(table: pd.DataFrame, expected_imgsz: int) -> Dict[str, Any]:
    signatures = [json.loads(value) for value in table["inference_signature_json"].unique()]
    sizes = {int(signature["imgsz"]) for signature in signatures}
    if sizes != {expected_imgsz}:
        raise ValueError("Expected imgsz=%d, observed=%s" % (expected_imgsz, sorted(sizes)))
    normalized = []
    for signature in signatures:
        normalized.append({
            key: value for key, value in signature.items()
            if key not in {
                "weights_sha256",
                "imgsz",
                "explicit_validation_study_count",
                "cohort_protocol",
                "study_inclusion_protocol",
            }
        })
    serialized = {json.dumps(value, sort_keys=True) for value in normalized}
    if len(serialized) != 1:
        raise ValueError("Within-run evaluation protocol differs across folds")
    return json.loads(next(iter(serialized)))


def exact_cache_slice_coverage(cache_dir: Path) -> Dict[tuple, tuple]:
    """Return ordered SOP coverage keyed by fold, Study and patient."""
    coverage = {}
    for path in sorted(cache_dir.glob("fold_*/studies/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (int(payload["fold"]), str(payload["series_id"]), str(payload["patient_id"]))
        if key in coverage:
            raise RuntimeError("Duplicate cached fold/Study/patient identity: %s" % (key,))
        sops = tuple(str(item["sop_uid"]) for item in payload.get("slices", []))
        if not sops or len(sops) != len(set(sops)):
            raise RuntimeError("Empty or duplicate SOP coverage in %s" % path)
        coverage[key] = (sops, int(payload.get("included_metadata_unknown_slices", 0)))
    if not coverage:
        raise RuntimeError("No cached Study coverage under %s" % cache_dir)
    return coverage


def validate_exact_paired_slice_coverage(cache_a: Path, cache_b: Path) -> Dict[str, Any]:
    """Prove paired Study and SOP coverage before accepting protocol aliases."""
    first = exact_cache_slice_coverage(cache_a)
    second = exact_cache_slice_coverage(cache_b)
    if set(first) != set(second):
        raise ValueError("Cache Study/patient/fold coverage differs")
    mismatches = [key for key in first if first[key] != second[key]]
    if mismatches:
        raise ValueError("Cache SOP sequence/metadata-unknown coverage differs for %s" % (mismatches[:5],))
    return {
        "exact_study_patient_fold_identity": True,
        "exact_ordered_sop_sequence_per_study": True,
        "exact_metadata_unknown_count_per_study": True,
        "studies": len(first),
        "slices": int(sum(len(value[0]) for value in first.values())),
        "metadata_unknown_slices": int(sum(value[1] for value in first.values())),
        "per_fold_studies": [sum(key[0] == fold for key in first) for fold in range(5)],
    }


def paired_tables(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    keys = ["study_id", "patient_id", "fold"]
    if set(map(tuple, a[keys].to_numpy())) != set(map(tuple, b[keys].to_numpy())):
        raise ValueError("Run A and B do not have identical held-out study/patient/fold coverage")
    keep = keys + [*('V_EDH', 'V_SDH', 'V_IPH', 'V_SAH', 'V_IVH', 'MLS_mm'), "fracture_true",
                   "included_target_series_slices", "included_metadata_unknown_slices",
                   "crossfit_fracture_prob"]
    merged = a[keep].merge(b[keep], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    truth_columns = ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH", "MLS_mm", "fracture_true"]
    for name in truth_columns:
        left, right = merged[name + "_a"], merged[name + "_b"]
        if name == "fracture_true":
            equal = left.astype(bool).equals(right.astype(bool))
        else:
            equal = bool(np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=0.0, atol=1e-9))
        if not equal:
            raise ValueError("Run A/B truth mismatch for %s" % name)
        merged[name] = left
    if not (
        merged["included_target_series_slices_a"].equals(merged["included_target_series_slices_b"])
        and merged["included_metadata_unknown_slices_a"].equals(merged["included_metadata_unknown_slices_b"])
    ):
        raise ValueError("Run A/B study inclusion differs; paired comparison is invalid")
    return merged


def metrics_for_probability(table: pd.DataFrame, probability_column: str) -> Dict[str, Any]:
    probabilities = table[probability_column].to_numpy(dtype=float)
    return {
        "triage": oracle_triage_report(table, probabilities),
        "fracture": binary_report(table["fracture_true"].astype(bool), probabilities),
    }


def flat_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    triage, fracture = metrics["triage"], metrics["fracture"]
    values = {
        "oracle_other_heads_macro_f1": triage["pooled_macro_f1"],
        "micro_f1": triage["micro_f1"], "accuracy": triage["accuracy"],
        "fracture_sensitivity": fracture["sensitivity"],
        "fracture_specificity": fracture["specificity"],
        "fracture_precision": fracture["precision"],
        "study_pr_auc": fracture.get("pr_auc"), "roc_auc": fracture.get("roc_auc"),
        "binary_qwk_diagnostic": fracture.get("binary_qwk_diagnostic"),
        "fracture_fp": float(fracture["fp"]), "fracture_fn": float(fracture["fn"]),
    }
    for label in (0, 1, 2):
        for metric in ("f1", "precision", "recall"):
            values["class_%d_%s" % (label, metric)] = triage["classwise"][str(label)][metric]
    return values


def delta_metrics(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    first, second = flat_metrics(a), flat_metrics(b)
    return {
        key: (float(second[key] - first[key]) if first[key] is not None and second[key] is not None else None)
        for key in first
    }


def patient_clustered_bootstrap(
    paired: pd.DataFrame, repeats: int, seed: int
) -> Dict[str, Any]:
    patient_truth = paired.groupby("patient_id")["fracture_true"].max().astype(bool)
    positive = patient_truth[patient_truth].index.to_numpy()
    negative = patient_truth[~patient_truth].index.to_numpy()
    if not len(positive) or not len(negative):
        raise ValueError("Both positive and negative patients are required for stratified bootstrap")
    groups = {patient: group for patient, group in paired.groupby("patient_id", sort=False)}
    rng = np.random.default_rng(seed)
    samples = {}  # type: Dict[str, List[float]]
    for _ in range(repeats):
        selected = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        sample = pd.concat([groups[patient] for patient in selected], ignore_index=True)
        delta = delta_metrics(
            metrics_for_probability(sample, "crossfit_fracture_prob_a"),
            metrics_for_probability(sample, "crossfit_fracture_prob_b"),
        )
        for key, value in delta.items():
            if value is not None and np.isfinite(value):
                samples.setdefault(key, []).append(float(value))
    return {
        "method": "paired stratified patient-clustered bootstrap of frozen cross-fitted held-out predictions",
        "direction": "Run B minus Run A", "seed": seed, "requested_repeats": repeats,
        "percentile_95_ci": {
            key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            for key, values in samples.items() if values
        },
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite %s" % args.output)
    manifest = pd.read_csv(args.manifest, dtype={"study_id": str, "patient_id": str, "image_path": str})
    cohort = derive_explicit_validation_cohort(manifest, args.val_lists)
    expected_by_fold = study_sets(cohort)
    validate_complete_cache(args.cache_a, expected_by_fold)
    validate_complete_cache(args.cache_b, expected_by_fold)
    paired_coverage = validate_exact_paired_slice_coverage(args.cache_a, args.cache_b)
    raw_a, raw_b = load_cache(args.cache_a), load_cache(args.cache_b)
    protocol_a = normalized_protocol(raw_a, args.expected_a_imgsz)
    protocol_b = normalized_protocol(raw_b, args.expected_b_imgsz)
    if protocol_a != protocol_b:
        raise ValueError("Run A/B score-generation or inclusion parameters differ beyond imgsz/weights")
    if str(raw_a["score_generation_protocol"].iloc[0]) != str(raw_b["score_generation_protocol"].iloc[0]):
        raise ValueError("Run A/B score-generation protocol identity differs")
    inclusion_a = str(raw_a["study_inclusion_protocol"].iloc[0])
    inclusion_b = str(raw_b["study_inclusion_protocol"].iloc[0])
    if inclusion_a != inclusion_b and {inclusion_a, inclusion_b} != EQUIVALENT_FULL_STUDY_INCLUSION_PROTOCOLS:
        raise ValueError("Run A/B study-inclusion protocol differs")

    table_a = attach_truth(raw_a, args.metadata)
    table_b = attach_truth(raw_b, args.metadata)
    crossfit_a, report_a = crossfit_cache_table(table_a)
    crossfit_b, report_b = crossfit_cache_table(table_b)
    paired = paired_tables(crossfit_a, crossfit_b)
    metrics_a = metrics_for_probability(paired, "crossfit_fracture_prob_a")
    metrics_b = metrics_for_probability(paired, "crossfit_fracture_prob_b")
    report = {
        "primary_decision_metric": "oracle_other_heads_macro_f1",
        "authoritative_cohort": {
            "cohort_protocol": cohort["cohort_protocol"],
            "total_studies": cohort["total_unique_validation_studies"],
            "per_fold_study_counts": cohort["per_fold_study_counts"],
        },
        "direction": "Run B minus Run A",
        "score_generation_protocol": str(raw_a["score_generation_protocol"].iloc[0]),
        "study_inclusion_protocols": {"first": inclusion_a, "second": inclusion_b},
        "paired_slice_coverage_proof": paired_coverage,
        "study_inclusion_protocol_equivalence": (
            "Labels differ because v2 names explicit-Study cohort selection; exact paired SOP coverage was proven."
            if inclusion_a != inclusion_b else "Protocol labels are identical."
        ),
        "official_downstream_fracture_threshold": 0.5,
        "nms_reconstruction_policy": "Different NMS settings require distinct inference caches and are never reconstructed post hoc.",
        "run_a": {"metrics": metrics_a, "fold_selections": report_a["fold_selections"]},
        "run_b": {"metrics": metrics_b, "fold_selections": report_b["fold_selections"]},
        "paired_deltas": delta_metrics(metrics_a, metrics_b),
        "patient_clustered_bootstrap": patient_clustered_bootstrap(paired, args.bootstrap_repeats, args.seed),
        "binary_qwk_role": "diagnostic only",
        "proposal_recall_role": "mechanistic guardrail reported by the separate proposal_diagnostic protocol",
    }
    args.output.mkdir(parents=True)
    paired.to_csv(args.output / "paired_cross_fitted_predictions.csv", index=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--cache-b", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "iaaa-contest-bct/Data/training_df.pkl")
    parser.add_argument("--manifest", type=Path, default=DATASET / "manifest.csv")
    parser.add_argument("--val-list", dest="val_lists", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-a-imgsz", type=int, default=768)
    parser.add_argument("--expected-b-imgsz", type=int, default=1024)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if not args.val_lists:
        args.val_lists = [DATASET / ("kfold/fold_%d_val.txt" % fold) for fold in range(5)]
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
