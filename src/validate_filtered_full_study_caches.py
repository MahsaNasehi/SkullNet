"""Strict CPU-only provenance gate for paired authoritative full-study caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from aggregate_full_study_oof_cache import load_cache
from compare_full_study_oof_caches import normalized_protocol, validate_complete_cache
from oof_cohort import derive_explicit_validation_cohort, study_sets


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data_prepared/skull_hu800_ww1600"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = pd.read_csv(
        args.manifest, dtype={"study_id": str, "patient_id": str, "image_path": str}
    )
    cohort = derive_explicit_validation_cohort(manifest, args.val_lists)
    expected = study_sets(cohort)
    validate_complete_cache(args.cache_a, expected)
    validate_complete_cache(args.cache_b, expected)
    a, b = load_cache(args.cache_a), load_cache(args.cache_b)

    keys = ["study_id", "patient_id", "fold"]
    key_a, key_b = set(map(tuple, a[keys].to_numpy())), set(map(tuple, b[keys].to_numpy()))
    if key_a != key_b:
        raise RuntimeError("Run A/B Study-patient-fold identities differ")
    if len(a) != 169 or len(b) != 169:
        raise RuntimeError("Both filtered caches must contain exactly 169 Studies")
    if a["study_id"].duplicated().any() or b["study_id"].duplicated().any():
        raise RuntimeError("Duplicate Study in filtered cache")

    manifest_work = manifest.copy()
    manifest_work["study_id"] = manifest_work["study_id"].astype(str)
    manifest_work["patient_id"] = manifest_work["patient_id"].astype(str)
    fixed = manifest_work[manifest_work["split"].astype(str).str.lower() == "test"]
    cached_studies = set(a["study_id"])
    cached_patients = set(a["patient_id"])
    fixed_study_overlap = cached_studies & set(fixed["study_id"])
    fixed_patient_overlap = cached_patients & set(fixed["patient_id"])
    if fixed_study_overlap or fixed_patient_overlap:
        raise RuntimeError(
            "Fixed-test Study/patient entered cache: studies=%s patients=%s"
            % (sorted(fixed_study_overlap), sorted(fixed_patient_overlap))
        )

    protocol_a = normalized_protocol(a, 768)
    protocol_b = normalized_protocol(b, 1024)
    if protocol_a != protocol_b:
        raise RuntimeError("A/B inference protocols differ beyond model weights and imgsz")
    if set(a["score_generation_protocol"]) != {"deployment_score"}:
        raise RuntimeError("Run A is not a pure deployment_score cache")
    if set(b["score_generation_protocol"]) != {"deployment_score"}:
        raise RuntimeError("Run B is not a pure deployment_score cache")

    paired = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    for column in ("included_target_series_slices", "included_metadata_unknown_slices"):
        if not paired[column + "_a"].equals(paired[column + "_b"]):
            raise RuntimeError("Run A/B included slice cohort differs for %s" % column)

    report = {
        "status": "PASS",
        "authoritative_total_studies": 169,
        "per_fold_study_counts": [int((a["fold"] == fold).sum()) for fold in range(5)],
        "run_a_studies": len(a),
        "run_b_studies": len(b),
        "a_b_study_patient_fold_identity": True,
        "missing_studies": [],
        "extra_studies": [],
        "duplicate_studies": 0,
        "fixed_test_study_overlap": [],
        "fixed_test_patient_overlap": [],
        "cross_fold_patient_leakage": False,
        "score_generation_protocol": "deployment_score",
        "conf": float(protocol_a["conf"]),
        "nms_iou": float(protocol_a["nms_iou"]),
        "run_a_imgsz": 768,
        "run_b_imgsz": 1024,
        "protocol_equal_beyond_weights_and_imgsz": True,
        "included_slice_counts_identical_per_study": True,
        "metadata_unknown_slice_counts_identical_per_study": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--cache-b", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DATASET / "manifest.csv")
    parser.add_argument("--val-list", dest="val_lists", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.val_lists:
        args.val_lists = [DATASET / ("kfold/fold_%d_val.txt" % fold) for fold in range(5)]
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
