"""CPU-only provenance audit of explicit OOF Studies versus cached content."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd

from oof_cohort import derive_explicit_validation_cohort, study_sets


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data_prepared/skull_hu800_ww1600"


def cached_content(cache_dir: Path, fold: int) -> Dict[str, str]:
    result = {}
    for path in sorted((cache_dir / ("fold_%d" % fold) / "studies").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("fold", -1)) != fold:
            raise ValueError("Cache content fold mismatch: %s" % path)
        study = str(payload.get("series_id", ""))
        patient = str(payload.get("patient_id", ""))
        if not study or not patient:
            raise ValueError("Cache content missing series_id/patient_id: %s" % path)
        if study in result:
            raise ValueError("Duplicate cached Study ID %s in Fold %d" % (study, fold))
        result[study] = patient
    return result


def audit_cache(cache_dir: Path, expected: Dict[int, set], study_patient: Dict[str, str]) -> Dict[str, Any]:
    folds = []
    for fold in range(5):
        content = cached_content(cache_dir, fold)
        actual = set(content)
        extra, missing = sorted(actual - expected[fold]), sorted(expected[fold] - actual)
        discrepancy_patients = {
            study: content.get(study, study_patient.get(study, "metadata/manifest patient unavailable"))
            for study in sorted(set(extra) | set(missing))
        }
        folds.append({
            "fold": fold,
            "expected_explicit_validation_study_ids": sorted(expected[fold]),
            "cached_study_ids_from_json_content": sorted(actual),
            "cached_but_not_expected": extra,
            "expected_but_not_cached": missing,
            "discrepancy_patient_ids": discrepancy_patients,
        })
    return {"cache_dir": str(cache_dir.resolve()), "folds": folds}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite %s" % args.output)
    manifest = pd.read_csv(args.manifest, dtype={"study_id": str, "patient_id": str, "image_path": str})
    cohort = derive_explicit_validation_cohort(manifest, args.val_lists)
    expected = study_sets(cohort)
    study_patient = dict(
        manifest[["study_id", "patient_id"]].drop_duplicates().astype(str).itertuples(index=False, name=None)
    )
    report = {
        "audit_protocol": "explicit_manifest_validation_studies_vs_cache_json_content_v1",
        "authoritative_cohort": cohort,
        "caches": [audit_cache(cache, expected, study_patient) for cache in args.cache],
        "root_cause_check": {
            "code_path": "cache_full_study_oof.py historical v1 selected metadata rows by held-out patient",
            "observed_behavior": "additional Studies belonging to held-out patients entered the 172-Study caches",
            "required_behavior": "explicit held-out Study IDs expand only to all valid target-series slices within each Study",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATASET / "manifest.csv")
    parser.add_argument("--val-list", dest="val_lists", type=Path, action="append", default=[])
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/full_study_oof_cohort_provenance_audit.json")
    args = parser.parse_args()
    if not args.val_lists:
        args.val_lists = [DATASET / ("kfold/fold_%d_val.txt" % fold) for fold in range(5)]
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))

