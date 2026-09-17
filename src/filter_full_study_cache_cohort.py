"""Materialize the authoritative 169-Study OOF cohort from an existing cache.

This is a CPU/filesystem-only operation.  Study JSON files are hard-linked when
possible (and metadata-preserving copied otherwise); no model is imported and
no prediction is recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Set

import pandas as pd

from oof_cohort import derive_explicit_validation_cohort, study_sets


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data_prepared/skull_hu800_ww1600"
EXPECTED_SOURCE_EXTRAS = {3: {"3122"}, 4: {"2023", "2557"}}


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_source_cache(source: Path) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """Index Study files by their JSON identities, never by filenames."""
    indexed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    globally_seen: Set[str] = set()
    for fold in range(5):
        marker_path = source / ("fold_%d" % fold) / "COMPLETE.json"
        if not marker_path.is_file():
            raise FileNotFoundError("Missing source completion marker %s" % marker_path)
        marker = _json(marker_path)
        if int(marker.get("fold", -1)) != fold:
            raise RuntimeError("Source completion marker Fold mismatch: %s" % marker_path)
        fold_items: Dict[str, Dict[str, Any]] = {}
        for path in sorted((source / ("fold_%d" % fold) / "studies").glob("*.json")):
            payload = _json(path)
            study = str(payload.get("series_id", ""))
            patient = str(payload.get("patient_id", ""))
            if not study or not patient or int(payload.get("fold", -1)) != fold:
                raise RuntimeError("Invalid Study/fold/patient identity in %s" % path)
            if study in fold_items or study in globally_seen:
                raise RuntimeError("Duplicate cached Study %s" % study)
            fold_items[study] = {"path": path, "payload": payload}
            globally_seen.add(study)
        if int(marker.get("studies", -1)) != len(fold_items):
            raise RuntimeError("Source completion marker count mismatch in Fold %d" % fold)
        indexed[fold] = fold_items
    return indexed


def _validate_payload_protocol(payload: Mapping[str, Any], expected_imgsz: int) -> None:
    signature = payload.get("inference_signature", {})
    checks = {
        "score_generation_protocol": payload.get("score_generation_protocol") == "deployment_score",
        "signature_score_generation_protocol": signature.get("score_generation_protocol") == "deployment_score",
        "confidence_floor": float(signature.get("conf", -1)) == 0.01,
        "nms_iou": float(signature.get("nms_iou", -1)) == 0.5,
        "imgsz": int(signature.get("imgsz", -1)) == expected_imgsz,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Cache protocol mismatch (%s): Study %s" % (", ".join(failed), payload.get("series_id")))


def materialize_filtered_cache(
    source: Path,
    destination: Path,
    cohort: Mapping[str, Any],
    expected_imgsz: int,
    allowed_source_extras: Mapping[int, Set[str]] = EXPECTED_SOURCE_EXTRAS,
) -> Dict[str, Any]:
    """Create an exact-cohort cache atomically from already-computed Studies."""
    if destination.exists():
        raise FileExistsError("Refusing to overwrite %s" % destination)
    expected = study_sets(dict(cohort))
    source_index = scan_source_cache(source)
    observed_extras: Dict[int, list] = {}
    for fold in range(5):
        actual = set(source_index[fold])
        missing = expected[fold] - actual
        extras = actual - expected[fold]
        if missing:
            raise RuntimeError("Source cache is missing Fold %d Studies: %s" % (fold, sorted(missing)))
        allowed = set(allowed_source_extras.get(fold, set()))
        if extras != allowed:
            raise RuntimeError(
                "Source cache extras differ from the confirmed diagnostic set in Fold %d: observed=%s expected=%s"
                % (fold, sorted(extras), sorted(allowed))
            )
        observed_extras[fold] = sorted(extras)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=str(destination.parent)))
    link_count = 0
    copy_count = 0
    try:
        for fold_info in cohort["folds"]:
            fold = int(fold_info["fold"])
            expected_patients = {str(k): str(v) for k, v in fold_info["study_to_patient"].items()}
            study_dir = temporary / ("fold_%d" % fold) / "studies"
            study_dir.mkdir(parents=True)
            signatures = []
            inclusion_protocols = set()
            prediction_protocols = set()
            for study in sorted(expected[fold]):
                item = source_index[fold][study]
                payload = item["payload"]
                if str(payload["patient_id"]) != expected_patients[study]:
                    raise RuntimeError("Manifest/cache patient mismatch for Study %s" % study)
                _validate_payload_protocol(payload, expected_imgsz)
                signatures.append(payload["inference_signature"])
                inclusion_protocols.add(str(payload.get("study_inclusion_protocol", "")))
                prediction_protocols.add(str(payload.get("prediction_protocol", "")))
                target = study_dir / (study + ".json")
                try:
                    os.link(str(item["path"]), str(target))
                    link_count += 1
                except OSError:
                    shutil.copy2(str(item["path"]), str(target))
                    copy_count += 1
                if _sha256(item["path"]) != _sha256(target):
                    raise RuntimeError("Materialized Study is not byte-identical: %s" % study)
            if len({json.dumps(value, sort_keys=True) for value in signatures}) != 1:
                raise RuntimeError("Mixed inference signatures within Fold %d" % fold)
            if len(inclusion_protocols) != 1 or len(prediction_protocols) != 1:
                raise RuntimeError("Mixed cache protocols within Fold %d" % fold)
            source_marker = source / ("fold_%d" % fold) / "COMPLETE.json"
            marker = {
                "fold": fold,
                "prediction_protocol": next(iter(prediction_protocols)),
                "studies": len(expected[fold]),
                "expected_study_ids": sorted(expected[fold]),
                "score_generation_protocol": "deployment_score",
                "study_inclusion_protocol": next(iter(inclusion_protocols)),
                "inference_signature": signatures[0],
                "cohort_protocol": cohort["cohort_protocol"],
                "materialization_protocol": "filtered_from_existing_cache_no_inference_v1",
                "source_cache": str(source.resolve()),
                "source_completion_marker_sha256": _sha256(source_marker),
                "excluded_source_study_ids": observed_extras[fold],
            }
            (temporary / ("fold_%d" % fold) / "COMPLETE.json").write_text(
                json.dumps(marker, separators=(",", ":")) + "\n", encoding="utf-8"
            )
        provenance = {
            "materialization_protocol": "filtered_from_existing_cache_no_inference_v1",
            "source_cache": str(source.resolve()),
            "destination_cache": str(destination.resolve()),
            "cohort_protocol": cohort["cohort_protocol"],
            "total_studies": int(cohort["total_unique_validation_studies"]),
            "per_fold_study_counts": list(cohort["per_fold_study_counts"]),
            "excluded_source_studies_by_fold": {str(k): v for k, v in observed_extras.items()},
            "expected_imgsz": expected_imgsz,
            "score_generation_protocol": "deployment_score",
            "conf": 0.01,
            "nms_iou": 0.5,
            "hard_links": link_count,
            "fallback_copies": copy_count,
            "prediction_recomputation": False,
        }
        (temporary / "FILTERED_CACHE_PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(destination)
        return provenance
    except BaseException:
        shutil.rmtree(temporary)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-imgsz", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=DATASET / "manifest.csv")
    parser.add_argument("--val-list", dest="val_lists", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.val_lists:
        args.val_lists = [DATASET / ("kfold/fold_%d_val.txt" % fold) for fold in range(5)]
    return args


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"study_id": str, "patient_id": str, "image_path": str})
    cohort = derive_explicit_validation_cohort(manifest, args.val_lists)
    print(json.dumps(materialize_filtered_cache(
        args.source, args.destination, cohort, args.expected_imgsz
    ), indent=2))


if __name__ == "__main__":
    main()
