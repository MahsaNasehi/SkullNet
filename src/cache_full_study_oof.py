"""Resumable full-study held-out OOF inference scaffold.

This module deliberately imports Ultralytics only after split/leakage checks
and the free-VRAM guard pass. It is not intended to run while training is
using the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from oof_cohort import (
    AUTHORITATIVE_FOLD_STUDY_COUNTS,
    AUTHORITATIVE_TOTAL_STUDIES,
    COHORT_PROTOCOL,
    derive_explicit_validation_cohort,
    patient_sets,
    study_sets,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data_prepared/skull_hu800_ww1600"
DEFAULT_MANIFEST = DATASET / "manifest.csv"
DEFAULT_ASSIGNMENTS = DATASET / "kfold/patient_fold_assignments.csv"
DEFAULT_METADATA = ROOT / "iaaa-contest-bct/Data/training_df.pkl"
DEFAULT_DICOM_ROOT = ROOT / "iaaa-contest-bct/Data/training"
MINIMUM_FREE_VRAM_GIB = 8.0
SCORE_PROTOCOLS = {
    "proposal_diagnostic": {
        "purpose": "proposal-recall/localization diagnostics only",
        "confidence_floor": 0.001, "nms_iou": 0.7, "max_det": 300,
        "initial_aggregator": None,
    },
    "deployment_score": {
        "purpose": "Study-level score generation for Macro-F1/aggregation/calibration",
        "confidence_floor": 0.01, "nms_iou": 0.5, "max_det": 300,
        "initial_aggregator": "top3_mean",
    },
}
STUDY_INCLUSION_PROTOCOL = "explicit_validation_studies_v2_all_valid_target_series_dicoms"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_index(device: str) -> int:
    value = str(device).strip().lower()
    if value == "cpu":
        raise ValueError("gpu_index called for CPU")
    try:
        return int(value.split(",")[0])
    except ValueError as exc:
        raise ValueError("--device must be 'cpu' or a numeric CUDA device such as 0") from exc


def query_free_vram_gib(device: str) -> float:
    index = gpu_index(device)
    command = [
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(index)
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        mib = float(completed.stdout.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as exc:
        raise RuntimeError("Cannot establish free GPU memory before model loading") from exc
    return mib / 1024.0


def enforce_vram_guard(device: str, force: bool, minimum_gib: float = MINIMUM_FREE_VRAM_GIB) -> float:
    """Hard-fail before importing Ultralytics when CUDA has too little free VRAM."""
    if str(device).lower() == "cpu":
        return float("inf")
    free = query_free_vram_gib(device)
    if free < minimum_gib and not force:
        raise RuntimeError(
            "Refusing GPU inference: %.3f GiB free is below %.1f GiB. "
            "Wait for training to finish or explicitly pass --force." % (free, minimum_gib)
        )
    return free


def _read_paths(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [str(Path(line.strip()).resolve()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("Empty or duplicate paths in %s" % path)
    return rows


def validate_protocol(
    manifest_path: Path,
    assignments_path: Path,
    val_lists: Sequence[Path],
    train_lists: Sequence[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Validate patient-heldout folds and hard-fail on fixed-test inclusion."""
    if len(val_lists) != 5 or len(train_lists) != 5:
        raise ValueError("Exactly five train lists and five validation lists are required")
    manifest = pd.read_csv(
        manifest_path, dtype={"patient_id": str, "study_id": str, "sop_uid": str, "image_path": str}
    )
    required = {"split", "patient_id", "study_id", "image_path"}
    if required - set(manifest.columns):
        raise ValueError("Manifest missing %s" % sorted(required - set(manifest.columns)))
    manifest["resolved_image_path"] = manifest["image_path"].map(lambda value: str(Path(value).resolve()))
    if manifest["resolved_image_path"].duplicated().any():
        raise ValueError("Duplicate image paths in manifest")
    image_rows = manifest.set_index("resolved_image_path")
    cohort = derive_explicit_validation_cohort(manifest, val_lists)
    heldout_by_fold = patient_sets(cohort)
    assignments = pd.read_csv(assignments_path, dtype={"patient_id": str})
    if assignments["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient assignment")
    fixed_patients = set(assignments.loc[assignments["partition"] == "fixed_test", "patient_id"])
    fixed_manifest_patients = set(manifest.loc[manifest["split"] == "test", "patient_id"])
    if fixed_patients != fixed_manifest_patients:
        raise ValueError("Fixed-test patient definitions disagree between manifest and assignments")

    seen_val_patients = set()
    for fold, (val_path, train_path) in enumerate(zip(val_lists, train_lists)):
        val_paths = _read_paths(val_path)
        train_paths = _read_paths(train_path)
        unknown = (set(val_paths) | set(train_paths)) - set(image_rows.index)
        if unknown:
            raise ValueError("Fold %d contains paths absent from manifest: %s" % (fold, sorted(unknown)[:3]))
        val_rows = image_rows.loc[val_paths]
        train_rows = image_rows.loc[train_paths]
        val_patients = set(val_rows["patient_id"].astype(str))
        train_patients = set(train_rows["patient_id"].astype(str))
        expected_assignment_patients = set(assignments.loc[
            (assignments["partition"] == "trainval")
            & (assignments["validation_fold"].astype(int) == fold), "patient_id"
        ])
        if val_patients != expected_assignment_patients or val_patients != heldout_by_fold[fold]:
            raise ValueError("Fold %d validation patients disagree with frozen assignments" % fold)
        if val_patients & train_patients:
            raise ValueError("Patient leakage in fold %d" % fold)
        if (val_patients | train_patients) & fixed_patients:
            raise ValueError("Fixed-test patient included in fold %d" % fold)
        if seen_val_patients & val_patients:
            raise ValueError("Patient held out by more than one fold")
        seen_val_patients.update(val_patients)
    return manifest, assignments, cohort


def validate_cache_cohort(cache_dir: Path, expected_by_fold: Dict[int, set], require_complete: bool) -> None:
    """Read Study IDs from cache JSON content and reject extras/missing coverage."""
    seen = {}
    for fold in range(5):
        actual = set()
        studies_dir = cache_dir / ("fold_%d" % fold) / "studies"
        for path in sorted(studies_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            study = str(payload.get("series_id", ""))
            content_fold = int(payload.get("fold", -1))
            if not study or content_fold != fold:
                raise RuntimeError("Invalid Study/fold identity in cache content: %s" % path)
            if study in actual or study in seen:
                raise RuntimeError("Cached Study %s occurs more than once" % study)
            actual.add(study)
            seen[study] = fold
        extra = actual - expected_by_fold[fold]
        missing = expected_by_fold[fold] - actual
        if extra:
            raise RuntimeError("Unexpected cached Studies in Fold %d: %s" % (fold, sorted(extra)))
        marker = cache_dir / ("fold_%d" % fold) / "COMPLETE.json"
        if require_complete or marker.exists():
            if missing:
                raise RuntimeError("Expected Studies missing from completed Fold %d cache: %s" % (fold, sorted(missing)))
            if marker.exists():
                marker_payload = json.loads(marker.read_text(encoding="utf-8"))
                marker_expected = set(map(str, marker_payload.get("expected_study_ids", [])))
                if marker_expected != expected_by_fold[fold] or int(marker_payload.get("studies", -1)) != len(actual):
                    raise RuntimeError("Completion marker cohort mismatch in Fold %d" % fold)


def metadata_index(metadata_path: Path) -> Tuple[pd.DataFrame, set]:
    metadata = pd.read_pickle(metadata_path).copy()
    required = {"dicom_series.id", "dicom_series.PatientID", "dicom_series.SOPInstanceUID"}
    if required - set(metadata.columns):
        raise ValueError("Metadata missing %s" % sorted(required - set(metadata.columns)))
    metadata["study_id"] = metadata["dicom_series.id"].astype(str)
    metadata["patient_id"] = metadata["dicom_series.PatientID"].astype(str)
    metadata["sop_uid"] = metadata["dicom_series.SOPInstanceUID"].astype(str)
    if metadata[["study_id", "sop_uid"]].duplicated().any():
        raise ValueError("Duplicate study/SOP metadata key")
    return metadata, set(zip(metadata["study_id"], metadata["sop_uid"]))


def select_explicit_fold_studies(
    metadata: pd.DataFrame, expected_studies: set, heldout_patients: set, fold: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select explicit Study IDs only; never expand cohort by patient."""
    fold_metadata = metadata[metadata["study_id"].isin(expected_studies)].copy()
    studies = fold_metadata[["study_id", "patient_id"]].drop_duplicates()
    if studies["study_id"].duplicated().any():
        raise ValueError("A held-out study maps to multiple patients")
    if set(studies["study_id"].astype(str)) != set(map(str, expected_studies)):
        raise RuntimeError("Metadata does not exactly cover explicit validation Studies in Fold %d" % fold)
    if not set(studies["patient_id"].astype(str)) <= set(map(str, heldout_patients)):
        raise RuntimeError("Explicit validation Study has a non-held-out patient in Fold %d" % fold)
    return fold_metadata, studies


def cache_is_resumable(path: Path, signature: Dict[str, Any], expected_sops: set) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("inference_signature") != signature:
        raise RuntimeError("Existing cache has a different inference signature: %s" % path)
    cached_sops = {str(item["sop_uid"]) for item in payload.get("slices", [])}
    if cached_sops != expected_sops:
        raise RuntimeError("Existing cache slice coverage mismatch: %s" % path)
    return True


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def infer_study(
    model: Any,
    study_dir: Path,
    series_id: str,
    patient_id: str,
    fold: int,
    included_headers: Sequence[Dict[str, Any]],
    excluded_headers: Sequence[Dict[str, Any]],
    ordering_method: str,
    signature: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    from data.dicom import read_slice
    from data.windows import make_hu_input

    records = [read_slice(Path(header["filesystem_path"]), series_id) for header in included_headers]
    if not records:
        raise RuntimeError("Study %s has no metadata-backed DICOM slices" % series_id)
    hu = [record.hu for record in records]
    positions = [record.physical_position for record in records]
    monochrome1 = [record.photometric_interpretation == "MONOCHROME1" for record in records]
    slice_rows = []  # type: List[Dict[str, Any]]
    for start in range(0, len(records), args.batch):
        stop = min(start + args.batch, len(records))
        images = [
            make_hu_input(
                hu, index, level=args.window_level, width=args.window_width,
                mode="2.5d", boundary_mode="repeat", monochrome1=monochrome1,
                physical_positions=positions, context_distance_mm=args.context_distance_mm,
            )
            for index in range(start, stop)
        ]
        predictions = model.predict(
            source=images, imgsz=args.imgsz, device=args.device, batch=args.batch,
            conf=args.conf, iou=args.nms_iou, max_det=args.max_det,
            half=args.half, stream=False, verbose=False,
        )
        if len(predictions) != len(images):
            raise RuntimeError("Prediction count mismatch in study %s" % series_id)
        for record, prediction in zip(records[start:stop], predictions):
            if prediction.boxes is None or len(prediction.boxes) == 0:
                boxes, scores = [], []
            else:
                boxes = prediction.boxes.xyxy.detach().cpu().numpy().astype(float).tolist()
                scores = prediction.boxes.conf.detach().cpu().numpy().astype(float).tolist()
            slice_rows.append({
                "sop_uid": str(record.sop_uid),
                "physical_position": record.physical_position,
                "image_shape": list(record.shape),
                "max_confidence": float(max(scores, default=0.0)),
                "num_detections": len(scores),
                "boxes": boxes,
                "scores": scores,
                "source_confidence_floor": args.conf,
            })
    return {
        "prediction_protocol": "heldout_patient_full_study",
        "study_inclusion_protocol": STUDY_INCLUSION_PROTOCOL,
        "score_generation_protocol": args.score_protocol,
        "fold": int(fold), "series_id": series_id, "patient_id": patient_id,
        "included_target_series_slices": len(records),
        "included_metadata_unknown_slices": sum(
            not bool(header.get("metadata_backed")) for header in included_headers
        ),
        "slice_ordering_method": ordering_method,
        "excluded_dicoms": [
            {
                "filesystem_path": header["filesystem_path"],
                "SOPInstanceUID": header["SOPInstanceUID"],
                "classification": header["classification"],
                "reason": header["reason"],
            }
            for header in excluded_headers
        ],
        "inference_signature": signature,
        "slices": slice_rows,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from dicom_series_guard import canonical_profile, discover_headers, select_and_order_target_headers

    manifest, assignments, cohort = validate_protocol(
        args.manifest, args.assignments, args.val_lists, args.train_lists
    )
    expected_by_fold = study_sets(cohort)
    heldout = patient_sets(cohort)
    if cohort["total_unique_validation_studies"] != AUTHORITATIVE_TOTAL_STUDIES:
        raise RuntimeError("Expected exactly 169 explicit validation Studies")
    if tuple(cohort["per_fold_study_counts"]) != AUTHORITATIVE_FOLD_STUDY_COUNTS:
        raise RuntimeError("Explicit validation Fold Study-count invariant failed")
    validate_cache_cohort(args.cache_dir, expected_by_fold, require_complete=False)
    metadata, metadata_keys = metadata_index(args.metadata)
    protocol = SCORE_PROTOCOLS[args.score_protocol]
    args.conf = float(protocol["confidence_floor"])
    args.nms_iou = float(protocol["nms_iou"])
    args.max_det = int(protocol["max_det"])
    fixed_patients = set(assignments.loc[assignments["partition"] == "fixed_test", "patient_id"])
    requested_folds = sorted(set(args.fold if args.fold else range(5)))
    if not set(requested_folds) <= set(range(5)):
        raise ValueError("Fold ids must be in 0..4")
    if len(args.weights) != 5:
        raise ValueError("Exactly five --weights paths are required")

    # The guard must precede importing Ultralytics or constructing any model.
    free_vram = enforce_vram_guard(args.device, args.force)
    from ultralytics import YOLO

    summary = {"folds": {}, "free_vram_gib_at_start": free_vram}  # type: Dict[str, Any]
    for fold in requested_folds:
        weight = args.weights[fold]
        if not weight.is_file():
            raise FileNotFoundError(weight)
        patients = heldout[fold]
        if patients & fixed_patients:
            raise RuntimeError("Fixed-test patient reached inference plan")
        expected_studies = expected_by_fold[fold]
        fold_metadata, studies = select_explicit_fold_studies(
            metadata, expected_studies, patients, fold
        )
        signature = {
            "weights_sha256": sha256_file(weight), "imgsz": args.imgsz,
            "conf": args.conf, "nms_iou": args.nms_iou, "max_det": args.max_det,
            "window_level": args.window_level, "window_width": args.window_width,
            "context_distance_mm": args.context_distance_mm, "input_mode": "2.5d",
            "score_generation_protocol": args.score_protocol,
            "score_protocol_definition": protocol,
            "study_inclusion_protocol": STUDY_INCLUSION_PROTOCOL,
            "cohort_protocol": COHORT_PROTOCOL,
            "explicit_validation_study_count": len(expected_studies),
        }
        completed = 0
        model = None
        for row in studies.sort_values("study_id").itertuples(index=False):
            series_id, patient_id = str(row.study_id), str(row.patient_id)
            if patient_id in fixed_patients:
                raise RuntimeError("Fixed-test inclusion detected for study %s" % series_id)
            study_metadata = fold_metadata[fold_metadata["study_id"] == series_id]
            canonical_uids = sorted(set(study_metadata["dicom_series.SeriesInstanceUID"].astype(str)))
            if len(canonical_uids) != 1:
                raise RuntimeError("Study %s does not have exactly one canonical target SeriesInstanceUID" % series_id)
            metadata_sops = set(study_metadata["sop_uid"].astype(str))
            headers = discover_headers(args.dicom_root / series_id)
            profile = canonical_profile(headers, canonical_uids[0], metadata_sops)
            for header in headers:
                header["metadata_backed"] = (series_id, str(header["SOPInstanceUID"])) in metadata_keys
            included_headers, excluded_headers, ordering_method = select_and_order_target_headers(headers, profile)
            expected_sops = {str(header["SOPInstanceUID"]) for header in included_headers}
            cache_path = args.cache_dir / ("fold_%d" % fold) / "studies" / (series_id + ".json")
            if cache_is_resumable(cache_path, signature, expected_sops):
                completed += 1
                continue
            if model is None:
                # Re-check immediately before constructing each fold model in
                # case another process consumed memory after program start.
                enforce_vram_guard(args.device, args.force)
                model = YOLO(str(weight))
            payload = infer_study(
                model, args.dicom_root / series_id, series_id, patient_id, fold,
                included_headers, excluded_headers, ordering_method, signature, args,
            )
            cached_sops = {str(item["sop_uid"]) for item in payload["slices"]}
            if cached_sops != expected_sops:
                raise RuntimeError("DICOM/metadata slice coverage mismatch for study %s" % series_id)
            atomic_json(cache_path, payload)
            completed += 1
        marker = args.cache_dir / ("fold_%d" % fold) / "COMPLETE.json"
        atomic_json(marker, {
            "fold": fold, "prediction_protocol": "heldout_patient_full_study",
            "studies": completed, "expected_study_ids": sorted(expected_studies),
            "score_generation_protocol": args.score_protocol,
            "study_inclusion_protocol": STUDY_INCLUSION_PROTOCOL,
            "cohort_protocol": COHORT_PROTOCOL,
            "inference_signature": signature,
        })
        summary["folds"][str(fold)] = {"studies": completed, "completion_marker": str(marker)}
        del model
    validate_cache_cohort(
        args.cache_dir, expected_by_fold,
        require_complete=set(requested_folds) == set(range(5)),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, action="append", required=True)
    parser.add_argument("--val-list", dest="val_lists", type=Path, action="append", default=[])
    parser.add_argument("--train-list", dest="train_lists", type=Path, action="append", default=[])
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--dicom-root", type=Path, default=DEFAULT_DICOM_ROOT)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--score-protocol", choices=sorted(SCORE_PROTOCOLS), default="deployment_score")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--window-level", type=float, default=800.0)
    parser.add_argument("--window-width", type=float, default=1600.0)
    parser.add_argument("--context-distance-mm", type=float, default=5.0)
    parser.add_argument("--half", dest="half", action="store_true", default=True)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.val_lists:
        args.val_lists = [DATASET / ("kfold/fold_%d_val.txt" % fold) for fold in range(5)]
    if not args.train_lists:
        args.train_lists = [DATASET / ("kfold/fold_%d_train.txt" % fold) for fold in range(5)]
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
