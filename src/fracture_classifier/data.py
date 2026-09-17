"""Manifest-backed, annotation-only slice labels for the Run A OOF cohort."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from cache_full_study_oof import DATASET, DEFAULT_MANIFEST, ROOT
from data.annotations import parse_annotation
from oof_cohort import derive_explicit_validation_cohort, study_sets


CACHE = ROOT / "outputs/full_study_oof_run_a_768_deployment_score_cache_v2_169"
EXPERIMENT = ROOT / "outputs/run_a_fracture_classifier_oof"
FINAL_DEPLOYMENT = ROOT / "outputs/yolo26s_p2_hu800_ww1600_run_a_full169/weights/final_deployment.pt"
MANIFEST_COLUMNS = (
    "fold", "patient_id", "study_id", "sop_uid", "image_path", "label_path",
    "fracture_label", "label_source", "is_reviewed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def detector_checkpoints() -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in sorted((ROOT / "outputs").glob("yolo26s_p2_hu800_ww1600_fold*/weights/best.pt")):
        match = re.fullmatch(r"yolo26s_p2_hu800_ww1600_fold([0-4])", path.parent.parent.name)
        if match:
            fold = int(match.group(1))
            if fold in found:
                raise RuntimeError(f"Duplicate Run A checkpoint for Fold {fold}")
            found[fold] = path.resolve()
    if set(found) != set(range(5)):
        raise RuntimeError(f"Expected five historical Run A checkpoints; found {sorted(found)}")
    return found


def _paths(path: Path) -> list[str]:
    values = [str(Path(line).resolve()) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise RuntimeError(f"Duplicate image paths in {path}")
    return values


def reviewed_binary_label(annotation_path: Path, expected_boxes: int, label_path: Path) -> int:
    """Only an existing reviewed JSON with matching valid boxes can supply a label."""
    if not annotation_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing reviewed annotation/YOLO label: {annotation_path}")
    boxes = parse_annotation(annotation_path).boxes
    label_count = sum(bool(line.strip()) for line in label_path.read_text().splitlines())
    if len(boxes) != expected_boxes or len(boxes) != label_count:
        raise RuntimeError(f"Annotation/YOLO label count mismatch: {annotation_path}")
    return int(bool(boxes))


def build_slice_manifest(cache_dir: Path = CACHE) -> tuple[pd.DataFrame, dict]:
    """Unknown slices appear with null labels and are never classifier training examples."""
    vals = [DATASET / f"kfold/fold_{fold}_val.txt" for fold in range(5)]
    trains = [DATASET / f"kfold/fold_{fold}_train.txt" for fold in range(5)]
    manifest = pd.read_csv(DEFAULT_MANIFEST, dtype={
        "patient_id": str, "study_id": str, "sop_uid": str,
        "image_path": str, "label_path": str,
    })
    cohort = derive_explicit_validation_cohort(manifest, vals)
    if cohort["per_fold_study_counts"] != [33, 33, 32, 36, 35]:
        raise RuntimeError("Historical Run A cohort changed")
    # Validate the exact historical train lists using only selected development
    # rows. No fixed-test Study/patient identifiers or labels are enumerated.
    by_path = manifest.copy()
    by_path["resolved_image_path"] = by_path["image_path"].map(lambda value: str(Path(value).resolve()))
    if by_path["resolved_image_path"].duplicated().any():
        raise RuntimeError("Manifest image path is not one-to-one")
    by_path = by_path.set_index("resolved_image_path")
    development_paths = set().union(*(set(_paths(path)) for path in vals))
    for fold in range(5):
        train_paths, val_paths = set(_paths(trains[fold])), set(_paths(vals[fold]))
        if train_paths & val_paths or train_paths | val_paths != development_paths:
            raise RuntimeError(f"Historical Run A train/validation membership mismatch in Fold {fold}")
        selected = by_path.loc[list(train_paths | val_paths)]
        if (selected["split"] == "test").any():
            raise RuntimeError("Fixed-test row entered historical Run A development lists")
        train_patients = set(by_path.loc[list(train_paths), "patient_id"])
        val_patients = set(by_path.loc[list(val_paths), "patient_id"])
        if train_patients & val_patients or val_patients != set(cohort["folds"][fold]["patient_ids"]):
            raise RuntimeError(f"Patient leakage/held-out mismatch in Fold {fold}")
    expected = study_sets(cohort)
    dev = manifest[manifest["split"].isin(("train", "val"))].copy()
    dev["study_id"] = dev["study_id"].astype(str)
    dev["patient_id"] = dev["patient_id"].astype(str)
    dev["sop_uid"] = dev["sop_uid"].astype(str)
    if dev[["study_id", "sop_uid"]].duplicated().any():
        raise RuntimeError("Duplicate annotated Study/SOP in manifest")
    by_sop = dev.set_index(["study_id", "sop_uid"])
    rows = []
    fold_reports = []
    all_studies: set[str] = set()
    all_patients: dict[str, int] = {}
    for fold in range(5):
        marker = cache_dir / f"fold_{fold}/COMPLETE.json"
        if not marker.is_file():
            raise FileNotFoundError(marker)
        complete = json.loads(marker.read_text())
        if set(map(str, complete.get("expected_study_ids", []))) != expected[fold]:
            raise RuntimeError(f"Run A cache cohort mismatch in Fold {fold}")
        paths = sorted((cache_dir / f"fold_{fold}/studies").glob("*.json"))
        seen: set[str] = set()
        counts = {"positive": 0, "negative": 0, "ignored": 0}
        for path in paths:
            payload = json.loads(path.read_text())
            study, patient = str(payload["series_id"]), str(payload["patient_id"])
            if int(payload["fold"]) != fold or study not in expected[fold] or study in seen or study in all_studies:
                raise RuntimeError(f"Unexpected/duplicate cached Study in Fold {fold}: {study}")
            seen.add(study)
            all_studies.add(study)
            if patient in all_patients and all_patients[patient] != fold:
                raise RuntimeError(f"Cross-fold patient leakage: {patient}")
            all_patients[patient] = fold
            if cohort["folds"][fold]["study_to_patient"][study] != patient:
                raise RuntimeError(f"Cached patient mismatch for Study {study}")
            observed_sops: set[str] = set()
            for item in payload["slices"]:
                sop = str(item["sop_uid"])
                if sop in observed_sops:
                    raise RuntimeError(f"Duplicate SOP in Study {study}")
                observed_sops.add(sop)
                key = (study, sop)
                row = {"fold": fold, "patient_id": patient, "study_id": study, "sop_uid": sop,
                       "image_path": "", "label_path": "", "fracture_label": pd.NA,
                       "label_source": "ignored_no_annotation_json", "is_reviewed": False}
                if key in by_sop.index:
                    source = by_sop.loc[key]
                    if source["split"] == "test":
                        raise RuntimeError("Fixed-test slice entered classifier labels")
                    image = Path(source["image_path"])
                    label = Path(source["label_path"])
                    annotation = ROOT / "iaaa-contest-bct/Data/annotations" / study / f"{sop}.json"
                    if not image.is_file() or not label.is_file() or not annotation.is_file():
                        raise FileNotFoundError(f"Reviewed slice is missing image/label/JSON: {study}/{sop}")
                    binary = reviewed_binary_label(annotation, int(source["num_boxes"]), label)
                    row.update({"image_path": str(image.resolve()), "label_path": str(label.resolve()),
                                "fracture_label": binary, "label_source": "reviewed_annotation_json",
                                "is_reviewed": True})
                    counts["positive" if binary else "negative"] += 1
                else:
                    counts["ignored"] += 1
                rows.append(row)
            expected_annotated = set(dev.loc[dev["study_id"] == study, "sop_uid"])
            if not expected_annotated <= observed_sops:
                raise RuntimeError(f"Cached full Study misses annotation-backed slices: {study}")
        if seen != expected[fold]:
            raise RuntimeError(f"Missing/extra cached Study in Fold {fold}")
        fold_reports.append({"fold": fold, "studies": len(seen), "patients": len(cohort["folds"][fold]["patient_ids"]), **counts})
    table = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if len(all_studies) != 169 or len(all_patients) != 155:
        raise RuntimeError("Run A full-study development cohort is not 169 Studies/155 patients")
    if len(table[table["is_reviewed"]]) != sum(len(_paths(path)) for path in vals):
        raise RuntimeError("Reviewed-slice coverage is not exact")
    return table, {"experiment": "RUN_A_FRACTURE_CLASSIFIER_OOF", "per_fold": fold_reports,
                   "studies": len(all_studies), "patients": len(all_patients),
                   "patient_leakage": False,
                   "detector_sha256": {str(fold): sha256(path) for fold, path in detector_checkpoints().items()},
                   "full169_deployment_sha256": sha256(FINAL_DEPLOYMENT)}


def fold_rows(table: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if fold not in range(5):
        raise ValueError("Fold must be 0..4")
    reviewed = table[table["is_reviewed"]].copy()
    train = reviewed[reviewed["fold"] != fold]
    val = reviewed[reviewed["fold"] == fold]
    if set(train["patient_id"]) & set(val["patient_id"]):
        raise RuntimeError("Patient leakage between classifier training and validation")
    pos = int((train["fracture_label"] == 1).sum())
    neg = int((train["fracture_label"] == 0).sum())
    if pos == 0 or neg == 0 or len(val) == 0:
        raise RuntimeError("Cannot compute training-only pos_weight")
    return train, val, neg / pos


def write_manifest_once(table: pd.DataFrame, report: dict) -> None:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENT / "slice_label_manifest.csv"
    content = table.to_csv(index=False)
    if path.exists() and path.read_text() != content:
        raise RuntimeError("Existing classifier slice manifest differs; refusing overwrite")
    if not path.exists():
        path.write_text(content)
    report_path = EXPERIMENT / "slice_label_audit.json"
    audit = json.dumps(report, indent=2) + "\n"
    if report_path.exists() and json.loads(report_path.read_text()) != report:
        raise RuntimeError("Existing classifier audit differs; refusing overwrite")
    if not report_path.exists():
        report_path.write_text(audit)


if __name__ == "__main__":
    labels, summary = build_slice_manifest()
    write_manifest_once(labels, summary)
    print(json.dumps(summary, indent=2))
