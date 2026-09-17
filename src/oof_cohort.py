"""Authoritative explicit Study cohort recovered from validation image lists."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd


AUTHORITATIVE_FOLD_STUDY_COUNTS = (33, 33, 32, 36, 35)
AUTHORITATIVE_TOTAL_STUDIES = 169
COHORT_PROTOCOL = "manifest_mapped_explicit_validation_studies_v2_169"


def _resolved_lines(path: Path) -> list:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = [str(Path(line.strip()).resolve()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("Empty or duplicated validation image paths: %s" % path)
    return values


def derive_explicit_validation_cohort(
    manifest: pd.DataFrame,
    val_lists: Sequence[Path],
    expected_counts: Tuple[int, ...] = AUTHORITATIVE_FOLD_STUDY_COUNTS,
    expected_total: int = AUTHORITATIVE_TOTAL_STUDIES,
) -> Dict[str, Any]:
    """Map every explicit validation image path through the authoritative manifest."""
    if len(val_lists) != len(expected_counts):
        raise ValueError("Expected %d validation lists" % len(expected_counts))
    required = {"split", "patient_id", "study_id", "image_path"}
    if required - set(manifest.columns):
        raise ValueError("Manifest missing %s" % sorted(required - set(manifest.columns)))
    work = manifest.copy()
    work["patient_id"] = work["patient_id"].astype(str)
    work["study_id"] = work["study_id"].astype(str)
    work["resolved_image_path"] = work["image_path"].map(lambda value: str(Path(value).resolve()))
    if work["resolved_image_path"].duplicated().any():
        raise ValueError("Manifest image_path mapping is not one-to-one")
    by_path = work.set_index("resolved_image_path")

    folds = []
    study_to_fold = {}
    patient_to_fold = {}
    for fold, path in enumerate(val_lists):
        image_paths = _resolved_lines(path)
        missing_paths = set(image_paths) - set(by_path.index)
        if missing_paths:
            raise ValueError("Fold %d contains paths absent from manifest: %s" % (fold, sorted(missing_paths)[:5]))
        rows = by_path.loc[image_paths]
        if (rows["split"] == "test").any():
            raise RuntimeError("Fixed-test image/Study entered explicit validation Fold %d" % fold)
        study_patient = rows[["study_id", "patient_id"]].drop_duplicates()
        if study_patient["study_id"].duplicated().any():
            raise ValueError("A validation Study maps to multiple patients in Fold %d" % fold)
        studies = sorted(set(study_patient["study_id"]))
        patients = sorted(set(study_patient["patient_id"]))
        if len(studies) != expected_counts[fold]:
            raise RuntimeError(
                "Authoritative Fold %d Study count is %d, expected invariant %d"
                % (fold, len(studies), expected_counts[fold])
            )
        for study in studies:
            if study in study_to_fold:
                raise RuntimeError("Study %s occurs in validation Folds %d and %d" % (study, study_to_fold[study], fold))
            study_to_fold[study] = fold
        for patient in patients:
            if patient in patient_to_fold and patient_to_fold[patient] != fold:
                raise RuntimeError("Patient %s occurs in multiple validation Folds" % patient)
            patient_to_fold[patient] = fold
        folds.append({
            "fold": fold,
            "explicit_validation_slices": len(image_paths),
            "explicit_validation_studies": len(studies),
            "heldout_patients": len(patients),
            "study_ids": studies,
            "patient_ids": patients,
            "study_to_patient": dict(zip(study_patient["study_id"], study_patient["patient_id"])),
        })
    if len(study_to_fold) != expected_total:
        raise RuntimeError(
            "Total explicit validation Studies is %d, expected %d" % (len(study_to_fold), expected_total)
        )
    return {
        "cohort_protocol": COHORT_PROTOCOL,
        "authoritative_source": "fold_*_val.txt image paths mapped through manifest.csv; no filename parsing",
        "total_unique_validation_studies": len(study_to_fold),
        "per_fold_study_counts": [item["explicit_validation_studies"] for item in folds],
        "folds": folds,
    }


def study_sets(cohort: Dict[str, Any]) -> Dict[int, set]:
    return {int(item["fold"]): set(map(str, item["study_ids"])) for item in cohort["folds"]}


def patient_sets(cohort: Dict[str, Any]) -> Dict[int, set]:
    return {int(item["fold"]): set(map(str, item["patient_ids"])) for item in cohort["folds"]}

