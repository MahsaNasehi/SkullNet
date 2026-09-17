"""Fast CPU-only dataset and patient-fold coverage audit; never edits data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data_prepared/skull_hu800_ww1600"


def keyed_files(root: Path, suffix: str = "") -> set:
    keys = set()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for study_dir in root.iterdir():
        if not study_dir.is_dir():
            continue
        for path in study_dir.rglob("*"):
            if path.is_file() and (not suffix or path.suffix.lower() == suffix):
                keys.add((study_dir.name, path.stem))
    return keys


def read_fold_paths(paths: Sequence[Path], manifest: pd.DataFrame) -> Tuple[List[set], Dict[str, Any]]:
    by_image = manifest.set_index("resolved_image_path")
    fold_patients = []  # type: List[set]
    fold_studies = []  # type: List[set]
    for fold, path in enumerate(paths):
        values = [str(Path(line).resolve()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(values) != len(set(values)):
            raise ValueError("Duplicate images in fold %d" % fold)
        unknown = set(values) - set(by_image.index)
        if unknown:
            raise ValueError("Fold %d contains images absent from manifest" % fold)
        rows = by_image.loc[values]
        fold_patients.append(set(rows["patient_id"].astype(str)))
        fold_studies.append(set(rows["study_id"].astype(str)))
    patient_membership = {}  # type: Dict[str, List[int]]
    study_membership = {}  # type: Dict[str, List[int]]
    for fold in range(len(paths)):
        for patient in fold_patients[fold]:
            patient_membership.setdefault(patient, []).append(fold)
        for study in fold_studies[fold]:
            study_membership.setdefault(study, []).append(fold)
    development = manifest[manifest["split"] != "test"]
    expected_studies = set(development["study_id"].astype(str))
    observed_studies = set(study_membership)
    return fold_patients, {
        "patient_leakage_across_validation_folds": {
            patient: folds for patient, folds in patient_membership.items() if len(folds) > 1
        },
        "study_duplication_across_validation_folds": {
            study: folds for study, folds in study_membership.items() if len(folds) > 1
        },
        "missing_development_studies_from_validation_coverage": sorted(expected_studies - observed_studies),
        "unexpected_validation_studies": sorted(observed_studies - expected_studies),
        "covered_development_studies": len(observed_studies & expected_studies),
        "expected_development_studies": len(expected_studies),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    metadata = pd.read_pickle(args.metadata).copy()
    metadata["study_id"] = metadata["dicom_series.id"].astype(str)
    metadata["patient_id"] = metadata["dicom_series.PatientID"].astype(str)
    metadata["sop_uid"] = metadata["dicom_series.SOPInstanceUID"].astype(str)
    metadata_keys = set(zip(metadata["study_id"], metadata["sop_uid"]))
    dicom_keys = keyed_files(args.dicom_root)
    json_keys = keyed_files(args.annotation_root, ".json")

    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str, "study_id": str, "image_path": str})
    manifest["resolved_image_path"] = manifest["image_path"].map(lambda value: str(Path(value).resolve()))
    fold_patients, coverage = read_fold_paths(args.val_lists, manifest)
    assignments = pd.read_csv(args.assignments, dtype={"patient_id": str})
    duplicate_assignments = assignments.loc[assignments["patient_id"].duplicated(False), "patient_id"].unique().tolist()

    positive_studies = set(metadata.groupby("study_id")["SkullFracture"].max().loc[lambda x: x.astype(bool)].index)
    patient_positive = metadata.groupby("patient_id")["SkullFracture"].max().astype(bool)
    fold_rows = []
    for fold, patients in enumerate(fold_patients):
        studies = set(metadata.loc[metadata["patient_id"].isin(patients), "study_id"])
        fold_rows.append({
            "fold": fold,
            "heldout_patients": len(patients),
            "heldout_studies_all_metadata": len(studies),
            "positive_studies": len(studies & positive_studies),
            "positive_patients": int(patient_positive.reindex(list(patients), fill_value=False).sum()),
        })

    assigned_trainval = set(assignments.loc[assignments["partition"] == "trainval", "patient_id"])
    fixed_patients = set(assignments.loc[assignments["partition"] == "fixed_test", "patient_id"])
    metadata_patients = set(metadata["patient_id"])
    report = {
        "mode": "CPU-only metadata/path audit; no DICOM pixel decoding and no dataset modification",
        "dataset": {
            "total_studies": int(metadata["study_id"].nunique()),
            "total_patients": int(metadata["patient_id"].nunique()),
            "total_dicom_studies": len({study for study, _ in dicom_keys}),
            "total_dicom_slices": len(dicom_keys),
            "metadata_rows": len(metadata),
            "metadata_backed_dicom_slices": len(dicom_keys & metadata_keys),
            "json_backed_dicom_slices": len(dicom_keys & json_keys),
            "metadata_backed_non_json_slices": len((dicom_keys & metadata_keys) - json_keys),
            "metadata_unknown_dicom_slices": len(dicom_keys - metadata_keys),
            "json_without_dicom": len(json_keys - dicom_keys),
            "metadata_without_dicom": len(metadata_keys - dicom_keys),
        },
        "split_coverage": coverage,
        "assignment_audit": {
            "duplicate_patient_assignments": duplicate_assignments,
            "trainval_assigned_patients": len(assigned_trainval),
            "fixed_test_patients": len(fixed_patients),
            "unassigned_metadata_patients": len(metadata_patients - assigned_trainval - fixed_patients),
            "trainval_fixed_patient_overlap": sorted(assigned_trainval & fixed_patients),
        },
        "folds": fold_rows,
    }
    if coverage["patient_leakage_across_validation_folds"]:
        raise RuntimeError("Patient leakage detected: %s" % coverage["patient_leakage_across_validation_folds"])
    if coverage["study_duplication_across_validation_folds"]:
        raise RuntimeError("Study duplication detected: %s" % coverage["study_duplication_across_validation_folds"])
    if duplicate_assignments or assigned_trainval & fixed_patients:
        raise RuntimeError("Invalid patient assignment table")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite %s" % args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "iaaa-contest-bct/Data/training_df.pkl")
    parser.add_argument("--dicom-root", type=Path, default=ROOT / "iaaa-contest-bct/Data/training")
    parser.add_argument("--annotation-root", type=Path, default=ROOT / "iaaa-contest-bct/Data/annotations")
    parser.add_argument("--manifest", type=Path, default=DATASET / "manifest.csv")
    parser.add_argument("--assignments", type=Path, default=DATASET / "kfold/patient_fold_assignments.csv")
    parser.add_argument("--val-list", dest="val_lists", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/oof_dataset_coverage_audit.json")
    args = parser.parse_args()
    if not args.val_lists:
        args.val_lists = [DATASET / ("kfold/fold_%d_val.txt" % fold) for fold in range(5)]
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
