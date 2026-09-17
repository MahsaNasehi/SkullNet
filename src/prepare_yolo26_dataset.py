"""Build a leakage-safe, HU-windowed YOLO dataset from the labeled CT slices."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from data.annotations import parse_annotation, xywh_to_yolo
from data.dicom import load_study
from data.windows import make_hu_input


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "iaaa-contest-bct" / "Data"


def _study_patient_map(metadata_path: Path) -> dict[str, str]:
    frame = pd.read_pickle(metadata_path)
    required = {"dicom_series.id", "dicom_series.PatientID"}
    if not required <= set(frame.columns):
        raise ValueError(f"Metadata is missing columns: {sorted(required - set(frame.columns))}")
    pairs = frame[["dicom_series.id", "dicom_series.PatientID"]].drop_duplicates()
    counts = pairs.groupby("dicom_series.id")["dicom_series.PatientID"].nunique()
    if (counts != 1).any():
        raise ValueError("At least one study maps to multiple patients")
    return {str(study): str(patient) for study, patient in pairs.itertuples(index=False, name=None)}


def _has_box(path: Path) -> bool:
    return bool(parse_annotation(path).boxes)


def _make_patient_split(
    annotated_studies: list[str], study_to_patient: dict[str, str], annotation_root: Path, seed: int
) -> tuple[dict[str, str], dict[str, object]]:
    positive_studies = {
        study
        for study in annotated_studies
        if any(_has_box(path) for path in (annotation_root / study).glob("*.json"))
    }
    patient_studies: dict[str, list[str]] = {}
    for study in annotated_studies:
        patient_studies.setdefault(study_to_patient[study], []).append(study)
    patients = sorted(patient_studies)
    labels = [int(any(s in positive_studies for s in patient_studies[p])) for p in patients]

    # 70/15/15, stratified by patient-level fracture status. A patient's studies
    # can never be separated across train, validation and test.
    train_patients, holdout_patients = train_test_split(
        patients, test_size=0.30, random_state=seed, stratify=labels
    )
    holdout_labels = [int(any(s in positive_studies for s in patient_studies[p])) for p in holdout_patients]
    val_patients, test_patients = train_test_split(
        holdout_patients, test_size=0.50, random_state=seed, stratify=holdout_labels
    )
    patient_split = {
        **{p: "train" for p in train_patients},
        **{p: "val" for p in val_patients},
        **{p: "test" for p in test_patients},
    }
    study_split = {study: patient_split[study_to_patient[study]] for study in annotated_studies}

    sets = {name: {p for p, value in patient_split.items() if value == name} for name in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    summary: dict[str, object] = {
        "seed": seed,
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "positive_definition": "patient has at least one non-empty boxes_xywh annotation",
        "counts": {},
    }
    for split in ("train", "val", "test"):
        split_patients = sets[split]
        split_studies = [s for s in annotated_studies if study_split[s] == split]
        summary["counts"][split] = {
            "patients": len(split_patients),
            "positive_patients": sum(
                any(s in positive_studies for s in patient_studies[p]) for p in split_patients
            ),
            "studies": len(split_studies),
            "positive_studies": sum(s in positive_studies for s in split_studies),
        }
    return study_split, summary


def _write_image(path: Path, image) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def build(args: argparse.Namespace) -> Path:
    data_root = args.data_root.resolve()
    dicom_root = data_root / "training"
    annotation_root = data_root / "annotations"
    metadata_path = data_root / "training_df.pkl"
    for path in (dicom_root, annotation_root, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)

    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {output}. Use --force to rebuild it.")
        # Guard against an accidentally broad deletion target.
        safe_root = (ROOT / "data_prepared").resolve()
        if output == safe_root or safe_root not in output.parents:
            raise ValueError(f"Refusing to replace unsafe output path: {output}")
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    study_to_patient = _study_patient_map(metadata_path)
    annotated_studies = sorted(
        path.name for path in annotation_root.iterdir()
        if path.is_dir() and any(path.glob("*.json"))
    )
    missing_patient = sorted(set(annotated_studies) - set(study_to_patient))
    if missing_patient:
        raise ValueError(f"No PatientID in metadata for studies: {missing_patient[:10]}")
    missing_dicom = sorted(s for s in annotated_studies if not (dicom_root / s).is_dir())
    if missing_dicom:
        raise ValueError(f"No DICOM directory for studies: {missing_dicom[:10]}")

    study_split, split_summary = _make_patient_split(
        annotated_studies, study_to_patient, annotation_root, args.seed
    )
    manifest: list[dict[str, object]] = []
    errors: list[str] = []
    for study_index, study in enumerate(annotated_studies, start=1):
        records = load_study(dicom_root / study)
        record_by_stem = {record.path.stem: i for i, record in enumerate(records)}
        annotation_paths = sorted((annotation_root / study).glob("*.json"))
        missing_slices = [path.stem for path in annotation_paths if path.stem not in record_by_stem]
        if missing_slices:
            errors.append(f"{study}: {len(missing_slices)} annotations have no matching DICOM")
            continue
        hu_images = [record.hu for record in records]
        positions = [record.physical_position for record in records]
        monochrome1 = [record.photometric_interpretation == "MONOCHROME1" for record in records]
        split = study_split[study]
        for annotation_path in annotation_paths:
            index = record_by_stem[annotation_path.stem]
            record = records[index]
            annotation = parse_annotation(annotation_path, image_shape=record.shape)
            image = make_hu_input(
                hu_images,
                index,
                level=args.window_level,
                width=args.window_width,
                mode="2.5d" if args.context else "single",
                boundary_mode="repeat",
                monochrome1=monochrome1,
                physical_positions=positions,
                context_distance_mm=args.context_distance_mm if args.context else None,
            )
            stem = f"{study}__{annotation_path.stem}"
            image_path = output / "images" / split / f"{stem}.png"
            label_path = output / "labels" / split / f"{stem}.txt"
            _write_image(image_path, image)
            label_lines = [
                "0 " + " ".join(f"{value:.8f}" for value in xywh_to_yolo(box, record.shape[1], record.shape[0]))
                for box in annotation.boxes
            ]
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            manifest.append(
                {
                    "split": split,
                    "patient_id": study_to_patient[study],
                    "study_id": study,
                    "sop_uid": annotation_path.stem,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "num_boxes": len(annotation.boxes),
                    "window_low_hu": args.window_level - args.window_width / 2,
                    "window_high_hu": args.window_level + args.window_width / 2,
                    "input_mode": "2.5d" if args.context else "single",
                }
            )
        print(f"[{study_index:03d}/{len(annotated_studies)}] {study}: {len(annotation_paths)} labeled slices -> {split}")

    if errors:
        raise RuntimeError("Dataset build aborted:\n" + "\n".join(errors))
    if not manifest:
        raise RuntimeError("No labeled slices were generated")
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    output.joinpath("dataset.yaml").write_text(
        f"path: {output}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: skull_fracture\n",
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        rows = [row for row in manifest if row["split"] == split]
        split_summary["counts"][split].update(
            {
                "labeled_slices": len(rows),
                "positive_slices": sum(int(row["num_boxes"]) > 0 for row in rows),
                "negative_slices": sum(int(row["num_boxes"]) == 0 for row in rows),
                "boxes": sum(int(row["num_boxes"]) for row in rows),
            }
        )
    split_summary.update(
        {
            "annotated_studies_only": True,
            "unannotated_studies_excluded_from_yolo_loss": len(
                [path for path in dicom_root.iterdir() if path.is_dir()]
            ) - len(annotated_studies),
            "window": {
                "level_hu": args.window_level,
                "width_hu": args.window_width,
                "low_hu": args.window_level - args.window_width / 2,
                "high_hu": args.window_level + args.window_width / 2,
            },
            "total": {
                "patients": len({str(row["patient_id"]) for row in manifest}),
                "studies": len({str(row["study_id"]) for row in manifest}),
                "labeled_slices": len(manifest),
                "positive_slices": sum(int(row["num_boxes"]) > 0 for row in manifest),
                "negative_slices": sum(int(row["num_boxes"]) == 0 for row in manifest),
                "boxes": sum(int(row["num_boxes"]) for row in manifest),
            },
        }
    )
    output.joinpath("split_summary.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(split_summary, indent=2, ensure_ascii=False))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "data_prepared" / "skull_hu800_ww1600")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-level", type=float, default=800.0)
    parser.add_argument("--window-width", type=float, default=1600.0)
    parser.add_argument("--context-distance-mm", type=float, default=5.0)
    parser.add_argument("--no-context", action="store_false", dest="context")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(context=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
