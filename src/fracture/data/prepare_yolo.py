"""Versioned single/2.5D YOLO dataset generation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import cv2

from fracture.utils.config import load_config, require_path

from .annotations import resolve_annotation, xywh_to_yolo
from .dicom import load_study
from .metadata import slice_label_index
from .windows import bone_window, make_input


def _load_hard_negative_keys(path: str | Path | None) -> set[tuple[str, str]]:
    if not path:
        return set()
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Hard-negative CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys: set[tuple[str, str]] = set()
    for row in rows:
        series_id = str(row.get("series_id", "")).strip()
        sop_uid = str(row.get("sop_uid", "")).strip()
        if series_id and sop_uid:
            keys.add((series_id, sop_uid))
    return keys


def generate(config: dict) -> Path:
    data, prep = config["data"], config["preprocessing"]
    root = Path(data["output_root"]) / data["annotation_version"]
    if root.exists():
        raise FileExistsError(f"Dataset version already exists and will not be overwritten: {root}")
    images_dir, labels_dir = root / "images", root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    dicom_root = require_path(config, "data", "dicom_root")
    original = require_path(config, "data", "annotation_root")
    corrected = data.get("corrected_annotation_root")
    metadata_labels = slice_label_index(data)
    hard_negatives = _load_hard_negative_keys(config.get("training", {}).get("hard_negative_csv"))
    boundary_mode = str(prep.get("boundary_mode", "repeat"))
    adjacent_factor = int(config.get("training", {}).get("adjacent_positive_oversample_factor", 1))
    hard_negative_factor = int(config.get("training", {}).get("hard_negative_oversample_factor", 3))

    manifest = []
    for study in sorted(x for x in dicom_root.iterdir() if x.is_dir()):
        records = load_study(study)
        windows = [
            bone_window(
                x.hu,
                prep["window_level"],
                prep["window_width"],
                x.photometric_interpretation == "MONOCHROME1",
            )
            for x in records
        ]
        for i, record in enumerate(records):
            ann = resolve_annotation(original, corrected, study.name, record.sop_uid, image_shape=record.shape)
            metadata_label = metadata_labels.get((study.name, record.sop_uid))
            if ann is None and metadata_label is not False and not data.get("missing_json_means_negative", False):
                continue
            image = (
                windows[i]
                if prep["input_mode"] == "single"
                else make_input(windows, i, prep["input_mode"], boundary_mode=boundary_mode)
            )
            stem = f"{study.name}__{record.sop_uid}"
            image_path, label_path = images_dir / f"{stem}.png", labels_dir / f"{stem}.txt"
            cv2.imwrite(str(image_path), image)
            lines = [
                "0 " + " ".join(f"{x:.8f}" for x in xywh_to_yolo(box, record.shape[1], record.shape[0]))
                for box in (ann.boxes if ann else ())
            ]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            provenance = (
                ann.provenance
                if ann
                else "metadata_verified_negative"
                if metadata_label is False
                else "configured_missing_as_negative"
            )
            is_hard_negative = (study.name, record.sop_uid) in hard_negatives and not lines
            manifest.append(
                {
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "series_id": study.name,
                    "sop_uid": record.sop_uid,
                    "slice_index": i,
                    "is_annotated": ann is not None,
                    "is_positive": bool(lines),
                    "num_boxes": len(lines),
                    "annotation_version": data["annotation_version"],
                    "provenance": provenance,
                    "is_hard_negative": is_hard_negative,
                }
            )

    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]) if manifest else ["series_id"])
        writer.writeheader()
        writer.writerows(manifest)
    digest = hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest()
    (root / "manifest.sha256").write_text(digest + "  manifest.csv\n", encoding="utf-8")

    split_path = Path(config.get("split", {}).get("path", "splits/folds.json"))
    if not split_path.is_file():
        raise FileNotFoundError(f"Fixed folds not found: {split_path}")
    folds = json.loads(split_path.read_text(encoding="utf-8"))
    fold_stats = []
    for fold in folds["folds"]:
        fold_id = int(fold["fold"])
        train_series, val_series = set(fold["train_series"]), set(fold["val_series"])
        train_rows = [row for row in manifest if row["series_id"] in train_series]
        unique_train_images = len(train_rows)
        # Mark fracture-adjacent negatives within the same study for mild oversampling.
        positives_by_series: dict[str, set[int]] = {}
        for row in train_rows:
            if row["is_positive"]:
                positives_by_series.setdefault(row["series_id"], set()).add(int(row["slice_index"]))

        negative_ratio = config.get("training", {}).get("negative_downsample_ratio")
        if negative_ratio is not None:
            positives = [row for row in train_rows if row["is_positive"]]
            negatives = [row for row in train_rows if not row["is_positive"]]
            hard = [row for row in negatives if row.get("is_hard_negative")]
            adjacent = [
                row
                for row in negatives
                if any(abs(int(row["slice_index"]) - idx) == 1 for idx in positives_by_series.get(row["series_id"], set()))
            ]
            rng = random.Random(int(config.get("split", {}).get("seed", 42)) + fold_id)
            rng.shuffle(negatives)
            negatives = negatives[: min(len(negatives), int(len(positives) * float(negative_ratio)))]
            positive_factor = int(config.get("training", {}).get("positive_oversample_factor", 1))
            if positive_factor < 1:
                raise ValueError("training.positive_oversample_factor must be >= 1")
            if adjacent_factor < 1 or hard_negative_factor < 1:
                raise ValueError("adjacent/hard-negative oversample factors must be >= 1")
            train_rows = (
                positives * positive_factor
                + adjacent * max(0, adjacent_factor - 1)
                + hard * max(0, hard_negative_factor - 1)
                + negatives
            )
            rng.shuffle(train_rows)

        train_images = [row["image_path"] for row in train_rows]
        val_rows = [row for row in manifest if row["series_id"] in val_series]
        val_images = [row["image_path"] for row in val_rows]
        if not train_images or not val_images:
            raise ValueError(f"Fold {fold_id} has an empty train or validation image list")
        train_list, val_list = root / f"fold_{fold_id}_train.txt", root / f"fold_{fold_id}_val.txt"
        train_list.write_text("\n".join(str(Path(x).resolve()) for x in train_images) + "\n", encoding="utf-8")
        val_list.write_text("\n".join(str(Path(x).resolve()) for x in val_images) + "\n", encoding="utf-8")
        dataset_yaml = root / f"fold_{fold_id}.yaml"
        dataset_yaml.write_text(
            f"path: {root.resolve()}\ntrain: {train_list.name}\nval: {val_list.name}\nnames:\n  0: skull_fracture\n",
            encoding="utf-8",
        )
        fold_stats.append(
            {
                "fold": fold_id,
                "unique_train_images": unique_train_images,
                "sampled_train_images": len(train_rows),
                "sampled_positive_images": sum(bool(row["is_positive"]) for row in train_rows),
                "sampled_negative_images": sum(not bool(row["is_positive"]) for row in train_rows),
                "sampled_hard_negatives": sum(bool(row.get("is_hard_negative")) for row in train_rows),
                "validation_images": len(val_rows),
                "validation_positive_images": sum(bool(row["is_positive"]) for row in val_rows),
            }
        )
    with (root / "fold_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fold_stats[0]))
        writer.writeheader()
        writer.writerows(fold_stats)
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(generate(load_config(args.config)))


if __name__ == "__main__":
    main()
