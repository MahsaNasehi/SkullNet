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
from .windows import make_hu_input


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


def _write_image(path: Path, image) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV failed to write rendered image: {path}")


def _slice_status(
    *,
    has_boxes: bool,
    has_annotation: bool,
    metadata_label: bool | None,
    missing_json_means_negative: bool,
) -> str:
    if has_boxes:
        return "positive"
    if has_annotation or metadata_label is False or missing_json_means_negative:
        return "negative"
    return "unknown"


def _jittered_window(prep: dict, seed: int, key: str, variant: int) -> tuple[float, float]:
    jitter = prep.get("training_window_jitter") or {}
    digest = hashlib.sha256(f"{seed}:{key}:{variant}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    level = float(prep["window_level"]) + rng.uniform(
        -float(jitter.get("level_delta", 0.0)), float(jitter.get("level_delta", 0.0))
    )
    fraction = float(jitter.get("width_fraction", 0.0))
    width = float(prep["window_width"]) * rng.uniform(1.0 - fraction, 1.0 + fraction)
    return level, max(float(jitter.get("minimum_width", 1.0)), width)


def _choose_rendered_variants(rows: list[dict]) -> list[dict]:
    """Use deterministic window variants for oversampled copies only.

    The first occurrence retains the fixed validation-compatible window.  A
    duplicated positive or hard-negative row cycles through its pre-rendered
    train-only window variants.  Dataset size and class balance therefore stay
    unchanged while exact oversampling copies no longer have identical pixels.
    """
    occurrences: dict[tuple[str, str], int] = {}
    selected: list[dict] = []
    for source in rows:
        row = dict(source)
        key = (str(row["series_id"]), str(row["sop_uid"]))
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        variants = [x for x in str(row.get("window_variant_paths", "")).split(";") if x]
        if occurrence > 0 and variants:
            row["image_path"] = variants[(occurrence - 1) % len(variants)]
        selected.append(row)
    return selected


def _sample_training_rows(rows: list[dict], config: dict, seed: int) -> list[dict]:
    """Balance one training partition without altering its unique records."""
    train_rows = list(rows)
    negative_ratio = config.get("training", {}).get("negative_downsample_ratio")
    if negative_ratio is None:
        return _choose_rendered_variants(train_rows)
    positives = [row for row in train_rows if row["is_positive"]]
    negatives = [row for row in train_rows if not row["is_positive"]]
    if not positives:
        raise ValueError("Training partition contains no positive fracture slices")
    positives_by_series: dict[str, set[int]] = {}
    for row in positives:
        positives_by_series.setdefault(str(row["series_id"]), set()).add(int(row["slice_index"]))
    hard = [row for row in negatives if row.get("is_hard_negative")]
    adjacent = [
        row
        for row in negatives
        if any(abs(int(row["slice_index"]) - index) == 1 for index in positives_by_series.get(str(row["series_id"]), set()))
    ]
    rng = random.Random(seed)
    priority_keys = {
        (str(row["series_id"]), str(row["sop_uid"]))
        for row in hard + adjacent
    }
    priority = [
        row for row in negatives
        if (str(row["series_id"]), str(row["sop_uid"])) in priority_keys
    ]
    random_pool = [
        row for row in negatives
        if (str(row["series_id"]), str(row["sop_uid"])) not in priority_keys
    ]
    rng.shuffle(random_pool)
    target_negatives = min(len(negatives), int(len(positives) * float(negative_ratio)))
    base_negatives = priority + random_pool[: max(0, target_negatives - len(priority))]
    positive_factor = int(config.get("training", {}).get("positive_oversample_factor", 1))
    adjacent_factor = int(config.get("training", {}).get("adjacent_positive_oversample_factor", 1))
    hard_negative_factor = int(config.get("training", {}).get("hard_negative_oversample_factor", 1))
    if min(positive_factor, adjacent_factor, hard_negative_factor) < 1:
        raise ValueError("All training oversample factors must be >= 1")
    hard_keys = {(str(row["series_id"]), str(row["sop_uid"])) for row in hard}
    adjacent_keys = {(str(row["series_id"]), str(row["sop_uid"])) for row in adjacent}
    sampled_negatives: list[dict] = []
    for row in base_negatives:
        key = (str(row["series_id"]), str(row["sop_uid"]))
        factor = max(
            hard_negative_factor if key in hard_keys else 1,
            adjacent_factor if key in adjacent_keys else 1,
        )
        sampled_negatives.extend([row] * factor)
    sampled = positives * positive_factor + sampled_negatives
    rng.shuffle(sampled)
    return _choose_rendered_variants(sampled)


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
    context_distance = prep.get("context_distance_mm")
    context_distance = float(context_distance) if context_distance is not None else None
    jitter = prep.get("training_window_jitter") or {}
    jitter_copies = int(jitter.get("copies", 0)) if bool(jitter.get("enabled", False)) else 0
    if jitter_copies < 0:
        raise ValueError("preprocessing.training_window_jitter.copies must be >= 0")
    variant_dirs: list[tuple[Path, Path]] = []
    for variant in range(jitter_copies):
        variant_images = root / f"window_variant_{variant + 1}" / "images"
        variant_labels = root / f"window_variant_{variant + 1}" / "labels"
        variant_images.mkdir(parents=True)
        variant_labels.mkdir()
        variant_dirs.append((variant_images, variant_labels))

    manifest = []
    for study in sorted(x for x in dicom_root.iterdir() if x.is_dir()):
        records = load_study(study)
        hu_images = [x.hu for x in records]
        monochrome1 = [x.photometric_interpretation == "MONOCHROME1" for x in records]
        positions = [x.physical_position for x in records]
        # Resolve the whole study once so train-only window variants are created
        # only for rows that can actually be repeated (positives, immediate
        # neighbours, and registered hard negatives).  Rendering variants for
        # every negative slice needlessly triples /dev/shm usage.
        annotations = [
            resolve_annotation(
                original,
                corrected,
                study.name,
                record.sop_uid,
                image_shape=record.shape,
            )
            for record in records
        ]
        positive_indices = {
            index for index, annotation in enumerate(annotations)
            if annotation is not None and bool(annotation.boxes)
        }
        adjacent_indices = {
            index + offset
            for index in positive_indices
            for offset in (-1, 1)
            if 0 <= index + offset < len(records)
        }
        for i, (record, ann) in enumerate(zip(records, annotations, strict=True)):
            metadata_label = metadata_labels.get((study.name, record.sop_uid))
            image = make_hu_input(
                hu_images,
                i,
                level=float(prep["window_level"]),
                width=float(prep["window_width"]),
                mode=str(prep["input_mode"]),
                boundary_mode=boundary_mode,
                monochrome1=monochrome1,
                physical_positions=positions,
                context_distance_mm=context_distance,
            )
            stem = f"{study.name}__{record.sop_uid}"
            image_path, label_path = images_dir / f"{stem}.png", labels_dir / f"{stem}.txt"
            _write_image(image_path, image)
            lines = [
                "0 " + " ".join(f"{x:.8f}" for x in xywh_to_yolo(box, record.shape[1], record.shape[0]))
                for box in (ann.boxes if ann else ())
            ]
            label_text = "\n".join(lines) + ("\n" if lines else "")
            label_path.write_text(label_text, encoding="utf-8")
            slice_status = _slice_status(
                has_boxes=bool(lines),
                has_annotation=ann is not None,
                metadata_label=metadata_label,
                missing_json_means_negative=bool(data.get("missing_json_means_negative", False)),
            )
            variant_paths: list[str] = []
            is_hard_negative = (study.name, record.sop_uid) in hard_negatives and slice_status == "negative"
            needs_variants = bool(lines) or i in adjacent_indices or is_hard_negative
            for variant, (variant_images, variant_labels) in enumerate(
                variant_dirs if needs_variants else (), start=1
            ):
                level, width = _jittered_window(
                    prep,
                    int(config.get("split", {}).get("seed", 42)),
                    stem,
                    variant,
                )
                variant_image = make_hu_input(
                    hu_images,
                    i,
                    level=level,
                    width=width,
                    mode=str(prep["input_mode"]),
                    boundary_mode=boundary_mode,
                    monochrome1=monochrome1,
                    physical_positions=positions,
                    context_distance_mm=context_distance,
                )
                variant_image_path = variant_images / f"{stem}.png"
                _write_image(variant_image_path, variant_image)
                (variant_labels / f"{stem}.txt").write_text(label_text, encoding="utf-8")
                variant_paths.append(str(variant_image_path))
            provenance = (
                ann.provenance
                if ann
                else "metadata_verified_negative"
                if metadata_label is False
                else "configured_missing_as_negative"
                if data.get("missing_json_means_negative", False)
                else "unknown_unlabeled"
            )
            manifest.append(
                {
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "series_id": study.name,
                    "sop_uid": record.sop_uid,
                    "slice_index": i,
                    "is_annotated": ann is not None,
                    "is_positive": bool(lines),
                    "slice_status": slice_status,
                    "detection_eligible": slice_status != "unknown",
                    "num_boxes": len(lines),
                    "annotation_version": data["annotation_version"],
                    "provenance": provenance,
                    "is_hard_negative": is_hard_negative,
                    "window_variant_paths": ";".join(variant_paths),
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
        train_rows = [
            row for row in manifest
            if row["series_id"] in train_series and row["detection_eligible"]
        ]
        unique_train_images = len(train_rows)
        train_rows = _sample_training_rows(
            train_rows,
            config,
            int(config.get("split", {}).get("seed", 42)) + fold_id,
        )

        train_images = [row["image_path"] for row in train_rows]
        val_rows = [
            row for row in manifest
            if row["series_id"] in val_series and row["detection_eligible"]
        ]
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

    # Final-fit data contains every study. Its validation list is deliberately
    # identical and must be used only for a fixed-epoch post-selection refit;
    # the resulting in-sample metrics are not model-selection evidence.
    detection_manifest = [row for row in manifest if row["detection_eligible"]]
    all_rows = _sample_training_rows(
        detection_manifest,
        config,
        int(config.get("split", {}).get("seed", 42)) + 10_000,
    )
    all_train = root / "all_train.txt"
    all_validation = root / "all_validation.txt"
    all_train.write_text("\n".join(str(Path(row["image_path"]).resolve()) for row in all_rows) + "\n", encoding="utf-8")
    all_validation.write_text("\n".join(str(Path(row["image_path"]).resolve()) for row in detection_manifest) + "\n", encoding="utf-8")
    (root / "all.yaml").write_text(
        f"path: {root.resolve()}\ntrain: {all_train.name}\nval: {all_validation.name}\nnames:\n  0: skull_fracture\n",
        encoding="utf-8",
    )
    (root / "all_stats.json").write_text(
        json.dumps(
            {
                "protocol": "fixed-epoch final refit only; validation metrics are in-sample",
                "manifest_images_for_study_model": len(manifest),
                "detector_eligible_images": len(detection_manifest),
                "unknown_images_excluded_from_detector": len(manifest) - len(detection_manifest),
                "sampled_train_images": len(all_rows),
                "sampled_positive_images": sum(bool(row["is_positive"]) for row in all_rows),
                "sampled_negative_images": sum(not bool(row["is_positive"]) for row in all_rows),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(generate(load_config(args.config)))


if __name__ == "__main__":
    main()
