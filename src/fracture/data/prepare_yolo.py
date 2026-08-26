"""Versioned single/2.5D YOLO dataset generation."""
from __future__ import annotations
import argparse, csv, hashlib, json, random
from pathlib import Path
import cv2
from .annotations import resolve_annotation, xywh_to_yolo
from .dicom import load_study
from .windows import bone_window, make_input
from fracture.utils.config import load_config, require_path
from .metadata import slice_label_index


def generate(config: dict) -> Path:
    data, prep = config["data"], config["preprocessing"]; root = Path(data["output_root"]) / data["annotation_version"]
    if root.exists(): raise FileExistsError(f"Dataset version already exists and will not be overwritten: {root}")
    images_dir, labels_dir = root / "images", root / "labels"; images_dir.mkdir(parents=True); labels_dir.mkdir()
    dicom_root, original = require_path(config, "data", "dicom_root"), require_path(config, "data", "annotation_root"); corrected = data.get("corrected_annotation_root"); metadata_labels = slice_label_index(data)
    manifest = []
    for study in sorted(x for x in dicom_root.iterdir() if x.is_dir()):
        records = load_study(study); windows = [bone_window(x.hu, prep["window_level"], prep["window_width"], x.photometric_interpretation == "MONOCHROME1") for x in records]
        for i, record in enumerate(records):
            ann = resolve_annotation(original, corrected, study.name, record.sop_uid, image_shape=record.shape)
            metadata_label = metadata_labels.get((study.name, record.sop_uid))
            if ann is None and metadata_label is not False and not data.get("missing_json_means_negative", False): continue
            # Store the baseline as lossless single-channel PNG; Ultralytics
            # expands it to three identical channels when loading. True 2.5D
            # inputs remain three-channel.
            image = windows[i] if prep["input_mode"] == "single" else make_input(windows, i, prep["input_mode"])
            stem = f"{study.name}__{record.sop_uid}"; image_path, label_path = images_dir / f"{stem}.png", labels_dir / f"{stem}.txt"
            cv2.imwrite(str(image_path), image)
            lines = ["0 " + " ".join(f"{x:.8f}" for x in xywh_to_yolo(box, record.shape[1], record.shape[0])) for box in (ann.boxes if ann else ())]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            provenance = ann.provenance if ann else "metadata_verified_negative" if metadata_label is False else "configured_missing_as_negative"
            manifest.append({"image_path": str(image_path), "label_path": str(label_path), "series_id": study.name, "sop_uid": record.sop_uid, "slice_index": i, "is_annotated": ann is not None, "is_positive": bool(lines), "num_boxes": len(lines), "annotation_version": data["annotation_version"], "provenance": provenance})
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream: writer = csv.DictWriter(stream, fieldnames=list(manifest[0]) if manifest else ["series_id"]); writer.writeheader(); writer.writerows(manifest)
    digest = hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest(); (root / "manifest.sha256").write_text(digest + "  manifest.csv\n", encoding="utf-8")
    split_path = Path(config.get("split", {}).get("path", "splits/folds.json"))
    if not split_path.is_file(): raise FileNotFoundError(f"Fixed folds not found: {split_path}")
    folds = json.loads(split_path.read_text(encoding="utf-8"))
    for fold in folds["folds"]:
        fold_id = int(fold["fold"]); train_series, val_series = set(fold["train_series"]), set(fold["val_series"])
        train_rows = [row for row in manifest if row["series_id"] in train_series]
        negative_ratio = config.get("training", {}).get("negative_downsample_ratio")
        if negative_ratio is not None:
            positives = [row for row in train_rows if row["is_positive"]]; negatives = [row for row in train_rows if not row["is_positive"]]
            rng = random.Random(int(config.get("split", {}).get("seed", 42)) + fold_id)
            rng.shuffle(negatives); negatives = negatives[: min(len(negatives), int(len(positives) * float(negative_ratio)))]
            train_rows = positives + negatives; rng.shuffle(train_rows)
        train_images = [row["image_path"] for row in train_rows]
        val_images = [row["image_path"] for row in manifest if row["series_id"] in val_series]
        if not train_images or not val_images: raise ValueError(f"Fold {fold_id} has an empty train or validation image list")
        train_list, val_list = root / f"fold_{fold_id}_train.txt", root / f"fold_{fold_id}_val.txt"
        train_list.write_text("\n".join(str(Path(x).resolve()) for x in train_images) + "\n", encoding="utf-8")
        val_list.write_text("\n".join(str(Path(x).resolve()) for x in val_images) + "\n", encoding="utf-8")
        dataset_yaml = root / f"fold_{fold_id}.yaml"
        dataset_yaml.write_text(f"path: {root.resolve()}\ntrain: {train_list.name}\nval: {val_list.name}\nnames:\n  0: skull_fracture\n", encoding="utf-8")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args(); print(generate(load_config(args.config)))
if __name__ == "__main__": main()
