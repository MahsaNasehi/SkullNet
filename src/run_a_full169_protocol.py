"""Frozen protocol and cohort guards for the Run A full-development refit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd
import yaml

from oof_cohort import derive_explicit_validation_cohort


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data_prepared/skull_hu800_ww1600"
MANIFEST = DATASET_ROOT / "manifest.csv"
KFOLD_DIR = DATASET_ROOT / "kfold"
CONFIG_DIR = ROOT / "configs/run_a_full169"
TRAIN_LIST = CONFIG_DIR / "full169_train.txt"
SYNTAX_VAL_LIST = CONFIG_DIR / "syntax_only_val.txt"
DATA_YAML = CONFIG_DIR / "dataset.yaml"
PREPARATION_REPORT = CONFIG_DIR / "cohort_report.json"

ARCHITECTURE = ROOT / "configs/yolo26s-p2.yaml"
COCO_WEIGHTS = ROOT / "weights/yolo26s.pt"
OUTPUT_DIR = ROOT / "outputs/yolo26s_p2_hu800_ww1600_run_a_full169"
RUN_NAME = OUTPUT_DIR.name
SEED = 42

EXPECTED_FOLD_STUDIES = [33, 33, 32, 36, 35]
EXPECTED_STUDIES = 169
EXPECTED_PATIENTS = 155
EXPECTED_IMAGES = 4418
RUN_A_BEST_EPOCHS = [42, 99, 61, 20, 59]
MEAN_BEST_EPOCH = mean(RUN_A_BEST_EPOCHS)
MEDIAN_BEST_EPOCH = int(median(RUN_A_BEST_EPOCHS))
FIXED_EPOCHS = MEDIAN_BEST_EPOCH

ARCHITECTURE_SHA256 = "2d57edc829bd8add780c6b60b043add292fd35373ed0f8b6f316287a4ba339e1"
COCO_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"

# Values recovered identically from all five historical Run A args.yaml files.
RUN_A_TRAIN_KWARGS: dict[str, Any] = {
    "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "cos_lr": True,
    "weight_decay": 0.0005, "warmup_epochs": 3.0,
    "momentum": 0.937, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
    "mosaic": 0.30, "close_mosaic": 10, "degrees": 5.0,
    "translate": 0.05, "scale": 0.15,
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
    "bgr": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "copy_paste_mode": "flip", "auto_augment": "randaugment", "erasing": 0.4,
    "amp": True, "deterministic": True, "fraction": 1.0, "nbs": 64,
    "cache": False, "save": True, "save_period": 10, "plots": True,
    "rect": False, "multi_scale": 0.0, "pretrained": True,
    "single_cls": False, "dropout": 0.0,
    "box": 7.5, "cls": 0.5, "dfl": 1.5,
}

FORBIDDEN_MODES = {
    "aug_mild": False, "hnm": False, "balanced_sampling_b25": False,
    "extra_metadata_negatives": False, "spatial_filtering": False,
    "verifier": False, "hu_jitter": False, "imgsz_1024": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonical_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def audit_run_a_results() -> dict[str, Any]:
    observed = []
    keys = {
        "optimizer": "optimizer", "lr0": "lr0", "lrf": "lrf", "cos_lr": "cos_lr",
        "weight_decay": "weight_decay", "warmup_epochs": "warmup_epochs",
        "amp": "amp", "deterministic": "deterministic", "mosaic": "mosaic",
        "degrees": "degrees", "translate": "translate", "scale": "scale",
        "batch": "batch", "imgsz": "imgsz", "workers": "workers",
    }
    expected = {**{k: RUN_A_TRAIN_KWARGS[k] for k in keys if k in RUN_A_TRAIN_KWARGS},
                "batch": 16, "imgsz": 768, "workers": 0}
    for fold in range(5):
        args_path = ROOT / f"outputs/yolo26s_p2_hu800_ww1600_fold{fold}/args.yaml"
        results_path = ROOT / f"outputs/yolo26s_p2_hu800_ww1600_fold{fold}/results.csv"
        args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if args.get(key) != value:
                raise RuntimeError(f"Historical Run A Fold {fold} drift for {key}: {args.get(key)!r} != {value!r}")
        results = pd.read_csv(results_path)
        results.columns = results.columns.str.strip()
        best = int(results.loc[results["metrics/mAP50-95(B)"].idxmax(), "epoch"])
        observed.append(best)
    if observed != RUN_A_BEST_EPOCHS:
        raise RuntimeError(f"Run A best epochs changed: {observed}")
    return {
        "best_epochs": observed, "mean_best_epoch": MEAN_BEST_EPOCH,
        "median_best_epoch": MEDIAN_BEST_EPOCH, "recommended_fixed_epochs": FIXED_EPOCHS,
    }


def audit_cohort() -> tuple[dict[str, Any], list[str]]:
    manifest = pd.read_csv(
        MANIFEST,
        dtype={"split": str, "patient_id": str, "study_id": str, "image_path": str, "label_path": str},
    )
    val_lists = [KFOLD_DIR / f"fold_{fold}_val.txt" for fold in range(5)]
    cohort = derive_explicit_validation_cohort(manifest, val_lists)
    if cohort["per_fold_study_counts"] != EXPECTED_FOLD_STUDIES:
        raise RuntimeError("Unexpected per-Fold Study counts")

    ordered_paths = [str(Path(value).resolve()) for path in val_lists for value in _lines(path)]
    if len(ordered_paths) != len(set(ordered_paths)):
        raise RuntimeError("Duplicate image across explicit validation cohorts")

    work = manifest.copy()
    work["patient_id"] = work["patient_id"].astype(str)
    work["study_id"] = work["study_id"].astype(str)
    work["resolved_image_path"] = work["image_path"].map(lambda value: str(Path(value).resolve()))
    if work["resolved_image_path"].duplicated().any():
        raise RuntimeError("Manifest image paths are not one-to-one")
    by_path = work.set_index("resolved_image_path")
    missing = set(ordered_paths) - set(by_path.index)
    if missing:
        raise RuntimeError(f"Explicit development paths absent from manifest: {sorted(missing)[:3]}")
    selected = by_path.loc[ordered_paths]
    studies = set(selected["study_id"])
    patients = set(selected["patient_id"])
    if len(studies) != EXPECTED_STUDIES or len(patients) != EXPECTED_PATIENTS:
        raise RuntimeError(f"Full-development identity mismatch: studies={len(studies)}, patients={len(patients)}")
    if len(ordered_paths) != EXPECTED_IMAGES:
        raise RuntimeError(f"Full-development image count is {len(ordered_paths)}, expected {EXPECTED_IMAGES}")
    if (selected["split"] == "test").any():
        raise RuntimeError("Fixed-test row entered full-development training cohort")

    test_rows = work[work["split"] == "test"]
    study_overlap = studies & set(test_rows["study_id"])
    patient_overlap = patients & set(test_rows["patient_id"])
    if study_overlap or patient_overlap:
        raise RuntimeError("Fixed-test Study/patient overlap in full-development cohort")

    # This is the exact manifest-backed development image set, not a metadata expansion.
    manifest_dev = work[(work["split"] != "test") & work["study_id"].isin(studies)]
    if set(manifest_dev["resolved_image_path"]) != set(ordered_paths) or len(manifest_dev) != EXPECTED_IMAGES:
        raise RuntimeError("Explicit Fold union does not equal all existing Run A development images")

    run_a_train_paths = {
        str(Path(value).resolve())
        for fold in range(5)
        for value in _lines(KFOLD_DIR / f"fold_{fold}_train.txt")
    }
    if not set(ordered_paths).issubset(run_a_train_paths):
        raise RuntimeError("An existing Run A development training image is missing")
    if any(not Path(path).is_file() for path in ordered_paths):
        raise FileNotFoundError("A full-development training image is missing")
    label_paths = [Path(value) for value in selected["label_path"]]
    if any(not path.is_file() for path in label_paths):
        raise FileNotFoundError("A manifest-backed Run A label is missing")

    payload = {
        "cohort_protocol": "exact_union_of_manifest_mapped_explicit_validation_studies_v1",
        "source": "fold_0_val.txt..fold_4_val.txt mapped through manifest.csv; no filename inference",
        "studies": len(studies), "patients": len(patients), "training_images": len(ordered_paths),
        "per_fold_studies": cohort["per_fold_study_counts"],
        "per_fold_images": [item["explicit_validation_slices"] for item in cohort["folds"]],
        "fixed_test_study_overlap": 0, "fixed_test_patient_overlap": 0,
        "duplicate_studies": 0, "duplicate_images": 0,
        "all_existing_run_a_development_images_included": True,
        "metadata_only_or_tier2_images_added": 0,
        "training_cohort_sha256": _canonical_hash(ordered_paths),
        "study_ids_sha256": _canonical_hash(sorted(studies)),
        "patient_ids_sha256": _canonical_hash(sorted(patients)),
        "manifest_sha256": sha256(MANIFEST),
        "source_val_list_sha256": {str(fold): sha256(val_lists[fold]) for fold in range(5)},
    }
    return payload, ordered_paths


def _write_new_or_identical(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite different prepared artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def prepare() -> dict[str, Any]:
    cohort, ordered_paths = audit_cohort()
    epochs = audit_run_a_results()
    if sha256(ARCHITECTURE) != ARCHITECTURE_SHA256 or sha256(COCO_WEIGHTS) != COCO_SHA256:
        raise RuntimeError("Architecture or COCO source differs from frozen Run A protocol")
    train_content = "\n".join(ordered_paths) + "\n"
    sentinel_content = ordered_paths[0] + "\n"
    yaml_payload = {
        "path": str(DATASET_ROOT.resolve()),
        "train": str(TRAIN_LIST.resolve()),
        # Ultralytics requires this key and runs a final validation even with val=False.
        # It is one in-cohort training image and is never used to select the deployment checkpoint.
        "val": str(SYNTAX_VAL_LIST.resolve()),
        "names": {0: "skull_fracture"},
    }
    _write_new_or_identical(TRAIN_LIST, train_content)
    _write_new_or_identical(SYNTAX_VAL_LIST, sentinel_content)
    _write_new_or_identical(DATA_YAML, yaml.safe_dump(yaml_payload, sort_keys=False))
    report = {
        "status": "prepared_training_not_started", **cohort, **epochs,
        "train_list": str(TRAIN_LIST.resolve()), "train_list_sha256": sha256(TRAIN_LIST),
        "syntax_only_val_list": str(SYNTAX_VAL_LIST.resolve()),
        "syntax_only_val_list_sha256": sha256(SYNTAX_VAL_LIST),
        "data_yaml": str(DATA_YAML.resolve()), "data_yaml_sha256": sha256(DATA_YAML),
        "architecture_sha256": ARCHITECTURE_SHA256, "source_coco_sha256": COCO_SHA256,
        "seed": SEED,
        "validation_policy": (
            "model.train(val=False); YAML val is one in-cohort syntax-only image. "
            "Ultralytics may run it only at finalization; no metric selects a checkpoint. "
            "weights/last.pt at fixed epoch 59 is copied to weights/final_deployment.pt."
        ),
        "forbidden_modes": FORBIDDEN_MODES,
    }
    _write_new_or_identical(PREPARATION_REPORT, json.dumps(report, indent=2) + "\n")
    return report


def validate_prepared() -> dict[str, Any]:
    expected = prepare()
    saved = json.loads(PREPARATION_REPORT.read_text(encoding="utf-8"))
    if saved != expected:
        raise RuntimeError("Prepared full169 report drift")
    if sha256(TRAIN_LIST) != saved["train_list_sha256"] or len(_lines(TRAIN_LIST)) != EXPECTED_IMAGES:
        raise RuntimeError("Prepared full169 training list drift")
    return saved

