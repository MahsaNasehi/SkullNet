"""Frozen protocol for the project-local FITNESS_BALANCED_768 experiment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from train_yolo26s_p2 import ROOT, sha256, split_audit


EXPERIMENT = "FITNESS_BALANCED_768"
RUN_PREFIX = "yolo26s_p2_hu800_ww1600_fitness_balanced_768_fold"
LOG_DIR = ROOT / "outputs/fitness_balanced_768_logs"
DATASET_ROOT = ROOT / "data_prepared/skull_hu800_ww1600"
CONFIG_DIR = ROOT / "configs/fitness_balanced_768_data"
ARCHITECTURE = ROOT / "configs/yolo26s-p2.yaml"
COCO_WEIGHTS = ROOT / "weights/yolo26s.pt"
ARCHITECTURE_SHA256 = "2d57edc829bd8add780c6b60b043add292fd35373ed0f8b6f316287a4ba339e1"
COCO_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"

FITNESS_WEIGHTS = {
    "metrics/precision(B)": 0.10,
    "metrics/recall(B)": 0.30,
    "metrics/mAP50(B)": 0.10,
    "metrics/mAP50-95(B)": 0.50,
}

EXPECTED_SPLIT_HASHES = {
    0: {"train": "e8a5ec7937ac1983c90e6f148219af6460c877cb009c72c1f32f51908e384e25",
        "val": "25cb6e49bc31e86028cd36c115f87820b9945aa310ad0603b76a9d3a88cf7f53"},
    1: {"train": "dae3d68519c00158d2262fb10ecce955f580ec47dfcdda1cb4b35795ab720694",
        "val": "040d9d152bece18c68358fc5d357e652769da7f7145bc45959f34ccb17341bc0"},
    2: {"train": "20b02f79f7afa607b1a174ac7bc3a92f8115d6b3dba54ca55c4c6d19344a219b",
        "val": "081f3511126c91732cc4b703a41cf30e089cafb3c97ec1f2c30e6dc82ac27ab5"},
    3: {"train": "5cb98ef0092d26be2becfcf425b5c51d4b1dfc0276fea9ab963a676bb798e1b6",
        "val": "d9da086967f8d72250671a706652499ce8b9b8b978c701d817e918084b2f31a6"},
    4: {"train": "b3c8b31c3cccdff0c7f7efbb1e4dcfded1016fed89fa942fc5576cad60090d8a",
        "val": "1cf9cdcacd74bd56ae9370bb5f1b76f7c5d38960d232e4d80627a33b7f8a1db6"},
}

# Recovered identically from the five historical Run A args.yaml files.
RUN_A_TRAINING: dict[str, Any] = {
    "epochs": 150, "patience": 30, "batch": 16, "imgsz": 768, "workers": 0,
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
    "single_cls": False, "dropout": 0.0, "box": 7.5, "cls": 0.5, "dfl": 1.5,
}

FORBIDDEN = {
    "balanced_sampling_b25": False, "hnm": False, "aug_mild": False,
    "imgsz_1024": False, "hu_jitter": False, "additional_metadata_negatives": False,
    "spatial_filtering": False, "verifier": False,
}


def balanced_fitness(metrics: dict[str, float]) -> float:
    missing = set(FITNESS_WEIGHTS) - set(metrics)
    if missing:
        raise KeyError("Balanced fitness missing metrics: %s" % sorted(missing))
    return float(sum(FITNESS_WEIGHTS[key] * float(metrics[key]) for key in FITNESS_WEIGHTS))


def run_dir(fold: int) -> Path:
    if fold not in range(5):
        raise ValueError("fold must be 0..4")
    return ROOT / "outputs" / (RUN_PREFIX + str(fold))


def data_yaml(fold: int) -> Path:
    return CONFIG_DIR / ("fold_%d.yaml" % fold)


def prepare_configs() -> None:
    if abs(sum(FITNESS_WEIGHTS.values()) - 1.0) > 1e-12:
        raise RuntimeError("Balanced fitness weights must sum to 1.0")
    if sha256(ARCHITECTURE) != ARCHITECTURE_SHA256 or sha256(COCO_WEIGHTS) != COCO_SHA256:
        raise RuntimeError("Architecture or COCO checkpoint drift")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        original = DATASET_ROOT / ("kfold/fold_%d.yaml" % fold)
        parsed = yaml.safe_load(original.read_text(encoding="utf-8"))
        clean = {
            "path": parsed["path"], "train": parsed["train"], "val": parsed["val"],
            "names": parsed["names"],
        }
        content = yaml.safe_dump(clean, sort_keys=False)
        target = data_yaml(fold)
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise FileExistsError("Refusing to overwrite changed FITNESS_BALANCED config: %s" % target)
        if not target.exists():
            target.write_text(content, encoding="utf-8")
    validate_configs()


def validate_configs() -> dict[str, Any]:
    records = {}
    protected_names = {
        "yolo26s_p2_hu800_ww1600_fold", "yolo26s_p2_hu800_ww1600_run_b_1024_fold",
        "yolo26s_p2_hu800_ww1600_aug_mild_768_fold",
        "yolo26s_p2_hu800_ww1600_run_a_full169",
    }
    if any(RUN_PREFIX.startswith(name) or RUN_PREFIX == name for name in protected_names):
        raise RuntimeError("FITNESS_BALANCED output prefix overlaps a protected experiment")
    for fold in range(5):
        clean = data_yaml(fold)
        original = DATASET_ROOT / ("kfold/fold_%d.yaml" % fold)
        if not clean.is_file():
            raise FileNotFoundError(clean)
        if "test:" in clean.read_text(encoding="utf-8"):
            raise RuntimeError("FITNESS_BALANCED YAML must not expose fixed test")
        clean_audit, original_audit = split_audit(clean), split_audit(original)
        if clean_audit != original_audit:
            raise RuntimeError("FITNESS_BALANCED membership differs from Run A Fold %d" % fold)
        hashes = {name: clean_audit[name]["sha256"] for name in ("train", "val")}
        if hashes != EXPECTED_SPLIT_HASHES[fold]:
            raise RuntimeError("Run A split hash drift for Fold %d" % fold)
        args_path = ROOT / ("outputs/yolo26s_p2_hu800_ww1600_fold%d/args.yaml" % fold)
        historical = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        for key, expected in RUN_A_TRAINING.items():
            if historical.get(key) != expected:
                raise RuntimeError("Historical Run A drift Fold %d key %s" % (fold, key))
        records[str(fold)] = clean_audit
    return records


def training_kwargs() -> dict[str, Any]:
    values = dict(RUN_A_TRAINING)
    for key in ("epochs", "patience", "batch", "imgsz", "workers"):
        values.pop(key)
    return values


def expected_protocol(fold: int) -> dict[str, Any]:
    splits = validate_configs()[str(fold)]
    return {
        "status": "frozen_before_training", "experiment": EXPERIMENT, "fold": fold,
        "seed": 42 + fold, "architecture": str(ARCHITECTURE.resolve()),
        "architecture_sha256": ARCHITECTURE_SHA256,
        "coco_source": str(COCO_WEIGHTS.resolve()), "source_coco_sha256": COCO_SHA256,
        "data": str(data_yaml(fold).resolve()), "splits": splits,
        "input": "2.5D physical +/-5 mm; HU WL=800 WW=1600",
        "run_a_training": RUN_A_TRAINING, "fitness_weights": FITNESS_WEIGHTS,
        "balanced_fitness_role": "additional checkpoint selector only",
        "default_fitness_role": "unchanged Ultralytics best.pt and early stopping",
        "ultralytics_default_detection_fitness_weights_P_R_mAP50_mAP50_95": [0.0, 0.0, 0.0, 1.0],
        "forbidden": FORBIDDEN,
        "output": str(run_dir(fold).resolve()),
    }


def validate_saved_protocol(fold: int) -> dict[str, Any]:
    path = run_dir(fold) / "run_protocol.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = expected_protocol(fold)
    for key, value in expected.items():
        if saved.get(key) != value:
            raise RuntimeError("Saved FITNESS_BALANCED protocol drift in %s" % key)
    return saved
