"""Frozen protocol for MOSAIC08_SCALE05_BALANCED_FOLD4."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from fitness_balanced_768_protocol import FITNESS_WEIGHTS, balanced_fitness
from train_yolo26s_p2 import ROOT, sha256, split_audit


EXPERIMENT = "MOSAIC08_SCALE05_BALANCED_FOLD4"
FOLD = 4
SEED = 46
MAX_EPOCHS = 40
PATIENCE = 30
SOURCE_CHECKPOINT = ROOT / "outputs/yolo26s_p2_hu800_ww1600_fold4/weights/best.pt"
SOURCE_SHA256 = "30e953e6f60b85b8e940559eb24c69b8817d46d27fcc3479b1fab0807c75ac16"
ARCHITECTURE = ROOT / "configs/yolo26s-p2.yaml"
ARCHITECTURE_SHA256 = "2d57edc829bd8add780c6b60b043add292fd35373ed0f8b6f316287a4ba339e1"
TRAIN_LIST = ROOT / "data_prepared/skull_hu800_ww1600/kfold/fold_4_train.txt"
VAL_LIST = ROOT / "data_prepared/skull_hu800_ww1600/kfold/fold_4_val.txt"
TRAIN_SHA256 = "b3c8b31c3cccdff0c7f7efbb1e4dcfded1016fed89fa942fc5576cad60090d8a"
VAL_SHA256 = "1cf9cdcacd74bd56ae9370bb5f1b76f7c5d38960d232e4d80627a33b7f8a1db6"
DATA_YAML = ROOT / "configs/mosaic08_scale05_balanced_fold4.yaml"
OUTPUT_DIR = ROOT / "outputs/yolo26s_p2_hu800_ww1600_mosaic08_scale05_balanced_fold4"
LOG_PATH = ROOT / "outputs/mosaic08_scale05_balanced_fold4.log"

TRAIN_KWARGS = {
    "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "cos_lr": True,
    "weight_decay": 0.0005, "warmup_epochs": 3.0,
    "momentum": 0.937, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
    "mosaic": 0.80, "close_mosaic": 10, "degrees": 5.0,
    "translate": 0.05, "scale": 0.50,
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
    "resume_optimizer": False, "aug_mild": False, "hnm": False,
    "balanced_sampling_b25": False, "imgsz_1024": False, "hu_jitter": False,
    "verifier": False, "spatial_filtering": False, "sahi_training": False,
    "tier2_or_metadata_negatives": False,
}


def prepare_config() -> None:
    validate_sources()
    payload = {
        "path": str((ROOT / "data_prepared/skull_hu800_ww1600").resolve()),
        "train": str(TRAIN_LIST.resolve()), "val": str(VAL_LIST.resolve()),
        "names": {0: "skull_fracture"},
    }
    content = yaml.safe_dump(payload, sort_keys=False)
    if DATA_YAML.exists() and DATA_YAML.read_text(encoding="utf-8") != content:
        raise FileExistsError("Refusing to overwrite changed experiment YAML: %s" % DATA_YAML)
    if not DATA_YAML.exists():
        DATA_YAML.write_text(content, encoding="utf-8")
    audit = split_audit(DATA_YAML)
    if audit["train"]["sha256"] != TRAIN_SHA256 or audit["val"]["sha256"] != VAL_SHA256:
        raise RuntimeError("Fold 4 membership drift")
    if "test:" in DATA_YAML.read_text(encoding="utf-8"):
        raise RuntimeError("Experiment YAML must not expose fixed test")


def validate_sources() -> None:
    if sha256(SOURCE_CHECKPOINT) != SOURCE_SHA256:
        raise RuntimeError("Historical Run A Fold 4 checkpoint changed")
    if sha256(ARCHITECTURE) != ARCHITECTURE_SHA256:
        raise RuntimeError("YOLO26s-P2 architecture changed")
    if sha256(TRAIN_LIST) != TRAIN_SHA256 or sha256(VAL_LIST) != VAL_SHA256:
        raise RuntimeError("Historical Run A Fold 4 split changed")
    if abs(sum(FITNESS_WEIGHTS.values()) - 1.0) > 1e-12:
        raise RuntimeError("Balanced fitness weights must sum to 1")
    protected = {
        ROOT / "outputs/yolo26s_p2_hu800_ww1600_fold4",
        ROOT / "outputs/yolo26s_p2_hu800_ww1600_run_a_full169",
    }
    if OUTPUT_DIR in protected or SOURCE_CHECKPOINT.is_relative_to(OUTPUT_DIR):
        raise RuntimeError("Experiment output is not isolated")


def expected_protocol(initialization_copy: Path) -> dict:
    prepare_config()
    return {
        "status": "frozen_before_training", "experiment_name": EXPERIMENT,
        "exploratory_post_hoc_single_fold": True,
        "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "historical_checkpoint_policy": "Historical Run A Fold 4 checkpoint preserved and untouched.",
        "initialization_copy": str(initialization_copy.resolve()),
        "initialization_copy_sha256": sha256(initialization_copy),
        "initialization_mode": "weights only; resume=False; fresh optimizer and fresh schedule",
        "architecture": str(ARCHITECTURE.resolve()), "architecture_sha256": ARCHITECTURE_SHA256,
        "data": str(DATA_YAML.resolve()), "train_split_sha256": TRAIN_SHA256,
        "validation_split_sha256": VAL_SHA256,
        "seed": SEED, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "imgsz": 768, "batch": 16, "automatic_batch_reduction": False,
        "input": "2.5D physical +/-5 mm; HU WL=800 WW=1600",
        "training": TRAIN_KWARGS,
        "augmentation_delta_from_run_a": {"mosaic": [0.30, 0.80], "scale": [0.15, 0.50]},
        "fitness_weights": FITNESS_WEIGHTS,
        "balanced_fitness_formula": "0.10*P + 0.30*R + 0.10*mAP50 + 0.50*mAP50_95",
        "balanced_fitness_role": "secondary checkpoint proxy only",
        "default_fitness_role": "unchanged best.pt and early stopping",
        "forbidden": FORBIDDEN,
        "output": str(OUTPUT_DIR.resolve()),
    }


def validate_saved_protocol() -> dict:
    path = OUTPUT_DIR / "run_protocol.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = expected_protocol(OUTPUT_DIR / "initialization_source.pt")
    for key, value in expected.items():
        if saved.get(key) != value:
            raise RuntimeError("Saved experiment protocol drift in %s" % key)
    return saved

