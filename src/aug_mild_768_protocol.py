"""Frozen protocol and guards for the AUG_MILD_768 five-fold ablation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from train_yolo26s_p2 import ROOT, sha256, split_audit


EXPERIMENT = "AUG_MILD_768"
RUN_PREFIX = "yolo26s_p2_hu800_ww1600_aug_mild_768_fold"
LOG_DIR = ROOT / "outputs/aug_mild_768_logs"
DATA_CONFIG_DIR = ROOT / "configs/aug_mild_768_data"
GLOBAL_PROTOCOL = ROOT / "outputs/aug_mild_768_protocol.json"
ALL_COMPLETED = ROOT / "outputs/aug_mild_768_all_completed.json"

ARCHITECTURE = ROOT / "configs/yolo26s-p2.yaml"
COCO_WEIGHTS = ROOT / "weights/yolo26s.pt"
ARCHITECTURE_SHA256 = "2d57edc829bd8add780c6b60b043add292fd35373ed0f8b6f316287a4ba339e1"
COCO_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"

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

BASELINE_AUGMENTATION: Dict[str, Any] = {
    "mosaic": 0.30, "degrees": 5.0, "translate": 0.05, "scale": 0.15,
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
    "bgr": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "copy_paste_mode": "flip", "auto_augment": "randaugment", "erasing": 0.4,
    "close_mosaic": 10,
}
INTENDED_CHANGES: Dict[str, Any] = {
    "mosaic": 0.0, "degrees": 3.0, "translate": 0.03, "scale": 0.08,
}
AUG_MILD_AUGMENTATION = {**BASELINE_AUGMENTATION, **INTENDED_CHANGES}

BASELINE_OPTIMIZATION: Dict[str, Any] = {
    "epochs": 150, "patience": 30, "batch": 16, "imgsz": 768, "workers": 0,
    "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "momentum": 0.937,
    "adam_beta2": 0.999, "weight_decay": 0.0005, "warmup_epochs": 3.0,
    "warmup_momentum": 0.8, "warmup_bias_lr": 0.1, "cos_lr": True,
    "amp": True, "deterministic": True, "fraction": 1.0, "nbs": 64,
    "cache": False, "save": True, "save_period": 10, "plots": True,
    "rect": False, "multi_scale": 0.0, "pretrained": True, "cls_remap": True,
    "single_cls": False, "freeze": None, "dropout": 0.0,
    "box": 7.5, "cls": 0.5, "dfl": 1.5, "cls_pw": 0.0,
    "overlap_mask": True, "mask_ratio": 4,
}


def augmentation_diff() -> Dict[str, Dict[str, Any]]:
    keys = set(BASELINE_AUGMENTATION) | set(AUG_MILD_AUGMENTATION)
    return {
        key: {"run_a": BASELINE_AUGMENTATION.get(key), "aug_mild_768": AUG_MILD_AUGMENTATION.get(key)}
        for key in sorted(keys) if BASELINE_AUGMENTATION.get(key) != AUG_MILD_AUGMENTATION.get(key)
    }


def train_kwargs() -> Dict[str, Any]:
    values = dict(BASELINE_OPTIMIZATION)
    values.pop("epochs")
    values.pop("patience")
    values.pop("batch")
    values.pop("imgsz")
    values.pop("workers")
    values.pop("adam_beta2")  # Ultralytics AdamW fixes beta2=0.999 internally.
    values.update(AUG_MILD_AUGMENTATION)
    return values


def validate_static_protocol() -> None:
    if set(augmentation_diff()) != set(INTENDED_CHANGES):
        raise RuntimeError("AUG_MILD_768 must differ from Run A in exactly four augmentation keys")
    if sha256(ARCHITECTURE) != ARCHITECTURE_SHA256:
        raise RuntimeError("Architecture changed since protocol recovery")
    if sha256(COCO_WEIGHTS) != COCO_SHA256:
        raise RuntimeError("COCO source checkpoint changed since protocol recovery")
    if "run_b" in RUN_PREFIX or RUN_PREFIX == "yolo26s_p2_hu800_ww1600_fold":
        raise RuntimeError("Experiment path could overwrite historical Run A/B")


def validate_data_configs() -> Dict[str, Any]:
    records = {}
    for fold in range(5):
        clean = DATA_CONFIG_DIR / ("fold_%d.yaml" % fold)
        original = ROOT / ("data_prepared/skull_hu800_ww1600/kfold/fold_%d.yaml" % fold)
        if not clean.is_file() or not original.is_file():
            raise FileNotFoundError(clean if not clean.is_file() else original)
        parsed = yaml.safe_load(clean.read_text(encoding="utf-8"))
        if "test" in parsed:
            raise RuntimeError("AUG_MILD_768 YAML must not expose fixed test")
        clean_audit, original_audit = split_audit(clean), split_audit(original)
        if clean_audit != original_audit:
            raise RuntimeError("AUG_MILD_768 split membership differs from Run A Fold %d" % fold)
        observed = {name: clean_audit[name]["sha256"] for name in ("train", "val")}
        if observed != EXPECTED_SPLIT_HASHES[fold]:
            raise RuntimeError("Run A split hash changed for Fold %d" % fold)
        records[str(fold)] = clean_audit
    return records


def expected_fold_protocol(fold: int) -> Dict[str, Any]:
    validate_static_protocol()
    splits = validate_data_configs()[str(fold)]
    return {
        "status": "frozen_before_training",
        "experiment": EXPERIMENT,
        "fold": fold,
        "seed": 42 + fold,
        "architecture": str(ARCHITECTURE.resolve()),
        "architecture_sha256": ARCHITECTURE_SHA256,
        "coco_source": str(COCO_WEIGHTS.resolve()),
        "source_coco_sha256": COCO_SHA256,
        "data": str((DATA_CONFIG_DIR / ("fold_%d.yaml" % fold)).resolve()),
        "splits": splits,
        "baseline_optimization": dict(BASELINE_OPTIMIZATION),
        "baseline_augmentation": dict(BASELINE_AUGMENTATION),
        "training_augmentation": dict(AUG_MILD_AUGMENTATION),
        "parameter_diff_from_run_a": augmentation_diff(),
        "fixed": {
            "model": "YOLO26s-P2", "heads": ["P2", "P3", "P4", "P5"],
            "input": "2.5D physical +/-5 mm; HU WL=800 WW=1600",
            "balanced_sampling": False, "hnm": False, "extra_negatives": False,
            "spatial_filter": False, "verifier": False, "hu_jitter": False,
            "automatic_batch_reduction": False,
        },
        "initialization_confound": (
            "Historical Run A did not seed before construction/save each random P2 initialization. "
            "AUG_MILD_768 seeds before model construction and saves coco_init.pt plus its SHA256; "
            "therefore comparison with historical Run A has an initialization confound."
        ),
    }


def assert_saved_protocol(path: Path, fold: int) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Missing frozen run protocol: %s" % path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = expected_fold_protocol(fold)
    for key in (
        "experiment", "fold", "seed", "architecture_sha256", "source_coco_sha256",
        "splits", "baseline_optimization", "baseline_augmentation", "training_augmentation",
        "parameter_diff_from_run_a", "fixed", "initialization_confound",
    ):
        if saved.get(key) != expected.get(key):
            raise RuntimeError("Saved AUG_MILD_768 protocol drift in %s" % key)
    return saved


def completion_is_valid(run_dir: Path, fold: int) -> bool:
    marker = run_dir / "training_completed.json"
    required = [run_dir / "weights/best.pt", run_dir / "results.csv", run_dir / "run_protocol.json"]
    if not marker.is_file():
        return False
    if any(not path.is_file() for path in required):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert_saved_protocol(run_dir / "run_protocol.json", fold)
    except (ValueError, OSError, RuntimeError):
        return False
    return payload.get("status") == "completed"
