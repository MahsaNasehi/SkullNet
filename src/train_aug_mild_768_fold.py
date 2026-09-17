"""Train or resume one frozen AUG_MILD_768 Fold."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.utils.torch_utils import init_seeds

from aug_mild_768_protocol import (
    ARCHITECTURE, AUG_MILD_AUGMENTATION, BASELINE_OPTIMIZATION, COCO_WEIGHTS,
    DATA_CONFIG_DIR, EXPERIMENT, RUN_PREFIX, assert_saved_protocol,
    expected_fold_protocol, train_kwargs, validate_static_protocol,
)
from train_yolo26s_p2 import ROOT, initialize, sha256, write_completion


def run_dir_for_fold(fold: int) -> Path:
    return ROOT / "outputs" / (RUN_PREFIX + str(fold))


def _validate_checkpoint_train_args(checkpoint: YOLO) -> None:
    train_args = checkpoint.ckpt.get("train_args", {})
    expected = {
        "imgsz": 768, "batch": 16, "epochs": 150, "patience": 30,
        "workers": 0, "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01,
        "momentum": 0.937, "weight_decay": 0.0005, "warmup_epochs": 3.0,
        "amp": True, "deterministic": True, **AUG_MILD_AUGMENTATION,
    }
    for key, value in expected.items():
        if train_args.get(key) != value:
            raise RuntimeError(
                "Resume checkpoint augmentation/training protocol drift for %s: %r != %r"
                % (key, train_args.get(key), value)
            )


def train_new(fold: int, device: str) -> None:
    validate_static_protocol()
    run_dir = run_dir_for_fold(fold)
    if run_dir.exists():
        raise FileExistsError("Refusing to overwrite AUG_MILD_768 output: %s" % run_dir)
    seed = 42 + fold
    # Historical Run A did not guarantee this pre-construction seed.  It is
    # explicit here, and the resulting initialized checkpoint is persisted.
    init_seeds(seed, deterministic=True)
    model = initialize(ARCHITECTURE, COCO_WEIGHTS, run_dir / "pretrained_transfer_audit.json")
    initialized = run_dir / "coco_init.pt"
    model.save(str(initialized))
    protocol = expected_fold_protocol(fold)
    protocol.update({
        "command": [sys.executable, *sys.argv],
        "initialized_checkpoint": str(initialized.resolve()),
        "initialized_checkpoint_sha256": sha256(initialized),
        "software": {
            "python": platform.python_version(), "torch": torch.__version__,
            "ultralytics": ultralytics_version,
        },
    })
    (run_dir / "run_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (run_dir / "training_started.json").write_text(json.dumps({
        "status": "started_not_completed", "experiment": EXPERIMENT, "fold": fold,
        "started_unix_time": time.time(), "completion_marker_required": "training_completed.json",
    }, indent=2) + "\n", encoding="utf-8")

    cuda_index = None
    if torch.cuda.is_available() and device.lower() != "cpu":
        cuda_index = int(device.split(",")[0])
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    started = time.perf_counter()
    model.train(
        data=str((DATA_CONFIG_DIR / ("fold_%d.yaml" % fold)).resolve()),
        project=str((ROOT / "outputs").resolve()), name=RUN_PREFIX + str(fold),
        epochs=BASELINE_OPTIMIZATION["epochs"], patience=BASELINE_OPTIMIZATION["patience"],
        batch=BASELINE_OPTIMIZATION["batch"], imgsz=BASELINE_OPTIMIZATION["imgsz"],
        workers=BASELINE_OPTIMIZATION["workers"], device=device, seed=seed,
        exist_ok=True, **train_kwargs()
    )
    assert_saved_protocol(run_dir / "run_protocol.json", fold)
    write_completion(run_dir, time.perf_counter() - started, cuda_index, resumed=False)


def resume_fold(fold: int) -> None:
    run_dir = run_dir_for_fold(fold)
    assert_saved_protocol(run_dir / "run_protocol.json", fold)
    last = run_dir / "weights/last.pt"
    if not last.is_file():
        raise FileNotFoundError("Cannot resume without %s" % last)
    if (run_dir / "training_completed.json").exists():
        raise FileExistsError("Completed AUG_MILD_768 Fold must not be resumed: %s" % run_dir)
    checkpoint = YOLO(str(last))
    if checkpoint.ckpt.get("optimizer") is None or checkpoint.ckpt.get("epoch", -1) < 0:
        raise RuntimeError("last.pt is not a resumable training checkpoint")
    _validate_checkpoint_train_args(checkpoint)
    cuda_index = None
    stored_device = str(checkpoint.ckpt.get("train_args", {}).get("device", "0"))
    if torch.cuda.is_available() and stored_device.lower() != "cpu":
        cuda_index = int(stored_device.split(",")[0])
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    events = run_dir / "resume_events.json"
    history = json.loads(events.read_text()) if events.is_file() else []
    history.append({
        "resume_unix_time": time.time(), "checkpoint": str(last.resolve()),
        "checkpoint_sha256_before_resume": sha256(last),
        "protocol_validated": True, "automatic_batch_reduction": False,
    })
    events.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    checkpoint.train(resume=True)
    assert_saved_protocol(run_dir / "run_protocol.json", fold)
    write_completion(run_dir, time.perf_counter() - started, cuda_index, resumed=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.resume:
        resume_fold(args.fold)
    else:
        train_new(args.fold, args.device)
