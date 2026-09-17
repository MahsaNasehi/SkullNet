"""Train/resume one isolated FITNESS_BALANCED_768 Fold."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import torch
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.utils.torch_utils import init_seeds, strip_optimizer

from balanced_fitness_checkpoint import BalancedCheckpointTrainer
from fitness_balanced_768_protocol import (
    ARCHITECTURE, COCO_WEIGHTS, EXPERIMENT, RUN_A_TRAINING, data_yaml,
    expected_protocol, run_dir, sha256, training_kwargs, validate_saved_protocol,
)
from train_yolo26s_p2 import ROOT, initialize, write_completion


def _finalize_balanced(run_path) -> dict:
    checkpoint = run_path / "weights/best_balanced.pt"
    metadata_path = run_path / "weights/best_balanced.json"
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise RuntimeError("Training completed without best_balanced checkpoint/metadata")
    strip_optimizer(checkpoint)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_sha256"] = sha256(checkpoint)
    metadata["optimizer_stripped_after_training"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def train_new(fold: int, device: str) -> None:
    target = run_dir(fold)
    if target.exists():
        raise FileExistsError("Refusing to overwrite FITNESS_BALANCED output: %s" % target)
    seed = 42 + fold
    init_seeds(seed, deterministic=True)
    model = initialize(ARCHITECTURE, COCO_WEIGHTS, target / "pretrained_transfer_audit.json")
    initialized = target / "coco_init.pt"
    model.save(str(initialized))
    protocol = expected_protocol(fold)
    protocol.update({
        "command": [sys.executable, *sys.argv],
        "initialized_checkpoint": str(initialized.resolve()),
        "initialized_checkpoint_sha256": sha256(initialized),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "ultralytics": ultralytics_version},
    })
    (target / "run_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (target / "training_started.json").write_text(json.dumps({
        "status": "started_not_completed", "experiment": EXPERIMENT, "fold": fold,
        "started_unix_time": time.time(), "completion_marker_required": "training_completed.json",
    }, indent=2) + "\n")
    cuda_index = None
    if torch.cuda.is_available() and device.lower() != "cpu":
        cuda_index = int(device.split(",")[0])
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    started = time.perf_counter()
    model.train(
        trainer=BalancedCheckpointTrainer,
        data=str(data_yaml(fold).resolve()), project=str((ROOT / "outputs").resolve()),
        name=target.name, epochs=150, patience=30, batch=16, imgsz=768,
        workers=0, device=device, seed=seed, exist_ok=True, **training_kwargs()
    )
    balanced = _finalize_balanced(target)
    completion = write_completion(target, time.perf_counter() - started, cuda_index, resumed=False)
    completion["best_balanced"] = balanced
    (target / "training_completed.json").write_text(json.dumps(completion, indent=2) + "\n")


def resume_fold(fold: int) -> None:
    target = run_dir(fold)
    validate_saved_protocol(fold)
    if (target / "training_completed.json").exists():
        raise FileExistsError("Completed FITNESS_BALANCED Fold must not be resumed")
    last = target / "weights/last.pt"
    if not last.is_file():
        raise FileNotFoundError(last)
    model = YOLO(str(last))
    if model.ckpt.get("optimizer") is None or model.ckpt.get("epoch", -1) < 0:
        raise RuntimeError("last.pt is not a resumable training checkpoint")
    args = model.ckpt.get("train_args", {})
    expected = {"epochs": 150, "patience": 30, "batch": 16, "imgsz": 768,
                "seed": 42 + fold, **{k: RUN_A_TRAINING[k] for k in training_kwargs()}}
    for key, value in expected.items():
        if args.get(key) != value:
            raise RuntimeError("Resume checkpoint Run A protocol drift in %s" % key)
    events = target / "resume_events.json"
    history = json.loads(events.read_text()) if events.is_file() else []
    history.append({"resume_unix_time": time.time(), "last_sha256": sha256(last)})
    events.write_text(json.dumps(history, indent=2) + "\n")
    cuda_index = None
    stored_device = str(args.get("device", "0"))
    if torch.cuda.is_available() and stored_device.lower() != "cpu":
        cuda_index = int(stored_device.split(",")[0])
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    started = time.perf_counter()
    model.train(resume=True, trainer=BalancedCheckpointTrainer)
    balanced = _finalize_balanced(target)
    completion = write_completion(target, time.perf_counter() - started, cuda_index, resumed=True)
    completion["best_balanced"] = balanced
    (target / "training_completed.json").write_text(json.dumps(completion, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume:
        resume_fold(args.fold)
    else:
        train_new(args.fold, args.device)


if __name__ == "__main__":
    main()

