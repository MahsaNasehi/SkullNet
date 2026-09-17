"""Train, resume, or report MOSAIC08_SCALE05_BALANCED_FOLD4 status."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time

import pandas as pd
import torch
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.utils.torch_utils import init_seeds, strip_optimizer

from balanced_fitness_checkpoint import FixedBatchBalancedCheckpointTrainer
from mosaic08_scale05_balanced_fold4_protocol import (
    DATA_YAML, EXPERIMENT, LOG_PATH, MAX_EPOCHS, OUTPUT_DIR, PATIENCE, SEED,
    SOURCE_CHECKPOINT, SOURCE_SHA256, TRAIN_KWARGS, expected_protocol,
    prepare_config, sha256, validate_saved_protocol, validate_sources,
)
from train_yolo26s_p2 import ROOT, write_completion


def _finalize(started: float, resumed: bool) -> None:
    validate_sources()
    checkpoint = OUTPUT_DIR / "weights/best_balanced.pt"
    metadata_path = OUTPUT_DIR / "best_balanced_metrics.json"
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise RuntimeError("Missing best_balanced checkpoint or metrics")
    strip_optimizer(checkpoint)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_sha256"] = sha256(checkpoint)
    metadata["optimizer_stripped_after_training"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    completion = write_completion(
        OUTPUT_DIR, time.perf_counter() - started,
        0 if torch.cuda.is_available() else None, resumed=resumed,
    )
    completion.update({
        "experiment": EXPERIMENT, "source_checkpoint_sha256_after_training": sha256(SOURCE_CHECKPOINT),
        "historical_checkpoint_untouched": sha256(SOURCE_CHECKPOINT) == SOURCE_SHA256,
        "best_balanced": metadata,
    })
    (OUTPUT_DIR / "training_completed.json").write_text(json.dumps(completion, indent=2) + "\n")


def train_new(device: str) -> None:
    prepare_config()
    if OUTPUT_DIR.exists():
        raise FileExistsError("Refusing to overwrite isolated output: %s" % OUTPUT_DIR)
    source_hash_before = sha256(SOURCE_CHECKPOINT)
    OUTPUT_DIR.mkdir(parents=True)
    initialization = OUTPUT_DIR / "initialization_source.pt"
    shutil.copy2(SOURCE_CHECKPOINT, initialization)
    if sha256(initialization) != SOURCE_SHA256 or sha256(SOURCE_CHECKPOINT) != source_hash_before:
        raise RuntimeError("Initialization copy/source identity failure")
    init_seeds(SEED, deterministic=True)
    model = YOLO(str(initialization), task="detect")
    protocol = expected_protocol(initialization)
    protocol.update({
        "command": [sys.executable, *sys.argv],
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "ultralytics": ultralytics_version},
    })
    (OUTPUT_DIR / "run_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (OUTPUT_DIR / "training_started.json").write_text(json.dumps({
        "status": "started_not_completed", "started_unix_time": time.time(),
        "resume": False, "fresh_optimizer": True,
    }, indent=2) + "\n")
    started = time.perf_counter()
    model.train(
        trainer=FixedBatchBalancedCheckpointTrainer,
        data=str(DATA_YAML.resolve()), project=str((ROOT / "outputs").resolve()), name=OUTPUT_DIR.name,
        epochs=MAX_EPOCHS, patience=PATIENCE, batch=16, imgsz=768, workers=0,
        device=device, seed=SEED, resume=False, exist_ok=True, **TRAIN_KWARGS,
    )
    _finalize(started, resumed=False)


def resume_training() -> None:
    validate_saved_protocol()
    if (OUTPUT_DIR / "training_completed.json").exists():
        raise FileExistsError("Completed experiment must not be resumed")
    last = OUTPUT_DIR / "weights/last.pt"
    if not last.is_file():
        raise FileNotFoundError(last)
    model = YOLO(str(last))
    if model.ckpt.get("optimizer") is None or model.ckpt.get("epoch", -1) < 0:
        raise RuntimeError("last.pt is not a resumable training checkpoint")
    args = model.ckpt.get("train_args", {})
    expected = {"epochs": MAX_EPOCHS, "patience": PATIENCE, "batch": 16,
                "imgsz": 768, "seed": SEED, "mosaic": 0.80, "scale": 0.50}
    for key, value in expected.items():
        if args.get(key) != value:
            raise RuntimeError("Resume checkpoint protocol drift in %s" % key)
    events = OUTPUT_DIR / "resume_events.json"
    history = json.loads(events.read_text()) if events.is_file() else []
    history.append({"resume_unix_time": time.time(), "last_sha256": sha256(last)})
    events.write_text(json.dumps(history, indent=2) + "\n")
    started = time.perf_counter()
    model.train(resume=True, trainer=FixedBatchBalancedCheckpointTrainer)
    _finalize(started, resumed=True)


def status() -> None:
    if not OUTPUT_DIR.exists():
        print(json.dumps({"experiment": EXPERIMENT, "status": "not_started"}))
        return
    complete = OUTPUT_DIR / "training_completed.json"
    metrics = OUTPUT_DIR / "best_balanced_metrics.json"
    results = OUTPUT_DIR / "results.csv"
    print(json.dumps({
        "experiment": EXPERIMENT,
        "status": "completed" if complete.is_file() else "started_not_complete",
        "epochs_recorded": len(pd.read_csv(results)) if results.is_file() else 0,
        "best_balanced": json.loads(metrics.read_text()) if metrics.is_file() else None,
        "source_checkpoint_sha256_current": sha256(SOURCE_CHECKPOINT),
        "source_checkpoint_untouched": sha256(SOURCE_CHECKPOINT) == SOURCE_SHA256,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
    elif args.resume:
        resume_training()
    else:
        train_new(args.device)


if __name__ == "__main__":
    main()
