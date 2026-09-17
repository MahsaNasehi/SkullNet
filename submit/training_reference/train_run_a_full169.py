"""Train/resume the fixed 59-epoch Run A full169 refit (manual launch only)."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import init_seeds

from run_a_full169_protocol import (
    ARCHITECTURE, ARCHITECTURE_SHA256, COCO_SHA256, COCO_WEIGHTS, DATA_YAML,
    FIXED_EPOCHS, FORBIDDEN_MODES, OUTPUT_DIR, PREPARATION_REPORT,
    RUN_A_TRAIN_KWARGS, RUN_NAME, SEED, sha256, validate_prepared,
)
from train_yolo26s_p2 import ROOT, initialize


class FixedBatchDetectionTrainer(DetectionTrainer):
    """Turn Ultralytics first-epoch OOM batch reduction into a hard failure."""

    def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode="train"):
        if mode == "train" and batch_size != 16:
            raise RuntimeError("FULL169 protocol forbids automatic batch reduction from batch=16")
        return super().get_dataloader(dataset_path, batch_size, rank, mode)


def _protocol(initialized: Path) -> dict:
    cohort = validate_prepared()
    return {
        "status": "frozen_before_training",
        "experiment": "RUN_A_FULL169_FIXED_EPOCH_REFIT",
        "output_dir": str(OUTPUT_DIR.resolve()),
        "model_family": "YOLO26s-P2 Run A 768",
        "input": "2.5D physical +/-5 mm; HU WL=800 WW=1600",
        "seed_before_model_construction": SEED,
        "fixed_epochs": FIXED_EPOCHS,
        "batch": 16, "imgsz": 768, "workers": 0, "patience_inert_with_val_false": 30,
        "validation_enabled_during_training": False,
        "validation_handling": cohort["validation_policy"],
        "checkpoint_selection": "none; final deployment checkpoint is fixed final epoch weights/last.pt",
        "final_deployment_checkpoint": str((OUTPUT_DIR / "weights/final_deployment.pt").resolve()),
        "architecture": str(ARCHITECTURE.resolve()), "architecture_sha256": ARCHITECTURE_SHA256,
        "coco_source": str(COCO_WEIGHTS.resolve()), "source_coco_sha256": COCO_SHA256,
        "initialized_checkpoint": str(initialized.resolve()),
        "initialized_checkpoint_sha256": sha256(initialized),
        "training_cohort_sha256": cohort["training_cohort_sha256"],
        "train_list_sha256": cohort["train_list_sha256"],
        "data_yaml_sha256": cohort["data_yaml_sha256"],
        "manifest_sha256": cohort["manifest_sha256"],
        "cohort": {key: cohort[key] for key in (
            "studies", "patients", "training_images", "per_fold_studies", "per_fold_images",
            "fixed_test_study_overlap", "fixed_test_patient_overlap",
            "metadata_only_or_tier2_images_added",
        )},
        "training_hyperparameters": RUN_A_TRAIN_KWARGS,
        "forbidden_modes": FORBIDDEN_MODES,
        "automatic_batch_reduction": False,
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "ultralytics": ultralytics_version},
        "preparation_report": str(PREPARATION_REPORT.resolve()),
    }


def _validate_saved_protocol() -> dict:
    path = OUTPUT_DIR / "run_protocol.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    cohort = validate_prepared()
    checks = {
        "fixed_epochs": FIXED_EPOCHS, "batch": 16, "imgsz": 768,
        "training_cohort_sha256": cohort["training_cohort_sha256"],
        "architecture_sha256": ARCHITECTURE_SHA256, "source_coco_sha256": COCO_SHA256,
        "training_hyperparameters": RUN_A_TRAIN_KWARGS,
        "forbidden_modes": FORBIDDEN_MODES, "automatic_batch_reduction": False,
    }
    for key, value in checks.items():
        if saved.get(key) != value:
            raise RuntimeError(f"Saved FULL169 protocol drift in {key}")
    return saved


def _finish(started: float | None, resumed: bool) -> None:
    last = OUTPUT_DIR / "weights/last.pt"
    results_path = OUTPUT_DIR / "results.csv"
    if not last.is_file() or not results_path.is_file():
        raise RuntimeError("Training returned without last.pt/results.csv")
    protocol_path = OUTPUT_DIR / "run_protocol.json"
    if not protocol_path.is_file():
        raise RuntimeError("Finalization requires the frozen run_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol.get("fixed_epochs"), protocol.get("batch"), protocol.get("imgsz")) != (FIXED_EPOCHS, 16, 768):
        raise RuntimeError("Frozen FULL169 protocol does not match the finalization target")
    results = pd.read_csv(results_path)
    results.columns = results.columns.str.strip()
    # Ultralytics records 1-based epochs in results.csv, but strips the final
    # checkpoint's zero-based epoch to -1 when it removes the optimizer.
    if len(results) != FIXED_EPOCHS or results["epoch"].tolist() != list(range(1, FIXED_EPOCHS + 1)):
        raise RuntimeError(f"Expected consecutive CSV epochs 1..{FIXED_EPOCHS}, got {results['epoch'].tolist()}")
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("model") is None:
        raise RuntimeError("Final last.pt is not a model checkpoint")
    if checkpoint.get("epoch") != -1 or checkpoint.get("optimizer") is not None:
        raise RuntimeError("Expected an optimizer-stripped final last.pt with epoch=-1")
    train_args = checkpoint.get("train_args") or {}
    required = {
        "epochs": FIXED_EPOCHS, "batch": 16, "imgsz": 768, "seed": SEED,
        "val": False, "name": RUN_NAME, "data": str(DATA_YAML.resolve()),
    }
    for key, value in required.items():
        if train_args.get(key) != value:
            raise RuntimeError(f"Final checkpoint training-argument drift in {key}")
    embedded = checkpoint.get("train_results")
    if not isinstance(embedded, dict):
        raise RuntimeError("Final checkpoint has no embedded training history")
    try:
        pd.testing.assert_frame_equal(results, pd.DataFrame(embedded), check_dtype=False)
    except AssertionError as exc:
        raise RuntimeError("Final checkpoint training history differs from results.csv") from exc
    deployment = OUTPUT_DIR / "weights/final_deployment.pt"
    completion_path = OUTPUT_DIR / "training_completed.json"
    if deployment.exists() or completion_path.exists():
        raise FileExistsError("Final deployment checkpoint/completion marker already exists")
    shutil.copy2(last, deployment)
    last_hash, deployment_hash = sha256(last), sha256(deployment)
    if last_hash != deployment_hash:
        raise RuntimeError("Final deployment copy differs from final-epoch last.pt")
    completion = {
        "status": "completed_fixed_epoch_refit", "fixed_epochs": FIXED_EPOCHS,
        "results_csv_first_epoch_one_based": 1,
        "results_csv_final_epoch_one_based": FIXED_EPOCHS,
        "final_epoch_zero_based_inferred_from_history": FIXED_EPOCHS - 1,
        "checkpoint_epoch_metadata": checkpoint["epoch"],
        "checkpoint_epoch_metadata_meaning": "-1 is the Ultralytics optimizer-stripped sentinel",
        "checkpoint_history_matches_results_csv": True,
        "final_checkpoint_validated": True,
        "final_deployment_is_byte_identical_copy_of_last_pt": True,
        "finalized_without_retraining": started is None,
        "resumed_final_process": resumed,
        "final_process_wall_seconds": None if started is None else time.perf_counter() - started,
        "deployment_checkpoint": str(deployment.resolve()),
        "deployment_checkpoint_sha256": deployment_hash,
        "source_final_epoch_last_pt": str(last.resolve()), "last_pt_sha256": last_hash,
        "run_protocol_sha256": sha256(protocol_path),
        "best_pt_is_not_used_for_deployment": True,
        "validation_was_not_used_for_checkpoint_selection": True,
    }
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion, indent=2), flush=True)


def train_new(device: str) -> None:
    cohort = validate_prepared()
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite FULL169 output: {OUTPUT_DIR}")
    init_seeds(SEED, deterministic=True)
    model = initialize(ARCHITECTURE, COCO_WEIGHTS, OUTPUT_DIR / "pretrained_transfer_audit.json")
    initialized = OUTPUT_DIR / "coco_init.pt"
    model.save(str(initialized))
    protocol = _protocol(initialized)
    protocol["command"] = [sys.executable, *sys.argv]
    (OUTPUT_DIR / "run_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (OUTPUT_DIR / "training_started.json").write_text(json.dumps({
        "status": "started_not_completed", "started_unix_time": time.time(),
        "completion_marker_required": "training_completed.json",
        "training_images": cohort["training_images"],
    }, indent=2) + "\n")
    started = time.perf_counter()
    model.train(
        trainer=FixedBatchDetectionTrainer,
        data=str(DATA_YAML.resolve()), project=str((ROOT / "outputs").resolve()), name=RUN_NAME,
        epochs=FIXED_EPOCHS, patience=30, batch=16, imgsz=768, workers=0,
        device=device, seed=SEED, val=False, exist_ok=True, **RUN_A_TRAIN_KWARGS,
    )
    _finish(started, resumed=False)


def resume_training() -> None:
    _validate_saved_protocol()
    if (OUTPUT_DIR / "training_completed.json").exists():
        raise FileExistsError("Completed FULL169 refit must not be resumed")
    last = OUTPUT_DIR / "weights/last.pt"
    if not last.is_file():
        raise FileNotFoundError(last)
    model = YOLO(str(last))
    if model.ckpt.get("optimizer") is None or model.ckpt.get("epoch", -1) < 0:
        raise RuntimeError("last.pt is not a resumable training checkpoint")
    args = model.ckpt.get("train_args", {})
    required = {"epochs": FIXED_EPOCHS, "batch": 16, "imgsz": 768, "seed": SEED, "val": False}
    for key, value in required.items():
        if args.get(key) != value:
            raise RuntimeError(f"Resume checkpoint protocol drift in {key}")
    events = OUTPUT_DIR / "resume_events.json"
    history = json.loads(events.read_text()) if events.is_file() else []
    history.append({"time": time.time(), "checkpoint_sha256_before_resume": sha256(last)})
    events.write_text(json.dumps(history, indent=2) + "\n")
    started = time.perf_counter()
    model.train(resume=True, trainer=FixedBatchDetectionTrainer)
    _finish(started, resumed=True)


def status() -> None:
    if not OUTPUT_DIR.exists():
        print(json.dumps({"status": "not_started", "output_dir": str(OUTPUT_DIR)}))
        return
    complete = OUTPUT_DIR / "training_completed.json"
    if complete.is_file():
        print(complete.read_text(encoding="utf-8"), end="")
        return
    results = OUTPUT_DIR / "results.csv"
    epochs = len(pd.read_csv(results)) if results.is_file() else 0
    print(json.dumps({
        "status": "started_not_complete", "epochs_recorded": epochs,
        "last_pt_exists": (OUTPUT_DIR / "weights/last.pt").is_file(),
        "protocol_exists": (OUTPUT_DIR / "run_protocol.json").is_file(),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--finalize-only", action="store_true",
                        help="Validate the finished checkpoint and copy it without training")
    args = parser.parse_args()
    if args.status:
        status()
    elif args.finalize_only:
        _finish(started=None, resumed=False)
    elif args.resume:
        resume_training()
    else:
        train_new(args.device)


if __name__ == "__main__":
    main()
