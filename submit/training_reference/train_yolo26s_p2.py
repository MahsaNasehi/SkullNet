"""Initialize YOLO26s-P2 from COCO YOLO26s weights and train it."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version
from ultralytics.utils.torch_utils import init_seeds


ROOT = Path(__file__).resolve().parents[1]

TRAINING_HYPERPARAMETERS = {
    "optimizer": "AdamW", "lr0": 1e-3, "lrf": 1e-2, "weight_decay": 5e-4,
    "warmup_epochs": 3.0, "cos_lr": True, "close_mosaic": 10,
    "mosaic": 0.30, "mixup": 0.0, "cutmix": 0.0,
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "degrees": 5.0, "translate": 0.05, "scale": 0.15,
    "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
    "amp": True, "cache": False, "plots": True, "save": True, "save_period": 10,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}


def split_audit(data_yaml: Path) -> dict[str, object]:
    """Hash train/val membership only; deliberately ignore any test key."""
    config = yaml.safe_load(data_yaml.read_text())
    dataset_root = Path(config.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    result = {}
    for split in ("train", "val"):
        value = config.get(split)
        if not isinstance(value, str):
            raise ValueError(f"Expected one {split} list in {data_yaml}")
        path = Path(value)
        if not path.is_absolute():
            path = (dataset_root / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        result[split] = {"path": str(path), "sha256": sha256(path), "images": len(lines)}
    return result


def write_completion(run_dir: Path, process_wall_seconds: float, cuda_index: int | None,
                     resumed: bool) -> dict[str, object]:
    results_path = run_dir / "results.csv"
    best_path = run_dir / "weights/best.pt"
    last_path = run_dir / "weights/last.pt"
    for required in (results_path, best_path, last_path):
        if not required.is_file():
            raise RuntimeError(f"Training returned without required artifact: {required}")
    results = pd.read_csv(results_path)
    results.columns = results.columns.str.strip()
    metric = "metrics/mAP50-95(B)"
    best_row = results.loc[results[metric].idxmax()]
    started_path = run_dir / "training_started.json"
    initial_started = json.loads(started_path.read_text())["started_unix_time"] if started_path.is_file() else None
    completion = {
        "status": "completed", "completed_unix_time": time.time(), "resumed_final_process": resumed,
        "final_process_wall_seconds": process_wall_seconds,
        "wall_since_initial_start_seconds": time.time() - initial_started if initial_started else None,
        "ultralytics_cumulative_time_seconds": (
            float(results.iloc[-1]["time"]) if "time" in results.columns else None
        ),
        "epochs_completed": len(results), "best_epoch_by_map50_95": int(best_row["epoch"]),
        "best_map50_95": float(best_row[metric]),
        "best_map50": float(best_row["metrics/mAP50(B)"]),
        "best_precision": float(best_row["metrics/precision(B)"]),
        "best_recall": float(best_row["metrics/recall(B)"]),
        "best_pt_sha256": sha256(best_path), "last_pt_sha256": sha256(last_path),
        "cuda_peak_allocated_gib_final_process": (
            torch.cuda.max_memory_allocated(cuda_index) / 2**30 if cuda_index is not None else None
        ),
        "cuda_peak_reserved_gib_final_process": (
            torch.cuda.max_memory_reserved(cuda_index) / 2**30 if cuda_index is not None else None
        ),
    }
    (run_dir / "training_completed.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion, indent=2), flush=True)
    return completion


def initialize(architecture: Path, pretrained: Path, audit_path: Path) -> YOLO:
    if not architecture.is_file():
        raise FileNotFoundError(architecture)
    if not pretrained.is_file():
        raise FileNotFoundError(
            f"{pretrained} not found. Run: .venv/bin/python scripts/download_weights.py"
        )
    model = YOLO(str(architecture), task="detect")
    source = YOLO(str(pretrained), task="detect")
    target_state = model.model.state_dict()
    source_state = source.model.state_dict()
    transferable = {
        key: value for key, value in source_state.items()
        if key in target_state and value.shape == target_state[key].shape
    }
    model.load(str(pretrained))
    total_tensors = len(target_state)
    total_params = sum(value.numel() for value in target_state.values())
    transferred_params = sum(value.numel() for value in transferable.values())
    audit = {
        "architecture": str(architecture.resolve()),
        "pretrained": str(pretrained.resolve()),
        "pretrained_sha256": sha256(pretrained),
        "source": "official Ultralytics YOLO26s COCO checkpoint",
        "matching_tensors": len(transferable),
        "total_target_tensors": total_tensors,
        "matching_state_elements": transferred_params,
        "total_target_state_elements": total_params,
        "state_element_transfer_fraction": transferred_params / total_params,
        "note": "P2-specific/new detection parameters remain randomly initialized.",
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return model


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data_prepared/skull_hu800_ww1600/dataset.yaml")
    parser.add_argument("--architecture", type=Path, default=ROOT / "configs/yolo26s-p2.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/yolo26s.pt")
    parser.add_argument("--project", type=Path, default=ROOT / "outputs")
    parser.add_argument("--name", default="yolo26s_p2_hu800_ww1600")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    # This server exposes only ~442 MiB at /dev/shm. Multiple workers prefetching
    # 768px batches exceed it and are killed with SIGBUS. workers=0 keeps loading
    # in the main process and does not use multiprocessing shared-memory queues.
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0, help="Training-data fraction; use <1 only for smoke tests")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing run directory")
    parser.add_argument("--positive-fraction", type=float, help="Enable patient-balanced primary-image sampling, e.g. 0.25")
    parser.add_argument("--sampling-manifest", type=Path, default=ROOT / "data_prepared/skull_hu800_ww1600/manifest.csv")
    parser.add_argument("--sampling-audit-only", action="store_true", help="Validate splits and report sampling without training")
    args = parser.parse_args(argv)

    training_extra = {}
    policy = None
    if args.positive_fraction is not None and not args.resume:
        from patient_balanced_sampling import sampling_policy
        if args.workers != 0 or args.fraction != 1.0:
            raise ValueError("Balanced experiment requires workers=0 and fraction=1")
        policy, sampler = sampling_policy(args.data, args.sampling_manifest, args.positive_fraction, args.batch, args.seed)
        if args.sampling_audit_only:
            print(json.dumps({"policy": policy, "epochs": [sampler.audit(e) for e in range(5)]}, indent=2))
            return
        from balanced_yolo_trainer import PatientBalancedDetectionTrainer
        training_extra["trainer"] = partial(PatientBalancedDetectionTrainer, sampling_config=policy)
    elif args.sampling_audit_only:
        raise ValueError("--sampling-audit-only requires --positive-fraction and a new experiment")

    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        resumed = YOLO(str(args.resume))
        if resumed.ckpt.get("optimizer") is None or resumed.ckpt.get("epoch", -1) < 0:
            raise ValueError("This checkpoint is completed/stripped and cannot resume training")
        policy = getattr(resumed.model, "sampling_policy", None)
        policy_path = args.resume.resolve().parents[1] / "sampling_policy.json"
        if policy is None and policy_path.is_file():
            policy = json.loads(policy_path.read_text())
        if policy is not None:
            if args.positive_fraction is not None and args.positive_fraction != policy["positive_fraction"]:
                raise ValueError("Cannot change the sampling ratio during resume")
            from balanced_yolo_trainer import PatientBalancedDetectionTrainer
            training_extra["trainer"] = partial(PatientBalancedDetectionTrainer, sampling_config=policy)
        elif args.positive_fraction is not None:
            raise ValueError("Start a separate COCO-initialized experiment to change the sampling policy")
        run_dir = args.resume.resolve().parents[1]
        if (run_dir / "training_completed.json").exists():
            raise FileExistsError(f"Completion marker already exists: {run_dir}")
        cuda_index = None
        resume_device = resumed.overrides.get("device", "0")
        if torch.cuda.is_available() and str(resume_device).lower() != "cpu":
            cuda_index = int(str(resume_device).split(",")[0])
            torch.cuda.set_device(cuda_index)
            torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(cuda_index)
        events_path = run_dir / "resume_events.json"
        events = json.loads(events_path.read_text()) if events_path.is_file() else []
        events.append({"resume_unix_time": time.time(), "checkpoint": str(args.resume.resolve()),
                       "checkpoint_sha256_before_resume": sha256(args.resume)})
        events_path.write_text(json.dumps(events, indent=2) + "\n")
        resume_start = time.perf_counter()
        resumed.train(resume=True, **training_extra)
        write_completion(run_dir, time.perf_counter() - resume_start, cuda_index, resumed=True)
        return
    if not args.init_only and not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}. Prepare the dataset first.")
    run_dir = args.project / args.name
    if run_dir.exists() and not (args.init_only or args.exist_ok):
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose --name or explicitly pass --exist-ok."
        )
    audit_path = args.project / args.name / "pretrained_transfer_audit.json"
    # Seed before model construction so P2-specific random parameters are also
    # reproducible. The initialized checkpoint is persisted per run.
    init_seeds(args.seed, deterministic=True)
    model = initialize(args.architecture, args.weights, audit_path)
    initialized_path = run_dir / "coco_init.pt"
    model.save(str(initialized_path))
    if args.init_only:
        print(f"Saved initialized model: {initialized_path}")
        return

    protocol = {
        "status": "frozen_before_training",
        "command": [sys.executable, *sys.argv],
        "arguments": _jsonable_args(args),
        "training_hyperparameters": TRAINING_HYPERPARAMETERS,
        "architecture_sha256": sha256(args.architecture),
        "source_coco_sha256": sha256(args.weights),
        "initialized_checkpoint": str(initialized_path.resolve()),
        "initialized_checkpoint_sha256": sha256(initialized_path),
        "splits": split_audit(args.data),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "ultralytics": ultralytics_version},
    }
    (run_dir / "run_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (run_dir / "training_started.json").write_text(json.dumps({
        "started_unix_time": time.time(), "completion_marker_required": "training_completed.json"
    }, indent=2) + "\n")
    cuda_index = None
    if torch.cuda.is_available() and str(args.device).lower() != "cpu":
        cuda_index = int(str(args.device).split(",")[0])
        torch.cuda.set_device(cuda_index)
        torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(cuda_index)
    training_start = time.perf_counter()

    # HU semantics are preserved: no HSV/color augmentation. Mild geometry is
    # used because large rotations and vertical flips are implausible for axial CT.
    model.train(
        **training_extra,
        data=str(args.data.resolve()),
        project=str(args.project.resolve()),
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        seed=args.seed,
        fraction=args.fraction,
        deterministic=True,
        **TRAINING_HYPERPARAMETERS,
        exist_ok=True,
    )
    write_completion(run_dir, time.perf_counter() - training_start, cuda_index, resumed=False)


if __name__ == "__main__":
    main()
