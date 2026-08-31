"""Offline, fold-specific Ultralytics training entry point."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from fracture.utils.config import load_config, save_config
from fracture.utils.pretrained import (
    resolve_initialization,
    sha256_file,
    verify_p2_architecture,
    verify_transfer,
)
from fracture.utils.seed import seed_everything


EXPERIMENT_FIELDS = [
    "experiment_id", "experiment_name", "run_name", "timestamp", "fold", "architecture",
    "input_mode", "image_size", "seed", "initialization_mode", "pretrained",
    "pretrained_source", "pretrained_dataset", "pretrained_checkpoint", "pretrained_sha256",
    "weight_origin", "transferred_items", "transfer_fraction", "transferred_parameters",
    "newly_initialized_items", "p2_stride4_active", "detection_strides",
    "external_resource_approval_reference", "annotation_version", "config_sha256",
    "dataset_manifest_sha256", "window_level", "window_width", "augmentation_config",
    "optimizer", "learning_rate", "batch", "epochs", "best_epoch",
    "mAP50", "mAP50_95", "detection_precision", "detection_recall", "study_AUROC",
    "study_PR_AUC", "sensitivity_at_0_5", "specificity_at_0_5", "precision_at_0_5",
    "F1_at_0_5", "Brier", "log_loss", "TP", "TN", "FP", "FN",
    "isolated_fracture_QWK", "training_seconds", "average_inference_time", "peak_VRAM", "notes",
]


def _append_experiment(
    cfg: dict,
    fold: int,
    run_name: str,
    result: object,
    initialization: dict,
    training_seconds: float,
) -> None:
    metrics = getattr(result, "results_dict", {}) or {}
    try:
        import torch

        peak_vram = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except ImportError:
        peak_vram = 0
    model = cfg["model"]
    prep = cfg["preprocessing"]
    checkpoint_metadata = initialization.get("checkpoint_metadata") or {}
    transfer = initialization.get("transfer") or {}
    results_path = Path(str(getattr(result, "save_dir", ""))) / "results.csv"
    best_row: dict[str, str] = {}
    if results_path.is_file():
        with results_path.open(newline="", encoding="utf-8") as stream:
            result_rows = list(csv.DictReader(stream))
        if result_rows:
            best_row = max(result_rows, key=lambda item: float(item["metrics/mAP50-95(B)"]))
    train = cfg["training"]
    dataset = initialization.get("dataset") or {}
    init_cfg = model.get("initialization") or {}
    row = {
        "experiment_id": f"fold{fold}_{run_name}",
        "experiment_name": run_name,
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fold": fold,
        "architecture": model.get("architecture", ""),
        "input_mode": prep["input_mode"],
        "image_size": prep["image_size"],
        "seed": train.get("seed", 42),
        "initialization_mode": initialization.get("mode", ""),
        "pretrained": initialization.get("mode") == "pretrained",
        "pretrained_source": checkpoint_metadata.get("source", init_cfg.get("pretrained_source", "")),
        "pretrained_dataset": checkpoint_metadata.get("pretraining_dataset", ""),
        "pretrained_checkpoint": checkpoint_metadata.get("local_path", ""),
        "pretrained_sha256": checkpoint_metadata.get("sha256", ""),
        "weight_origin": model.get("weight_origin", ""),
        "transferred_items": transfer.get("transferred_items", ""),
        "transfer_fraction": transfer.get("transfer_fraction", ""),
        "transferred_parameters": transfer.get("transferred_parameters", ""),
        "newly_initialized_items": transfer.get("newly_initialized_items", ""),
        "p2_stride4_active": initialization.get("p2_stride4_active", ""),
        "detection_strides": json.dumps(initialization.get("detection_strides", [])),
        "external_resource_approval_reference": model.get("external_resource_approval_reference") or "",
        "annotation_version": cfg["data"]["annotation_version"],
        "config_sha256": initialization.get("config_sha256", ""),
        "dataset_manifest_sha256": dataset.get("manifest_sha256", ""),
        "window_level": prep["window_level"],
        "window_width": prep["window_width"],
        "augmentation_config": json.dumps(train.get("augmentations", {}), sort_keys=True),
        "optimizer": train.get("optimizer", ""),
        "learning_rate": train.get("lr0", ""),
        "batch": train.get("batch_size", ""),
        "epochs": train.get("epochs", ""),
        "best_epoch": best_row.get("epoch", ""),
        "mAP50": best_row.get("metrics/mAP50(B)", metrics.get("metrics/mAP50(B)", "")),
        "mAP50_95": best_row.get("metrics/mAP50-95(B)", metrics.get("metrics/mAP50-95(B)", "")),
        "detection_precision": best_row.get("metrics/precision(B)", metrics.get("metrics/precision(B)", "")),
        "detection_recall": best_row.get("metrics/recall(B)", metrics.get("metrics/recall(B)", "")),
        "study_AUROC": "",
        "study_PR_AUC": "",
        "sensitivity_at_0_5": "",
        "specificity_at_0_5": "",
        "precision_at_0_5": "",
        "F1_at_0_5": "",
        "Brier": "",
        "log_loss": "",
        "TP": "", "TN": "", "FP": "", "FN": "",
        "isolated_fracture_QWK": "",
        "training_seconds": training_seconds,
        "average_inference_time": "",
        "peak_VRAM": peak_vram,
        "notes": "Study-level fields require held-out predict_fold evaluation.",
    }
    path = Path("reports/experiments.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    if path.is_file() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != EXPERIMENT_FIELDS:
                existing_rows = list(reader)
    if existing_rows or (path.is_file() and path.stat().st_size and path.read_text(encoding="utf-8").splitlines()[0].split(",") != EXPERIMENT_FIELDS):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=EXPERIMENT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPERIMENT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dataset_provenance(cfg: dict, dataset_yaml: str | Path) -> dict[str, object]:
    yaml_path = Path(dataset_yaml).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    dataset_root = (Path(cfg["data"]["output_root"]) / cfg["data"]["annotation_version"]).resolve()
    if yaml_path.parent != dataset_root:
        raise ValueError(f"Dataset YAML is not from configured immutable dataset version: {yaml_path} != {dataset_root}")
    manifest_path = dataset_root / "manifest.csv"
    manifest_hash_path = dataset_root / "manifest.sha256"
    if not manifest_path.is_file() or not manifest_hash_path.is_file():
        raise FileNotFoundError(f"Dataset manifest or checksum is missing under {dataset_root}")
    manifest_digest = sha256_file(manifest_path)
    expected_digest = manifest_hash_path.read_text(encoding="utf-8").split()[0]
    if manifest_digest != expected_digest:
        raise RuntimeError(f"Dataset manifest SHA256 mismatch: expected {expected_digest}, got {manifest_digest}")
    split_path = Path(cfg["split"].get("path", "splits/folds.json")).resolve()
    return {
        "dataset_yaml": str(yaml_path),
        "dataset_yaml_sha256": sha256_file(yaml_path) if yaml_path.is_file() else None,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_digest,
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path) if split_path.is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--dataset-yaml", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--run-name", default="detector")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume an interrupted Ultralytics run from a local last.pt checkpoint.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["training"].get("seed", 42))
    model_cfg = cfg["model"]
    weights = model_cfg.get("weights")
    if not weights:
        raise ValueError("model.weights must point to a local architecture YAML or weights file")
    if not Path(weights).is_file():
        raise FileNotFoundError(weights)

    resume_checkpoint = args.resume_from.resolve() if args.resume_from else None
    if resume_checkpoint and not resume_checkpoint.is_file():
        raise FileNotFoundError(resume_checkpoint)

    # Resolve/download development-time initialization before enforcing offline mode.
    plan = None if resume_checkpoint else resolve_initialization(model_cfg)
    os.environ["YOLO_OFFLINE"] = "1"
    from ultralytics import YOLO
    from ultralytics.data import utils as ultralytics_data_utils
    import torch, ultralytics

    # Ultralytics checks for Arial while validating every detection dataset and
    # downloads it when absent. Use the installed system font to keep training
    # strictly offline; plots are disabled for this memory-constrained run.
    system_font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not system_font.is_file():
        raise FileNotFoundError(f"Required offline system font is missing: {system_font}")
    ultralytics_data_utils.check_font = lambda _font="Arial.ttf": system_font

    print(f"Python={platform.python_version()} torch={torch.__version__} CUDA={torch.version.cuda} ultralytics={ultralytics.__version__}")
    output = (Path("outputs") / f"fold_{args.fold}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output / f"{args.run_name}_resolved_config.yaml")
    started = time.time()
    model = YOLO(str(resume_checkpoint or weights))
    strides = verify_p2_architecture(model)
    configured_mode = str((model_cfg.get("initialization") or {}).get("mode") or "").strip() or None
    initialization_report: dict[str, object] = {
        "mode": configured_mode if resume_checkpoint and configured_mode else "resume" if resume_checkpoint else plan.mode,
        "resumed_from_checkpoint": bool(resume_checkpoint),
        "target_architecture": str(Path(weights).resolve()),
        "detection_strides": strides,
        "p2_stride4_active": 4.0 in strides,
        "checkpoint_path": str(resume_checkpoint) if resume_checkpoint else None,
        "checkpoint_metadata": ({
            "name": resume_checkpoint.name,
            "local_path": str(resume_checkpoint),
            "source": "resumed_training_checkpoint",
            "sha256": sha256_file(resume_checkpoint),
            "file_size_bytes": resume_checkpoint.stat().st_size,
        } if resume_checkpoint else None),
        "transfer": None,
    }
    if plan and plan.checkpoint_path:
        print(f"Loading compatible parameters for {plan.mode}: {plan.checkpoint_path}")
        initialization_report["checkpoint_path"] = str(plan.checkpoint_path)
        initialization_report["checkpoint_metadata"] = plan.checkpoint_metadata
        initialization_report["transfer"] = verify_transfer(model, plan.checkpoint_path)
    elif plan:
        initialization_report["checkpoint_metadata"] = plan.checkpoint_metadata
    initialization_report["runtime"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    initialization_report["git_commit"] = _git_commit()
    initialization_report["config_path"] = str(Path(args.config).resolve())
    initialization_report["config_sha256"] = sha256_file(args.config)
    initialization_report["dataset"] = _dataset_provenance(cfg, args.dataset_yaml)
    initialization_report_path = output / f"{args.run_name}_initialization.json"
    initialization_report_path.write_text(
        json.dumps(initialization_report, indent=2) + "\n",
        encoding="utf-8",
    )

    # Ultralytics treats pretrained=False as an instruction to rebuild a model
    # created from YAML without the weights already loaded by verify_transfer().
    # Guard one known transferred tensor at the last callback before training so
    # an API behavior change cannot silently turn an initialization experiment
    # into seeded random initialization again.
    has_transferred_initialization = bool(plan and plan.checkpoint_path)
    if has_transferred_initialization:
        backbone_key = str(initialization_report["transfer"]["backbone_tensor_key"])
        expected_backbone = model.model.state_dict()[backbone_key].detach().cpu().clone()

        def verify_training_initialization(trainer: object) -> None:
            live_state = trainer.model.state_dict()
            live_backbone = live_state.get(backbone_key)
            if live_backbone is None or not torch.equal(expected_backbone, live_backbone.detach().cpu()):
                raise RuntimeError(
                    "Transferred initialization was not preserved by the Ultralytics trainer; "
                    "aborting before the first training epoch"
                )
            initialization_report["training_initialization_verified"] = True
            initialization_report_path.write_text(
                json.dumps(initialization_report, indent=2) + "\n",
                encoding="utf-8",
            )

        initialization_report["training_initialization_verified"] = False
        initialization_report_path.write_text(
            json.dumps(initialization_report, indent=2) + "\n",
            encoding="utf-8",
        )
        model.add_callback("on_pretrain_routine_end", verify_training_initialization)
    train = cfg["training"]
    augmentations = train.get("augmentations", {})
    if resume_checkpoint:
        # The checkpoint contains the original dataset, optimizer, scheduler,
        # epoch, output directory, and augmentation arguments.
        result = model.train(resume=True)
    else:
        train_kwargs = {
            "data": args.dataset_yaml,
            "epochs": args.epochs or train["epochs"],
            "imgsz": cfg["preprocessing"]["image_size"],
            "batch": args.batch_size or train["batch_size"],
            "workers": args.workers if args.workers is not None else train["workers"],
            "patience": train["patience"],
            "device": train["device"],
            "seed": train.get("seed", 42),
            "project": str(output),
            "name": args.run_name,
            "exist_ok": False,
            # True here means "preserve the already-loaded in-memory model" in
            # Ultralytics Model.train(); it does not download another checkpoint.
            "pretrained": has_transferred_initialization,
            "amp": bool(train.get("amp", False)),
            "plots": bool(train.get("plots", False)),
            "cache": False,
        }
        for key in (
            "optimizer", "lr0", "lrf", "cos_lr", "warmup_epochs",
            "weight_decay", "close_mosaic", "box", "cls", "dfl",
            "save_period",
        ):
            if key in train:
                train_kwargs[key] = train[key]
        result = model.train(**train_kwargs, **augmentations)
    training_seconds = time.time() - started
    with (output / f"{args.run_name}_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fold", "training_seconds", "annotation_version", "best_model"]); writer.writeheader(); writer.writerow({"fold": args.fold, "training_seconds": training_seconds, "annotation_version": cfg["data"]["annotation_version"], "best_model": str(getattr(result, "save_dir", ""))})
    _append_experiment(cfg, args.fold, args.run_name, result, initialization_report, training_seconds)

if __name__ == "__main__": main()
