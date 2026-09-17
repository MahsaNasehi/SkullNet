"""Run the frozen 1024-only five-fold ablation without reading fixed test."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from train_yolo26s_p2 import ROOT, sha256, split_audit


DATA_DIR = ROOT / "data_prepared/skull_hu800_ww1600"
CONFIG_DIR = DATA_DIR / "run_b_1024"
RUN_PREFIX = "yolo26s_p2_hu800_ww1600_run_b_1024_fold"


def command_for_fold(fold: int, args: argparse.Namespace) -> list[str]:
    last = ROOT / "outputs" / f"{RUN_PREFIX}{fold}/weights/last.pt"
    if getattr(args, "resume_existing", False) and last.is_file():
        return [sys.executable, str(ROOT / "src/train_yolo26s_p2.py"), "--resume", str(last)]
    return [
        sys.executable, str(ROOT / "src/train_yolo26s_p2.py"),
        "--data", str(CONFIG_DIR / f"fold_{fold}.yaml"),
        "--name", f"{RUN_PREFIX}{fold}",
        "--epochs", str(args.epochs), "--imgsz", "1024", "--batch", str(args.batch),
        "--workers", "0", "--device", args.device, "--patience", str(args.patience),
        "--seed", str(42 + fold),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="all", help="0..4 or all")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--parallel", type=int, choices=(1, 2), default=2)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    if args.batch != 16 or args.epochs != 150 or args.patience != 30:
        raise ValueError("Frozen Run B requires batch=16, epochs=150 and patience=30")
    folds = list(range(5)) if args.fold == "all" else [int(args.fold)]
    if any(fold not in range(5) for fold in folds):
        raise ValueError("--fold must be 0..4 or all")

    split_records = {}
    for fold in range(5):
        sanitized = CONFIG_DIR / f"fold_{fold}.yaml"
        original = DATA_DIR / f"kfold/fold_{fold}.yaml"
        if not sanitized.is_file() or not original.is_file():
            raise FileNotFoundError(sanitized if not sanitized.is_file() else original)
        clean_audit, original_audit = split_audit(sanitized), split_audit(original)
        if clean_audit != original_audit:
            raise ValueError(f"Run B membership differs from Run A fold {fold}")
        split_records[str(fold)] = clean_audit

    for fold in folds:
        run_dir = ROOT / "outputs" / f"{RUN_PREFIX}{fold}"
        if run_dir.exists() and not (args.resume_existing and (run_dir / "weights/last.pt").is_file()):
            raise FileExistsError(f"Run B output already exists; refusing overwrite: {run_dir}")
    log_dir = ROOT / "outputs/run_b_1024_logs"
    if (not args.resume_existing and log_dir.exists()
            and any((log_dir / f"fold_{fold}.log").exists() for fold in folds)):
        raise FileExistsError(f"Run B log already exists; refusing overwrite: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "status": "frozen_before_training", "experiment": "B_1024_no_hnm",
        "hypothesis": "1024 canvas may improve feature-map sampling; no new physical CT information",
        "only_primary_change_from_A": {"train_imgsz": [768, 1024], "inference_imgsz": [768, 1024]},
        "fixed": {"architecture": "YOLO26s-P2", "batch": 16, "epochs": 150, "patience": 30,
                  "workers": 0, "fold_seeds": {str(f): 42 + f for f in range(5)},
                  "preprocessing": "HU WL800/WW1600, 2.5D +/-5mm, stored 512x512 PNG",
                  "hnm": False, "balanced_sampling": False, "spatial_filter": False,
                  "verifier": False, "new_calibration": False, "aggregator_change": False},
        "source_coco_sha256": sha256(ROOT / "weights/yolo26s.pt"),
        "architecture_sha256": sha256(ROOT / "configs/yolo26s-p2.yaml"),
        "splits": split_records,
        "fixed_test_policy": "not present in sanitized YAML; never read",
        "run_a_initialization_limitation": (
            "Run A recorded the COCO source hash but not each pre-training random P2 checkpoint hash; "
            "exact per-fold random P2 initialization cannot be reconstructed. Run B persists it."
        ),
        "commands": {str(f): command_for_fold(f, args) for f in folds},
    }
    protocol_path = ROOT / "outputs/run_b_1024_protocol.json"
    if protocol_path.exists() and not args.resume_existing:
        raise FileExistsError(protocol_path)
    if not protocol_path.exists():
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    start_all = time.time()
    for start in range(0, len(folds), args.parallel):
        wave = folds[start:start + args.parallel]
        running = []
        for fold in wave:
            log_path = log_dir / f"fold_{fold}.log"
            stream = log_path.open("a" if args.resume_existing and log_path.exists() else "x", encoding="utf-8")
            command = command_for_fold(fold, args)
            print(f"Starting Run B fold {fold}, log={log_path}", flush=True)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            running.append((fold, process, stream, log_path))
        failures = []
        for fold, process, stream, log_path in running:
            code = process.wait()
            stream.close()
            marker = ROOT / "outputs" / f"{RUN_PREFIX}{fold}/training_completed.json"
            print(f"Run B fold {fold} exit={code}, completion_marker={marker.is_file()}", flush=True)
            if code or not marker.is_file():
                failures.append({"fold": fold, "exit_code": code, "log": str(log_path),
                                 "completion_marker": marker.is_file()})
        if failures:
            failure_path = ROOT / "outputs/run_b_1024_failure.json"
            failure_path.write_text(json.dumps({
                "failures": failures, "automatic_batch_reduction": False,
                "instruction": "Inspect and record OOM before any explicit protocol change."
            }, indent=2) + "\n")
            raise RuntimeError(f"Run B wave failed: {failures}")
    completion = {
        "status": "all_five_folds_completed", "wall_seconds": time.time() - start_all,
        "fold_completion_markers": {
            str(f): str(ROOT / "outputs" / f"{RUN_PREFIX}{f}/training_completed.json") for f in folds
        },
    }
    (ROOT / "outputs/run_b_1024_all_completed.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
