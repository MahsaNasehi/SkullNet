"""Reliably run AUG_MILD_768 one Fold at a time: 0 -> 1 -> 2 -> 3 -> 4."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from aug_mild_768_protocol import (
    ALL_COMPLETED, ARCHITECTURE_SHA256, COCO_SHA256, EXPERIMENT, GLOBAL_PROTOCOL,
    LOG_DIR, RUN_PREFIX, augmentation_diff, completion_is_valid,
    expected_fold_protocol, validate_data_configs, validate_static_protocol,
)
from train_yolo26s_p2 import ROOT


def run_dir_for_fold(fold: int) -> Path:
    return ROOT / "outputs" / (RUN_PREFIX + str(fold))


def command_for_fold(fold: int, device: str, resume: bool) -> list:
    command = [
        sys.executable, str(ROOT / "src/train_aug_mild_768_fold.py"),
        "--fold", str(fold), "--device", device,
    ]
    if resume:
        command.append("--resume")
    return command


def decide_fold_action(fold: int, resume_existing: bool) -> str:
    run_dir = run_dir_for_fold(fold)
    if completion_is_valid(run_dir, fold):
        if resume_existing:
            return "skip_completed"
        raise FileExistsError("Completed output exists; use --resume-existing to skip it: %s" % run_dir)
    if not run_dir.exists():
        return "new"
    if not resume_existing:
        raise FileExistsError("Incomplete output exists; use --resume-existing: %s" % run_dir)
    if not (run_dir / "weights/last.pt").is_file():
        raise RuntimeError("Incomplete Fold has no valid last.pt path: %s" % run_dir)
    return "resume"


def write_global_protocol() -> None:
    validate_static_protocol()
    splits = validate_data_configs()
    payload = {
        "status": "frozen_before_training", "experiment": EXPERIMENT,
        "execution_order": [0, 1, 2, 3, 4], "parallel_folds": 1,
        "run_prefix": RUN_PREFIX, "architecture_sha256": ARCHITECTURE_SHA256,
        "source_coco_sha256": COCO_SHA256, "splits": splits,
        "parameter_diff_from_historical_run_a": augmentation_diff(),
        "fold_protocols": {str(fold): expected_fold_protocol(fold) for fold in range(5)},
        "automatic_batch_reduction": False,
        "initialization_confound": expected_fold_protocol(0)["initialization_confound"],
    }
    if GLOBAL_PROTOCOL.exists():
        existing = json.loads(GLOBAL_PROTOCOL.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("Existing global AUG_MILD_768 protocol differs")
    else:
        GLOBAL_PROTOCOL.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    if args.device != "0":
        raise ValueError("Frozen manual protocol requires --device 0")
    write_global_protocol()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    start = time.time()
    for fold in range(5):
        action = decide_fold_action(fold, args.resume_existing)
        if action == "skip_completed":
            print("AUG_MILD_768 Fold %d already complete; skipping." % fold, flush=True)
            continue
        log_path = LOG_DIR / ("fold_%d.log" % fold)
        mode = "a" if action == "resume" else "x"
        command = command_for_fold(fold, args.device, resume=action == "resume")
        print("Starting AUG_MILD_768 Fold %d (%s), log=%s" % (fold, action, log_path), flush=True)
        with log_path.open(mode, encoding="utf-8") as stream:
            process = subprocess.run(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode != 0 or not completion_is_valid(run_dir_for_fold(fold), fold):
            failure = {
                "experiment": EXPERIMENT, "fold": fold, "action": action,
                "exit_code": process.returncode, "log": str(log_path),
                "automatic_batch_reduction": False,
            }
            (ROOT / "outputs/aug_mild_768_failure.json").write_text(
                json.dumps(failure, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError("AUG_MILD_768 Fold failed: %s" % failure)
    if not all(completion_is_valid(run_dir_for_fold(fold), fold) for fold in range(5)):
        raise RuntimeError("Not all five AUG_MILD_768 Folds have valid completion artifacts")
    ALL_COMPLETED.write_text(json.dumps({
        "status": "all_five_folds_completed", "experiment": EXPERIMENT,
        "folds": [0, 1, 2, 3, 4], "wall_seconds_this_runner": time.time() - start,
    }, indent=2) + "\n", encoding="utf-8")
    print("All five AUG_MILD_768 Folds completed.", flush=True)


if __name__ == "__main__":
    main()
