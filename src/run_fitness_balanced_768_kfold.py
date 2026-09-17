"""Run FITNESS_BALANCED_768 sequentially over Fold 0 -> 4."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from fitness_balanced_768_protocol import (
    EXPERIMENT, LOG_DIR, RUN_PREFIX, expected_protocol, prepare_configs, run_dir,
    validate_saved_protocol,
)
from train_yolo26s_p2 import ROOT


def fold_action(fold: int, resume_existing: bool) -> str:
    target = run_dir(fold)
    if (target / "training_completed.json").is_file():
        return "skip" if resume_existing else "exists"
    if not target.exists():
        return "new"
    if resume_existing and (target / "weights/last.pt").is_file():
        validate_saved_protocol(fold)
        return "resume"
    return "exists"


def command_for_fold(fold: int, device: str, resume: bool) -> list[str]:
    command = [sys.executable, str(ROOT / "src/train_fitness_balanced_768_fold.py"),
               "--fold", str(fold), "--device", device]
    if resume:
        command.append("--resume")
    return command


def print_status() -> None:
    rows = []
    for fold in range(5):
        target = run_dir(fold)
        metadata = target / "weights/best_balanced.json"
        rows.append({
            "fold": fold,
            "status": "completed" if (target / "training_completed.json").is_file()
                      else "started_not_complete" if target.exists() else "not_started",
            "best_balanced": json.loads(metadata.read_text()) if metadata.is_file() else None,
        })
    print(json.dumps({"experiment": EXPERIMENT, "folds": rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print_status()
        return
    prepare_configs()
    actions = [fold_action(fold, args.resume_existing) for fold in range(5)]
    if "exists" in actions:
        raise FileExistsError("A FITNESS_BALANCED output exists but is not eligible for requested resume")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    for fold, action in enumerate(actions):
        if action == "skip":
            continue
        log = LOG_DIR / ("fold_%d.log" % fold)
        mode = "a" if action == "resume" else "x"
        with log.open(mode, encoding="utf-8") as stream:
            process = subprocess.run(
                command_for_fold(fold, args.device, action == "resume"),
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
        if process.returncode or not (run_dir(fold) / "training_completed.json").is_file():
            raise RuntimeError("FITNESS_BALANCED Fold %d failed; inspect %s" % (fold, log))
    print_status()


if __name__ == "__main__":
    main()

