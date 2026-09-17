"""Train one or all patient-grouped YOLO26s-P2 folds, optionally in parallel."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="all", help="0..4 or all")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()

    folds = list(range(5)) if args.fold == "all" else [int(args.fold)]
    if any(fold not in range(5) for fold in folds):
        raise ValueError("--fold must be 0, 1, 2, 3, 4, or all")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    def fold_command(fold: int) -> tuple[list[str], bool]:
        data = ROOT / "data_prepared/skull_hu800_ww1600/kfold" / f"fold_{fold}.yaml"
        if not data.is_file():
            raise FileNotFoundError(f"{data} not found; run src/prepare_yolo26_kfold.py first")
        last = ROOT / "outputs" / f"yolo26s_p2_hu800_ww1600_fold{fold}" / "weights/last.pt"
        if args.resume_existing and last.is_file():
            return [
                sys.executable,
                str(ROOT / "src/train_yolo26s_p2.py"),
                "--resume", str(last),
            ], True
        command = [
            sys.executable,
            str(ROOT / "src/train_yolo26s_p2.py"),
            "--data", str(data),
            "--name", f"yolo26s_p2_hu800_ww1600_fold{fold}",
            "--epochs", str(args.epochs),
            "--imgsz", str(args.imgsz),
            "--batch", str(args.batch),
            "--workers", str(args.workers),
            "--device", args.device,
            "--patience", str(args.patience),
            "--seed", str(42 + fold),
        ]
        if args.exist_ok:
            command.append("--exist-ok")
        return command, False

    # Start folds in fixed-size waves: with --parallel 2 this is (0,1),
    # then (2,3), then (4). Each child has a separate log to avoid interleaving.
    log_root = ROOT / "outputs" / "kfold_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(folds), args.parallel):
        wave = folds[start : start + args.parallel]
        running: list[tuple[int, subprocess.Popen, object, Path]] = []
        for fold in wave:
            command, resuming = fold_command(fold)
            log_path = log_root / f"fold_{fold}.log"
            stream = log_path.open("a" if resuming else "w", encoding="utf-8")
            print(
                f"Starting fold {fold}/4 ({'resume' if resuming else 'new'}), log={log_path}",
                flush=True,
            )
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            running.append((fold, process, stream, log_path))
        failures: list[str] = []
        for fold, process, stream, log_path in running:
            return_code = process.wait()
            stream.close()
            print(f"Fold {fold} exited with code {return_code}", flush=True)
            if return_code:
                failures.append(f"fold {fold}: exit={return_code}, log={log_path}")
        if failures:
            raise RuntimeError("K-fold wave failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
