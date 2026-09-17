"""Run one balanced-sampling ablation, then evaluate fold 0 and compare to baseline.

Use nohup to detach. An advisory lock prevents duplicate launches. No fixed-test
inference, threshold search selection, or subsequent-fold training is performed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "yolo26s_p2_hu800_ww1600_fold0_balanced25"


def record_event(stage, detail):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with (ROOT / "agent.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n### ثبت خودکار B25 — {stage} — {timestamp}\n\n{detail}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    run = ROOT / "outputs" / NAME
    lock_path = ROOT / "outputs" / (NAME + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This experiment is already running (lock held)")
        import torch
        if args.device != "cpu" and not torch.cuda.is_available():
            raise SystemExit("CUDA is not accessible in this environment. Run this command in the GPU server terminal.")
        if not args.evaluate_only and run.exists() and not args.resume:
            raise SystemExit(f"Run already exists: {run}. Use --resume for unfinished training, or --evaluate-only.")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        def execute(command):
            print("Running:", " ".join(command), flush=True)
            # The child keeps the lock if the supervising process is interrupted.
            try:
                subprocess.run(command, cwd=ROOT, env=env, check=True, pass_fds=(lock.fileno(),))
            except subprocess.CalledProcessError as error:
                record_event("خطای اجرا", f"خروج با کد {error.returncode}. فرمان: `{' '.join(command)}`. نتیجه این مرحله کامل نشده است؛ جزئیات در لاگ آزمایش است.")
                raise
        train = [sys.executable, str(ROOT / "src/train_yolo26s_p2.py")]
        record_event("شروع runner", f"PID={os.getpid()}؛ device={args.device}؛ resume={args.resume}؛ evaluate_only={args.evaluate_only}. مسیر اجرا: `{run}`. فرضیه: افزایش سهم مثبت‌ها با توازن بیمار، حساسیت مطالعه را بهتر کند؛ هنوز نتیجه آموزشی جدیدی موجود نیست.")
        if not args.evaluate_only:
            if args.resume:
                execute(train + ["--resume", str(run / "weights/last.pt")])
            else:
                execute(train + ["--data", str(ROOT / "data_prepared/skull_hu800_ww1600/kfold/fold_0.yaml"),
                                 "--name", NAME, "--device", args.device, "--batch", "16",
                                 "--imgsz", "768", "--epochs", "150", "--patience", "30",
                                 "--workers", "0", "--seed", "42", "--positive-fraction", "0.25"])
            record_event("پایان آموزش", "فرایند آموزش با کد صفر تمام شد؛ ارزیابی مستقل از loader آموزش روی validation همان Fold 0 آغاز می‌شود. این رخداد هنوز به معنی بهبود دقت نیست.")
        output = run / "study_evaluation"
        execute([sys.executable, str(ROOT / "src/evaluate_fold_qwk.py"), "--fold", "0",
                 "--weights", str(run / "weights/best.pt"), "--output", str(output),
                 "--device", args.device, "--batch", "1", "--imgsz", "768",
                 "--aggregator", "top3_mean", "--threshold", "0.5"])
        evaluated = json.loads((output / "metrics.json").read_text())
        record_event("پایان ارزیابی validation", "خروجی واقعی معیارهای مطالعه و QWK ایزوله (ICH/MLS واقعی):\n\n```json\n" + json.dumps({"fracture": evaluated["fracture_study_metrics"], "isolated_qwk": evaluated["official_triage_qwk_isolated_fracture"]}, indent=2) + "\n```\n\nاین امتیاز نتیجه Fold 0 است؛ پنج‌فولدی یا test نهایی نیست.")
        baseline_dir = ROOT / "outputs/yolo26s_p2_hu800_ww1600_fold0/study_evaluation"
        if not (baseline_dir / "metrics.json").is_file():
            print("Evaluation saved; baseline study_evaluation is missing, comparison skipped", flush=True)
            return
        import pandas as pd
        baseline_studies = pd.read_csv(baseline_dir / "study_predictions.csv", dtype={"study_id": str})
        new_studies = pd.read_csv(output / "study_predictions.csv", dtype={"study_id": str})
        if set(baseline_studies.study_id) != set(new_studies.study_id):
            raise ValueError("Baseline and balanced experiment evaluated different studies")
        baseline = json.loads((baseline_dir / "metrics.json").read_text())
        current = json.loads((output / "metrics.json").read_text())
        if baseline["aggregator"] != current["aggregator"] or baseline["fracture_study_metrics"]["threshold"] != .5:
            raise ValueError("Baseline aggregation or threshold differs")
        fields = ["tp", "fp", "fn", "tn", "sensitivity", "specificity", "precision", "study_pr_auc", "study_roc_auc"]
        comparison = {"protocol": "exploratory fold-0 development ablation; not final test or five-fold evidence",
                      "baseline_report": str(baseline_dir / "metrics.json"), "experiment_report": str(output / "metrics.json"),
                      "metrics": {k: {"baseline": baseline["fracture_study_metrics"][k],
                                      "balanced25": current["fracture_study_metrics"][k]} for k in fields},
                      "isolated_triage_qwk": {"baseline": baseline["official_triage_qwk_isolated_fracture"],
                                               "balanced25": current["official_triage_qwk_isolated_fracture"]},
                      "initialization_note": "New P2 initialization is explicitly seeded; historical baseline did not seed before initialization. Confirm small gains with matched-seed controls."}
        (run / "comparison_to_baseline.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison, indent=2), flush=True)
        record_event("مقایسه با baseline", "نتیجه مقایسه در `outputs/" + NAME + "/comparison_to_baseline.json` ذخیره شد.\n\n```json\n" + json.dumps(comparison, indent=2) + "\n```")
        (run / "experiment_complete.json").write_text(json.dumps({"evaluation_and_comparison_complete": True}) + "\n")


if __name__ == "__main__":
    main()
