"""Short CPU integration test using annotated train/val subsets (not an accuracy experiment)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

import balanced_yolo_trainer
from train_yolo26s_p2 import main as train


def main():
    torch.set_num_threads(2)
    data_root = ROOT / "data_prepared/skull_hu800_ww1600"
    manifest = pd.read_csv(data_root / "manifest.csv")
    source = yaml.safe_load((data_root / "kfold/fold_0.yaml").read_text())
    def subset(split, positive, negative):
        paths = set(Path(source[split]).read_text().splitlines())
        rows = manifest[manifest.image_path.isin(paths)]
        return rows[rows.num_boxes > 0].head(positive).image_path.tolist() + rows[rows.num_boxes == 0].head(negative).image_path.tolist()
    with tempfile.TemporaryDirectory(prefix="yolo26_balanced_smoke_") as directory:
        scratch = Path(directory)
        for split, counts in (("train", (8, 24)), ("val", (4, 4))):
            file = scratch / (split + ".txt")
            file.write_text("\n".join(subset(split, *counts)) + "\n")
            source[split] = str(file)
        config = scratch / "data.yaml"
        config.write_text(yaml.safe_dump(source))
        run = scratch / "runs/smoke"
        # Interrupt after epoch 0 has saved a resumable checkpoint, then restore
        # via the production CLI. This tests policy serialization and reinstall.
        original_trainer = balanced_yolo_trainer.PatientBalancedDetectionTrainer
        class InterruptedTraining(RuntimeError):
            pass
        def interrupt_after_checkpoint(trainer):
            if trainer.epoch == 1:
                raise InterruptedTraining("Intentional smoke-test interruption")
        class InterruptingTrainer(original_trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.add_callback("on_train_epoch_start", interrupt_after_checkpoint)
        balanced_yolo_trainer.PatientBalancedDetectionTrainer = InterruptingTrainer
        try:
            train(["--data", str(config), "--project", str(scratch / "runs"), "--name", "smoke",
                   "--positive-fraction", ".25", "--epochs", "2", "--batch", "16", "--imgsz", "128",
                   "--workers", "0", "--device", "cpu"])
        except InterruptedTraining:
            pass
        else:
            raise AssertionError("Smoke interruption did not execute")
        finally:
            balanced_yolo_trainer.PatientBalancedDetectionTrainer = original_trainer
        last = run / "weights/last.pt"
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 0 and checkpoint["optimizer"] is not None
        # Main receives no sampler arguments here; it must recover the policy.
        train(["--resume", str(last)])
        results = pd.read_csv(run / "results.csv")
        assert len(results) == 2, results
        model = YOLO(str(run / "weights/best.pt"))
        assert model.model.sampling_policy["positive_fraction"] == .25
        history = [json.loads(s) for s in (run / "sampling_epochs.jsonl").read_text().splitlines()]
        assert [h["epoch"] for h in history] == [0, 1, 1]
        assert history[1]["indices_sha256"] == history[2]["indices_sha256"]
        assert all(h["positive_draws"] == 8 and h["negative_draws"] == 24 for h in history)
        assert all(h["batches"] == 2 for h in history)
        print("SMOKE PASSED: CPU train/val, checkpoint, interrupted resume, identical sampling schedule; 2 epochs, 32 train / 8 val images at 128px.")


if __name__ == "__main__":
    main()
