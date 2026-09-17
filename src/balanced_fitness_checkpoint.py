"""Project-local secondary checkpoint selector; never changes trainer fitness."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ultralytics.models.yolo.detect.train import DetectionTrainer

from fitness_balanced_768_protocol import FITNESS_WEIGHTS, balanced_fitness
from train_yolo26s_p2 import sha256


class BalancedCheckpointSelector:
    def __init__(self, save_dir: Path, metadata_name: str = "best_balanced.json", metadata_in_weights: bool = True):
        self.weights_dir = Path(save_dir) / "weights"
        self.checkpoint = self.weights_dir / "best_balanced.pt"
        self.metadata = (self.weights_dir if metadata_in_weights else Path(save_dir)) / metadata_name
        self.best = float("-inf")
        if self.metadata.is_file():
            self.best = float(json.loads(self.metadata.read_text(encoding="utf-8"))["balanced_fitness"])

    def __call__(self, trainer) -> None:
        # BaseTrainer.final_eval emits on_fit_epoch_end once more at epoch==epochs;
        # exclude that event because trainer.last then refers to the final training epoch.
        if int(trainer.epoch) >= int(trainer.epochs):
            return
        score = balanced_fitness(trainer.metrics)
        if score <= self.best:
            return
        source = Path(trainer.last)
        if not source.is_file():
            raise FileNotFoundError("Current-epoch last.pt is unavailable: %s" % source)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.weights_dir / ".best_balanced.pt.tmp"
        shutil.copy2(source, temporary)
        temporary.replace(self.checkpoint)
        values = {
            "epoch": int(trainer.epoch),
            "epoch_one_based": int(trainer.epoch) + 1,
            "precision": float(trainer.metrics["metrics/precision(B)"]),
            "recall": float(trainer.metrics["metrics/recall(B)"]),
            "mAP50": float(trainer.metrics["metrics/mAP50(B)"]),
            "mAP50_95": float(trainer.metrics["metrics/mAP50-95(B)"]),
            "balanced_fitness": score,
            "fitness_weights": FITNESS_WEIGHTS,
            "checkpoint": str(self.checkpoint.resolve()),
            "checkpoint_sha256": sha256(self.checkpoint),
            "normal_best_and_early_stopping_unchanged": True,
        }
        temporary_json = self.weights_dir / ".best_balanced.json.tmp"
        temporary_json.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
        temporary_json.replace(self.metadata)
        self.best = score


class BalancedCheckpointTrainer(DetectionTrainer):
    """Base DetectionTrainer plus one post-save observer callback."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.balanced_checkpoint_selector = BalancedCheckpointSelector(Path(self.save_dir))
        self.add_callback("on_fit_epoch_end", self.balanced_checkpoint_selector)


class FixedBatchBalancedCheckpointTrainer(BalancedCheckpointTrainer):
    """Balanced selector with a hard failure instead of automatic batch reduction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This experiment's requested metadata filename is explicit and isolated.
        self.callbacks["on_fit_epoch_end"].remove(self.balanced_checkpoint_selector)
        self.balanced_checkpoint_selector = BalancedCheckpointSelector(
            Path(self.save_dir), metadata_name="best_balanced_metrics.json", metadata_in_weights=False
        )
        self.add_callback("on_fit_epoch_end", self.balanced_checkpoint_selector)

    def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode="train"):
        if mode == "train" and batch_size != 16:
            raise RuntimeError("MOSAIC08_SCALE05 protocol forbids automatic batch reduction from batch=16")
        return super().get_dataloader(dataset_path, batch_size, rank, mode)
