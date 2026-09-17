import inspect
import json
import tempfile
import unittest
from pathlib import Path

from ultralytics.models.yolo.detect.train import DetectionTrainer

from balanced_fitness_checkpoint import (
    BalancedCheckpointSelector, FixedBatchBalancedCheckpointTrainer,
)
from fitness_balanced_768_protocol import FITNESS_WEIGHTS, balanced_fitness
from mosaic08_scale05_balanced_fold4_protocol import (
    DATA_YAML, MAX_EPOCHS, OUTPUT_DIR, SOURCE_CHECKPOINT, SOURCE_SHA256,
    TRAIN_KWARGS, prepare_config, sha256, validate_sources,
)
from train_mosaic08_scale05_balanced_fold4 import train_new


class Mosaic08Scale05BalancedFold4Tests(unittest.TestCase):
    def test_source_identity_is_read_only_and_output_isolated(self):
        validate_sources()
        self.assertEqual(sha256(SOURCE_CHECKPOINT), SOURCE_SHA256)
        self.assertFalse(SOURCE_CHECKPOINT.is_relative_to(OUTPUT_DIR))
        self.assertNotEqual(SOURCE_CHECKPOINT.parent.parent, OUTPUT_DIR)

    def test_initialization_is_copy_then_weight_load_with_fresh_optimizer(self):
        source = inspect.getsource(train_new)
        self.assertIn("shutil.copy2(SOURCE_CHECKPOINT, initialization)", source)
        self.assertIn("YOLO(str(initialization)", source)
        self.assertIn("resume=False", source)
        self.assertNotIn("resume=True", source)

    def test_exact_augmentation_and_duration(self):
        self.assertEqual(MAX_EPOCHS, 40)
        self.assertEqual(TRAIN_KWARGS["mosaic"], 0.80)
        self.assertEqual(TRAIN_KWARGS["scale"], 0.50)
        self.assertEqual(TRAIN_KWARGS["degrees"], 5.0)
        self.assertEqual(TRAIN_KWARGS["translate"], 0.05)

    def test_exact_balanced_fitness(self):
        metrics = {
            "metrics/precision(B)": 0.2, "metrics/recall(B)": 0.4,
            "metrics/mAP50(B)": 0.6, "metrics/mAP50-95(B)": 0.8,
        }
        self.assertAlmostEqual(balanced_fitness(metrics), 0.10*0.2 + 0.30*0.4 + 0.10*0.6 + 0.50*0.8)
        self.assertAlmostEqual(sum(FITNESS_WEIGHTS.values()), 1.0)

    def test_best_balanced_strict_improvement_and_requested_metadata_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights").mkdir()
            selector = BalancedCheckpointSelector(
                root, "best_balanced_metrics.json", metadata_in_weights=False
            )
            self.assertEqual(selector.metadata, root / "best_balanced_metrics.json")

            class Trainer:
                epoch = 0
                epochs = 4
                metrics = {key: 0.2 for key in FITNESS_WEIGHTS}

            trainer = Trainer()
            trainer.last = root / "weights/last.pt"
            trainer.last.write_bytes(b"first")
            selector(trainer)
            first = json.loads(selector.metadata.read_text())
            trainer.epoch = 1
            trainer.last.write_bytes(b"worse")
            trainer.metrics = {key: 0.1 for key in FITNESS_WEIGHTS}
            selector(trainer)
            self.assertEqual(selector.checkpoint.read_bytes(), b"first")
            self.assertEqual(json.loads(selector.metadata.read_text()), first)

    def test_default_selection_and_early_stopping_are_unchanged(self):
        self.assertIs(FixedBatchBalancedCheckpointTrainer.save_model, DetectionTrainer.save_model)
        self.assertIs(FixedBatchBalancedCheckpointTrainer.validate, DetectionTrainer.validate)
        self.assertIs(FixedBatchBalancedCheckpointTrainer._do_train, DetectionTrainer._do_train)

    def test_automatic_batch_reduction_hard_fails(self):
        trainer = FixedBatchBalancedCheckpointTrainer.__new__(FixedBatchBalancedCheckpointTrainer)
        with self.assertRaisesRegex(RuntimeError, "forbids automatic batch reduction"):
            trainer.get_dataloader(None, batch_size=8, mode="train")

    def test_fold4_config_matches_lists_and_excludes_fixed_test(self):
        prepare_config()
        text = DATA_YAML.read_text()
        self.assertNotIn("test:", text)
        self.assertIn("fold_4_train.txt", text)
        self.assertIn("fold_4_val.txt", text)


if __name__ == "__main__":
    unittest.main()
