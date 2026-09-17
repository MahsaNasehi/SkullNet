import json
import tempfile
import unittest
from pathlib import Path

from ultralytics.models.yolo.detect.train import DetectionTrainer

from balanced_fitness_checkpoint import BalancedCheckpointSelector, BalancedCheckpointTrainer
from fitness_balanced_768_protocol import (
    FITNESS_WEIGHTS, RUN_PREFIX, balanced_fitness, data_yaml, expected_protocol,
    prepare_configs, run_dir, validate_configs,
)


class FitnessBalanced768Tests(unittest.TestCase):
    def test_exact_formula_and_unit_sum(self):
        metrics = {
            "metrics/precision(B)": 0.2, "metrics/recall(B)": 0.4,
            "metrics/mAP50(B)": 0.6, "metrics/mAP50-95(B)": 0.8,
        }
        self.assertAlmostEqual(balanced_fitness(metrics), 0.10*0.2 + 0.30*0.4 + 0.10*0.6 + 0.50*0.8)
        self.assertAlmostEqual(sum(FITNESS_WEIGHTS.values()), 1.0)

    def test_normal_best_and_early_stopping_methods_are_inherited(self):
        self.assertIs(BalancedCheckpointTrainer.save_model, DetectionTrainer.save_model)
        self.assertIs(BalancedCheckpointTrainer.validate, DetectionTrainer.validate)
        self.assertIs(BalancedCheckpointTrainer._do_train, DetectionTrainer._do_train)
        self.assertEqual(
            expected_protocol(0)["ultralytics_default_detection_fitness_weights_P_R_mAP50_mAP50_95"],
            [0.0, 0.0, 0.0, 1.0],
        )

    def test_secondary_checkpoint_updates_only_on_strict_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights"
            weights.mkdir()
            last = weights / "last.pt"
            selector = BalancedCheckpointSelector(root)

            class Trainer:
                epoch = 0
                epochs = 3
                metrics = {
                    "metrics/precision(B)": 0.1, "metrics/recall(B)": 0.2,
                    "metrics/mAP50(B)": 0.3, "metrics/mAP50-95(B)": 0.4,
                }

            trainer = Trainer()
            trainer.last = last
            last.write_bytes(b"epoch0")
            selector(trainer)
            first = json.loads(selector.metadata.read_text())
            self.assertEqual(selector.checkpoint.read_bytes(), b"epoch0")

            trainer.epoch = 1
            last.write_bytes(b"epoch1-worse")
            trainer.metrics = {key: 0.0 for key in FITNESS_WEIGHTS}
            selector(trainer)
            self.assertEqual(selector.checkpoint.read_bytes(), b"epoch0")
            self.assertEqual(json.loads(selector.metadata.read_text()), first)

            trainer.epoch = 2
            last.write_bytes(b"epoch2-better")
            trainer.metrics = {key: 0.9 for key in FITNESS_WEIGHTS}
            selector(trainer)
            self.assertEqual(selector.checkpoint.read_bytes(), b"epoch2-better")
            self.assertGreater(json.loads(selector.metadata.read_text())["balanced_fitness"], first["balanced_fitness"])

    def test_configs_match_run_a_and_exclude_test(self):
        prepare_configs()
        records = validate_configs()
        self.assertEqual(len(records), 5)
        for fold in range(5):
            self.assertNotIn("test:", data_yaml(fold).read_text())

    def test_output_prefix_is_isolated(self):
        self.assertIn("fitness_balanced_768", RUN_PREFIX)
        protected = {
            "yolo26s_p2_hu800_ww1600_fold0",
            "yolo26s_p2_hu800_ww1600_run_b_1024_fold0",
            "yolo26s_p2_hu800_ww1600_aug_mild_768_fold0",
            "yolo26s_p2_hu800_ww1600_run_a_full169",
        }
        self.assertNotIn(run_dir(0).name, protected)


if __name__ == "__main__":
    unittest.main()
