import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aug_mild_768_protocol import (
    AUG_MILD_AUGMENTATION, BASELINE_AUGMENTATION, BASELINE_OPTIMIZATION,
    DATA_CONFIG_DIR, EXPECTED_SPLIT_HASHES, INTENDED_CHANGES, RUN_PREFIX,
    assert_saved_protocol, augmentation_diff, completion_is_valid,
    expected_fold_protocol, train_kwargs, validate_data_configs,
    validate_static_protocol,
)
from oof_cohort import derive_explicit_validation_cohort
from run_aug_mild_768_kfold import command_for_fold
from train_yolo26s_p2 import ROOT
from train_aug_mild_768_fold import _validate_checkpoint_train_args


class AugMild768ProtocolTests(unittest.TestCase):
    def test_exactly_four_augmentation_changes(self):
        validate_static_protocol()
        self.assertEqual(set(augmentation_diff()), {"mosaic", "degrees", "translate", "scale"})
        self.assertEqual(
            {key: AUG_MILD_AUGMENTATION[key] for key in INTENDED_CHANGES},
            {"mosaic": 0.0, "degrees": 3.0, "translate": 0.03, "scale": 0.08},
        )
        for key in set(BASELINE_AUGMENTATION) - set(INTENDED_CHANGES):
            self.assertEqual(AUG_MILD_AUGMENTATION[key], BASELINE_AUGMENTATION[key])

    def test_explicit_training_configuration_disables_forbidden_modes(self):
        kwargs = train_kwargs()
        self.assertEqual(BASELINE_OPTIMIZATION["batch"], 16)
        self.assertEqual(BASELINE_OPTIMIZATION["imgsz"], 768)
        self.assertNotIn("positive_fraction", kwargs)
        self.assertNotIn("hnm", kwargs)
        self.assertNotIn("extra_negatives", kwargs)
        command = command_for_fold(2, "0", resume=False)
        joined = " ".join(command).lower()
        self.assertNotIn("positive-fraction", joined)
        self.assertNotIn("hnm", joined)
        self.assertNotIn("run_b", RUN_PREFIX)
        self.assertNotEqual(RUN_PREFIX, "yolo26s_p2_hu800_ww1600_fold")

    def test_sanitized_configs_preserve_authoritative_split_hashes(self):
        records = validate_data_configs()
        for fold in range(5):
            self.assertNotIn("test:", (DATA_CONFIG_DIR / ("fold_%d.yaml" % fold)).read_text())
            observed = {name: records[str(fold)][name]["sha256"] for name in ("train", "val")}
            self.assertEqual(observed, EXPECTED_SPLIT_HASHES[fold])

    def test_resume_refuses_augmentation_protocol_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = expected_fold_protocol(0)
            path = Path(directory) / "run_protocol.json"
            path.write_text(json.dumps(protocol))
            assert_saved_protocol(path, 0)
            protocol["training_augmentation"]["mosaic"] = 0.1
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(RuntimeError, "protocol drift"):
                assert_saved_protocol(path, 0)

    def test_resume_checkpoint_must_preserve_mild_augmentation(self):
        class Checkpoint:
            pass
        checkpoint = Checkpoint()
        checkpoint.ckpt = {"train_args": {
            "imgsz": 768, "batch": 16, "epochs": 150, "patience": 30,
            "workers": 0, "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01,
            "momentum": 0.937, "weight_decay": 0.0005, "warmup_epochs": 3.0,
            "amp": True, "deterministic": True, **AUG_MILD_AUGMENTATION,
        }}
        _validate_checkpoint_train_args(checkpoint)
        checkpoint.ckpt["train_args"]["scale"] = 0.15
        with self.assertRaisesRegex(RuntimeError, "checkpoint.*protocol drift"):
            _validate_checkpoint_train_args(checkpoint)

    def test_completion_requires_best_results_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "weights").mkdir()
            (run_dir / "training_completed.json").write_text('{"status":"completed"}')
            self.assertFalse(completion_is_valid(run_dir, 0))
            (run_dir / "weights/best.pt").touch()
            (run_dir / "results.csv").write_text("epoch\n0\n")
            (run_dir / "run_protocol.json").write_text(json.dumps(expected_fold_protocol(0)))
            self.assertTrue(completion_is_valid(run_dir, 0))

    def test_full_study_evaluator_authoritative_cohort_is_169(self):
        dataset = ROOT / "data_prepared/skull_hu800_ww1600"
        manifest = pd.read_csv(
            dataset / "manifest.csv", dtype={"study_id": str, "patient_id": str, "image_path": str}
        )
        cohort = derive_explicit_validation_cohort(
            manifest, [dataset / ("kfold/fold_%d_val.txt" % fold) for fold in range(5)]
        )
        self.assertEqual(cohort["total_unique_validation_studies"], 169)
        self.assertEqual(cohort["per_fold_study_counts"], [33, 33, 32, 36, 35])


if __name__ == "__main__":
    unittest.main()
