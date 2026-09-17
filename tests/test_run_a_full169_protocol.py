import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

import train_run_a_full169

from run_a_full169_protocol import (
    DATA_YAML, EXPECTED_IMAGES, EXPECTED_PATIENTS, EXPECTED_STUDIES,
    FIXED_EPOCHS, FORBIDDEN_MODES, MEAN_BEST_EPOCH, MEDIAN_BEST_EPOCH,
    OUTPUT_DIR, RUN_A_BEST_EPOCHS, RUN_A_TRAIN_KWARGS, TRAIN_LIST,
    audit_cohort, audit_run_a_results, prepare, validate_prepared,
)


class RunAFull169ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = prepare()

    def test_exact_authoritative_cohort(self):
        cohort, paths = audit_cohort()
        self.assertEqual((cohort["studies"], cohort["patients"], len(paths)),
                         (EXPECTED_STUDIES, EXPECTED_PATIENTS, EXPECTED_IMAGES))
        self.assertEqual(cohort["per_fold_studies"], [33, 33, 32, 36, 35])
        self.assertEqual(cohort["fixed_test_study_overlap"], 0)
        self.assertEqual(cohort["fixed_test_patient_overlap"], 0)
        self.assertEqual(cohort["metadata_only_or_tier2_images_added"], 0)

    def test_epoch_derivation(self):
        epochs = audit_run_a_results()
        self.assertEqual(RUN_A_BEST_EPOCHS, [42, 99, 61, 20, 59])
        self.assertEqual(MEAN_BEST_EPOCH, 56.2)
        self.assertEqual(MEDIAN_BEST_EPOCH, 59)
        self.assertEqual(FIXED_EPOCHS, 59)
        self.assertEqual(epochs["recommended_fixed_epochs"], 59)

    def test_prepared_artifacts_are_exact_and_hide_test(self):
        saved = validate_prepared()
        self.assertEqual(len(TRAIN_LIST.read_text().splitlines()), EXPECTED_IMAGES)
        self.assertNotIn("test:", DATA_YAML.read_text())
        self.assertEqual(saved["status"], "prepared_training_not_started")

    def test_frozen_run_a_protocol_and_isolated_output(self):
        self.assertEqual(RUN_A_TRAIN_KWARGS["mosaic"], 0.30)
        self.assertEqual(RUN_A_TRAIN_KWARGS["degrees"], 5.0)
        self.assertEqual(RUN_A_TRAIN_KWARGS["translate"], 0.05)
        self.assertEqual(RUN_A_TRAIN_KWARGS["scale"], 0.15)
        self.assertTrue(RUN_A_TRAIN_KWARGS["amp"])
        self.assertTrue(RUN_A_TRAIN_KWARGS["deterministic"])
        self.assertTrue(all(value is False for value in FORBIDDEN_MODES.values()))
        self.assertEqual(OUTPUT_DIR.name, "yolo26s_p2_hu800_ww1600_run_a_full169")


class RunAFull169FinalizationTests(unittest.TestCase):
    def test_finalization_accepts_one_based_history_and_stripped_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights").mkdir()
            (root / "run_protocol.json").write_text(json.dumps({
                "fixed_epochs": FIXED_EPOCHS, "batch": 16, "imgsz": 768,
            }))
            history = pd.DataFrame({"epoch": range(1, FIXED_EPOCHS + 1)})
            history.to_csv(root / "results.csv", index=False)
            torch.save({
                "epoch": -1, "optimizer": None, "model": "synthetic-model",
                "train_args": {
                    "epochs": FIXED_EPOCHS, "batch": 16, "imgsz": 768,
                    "seed": train_run_a_full169.SEED, "val": False,
                    "name": train_run_a_full169.RUN_NAME,
                    "data": str(train_run_a_full169.DATA_YAML.resolve()),
                },
                "train_results": history.to_dict(orient="list"),
            }, root / "weights/last.pt")
            with patch.object(train_run_a_full169, "OUTPUT_DIR", root):
                train_run_a_full169._finish(started=None, resumed=False)
            completion = json.loads((root / "training_completed.json").read_text())
            self.assertTrue(completion["finalized_without_retraining"])
            self.assertEqual(completion["results_csv_final_epoch_one_based"], 59)
            self.assertEqual(completion["checkpoint_epoch_metadata"], -1)
            self.assertEqual((root / "weights/last.pt").read_bytes(),
                             (root / "weights/final_deployment.pt").read_bytes())

    def test_finalization_rejects_zero_based_csv_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights").mkdir()
            (root / "run_protocol.json").write_text(json.dumps({
                "fixed_epochs": FIXED_EPOCHS, "batch": 16, "imgsz": 768,
            }))
            pd.DataFrame({"epoch": range(FIXED_EPOCHS)}).to_csv(root / "results.csv", index=False)
            (root / "weights/last.pt").touch()
            with patch.object(train_run_a_full169, "OUTPUT_DIR", root):
                with self.assertRaisesRegex(RuntimeError, "consecutive CSV epochs"):
                    train_run_a_full169._finish(started=None, resumed=False)
            self.assertFalse((root / "weights/final_deployment.pt").exists())
            self.assertFalse((root / "training_completed.json").exists())


if __name__ == "__main__":
    unittest.main()
