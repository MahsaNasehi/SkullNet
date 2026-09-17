import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from data.windows import make_hu_input as project_make_hu_input
from submit import model as submission


class SubmissionPackageTests(unittest.TestCase):
    def test_checkpoint_identity_and_frozen_protocol(self):
        digest = hashlib.sha256(submission.resolve_weights().read_bytes()).hexdigest()
        self.assertEqual(digest, "6c3635082f4ed0e15d549e02e3a7faeaec0293299a06b01d342d8f2e3853f206")
        self.assertEqual(submission.IMAGE_SIZE, 768)
        self.assertEqual(submission.CONFIDENCE, 0.01)
        self.assertEqual(submission.NMS_IOU, 0.5)
        self.assertEqual(submission.MAX_DETECTIONS, 300)
        self.assertEqual(submission.AGGREGATION_METHOD, "top10_percent_mean")

    def test_top10_percent_aggregation_and_mapping(self):
        scores = list(np.linspace(0.0, 0.99, 20))
        expected = np.mean(sorted(scores, reverse=True)[:2])
        self.assertAlmostEqual(submission.aggregate_top10_percent_mean(scores), expected)
        self.assertAlmostEqual(
            submission.rescale_for_macro_f1(submission.RAW_MACRO_F1_THRESHOLD), 0.5
        )

    def test_preprocessing_matches_project_implementation(self):
        hu = [np.arange(64, dtype=np.float32).reshape(8, 8) * 30 + shift for shift in (-10, 0, 10)]
        positions = [-5.0, 0.0, 5.0]
        actual = submission.make_hu_input(
            hu, 1, level=800.0, width=1600.0, mode="2.5d",
            physical_positions=positions, context_distance_mm=5.0,
        )
        expected = project_make_hu_input(
            hu, 1, level=800.0, width=1600.0, mode="2.5d",
            physical_positions=positions, context_distance_mm=5.0,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_official_api_has_exact_schema(self):
        instance = submission.Model.__new__(submission.Model)
        with patch.object(submission.Model, "predict_fracture_prob", return_value=0.75):
            result = instance.predict("unused")
        self.assertEqual(tuple(result), submission.INTERMEDIATE_KEYS)
        self.assertEqual(result["fracture_prob"], 0.75)
        self.assertTrue(all(result[key] == 0.0 for key in submission.INTERMEDIATE_KEYS if key != "fracture_prob"))


if __name__ == "__main__":
    unittest.main()
