import unittest

import numpy as np
import pandas as pd

from select_oof_fracture_threshold import (
    cross_fitted_predictions,
    decision_metrics,
    patient_bootstrap_decisions,
    select_threshold,
    threshold_candidates,
)


class OOFThresholdTests(unittest.TestCase):
    def test_candidates_cover_all_positive_and_all_negative(self):
        candidates = threshold_candidates(np.array([0.1, 0.2, 0.9]))
        self.assertIn(0.0, candidates)
        self.assertGreater(candidates.max(), 0.9)

    def test_select_threshold_finds_perfect_split(self):
        y = np.array([False, False, True, True])
        scores = np.array([0.1, 0.2, 0.7, 0.8])
        threshold, metrics, _ = select_threshold(y, scores)
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.7)
        self.assertEqual(metrics["binary_qwk"], 1.0)

    def test_decision_metrics_uses_given_decisions(self):
        result = decision_metrics(
            np.array([False, True, False, True]),
            np.array([False, True, True, False]),
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (1, 1, 1, 1))

    def test_crossfit_never_selects_on_heldout_fold(self):
        rows = []
        for fold in range(5):
            rows.extend([
                {"study_id": f"n{fold}", "patient_id": f"pn{fold}", "fold": fold,
                 "fracture_true": False, "top3_mean": 0.1},
                {"study_id": f"p{fold}", "patient_id": f"pp{fold}", "fold": fold,
                 "fracture_true": True, "top3_mean": 0.9},
            ])
        predictions, thresholds = cross_fitted_predictions(pd.DataFrame(rows))
        self.assertEqual(len(predictions), 10)
        self.assertTrue(predictions["crossfit_predicted"].eq(predictions["fracture_true"]).all())
        self.assertEqual({x["selection_data_studies"] for x in thresholds}, {8})

    def test_patient_bootstrap_of_frozen_decisions(self):
        table = pd.DataFrame({
            "patient_id": ["p1", "p1", "p2", "p3"],
            "fracture_true": [True, False, False, True],
            "predicted": [True, False, False, True],
        })
        result = patient_bootstrap_decisions(table, "predicted", repeats=20, seed=1)
        self.assertEqual(result["percentile_95_ci"]["binary_qwk"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
