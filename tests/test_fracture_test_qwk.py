import unittest

import numpy as np
import pandas as pd

from evaluate_fracture_test_qwk import binary_metrics, patient_bootstrap_qwk, top3_mean


class FractureTestQWKTests(unittest.TestCase):
    def test_top3_mean(self):
        self.assertAlmostEqual(top3_mean(pd.Series([.1, .9, .5, .7])), .7)
        self.assertAlmostEqual(top3_mean(pd.Series([.2])), .2)

    def test_perfect_binary_qwk(self):
        result = binary_metrics(np.array([False, True, False, True]), np.array([.1, .9, .2, .8]), .5)
        self.assertEqual(result["binary_qwk"], 1.0)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (2, 0, 0, 2))

    def test_patient_cluster_bootstrap(self):
        table = pd.DataFrame({"patient_id": ["p1", "p1", "p2", "p3"],
                              "fracture_true": [True, False, False, True],
                              "score": [.9, .1, .2, .8]})
        result = patient_bootstrap_qwk(table, "score", .5, repeats=20, seed=1)
        self.assertEqual(result["valid_repeats"], 20)
        self.assertEqual(result["qwk_percentile_95_ci"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
