import unittest
from types import SimpleNamespace

from evaluate_fold_qwk import _predict_triage


class ThresholdTests(unittest.TestCase):
    def test_custom_threshold_changes_official_fracture_bit(self):
        row = SimpleNamespace(V_EDH=0., V_SDH=0., V_IPH=20., V_SAH=0., V_IVH=0.,
                              MLS_mm=0., top3_mean=.3)
        self.assertEqual(_predict_triage(row, "top3_mean", .5), 1)
        self.assertEqual(_predict_triage(row, "top3_mean", .25), 2)
        self.assertEqual(_predict_triage(row, "top3_mean", .3), 2)

    def test_high_mls_without_fracture_or_ich_is_urgent(self):
        row = SimpleNamespace(V_EDH=0., V_SDH=0., V_IPH=0., V_SAH=0., V_IVH=0.,
                              MLS_mm=6., top3_mean=.4)
        self.assertEqual(_predict_triage(row, "top3_mean", .5), 1)
        self.assertEqual(_predict_triage(row, "top3_mean", .3), 2)


if __name__ == "__main__":
    unittest.main()
