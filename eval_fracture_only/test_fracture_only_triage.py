"""Unit checks for fracture-only official triage isolation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fracture_only_triage import fracture_only_triage
from evaluate_fracture_only import fracture_only_triage_report, two_class_macro_f1_report


class FractureOnlyTriageTests(unittest.TestCase):
    def test_official_rule_reduces_to_binary(self):
        self.assertEqual(fracture_only_triage(0.0), 0)
        self.assertEqual(fracture_only_triage(0.499999), 0)
        self.assertEqual(fracture_only_triage(0.5), 1)
        self.assertEqual(fracture_only_triage(1.0), 1)

    def test_class_2_unreachable(self):
        for probability in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertNotEqual(fracture_only_triage(probability), 2)

    def test_perfect_scores(self):
        report = fracture_only_triage_report([False, True, False, True], [0.1, 0.9, 0.2, 0.8])
        self.assertAlmostEqual(report["primary_two_class_macro_f1"]["pooled_macro_f1"], 1.0)
        self.assertEqual(report["binary_fracture_metrics"]["tp"], 2)
        self.assertEqual(report["binary_fracture_metrics"]["fn"], 0)

    def test_two_class_rejects_critical(self):
        with self.assertRaises(ValueError):
            two_class_macro_f1_report([0, 2], [0, 1])


if __name__ == "__main__":
    unittest.main()
