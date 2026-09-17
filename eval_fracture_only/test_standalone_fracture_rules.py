"""Tests for standalone fracture rules (no ICH / MLS dependency)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from standalone_fracture_rules import explain_rules, standalone_fracture_triage
from evaluate_standalone_fracture import model_only_report


class StandaloneRulesTests(unittest.TestCase):
    def test_no_external_dependencies_declared(self):
        meta = explain_rules()
        self.assertFalse(meta["depends_on_ICH"])
        self.assertFalse(meta["depends_on_MLS"])
        self.assertFalse(meta["calls_official_triage"])
        self.assertEqual(meta["inputs"], ["fracture_prob"])

    def test_binary_mapping(self):
        self.assertEqual(standalone_fracture_triage(0.0), 0)
        self.assertEqual(standalone_fracture_triage(0.499), 0)
        self.assertEqual(standalone_fracture_triage(0.5), 1)
        self.assertEqual(standalone_fracture_triage(1.0), 1)

    def test_never_critical(self):
        for p in (0.0, 0.5, 1.0):
            self.assertIn(standalone_fracture_triage(p), (0, 1))

    def test_model_report_perfect(self):
        report = model_only_report([False, True, False, True], [0.1, 0.9, 0.0, 0.8])
        self.assertAlmostEqual(report["model_accuracy"], 1.0)
        self.assertAlmostEqual(report["macro_f1_binary"], 1.0)
        self.assertAlmostEqual(report["standalone_triage_macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
