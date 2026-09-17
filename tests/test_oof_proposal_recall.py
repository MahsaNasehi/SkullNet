import unittest

import numpy as np
import pandas as pd

from evaluate_oof_proposal_recall import (
    box_size_category,
    confidence_distribution,
    geometry_diagnostics,
    proposal_metrics,
)


class OOFProposalRecallTests(unittest.TestCase):
    def test_size_bins_use_original_pixel_area(self):
        self.assertEqual(box_size_category(np.array([0, 0, 31, 31])), "small")
        self.assertEqual(box_size_category(np.array([0, 0, 32, 32])), "medium")
        self.assertEqual(box_size_category(np.array([0, 0, 96, 96])), "large")

    def test_confidence_distribution(self):
        result = confidence_distribution(pd.Series([0.1, 0.9]))
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["quantiles"]["0.5"], 0.5)

    def test_proposal_recall_and_missing_counts(self):
        gt = pd.DataFrame({
            "best_iou": [0.6, 0.4, 0.0],
            "proposals": [2, 1, 0],
            "best_iou_proposal_confidence": [0.2, 0.1, 0.0],
            "max_confidence_iou30": [0.2, 0.1, 0.0],
            "max_confidence_iou50": [0.2, 0.0, 0.0],
            "size_category": ["small", "medium", "large"],
        })
        result = proposal_metrics(gt)["overall"]
        self.assertAlmostEqual(result["proposal_recall_iou30"], 2 / 3)
        self.assertAlmostEqual(result["proposal_recall_iou50"], 1 / 3)
        self.assertEqual(result["gt_on_slices_with_zero_proposals"], 1)
        self.assertEqual(result["gt_without_iou30_proposal"], 1)

    def test_geometry_diagnostics_for_small_proposal_inside_gt(self):
        result = geometry_diagnostics(np.array([0, 0, 10, 10]), np.array([2, 2, 8, 8]))
        self.assertAlmostEqual(result["best_proposal_area_over_gt_area"], 0.36)
        self.assertAlmostEqual(result["intersection_over_gt_area"], 0.36)
        self.assertEqual(result["intersection_over_proposal_area"], 1.0)
        self.assertTrue(result["proposal_center_inside_gt"])
        self.assertTrue(result["gt_center_inside_proposal"])
        self.assertEqual(result["normalized_center_distance_by_gt_diagonal"], 0.0)

    def test_geometry_diagnostics_without_proposal(self):
        result = geometry_diagnostics(np.array([0, 0, 10, 10]), None)
        self.assertEqual(result["intersection_over_gt_area"], 0.0)
        self.assertIsNone(result["best_proposal_area_over_gt_area"])


if __name__ == "__main__":
    unittest.main()
