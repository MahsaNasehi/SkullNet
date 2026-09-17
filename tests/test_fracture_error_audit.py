import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_fracture_test_errors import box_iou, proposal_summary, yolo_labels_to_xyxy


class FractureErrorAuditTests(unittest.TestCase):
    def test_iou(self):
        first = np.array([[0, 0, 10, 10]])
        second = np.array([[0, 0, 10, 10], [10, 10, 20, 20]])
        np.testing.assert_allclose(box_iou(first, second), [[1.0, 0.0]])

    def test_yolo_label_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("0 0.5 0.5 0.5 0.5\n")
            np.testing.assert_allclose(yolo_labels_to_xyxy(path, 100, 200), [[25, 50, 75, 150]])

    def test_proposal_matching_retains_confidence(self):
        summary, details = proposal_summary(
            np.array([[0, 0, 10, 10]]),
            np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            np.array([0.2, 0.9]),
        )
        self.assertEqual(summary["gt_boxes_matched_iou50"], 1)
        self.assertAlmostEqual(summary["max_matched_confidence_iou50"], 0.2)
        self.assertAlmostEqual(details[0]["best_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
