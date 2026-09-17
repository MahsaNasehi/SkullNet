import unittest

import pandas as pd

from compare_run_a_b_1024 import transitions


class CompareRunABTests(unittest.TestCase):
    def test_transition_counts(self):
        frame = pd.DataFrame({"best_iou_A": [0.1, 0.6, 0.7, 0.1],
                              "best_iou_B": [0.6, 0.1, 0.8, 0.2]})
        result = transitions(frame, 0.5)
        self.assertEqual(result, {"miss_A_to_hit_B": 1, "hit_A_to_miss_B": 1,
                                  "hit_both": 1, "miss_both": 1})


if __name__ == "__main__":
    unittest.main()
