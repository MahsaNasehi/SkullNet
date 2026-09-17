import argparse
import unittest

from run_b_1024_kfold import CONFIG_DIR, command_for_fold
from train_yolo26s_p2 import ROOT, split_audit


class RunBProtocolTests(unittest.TestCase):
    def test_sanitized_yaml_has_identical_train_val_membership_and_no_test(self):
        for fold in range(5):
            clean = CONFIG_DIR / f"fold_{fold}.yaml"
            original = ROOT / f"data_prepared/skull_hu800_ww1600/kfold/fold_{fold}.yaml"
            self.assertEqual(split_audit(clean), split_audit(original))
            self.assertNotIn("test:", clean.read_text())

    def test_fold_command_changes_only_resolution_protocol_fields(self):
        args = argparse.Namespace(epochs=150, batch=16, device="0", patience=30)
        command = command_for_fold(3, args)
        self.assertEqual(command[command.index("--imgsz") + 1], "1024")
        self.assertEqual(command[command.index("--batch") + 1], "16")
        self.assertEqual(command[command.index("--seed") + 1], "45")
        self.assertNotIn("--positive-fraction", command)


if __name__ == "__main__":
    unittest.main()
