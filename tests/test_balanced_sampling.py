import math
import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import yaml

from patient_balanced_sampling import PatientBalancedSampler, read_partition


class BalancedSamplingTests(unittest.TestCase):
    def make_sampler(self, seed=42):
        # Patients deliberately have very different slice counts.
        return PatientBalancedSampler([True] * 10 + [False] * 43,
                                      ["p1"] * 2 + ["p2"] * 8 + ["n1"] * 3 + ["n2"] * 40,
                                      batch_size=16, positive_fraction=0.25, seed=seed)

    def test_batch_ratio_and_same_epoch_budget(self):
        sampler = self.make_sampler()
        indices = list(sampler)
        self.assertEqual(len(indices), 53)
        self.assertEqual(math.ceil(len(indices) / 16), 4)
        for start in range(0, len(indices), 16):
            batch = indices[start:start + 16]
            self.assertEqual(sum(sampler.positives[i] for i in batch), math.floor(len(batch) * .25 + .5))

    def test_long_studies_do_not_dominate(self):
        sampler = self.make_sampler()
        for epoch in range(5):
            indices = sampler.indices(epoch)
            for label in (True, False):
                counts = Counter(sampler.patient_ids[i] for i in indices if sampler.positives[i] == label)
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_rotation_eventually_covers_every_slice(self):
        sampler = self.make_sampler()
        bound = max(sampler.audit()[name]["all_slices_coverage_within_epochs"] for name in ("positive", "negative"))
        visited = {i for epoch in range(bound) for i in sampler.indices(epoch)}
        self.assertEqual(visited, set(range(len(sampler))))

    def test_resume_reproduces_schedule_without_rng_history(self):
        original = self.make_sampler()
        for epoch in range(5):
            original.set_epoch(epoch)
            expected = list(original)
            restored = self.make_sampler()
            restored.set_epoch(epoch)
            self.assertEqual(expected, list(restored))
        self.assertNotEqual(original.indices(0), original.indices(1))
        self.assertNotEqual(original.indices(0), self.make_sampler(43).indices(0))

    def test_invalid_policies_fail(self):
        for labels, patients, batch, fraction in (([True], ["p"], 16, .25),
                                                  ([True, False], ["p"], 16, .25),
                                                  ([True, False], ["p", "n"], 1, .25),
                                                  ([True, False], ["p", "n"], 16, 1.0)):
            with self.assertRaises(ValueError):
                PatientBalancedSampler(labels, patients, batch, fraction)

    def test_real_loader_consumes_new_epoch_schedule(self):
        import torch
        from balanced_yolo_trainer import EpochDataLoader
        sampler = self.make_sampler()
        dataset = torch.utils.data.TensorDataset(torch.arange(len(sampler)))
        loader = EpochDataLoader(dataset, batch_size=16, sampler=sampler, num_workers=0)
        for epoch in (0, 1, 6):
            sampler.set_epoch(epoch)
            loader.reset()
            received = torch.cat([batch[0] for batch in loader]).tolist()
            self.assertEqual(received, sampler.indices(epoch))

    def partition_fixture(self, root, overlap=False, fixed_test=False):
        rows, config = [], {"path": str(root)}
        for index, split in enumerate(("train", "val", "test")):
            image = root / (split + ".png")
            image.touch()
            label = root / (split + "_label.txt")
            label.touch()
            row = {"image_path": str(image), "label_path": str(label), "num_boxes": "0",
                   "patient_id": "same" if overlap and split in ("train", "val") else str(index),
                   "split": "test" if fixed_test and split == "train" else split,
                   "study_id": "nonexistent_smoke_study", "sop_uid": split}
            rows.append(row)
            paths = root / (split + ".txt")
            paths.write_text(str(image) + "\n")
            config[split] = str(paths)
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        data = root / "data.yaml"
        data.write_text(yaml.safe_dump(config))
        return data, manifest

    def test_patient_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data, manifest = self.partition_fixture(Path(directory), overlap=True)
            with self.assertRaisesRegex(ValueError, "Patient leakage"):
                read_partition(data, manifest)

    def test_fixed_test_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as directory:
            data, manifest = self.partition_fixture(Path(directory), fixed_test=True)
            with self.assertRaisesRegex(ValueError, "Fixed test images"):
                read_partition(data, manifest)

    def test_missing_annotation_is_not_a_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            data, manifest = self.partition_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "JSON annotation"):
                read_partition(data, manifest)


if __name__ == "__main__":
    unittest.main()
