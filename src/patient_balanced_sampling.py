"""Patient-balanced, rotating slice sampling for the annotated YOLO training split.

Epoch length is unchanged. Ratios refer to primary images BEFORE mosaic and
geometric augmentation; neither auxiliary mosaic images nor surviving boxes
are forced to follow that ratio. Epoch schedules do not depend on process RNG.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml


SAMPLER_VERSION = 1


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_partition(data: Path, manifest: Path) -> dict:
    """Validate explicit patient partitions and annotation-backed train images."""
    config = yaml.safe_load(data.read_text())
    root = Path(config.get("path", data.parent))
    if not root.is_absolute():
        root = (data.parent / root).resolve()
    with manifest.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    by_path = {str(Path(row["image_path"]).resolve()): row for row in records}
    if len(by_path) != len(records):
        raise ValueError("Duplicate images in sampling manifest")
    paths, patients, split_hashes = {}, {}, {}
    for split in ("train", "val", "test"):
        source = config.get(split)
        if not isinstance(source, str):
            raise ValueError("Balanced training requires explicit train/val/test text lists")
        source = Path(source)
        source = source if source.is_absolute() else root / source
        if source.suffix != ".txt":
            raise ValueError(f"Expected a split text list: {source}")
        items = [str(Path(s.strip()).resolve()) for s in source.read_text().splitlines() if s.strip()]
        if not items or len(set(items)) != len(items):
            raise ValueError(f"Empty or duplicate images in {split}")
        if any(p not in by_path for p in items):
            raise ValueError(f"{split} contains an image absent from the manifest")
        if any(not by_path[p]["patient_id"].strip() for p in items):
            raise ValueError(f"Missing PatientID in {split}")
        paths[split] = items
        patients[split] = {by_path[p]["patient_id"] for p in items}
        split_hashes[split] = fingerprint(source)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if patients[a] & patients[b]:
            raise ValueError(f"Patient leakage between {a} and {b}")
    if any(by_path[p]["split"] == "test" for p in paths["train"] + paths["val"]):
        raise ValueError("Fixed test images cannot be used for development")
    label_hash = hashlib.sha256()
    project = Path(__file__).resolve().parents[1]
    for p in paths["train"]:
        row = by_path[p]
        annotation = project / "iaaa-contest-bct/Data/annotations" / row["study_id"] / (row["sop_uid"] + ".json")
        if not Path(p).is_file() or not annotation.is_file():
            raise ValueError(f"Missing image or JSON annotation for {p}")
        label = Path(row["label_path"]).read_bytes()
        count = sum(bool(line.strip()) for line in label.splitlines())
        if count != int(row["num_boxes"]):
            raise ValueError(f"Label count disagrees with manifest: {p}")
        label_hash.update(p.encode() + b"\0" + label + b"\0")
    return {
        "rows": by_path, "paths": paths,
        "hashes": {"manifest": fingerprint(manifest), "data": fingerprint(data),
                   "split_lists": split_hashes, "train_labels": label_hash.hexdigest()},
        "counts": {s: {"patients": len(patients[s]), "images": len(paths[s])} for s in paths},
        "patient_overlap": 0,
    }


class PatientBalancedSampler:
    """Exact within-batch primary-image ratio with deterministic patient rotation.

Patients alternate within each class. Each patient's slices are independently
shuffled once, then traversed cyclically. A patient receives at most ceil(N/P)
draws in a class per epoch; long CT series cannot dominate by their slice count.
Every available slice is eventually visited. Epoch number alone restores the
sampler on resume (augmentation RNG is handled separately by Ultralytics).
"""

    def __init__(self, positives, patient_ids, batch_size=16, positive_fraction=0.25, seed=42):
        if len(positives) != len(patient_ids) or not positives:
            raise ValueError("Nonempty, aligned labels and PatientIDs are required")
        if batch_size < 2 or not 0 < positive_fraction < 1:
            raise ValueError("Require batch_size >= 2 and 0 < positive_fraction < 1")
        self.positives = [bool(x) for x in positives]
        self.patient_ids = list(patient_ids)
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction
        self.seed, self.epoch = seed, 0
        self.groups = {}
        self.orders = {}
        for label in (False, True):
            groups = defaultdict(list)
            for index, (positive, patient) in enumerate(zip(self.positives, self.patient_ids)):
                if positive == label:
                    groups[patient].append(index)
            if not groups:
                raise ValueError("Training needs both positive and annotated negative images")
            rng = random.Random(seed + 1009 * int(label))
            order = sorted(groups)
            rng.shuffle(order)
            for patient in sorted(groups):
                rng.shuffle(groups[patient])
            self.groups[label], self.orders[label] = groups, order
        self.batch_counts = []
        for start in range(0, len(self), batch_size):
            size = min(batch_size, len(self) - start)
            positive = int(math.floor(size * positive_fraction + 0.5))
            self.batch_counts.append((positive, size - positive))
        self.epoch_positive = sum(p for p, _ in self.batch_counts)
        self.epoch_negative = len(self) - self.epoch_positive
        if not self.epoch_positive or not self.epoch_negative:
            raise ValueError("Requested batch/ratio rounds to a single class")

    def __len__(self):
        return len(self.positives)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def _draw(self, label, offset, count):
        order, groups = self.orders[label], self.groups[label]
        output = []
        for position in range(offset, offset + count):
            patient = order[position % len(order)]
            slices = groups[patient]
            output.append(slices[(position // len(order)) % len(slices)])
        return output

    def indices(self, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        positive_offset, negative_offset = epoch * self.epoch_positive, epoch * self.epoch_negative
        rng = random.Random(self.seed + 1000003 * (epoch + 1))
        result = []
        for positive, negative in self.batch_counts:
            batch = self._draw(True, positive_offset, positive) + self._draw(False, negative_offset, negative)
            rng.shuffle(batch)
            result.extend(batch)
            positive_offset += positive
            negative_offset += negative
        return result

    def __iter__(self):
        return iter(self.indices())

    def audit(self, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        indices = self.indices(epoch)
        report = {"epoch": epoch, "images": len(indices), "batches": len(self.batch_counts),
                  "positive_draws": self.epoch_positive, "negative_draws": self.epoch_negative,
                  "actual_primary_positive_fraction": self.epoch_positive / len(self),
                  "indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest()}
        for label, name, per_epoch in ((True, "positive", self.epoch_positive), (False, "negative", self.epoch_negative)):
            selected = [i for i in indices if self.positives[i] == label]
            counts = Counter(self.patient_ids[i] for i in selected)
            report[name] = {"unique_images": len(set(selected)), "pool_images": sum(map(len, self.groups[label].values())),
                            "patients_seen": len(counts), "pool_patients": len(self.orders[label]),
                            "max_patient_draws": max(counts.values()), "min_patient_draws": min(counts.values()),
                            "all_slices_coverage_within_epochs": math.ceil(
                                len(self.orders[label]) * max(map(len, self.groups[label].values())) / per_epoch)}
        return report


def sampling_policy(data, manifest, fraction, batch_size, seed):
    partition = read_partition(data, manifest)
    policy = {"version": SAMPLER_VERSION, "data": str(data.resolve()), "manifest": str(manifest.resolve()),
              "positive_fraction": fraction, "batch_size": batch_size, "seed": seed,
              "hashes": partition["hashes"], "partition_counts": partition["counts"],
              "patient_overlap": 0, "ratio_scope": "primary images before unchanged mosaic/geometry"}
    rows = [partition["rows"][p] for p in partition["paths"]["train"]]
    sampler = PatientBalancedSampler([int(r["num_boxes"]) > 0 for r in rows],
                                     [r["patient_id"] for r in rows], batch_size, fraction, seed)
    return policy, sampler
