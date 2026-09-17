"""Ultralytics integration for the patient-balanced sampler (single GPU, workers=0)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

from patient_balanced_sampling import SAMPLER_VERSION, PatientBalancedSampler, read_partition


class EpochDataLoader(DataLoader):
    # Finite iterators are rebuilt each epoch; no prefetched next-epoch batches.
    def reset(self):
        pass

    def close(self):
        pass


def log_sampling_epoch(trainer):
    sampler = trainer.train_loader.sampler
    sampler.set_epoch(trainer.epoch)
    report = sampler.audit()
    with (trainer.save_dir / "sampling_epochs.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report) + "\n")
    LOGGER.info(
        f"Balanced primary images, epoch {trainer.epoch + 1}: "
        f"{report['positive_draws']} positive + {report['negative_draws']} negative, "
        f"{report['batches']} batches; unique negatives={report['negative']['unique_images']}"
    )


class PatientBalancedDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, sampling_config, **kwargs):
        if sampling_config["version"] != SAMPLER_VERSION:
            raise ValueError("Sampler version differs from the checkpoint policy")
        self.sampling_config = sampling_config
        super().__init__(*args, **kwargs)
        if self.world_size > 1 or self.args.workers != 0 or self.args.fraction != 1.0 or self.args.rect:
            raise ValueError("Balanced experiment requires one device, workers=0, fraction=1, rect=False")
        if self.args.batch != sampling_config["batch_size"] or self.args.seed != sampling_config["seed"]:
            raise ValueError("Batch/seed must match the saved sampling policy")
        self.add_callback("on_train_epoch_start", log_sampling_epoch)

    def set_model_attributes(self):
        super().set_model_attributes()
        # Persist the policy in EMA/checkpoints too, including copied checkpoints.
        self.model.sampling_policy = self.sampling_config

    def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode="train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank != -1 or batch_size != self.sampling_config["batch_size"]:
            raise ValueError("DDP and automatic batch-size changes are not supported for this controlled experiment")
        config = self.sampling_config
        partition = read_partition(Path(self.args.data), Path(config["manifest"]))
        if partition["hashes"] != config["hashes"]:
            raise ValueError("Dataset/splits/labels changed since sampling policy was created")
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        image_paths = [str(Path(p).resolve()) for p in dataset.im_files]
        if set(image_paths) != set(partition["paths"]["train"]) or len(image_paths) != len(set(image_paths)):
            raise ValueError("Loader removed/duplicated training images; refusing a changed experiment")
        positives, patients = [], []
        for p, label in zip(image_paths, dataset.labels):
            row = partition["rows"][p]
            if len(label["cls"]) != int(row["num_boxes"]):
                raise ValueError(f"Loader labels disagree with the manifest: {p}")
            positives.append(int(row["num_boxes"]) > 0)
            patients.append(row["patient_id"])
        sampler = PatientBalancedSampler(positives, patients, batch_size, config["positive_fraction"], config["seed"])
        (self.save_dir / "sampling_policy.json").write_text(json.dumps(config, indent=2) + "\n")
        generator = torch.Generator().manual_seed(6148914691236517205 + self.args.seed)
        return EpochDataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
                               pin_memory=self.device.type == "cuda", collate_fn=dataset.collate_fn,
                               generator=generator, drop_last=False)
