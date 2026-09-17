"""Manual-only fixed nine-epoch FULL169 classifier refit and CPU-only finalization."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fracture_classifier.data import (EXPERIMENT, FINAL_DEPLOYMENT, build_slice_manifest,
                                      detector_checkpoints, sha256)
from fracture_classifier.model import FractureSliceClassifier
from fracture_classifier.training import ReviewedSlices


OUTPUT = EXPERIMENT.parent / "run_a_fracture_classifier_full169"
EXPECTED_DETECTOR_SHA256 = "109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640"
EXPECTED_BEST_EPOCHS = [3, 7, 9, 9, 10]
FIXED_EPOCHS = 9
EXPECTED_COUNTS = {"positive": 223, "negative": 4195, "ignored": 24,
                   "studies": 169, "patients": 155}
SEED = 42
BATCH = 8
LR = 1e-4
WEIGHT_DECAY = 5e-4


def audit_training_inputs() -> tuple[pd.DataFrame, dict]:
    if sha256(FINAL_DEPLOYMENT) != EXPECTED_DETECTOR_SHA256:
        raise RuntimeError("Protected FULL169 detector checksum mismatch")
    table, cohort = build_slice_manifest()
    manifest_path = EXPERIMENT / "slice_label_manifest.csv"
    if not manifest_path.is_file() or table.to_csv(index=False) != manifest_path.read_text():
        raise RuntimeError("FULL169 labels differ from frozen five-fold reviewed-slice manifest")
    observed = {"positive": int(((table["is_reviewed"]) & (table["fracture_label"] == 1)).sum()),
                "negative": int(((table["is_reviewed"]) & (table["fracture_label"] == 0)).sum()),
                "ignored": int((~table["is_reviewed"]).sum()),
                "studies": int(table["study_id"].nunique()),
                "patients": int(table["patient_id"].nunique())}
    if observed != EXPECTED_COUNTS or cohort["patient_leakage"]:
        raise RuntimeError(f"Unexpected FULL169 reviewed-slice counts: {observed}")
    best_epochs = []
    for fold in range(5):
        path = EXPERIMENT.parent / f"run_a_fracture_classifier_fold{fold}/classifier_metrics.json"
        completion = path.parent / "COMPLETE.json"
        if not completion.is_file() or not json.loads(completion.read_text()).get("stage1_completed"):
            raise RuntimeError(f"Classifier OOF Fold {fold} not complete")
        best_epochs.append(int(json.loads(path.read_text())["stage1"]["best_epoch"]))
    if best_epochs != EXPECTED_BEST_EPOCHS or int(np.median(best_epochs)) != FIXED_EPOCHS:
        raise RuntimeError(f"OOF-derived fixed epoch protocol changed: {best_epochs}")
    reviewed = table[table["is_reviewed"]].copy()
    if reviewed["fracture_label"].isna().any() or len(reviewed) != 4418:
        raise RuntimeError("Unknown/unreviewed slice reached FULL169 classifier training")
    return reviewed, {**observed, "oof_best_epochs": best_epochs,
                      "pos_weight": observed["negative"] / observed["positive"],
                      "detector_sha256": sha256(FINAL_DEPLOYMENT),
                      "slice_label_manifest_sha256": sha256(manifest_path),
                      "oof_checkpoint_sha256": {str(fold): sha256(detector_checkpoints()[fold])
                                                for fold in range(5)}}


def _atomic_torch_save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(device: str) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite FULL169 classifier output: {OUTPUT}")
    reviewed, audit = audit_training_inputs()
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = FractureSliceClassifier.from_detector(FINAL_DEPLOYMENT).to(device)
    model.freeze_backbone()
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("FULL169 classifier backbone is not frozen")
    original_backbone = {key: value.detach().cpu().clone() for key, value in model.backbone.state_dict().items()}
    protocol = {"experiment": "RUN_A_FRACTURE_CLASSIFIER_FULL169_FIXED9",
                "source_detector": str(FINAL_DEPLOYMENT), "source_detector_sha256": audit["detector_sha256"],
                "oof_best_epochs": audit["oof_best_epochs"], "fixed_epochs": FIXED_EPOCHS,
                "seed_before_model_construction": SEED, "batch": BATCH,
                "optimizer": "AdamW", "head_lr": LR, "weight_decay": WEIGHT_DECAY,
                "loss": "BCEWithLogitsLoss", "pos_weight": audit["pos_weight"],
                "training_counts": {key: audit[key] for key in EXPECTED_COUNTS},
                "slice_label_manifest_sha256": audit["slice_label_manifest_sha256"],
                "oof_checkpoint_sha256": audit["oof_checkpoint_sha256"],
                "architecture": "Run A backbone layers 0..10 P5 C2PSA 512ch -> GAP -> LN512 -> Linear512x128 -> SiLU -> Dropout0.1 -> Linear128x1",
                "input": "reviewed-only 2.5D +/-5mm HU WL800 WW1600 prepared PNG; resize 768",
                "augmentation": "none; no Mosaic/HU jitter/HNM/B25",
                "backbone_frozen": True, "trainable_parameters": "classifier head only",
                "validation_used": False, "checkpoint_selection": "fixed final epoch 9 only",
                "official_fracture_threshold": 0.5}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    loader = DataLoader(ReviewedSlices(reviewed), batch_size=BATCH, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(audit["pos_weight"], device=device))
    history = []
    for epoch in range(1, FIXED_EPOCHS + 1):
        model.train()
        losses = []
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite FULL169 classifier training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        item = {"epoch_one_based": epoch, "train_loss": float(np.mean(losses))}
        history.append(item)
        _atomic_torch_save(OUTPUT / "last_classifier.pt", {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "stage": 1, "epoch_one_based": epoch, "fixed_epochs": FIXED_EPOCHS,
            "source_detector_sha256": audit["detector_sha256"],
            "slice_label_manifest_sha256": audit["slice_label_manifest_sha256"],
            "training_history": list(history),
        })
        print(f"FULL169 classifier epoch {epoch}/{FIXED_EPOCHS} train_loss={item['train_loss']:.6f}", flush=True)
    if any(not torch.equal(value, model.backbone.state_dict()[key].detach().cpu())
           for key, value in original_backbone.items()):
        raise RuntimeError("Frozen FULL169 backbone changed during head training")
    if sha256(FINAL_DEPLOYMENT) != audit["detector_sha256"]:
        raise RuntimeError("Protected FULL169 detector changed during classifier training")
    (OUTPUT / "classifier_training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (OUTPUT / "training_finished.json").write_text(json.dumps({
        "status": "nine_training_epochs_finished_not_finalized", "epochs": FIXED_EPOCHS,
        "last_classifier_sha256": sha256(OUTPUT / "last_classifier.pt"),
        "source_detector_sha256": audit["detector_sha256"],
        "frozen_backbone_unchanged": True,
    }, indent=2) + "\n")


def finalize() -> dict:
    protocol_path = OUTPUT / "protocol.json"
    history_path = OUTPUT / "classifier_training_history.json"
    finished_path = OUTPUT / "training_finished.json"
    last = OUTPUT / "last_classifier.pt"
    final = OUTPUT / "final_classifier.pt"
    completion = OUTPUT / "training_completed.json"
    if final.exists() or completion.exists():
        raise FileExistsError("FULL169 classifier is already finalized; refusing overwrite")
    if not all(path.is_file() for path in (protocol_path, history_path, finished_path, last)):
        raise RuntimeError("FULL169 classifier training artifacts are incomplete")
    protocol = json.loads(protocol_path.read_text())
    history = json.loads(history_path.read_text())
    finished = json.loads(finished_path.read_text())
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    if (protocol.get("fixed_epochs") != FIXED_EPOCHS or protocol.get("validation_used") is not False
            or protocol.get("checkpoint_selection") != "fixed final epoch 9 only"):
        raise RuntimeError("FULL169 classifier protocol drift")
    if ([int(item["epoch_one_based"]) for item in history] != list(range(1, FIXED_EPOCHS + 1))
            or checkpoint.get("training_history") != history
            or checkpoint.get("epoch_one_based") != FIXED_EPOCHS
            or checkpoint.get("fixed_epochs") != FIXED_EPOCHS
            or checkpoint.get("stage") != 1):
        raise RuntimeError("last_classifier.pt does not represent fixed final epoch 9")
    if (finished.get("epochs") != FIXED_EPOCHS
            or finished.get("frozen_backbone_unchanged") is not True
            or finished.get("last_classifier_sha256") != sha256(last)
            or checkpoint.get("source_detector_sha256") != EXPECTED_DETECTOR_SHA256
            or protocol.get("source_detector_sha256") != EXPECTED_DETECTOR_SHA256
            or sha256(FINAL_DEPLOYMENT) != EXPECTED_DETECTOR_SHA256):
        raise RuntimeError("FULL169 classifier/detector provenance mismatch")
    manifest_hash = sha256(EXPERIMENT / "slice_label_manifest.csv")
    if checkpoint.get("slice_label_manifest_sha256") != manifest_hash or protocol.get("slice_label_manifest_sha256") != manifest_hash:
        raise RuntimeError("FULL169 classifier label provenance mismatch")
    shutil.copy2(last, final)
    if sha256(final) != sha256(last):
        raise RuntimeError("Final classifier copy differs from final epoch checkpoint")
    report = {"status": "completed_fixed_nine_epoch_classifier_refit",
              "final_epoch_one_based": FIXED_EPOCHS, "validation_used": False,
              "final_classifier_is_byte_identical_copy_of_last": True,
              "final_classifier_sha256": sha256(final), "last_classifier_sha256": sha256(last),
              "source_detector_sha256": EXPECTED_DETECTOR_SHA256,
              "training_cohort_studies": 169, "training_cohort_patients": 155,
              "no_retraining_during_finalization": True}
    completion.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.audit:
        _, audit = audit_training_inputs()
        print(json.dumps(audit, indent=2))
    elif args.train:
        train(args.device)
    elif args.finalize:
        print(json.dumps(finalize(), indent=2))
    else:
        state = ("complete" if (OUTPUT / "training_completed.json").is_file()
                 else "finished_not_finalized" if (OUTPUT / "training_finished.json").is_file()
                 else "incomplete" if OUTPUT.exists() else "not_started")
        print(json.dumps({"status": state, "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
