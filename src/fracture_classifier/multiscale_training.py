"""Manual, isolated Stage-1-only Run A P3+P4+P5 OOF training."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fracture_classifier.data import EXPERIMENT, FINAL_DEPLOYMENT, detector_checkpoints, fold_rows, sha256
from fracture_classifier.feature_hooks import SPECS, verify_backbone_graph
from fracture_classifier.multiscale_model import MultiScaleFractureClassifier
from fracture_classifier.training import ReviewedSlices, slice_metrics


EPOCHS = 10
BATCH = 8
LR = 1e-4
WEIGHT_DECAY = 5e-4
OUTPUT_PREFIX = "run_a_fracture_multiscale_fold"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def audited_fold(fold: int) -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    if fold not in range(5):
        raise ValueError("Fold must be 0..4")
    manifest_path = EXPERIMENT / "slice_label_manifest.csv"
    audit_path = EXPERIMENT / "slice_label_audit.json"
    if not manifest_path.is_file() or not audit_path.is_file():
        raise RuntimeError("Existing Run A P5 slice-label audit is required")
    table = pd.read_csv(manifest_path, dtype={"patient_id": str, "study_id": str, "sop_uid": str,
                                               "image_path": str, "label_path": str})
    if table.study_id.nunique() != 169 or table.patient_id.nunique() != 155:
        raise RuntimeError("Run A classifier cohort is not 169 Studies/155 patients")
    if table.groupby("patient_id").fold.nunique().max() != 1:
        raise RuntimeError("Cross-fold patient leakage")
    if table.groupby("fold").study_id.nunique().tolist() != [33, 33, 32, 36, 35]:
        raise RuntimeError("Run A fold Study counts changed")
    train, val, pos_weight = fold_rows(table, fold)
    if not train.is_reviewed.astype(bool).all() or not val.is_reviewed.astype(bool).all():
        raise RuntimeError("Unreviewed slice entered multi-scale training/validation")
    audit = json.loads(audit_path.read_text())
    source = detector_checkpoints()[fold]
    p5_protocol_path = EXPERIMENT.parent / f"run_a_fracture_classifier_fold{fold}/protocol.json"
    p5 = json.loads(p5_protocol_path.read_text())
    if (sha256(source) != audit["detector_sha256"][str(fold)]
            or sha256(source) != p5["source_detector_sha256"]
            or sha256(manifest_path) != p5["slice_label_manifest_sha256"]
            or sha256(FINAL_DEPLOYMENT) != audit["full169_deployment_sha256"]):
        raise RuntimeError("Run A source/manifest/FULL169 provenance mismatch")
    if (int((train.fracture_label == 1).sum()) != p5["train_positive_slices"]
            or int((train.fracture_label == 0).sum()) != p5["train_negative_slices"]
            or int((val.fracture_label == 1).sum()) != p5["validation_positive_slices"]
            or int((val.fracture_label == 0).sum()) != p5["validation_negative_slices"]):
        raise RuntimeError("Multi-scale training labels differ from historical P5 experiment")
    protocol = {
        "experiment": "RUN_A_FROZEN_P3_P4_P5_OOF", "fold": fold,
        "source_detector": str(source), "source_detector_sha256": sha256(source),
        "full169_deployment_sha256_at_start": sha256(FINAL_DEPLOYMENT),
        "slice_label_manifest": str(manifest_path),
        "slice_label_manifest_sha256": sha256(manifest_path),
        "train_studies": int(train.study_id.nunique()), "val_studies": int(val.study_id.nunique()),
        "train_patients": int(train.patient_id.nunique()), "val_patients": int(val.patient_id.nunique()),
        "train_positive_slices": int((train.fracture_label == 1).sum()),
        "train_negative_slices": int((train.fracture_label == 0).sum()),
        "val_positive_slices": int((val.fracture_label == 1).sum()),
        "val_negative_slices": int((val.fracture_label == 0).sum()),
        "ignored_unreviewed_slices": int((~table.is_reviewed.astype(bool)).sum()),
        "pos_weight_train_only": pos_weight, "seed": 42 + fold,
        "features": [{"name": s.name, "backbone_layer": s.layer, "stride": s.stride,
                      "channels": s.channels, "module_type": s.module_type} for s in SPECS],
        "architecture": "GAP(P3/P4/P5)->separate LN(C)-Linear(C,128)-SiLU->concat384->LN384-Linear384,128-SiLU-Dropout0.10-Linear128,1",
        "input": "same reviewed Run A 2.5D physical +/-5mm HU WL800 WW1600 PNG; RGB; resize768",
        "preprocessing_code": "fracture_classifier.training.ReviewedSlices/image_tensor",
        "augmentation": "same as P5: no Mosaic, MixUp or intensity jitter",
        "backbone": "100% frozen and eval, no Stage 2",
        "epochs": EPOCHS, "batch": BATCH, "optimizer": "AdamW", "head_lr": LR,
        "weight_decay": WEIGHT_DECAY, "loss": "BCEWithLogitsLoss(pos_weight=train_neg/train_pos)",
        "checkpoint_selection": "maximum held-out validation slice PR-AUC; strict improvement; earlier epoch wins tie",
        "official_fracture_threshold": 0.5,
    }
    return train, val, pos_weight, protocol


def audit(fold: int) -> dict:
    _train, _val, _weight, protocol = audited_fold(fold)
    checkpoint = torch.load(protocol["source_detector"], map_location="cpu", weights_only=False)
    verify_backbone_graph(checkpoint["model"])
    return protocol


@torch.no_grad()
def evaluate(model: MultiScaleFractureClassifier, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    truth, scores = [], []
    for images, labels in loader:
        probabilities = model.probabilities(images.to(device)).cpu().numpy()
        truth.extend(labels.numpy().tolist())
        scores.extend(probabilities.tolist())
    return slice_metrics(np.asarray(truth, dtype=int), np.asarray(scores, dtype=float))


def _save(path: Path, model: MultiScaleFractureClassifier, epoch: int,
          metrics: dict, protocol: dict) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"model_state": model.state_dict(), "epoch_one_based": epoch,
                "stage": 1, "validation_slice_metrics": metrics,
                "source_detector_sha256": protocol["source_detector_sha256"],
                "slice_label_manifest_sha256": protocol["slice_label_manifest_sha256"]}, temporary)
    os.replace(temporary, path)


def train_fold(fold: int, device_name: str = "cuda:0", batch: int = BATCH) -> dict:
    if batch != BATCH:
        raise ValueError("This matched experiment requires batch=8")
    train, val, pos_weight, protocol = audited_fold(fold)
    output = EXPERIMENT.parent / f"{OUTPUT_PREFIX}{fold}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite multi-scale Fold output: {output}")
    if device_name != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but unavailable")
    seed_all(protocol["seed"])
    model = MultiScaleFractureClassifier.from_detector(Path(protocol["source_detector"])).to(device_name)
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("Frozen backbone unexpectedly trainable")
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    training = DataLoader(ReviewedSlices(train), batch_size=BATCH, shuffle=True, num_workers=0)
    validation = DataLoader(ReviewedSlices(val), batch_size=BATCH, shuffle=False, num_workers=0)
    device = torch.device(device_name)
    optimizer = torch.optim.AdamW(list(model.branches.parameters()) + list(model.head.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    history, best_epoch, best_pr_auc = [], None, -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for images, labels in training:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite multi-scale loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, validation, device)
        metrics.update({"epoch": epoch, "train_loss": float(np.mean(losses))})
        history.append(metrics)
        _save(output / "last_classifier.pt", model, epoch, metrics, protocol)
        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc, best_epoch = metrics["pr_auc"], epoch
            _save(output / "best_classifier.pt", model, epoch, metrics, protocol)
        print(f"fold={fold} epoch={epoch}/{EPOCHS} loss={metrics['train_loss']:.5f} "
              f"slice_PR_AUC={metrics['pr_auc']:.5f}", flush=True)
    if (sha256(Path(protocol["source_detector"])) != protocol["source_detector_sha256"]
            or sha256(FINAL_DEPLOYMENT) != protocol["full169_deployment_sha256_at_start"]):
        raise RuntimeError("Protected detector/FULL169 checkpoint changed during training")
    result = {"fold": fold, "best_epoch": best_epoch,
              "best_validation_slice_pr_auc": best_pr_auc, "history": history,
              "best_checkpoint": str(output / "best_classifier.pt")}
    (output / "classifier_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "COMPLETE.json").write_text(json.dumps({"fold": fold, "stage1_completed": True,
                                                        "epochs_completed": EPOCHS,
                                                        "best_epoch": best_epoch}, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--audit", action="store_true", help="CPU-only audit; do not train or create output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args()
    print(json.dumps(audit(args.fold) if args.audit else train_fold(args.fold, args.device, args.batch),
                     indent=2))


if __name__ == "__main__":
    main()
