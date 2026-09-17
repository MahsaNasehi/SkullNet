"""Manual-only, one-fold-at-a-time Run A frozen-backbone classifier training."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fracture_classifier.data import EXPERIMENT, FINAL_DEPLOYMENT, detector_checkpoints, fold_rows, sha256
from fracture_classifier.model import FractureSliceClassifier


IMG_SIZE = 768
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 5


def image_tensor(image: np.ndarray) -> torch.Tensor:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel Run A 2.5D image")
    # Prepared PNGs are read BGR by OpenCV; Ultralytics reverses to original RGB.
    if image.shape[:2] != (IMG_SIZE, IMG_SIZE):
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)


class ReviewedSlices(Dataset):
    def __init__(self, rows: pd.DataFrame):
        if not rows["is_reviewed"].astype(bool).all() or rows["fracture_label"].isna().any():
            raise ValueError("Unknown/unreviewed slice cannot enter classifier training")
        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[index]
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(row["image_path"])
        # No Mosaic or HU/intensity jitter. BGR readback is reversed as in YOLO.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_tensor(rgb), torch.tensor(float(row["fracture_label"]), dtype=torch.float32)


def slice_metrics(truth: np.ndarray, scores: np.ndarray) -> dict:
    if set(np.unique(truth)) != {0, 1}:
        raise RuntimeError("Slice validation must contain both classes for PR-AUC")
    labels = scores >= 0.5
    return {
        "pr_auc": float(average_precision_score(truth, scores)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "precision_at_0_5": float(precision_score(truth, labels, zero_division=0)),
        "recall_at_0_5": float(recall_score(truth, labels, zero_division=0)),
        "f1_at_0_5": float(f1_score(truth, labels, zero_division=0)),
    }


def _save(path: Path, model: nn.Module, epoch: int, metrics: dict, protocol: dict, stage: int) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"model_state": model.state_dict(), "epoch_one_based": epoch,
                "stage": stage, "validation_slice_metrics": metrics,
                "source_detector_sha256": protocol["source_detector_sha256"],
                "slice_label_manifest_sha256": protocol["slice_label_manifest_sha256"]}, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(model: FractureSliceClassifier, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    truth, scores = [], []
    for images, labels in loader:
        probabilities = model.probabilities(images.to(device)).cpu().numpy()
        truth.extend(labels.numpy().tolist())
        scores.extend(probabilities.tolist())
    return slice_metrics(np.asarray(truth, dtype=int), np.asarray(scores, dtype=float))


def _fit_stage(model: FractureSliceClassifier, train: DataLoader, val: DataLoader,
               device: torch.device, pos_weight: float, output: Path,
               protocol: dict, stage: int, epochs: int) -> dict:
    if stage == 1:
        model.freeze_backbone()
        groups = [{"params": model.head.parameters(), "lr": 1e-4}]
        best_path, last_path = output / "best_classifier.pt", output / "last_classifier.pt"
    else:
        model.unfreeze_last_stage()
        groups = [{"params": model.head.parameters(), "lr": 1e-4},
                  {"params": model.backbone[10].parameters(), "lr": 1e-5}]
        best_path, last_path = output / "best_stage2_classifier.pt", output / "last_stage2_classifier.pt"
    optimizer = torch.optim.AdamW(groups, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    best_pr_auc, best_epoch, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, labels in train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite classifier loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(losses))
        history.append(metrics)
        _save(last_path, model, epoch, metrics, protocol, stage)
        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc, best_epoch = metrics["pr_auc"], epoch
            _save(best_path, model, epoch, metrics, protocol, stage)
        print(f"fold={protocol['fold']} stage={stage} epoch={epoch}/{epochs} "
              f"loss={metrics['train_loss']:.5f} val_PR_AUC={metrics['pr_auc']:.5f}", flush=True)
    return {"best_epoch": best_epoch, "best_validation_slice_pr_auc": best_pr_auc,
            "history": history, "best_checkpoint": str(best_path), "last_checkpoint": str(last_path)}


def train_fold(fold: int, device_name: str, batch: int, stage2: bool) -> dict:
    if fold not in range(5) or batch <= 0:
        raise ValueError("Invalid fold or batch")
    labels_path = EXPERIMENT / "slice_label_manifest.csv"
    audit_path = EXPERIMENT / "slice_label_audit.json"
    if not labels_path.is_file() or not audit_path.is_file():
        raise RuntimeError("Run CPU label audit first: python -m fracture_classifier.data")
    table = pd.read_csv(labels_path, dtype={"patient_id": str, "study_id": str, "sop_uid": str,
                                             "image_path": str, "label_path": str})
    train, val, pos_weight = fold_rows(table, fold)
    audit = json.loads(audit_path.read_text())
    source = detector_checkpoints()[fold]
    source_hash = sha256(source)
    deployment_hash = sha256(FINAL_DEPLOYMENT)
    if source_hash != audit["detector_sha256"][str(fold)] or deployment_hash != audit["full169_deployment_sha256"]:
        raise RuntimeError("Source Run A/FULL169 checkpoints changed since CPU audit")
    output = EXPERIMENT.parent / f"run_a_fracture_classifier_fold{fold}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite classifier Fold output: {output}")
    if device_name != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but unavailable")
    seed = 42 + fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = FractureSliceClassifier.from_detector(source).to(device_name)
    protocol = {
        "experiment": "RUN_A_FRACTURE_CLASSIFIER_OOF", "fold": fold,
        "source_detector": str(source), "source_detector_sha256": source_hash,
        "full169_deployment_sha256_at_start": deployment_hash,
        "slice_label_manifest_sha256": sha256(labels_path),
        "training_patients": int(train["patient_id"].nunique()),
        "heldout_patients": int(val["patient_id"].nunique()),
        "patient_overlap": 0, "train_positive_slices": int((train["fracture_label"] == 1).sum()),
        "train_negative_slices": int((train["fracture_label"] == 0).sum()),
        "validation_positive_slices": int((val["fracture_label"] == 1).sum()),
        "validation_negative_slices": int((val["fracture_label"] == 0).sum()),
        "pos_weight_train_only": pos_weight, "seed": seed,
        "architecture": "Run A backbone layers 0..10 P5 C2PSA 512ch -> GAP -> LN512 -> Linear512x128 -> SiLU -> Dropout0.1 -> Linear128x1",
        "input": "reviewed prepared 2.5D +/-5mm HU WL800 WW1600 PNG; RGB; resize 768; no Mosaic or intensity jitter",
        "stage1": {"epochs": STAGE1_EPOCHS, "head_lr": 1e-4, "weight_decay": 5e-4,
                   "criterion": "weighted BCEWithLogitsLoss", "checkpoint_metric": "slice PR-AUC"},
        "stage2": {"enabled": stage2, "epochs": STAGE2_EPOCHS if stage2 else 0,
                   "head_lr": 1e-4, "last_backbone_block_lr": 1e-5,
                   "weight_decay": 5e-4},
        "batch": batch, "official_fracture_threshold": 0.5,
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    training_loader = DataLoader(ReviewedSlices(train), batch_size=batch, shuffle=True, num_workers=0)
    validation_loader = DataLoader(ReviewedSlices(val), batch_size=batch, shuffle=False, num_workers=0)
    device = torch.device(device_name)
    stage1_result = _fit_stage(model, training_loader, validation_loader, device,
                               pos_weight, output, protocol, 1, STAGE1_EPOCHS)
    report = {"fold": fold, "stage1": stage1_result, "stage2": None}
    if stage2:
        best = torch.load(output / "best_classifier.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best["model_state"], strict=True)
        report["stage2"] = _fit_stage(model, training_loader, validation_loader, device,
                                      pos_weight, output, protocol, 2, STAGE2_EPOCHS)
    if sha256(source) != source_hash or sha256(FINAL_DEPLOYMENT) != deployment_hash:
        raise RuntimeError("Protected detector checkpoint changed during classifier training")
    (output / "classifier_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "COMPLETE.json").write_text(json.dumps({"fold": fold, "stage1_completed": True,
                                                        "stage2_completed": stage2}, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--stage2", action="store_true", help="Optional isolated last-backbone-block fine-tuning")
    args = parser.parse_args()
    print(json.dumps(train_fold(args.fold, args.device, args.batch, args.stage2), indent=2))


if __name__ == "__main__":
    main()
