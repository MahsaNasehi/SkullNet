"""Train a compact patient-folded study-level MIL classifier on rendered CT slices."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from fracture.data.metadata import load_metadata
from fracture.models.study_mil import StudyMIL
from fracture.utils.config import load_config, require_path
from fracture.utils.seed import seed_everything


class StudyDataset(Dataset):
    def __init__(
        self,
        series_ids: list[str],
        rows_by_series: dict[str, list[dict[str, str]]],
        labels: dict[str, int],
        *,
        image_size: int,
        training: bool,
    ):
        self.series_ids = series_ids
        self.rows_by_series = rows_by_series
        self.labels = labels
        self.image_size = int(image_size)
        self.training = training

    def __len__(self) -> int:
        return len(self.series_ids)

    def __getitem__(self, index: int):
        series_id = self.series_ids[index]
        rows = sorted(self.rows_by_series[series_id], key=lambda row: int(row["slice_index"]))
        images = []
        slice_labels = []
        slice_loss_mask = []
        horizontal_flip = self.training and random.random() < 0.5
        for row in rows:
            image_path = row["image_path"]
            variants = [value for value in row.get("window_variant_paths", "").split(";") if value]
            if self.training and variants:
                image_path = random.choice([image_path, *variants])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot read rendered study image: {image_path}")
            if horizontal_flip:
                image = np.ascontiguousarray(image[:, ::-1])
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            images.append(np.moveaxis(image.astype(np.float32) / 255.0, -1, 0))
            slice_labels.append(float(str(row["is_positive"]).lower() in {"1", "true", "yes"}))
            slice_loss_mask.append(str(row.get("slice_status", "unknown")) != "unknown")
        return (
            series_id,
            torch.from_numpy(np.stack(images)),
            torch.tensor(slice_labels, dtype=torch.float32),
            torch.tensor(slice_loss_mask, dtype=torch.bool),
            torch.tensor(float(self.labels[series_id]), dtype=torch.float32),
        )


def _collate(batch):
    if len(batch) != 1:
        raise ValueError("StudyMIL uses batch_size=1 to bound memory for variable-length CT studies")
    series_id, images, slice_labels, slice_loss_mask, study_label = batch[0]
    return (
        series_id,
        images.unsqueeze(0),
        slice_labels.unsqueeze(0),
        slice_loss_mask.unsqueeze(0),
        study_label.unsqueeze(0),
    )


def _read_manifest(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["series_id"])].append(row)
    return grouped


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def _evaluate(model: StudyMIL, loader: DataLoader, device: torch.device, fp16: bool):
    model.eval()
    rows = []
    for series_id, images, _, _, label in loader:
        images = images.to(device)
        mask = torch.ones(images.shape[:2], dtype=torch.bool, device=device)
        with torch.autocast(device_type=device.type, enabled=fp16):
            logit = model(images, mask)["study_logits"]
        rows.append({
            "series_id": series_id,
            "y_true": int(label.item()),
            "study_model_probability": float(torch.sigmoid(logit.float())[0].cpu()),
        })
    y = np.asarray([row["y_true"] for row in rows], dtype=int)
    p = np.asarray([row["study_model_probability"] for row in rows], dtype=float)
    return rows, {
        "pr_auc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Fixed-epoch final refit on all studies after OOF model selection; writes no held-out metrics.",
    )
    parser.add_argument("--run-name", default="study_mil")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if args.all_data == (args.fold is not None):
        raise ValueError("Specify exactly one of --fold or --all-data")

    cfg = load_config(args.config)
    seed = int(cfg.get("training", {}).get("seed", 42))
    seed_everything(seed)
    mil = cfg.get("study_model") or {}
    device_value = mil.get("device", cfg.get("training", {}).get("device", 0))
    device = torch.device(f"cuda:{device_value}" if isinstance(device_value, int) or str(device_value).isdigit() else str(device_value))
    fp16 = bool(mil.get("amp", True) and device.type == "cuda")

    fold_payload = json.loads(Path(cfg["split"]["path"]).read_text(encoding="utf-8"))["folds"]
    fold = next((item for item in fold_payload if int(item["fold"]) == args.fold), None) if not args.all_data else None
    if not args.all_data and fold is None:
        raise ValueError(f"Fold {args.fold} is not configured")
    manifest_path = Path(args.dataset_root) / "manifest.csv"
    rows_by_series = _read_manifest(manifest_path)
    metadata = load_metadata(require_path(cfg, "data", "metadata_path"))
    series_col = cfg["data"]["series_id_column"]
    label_col = cfg["data"]["fracture_label_column"]
    labels = {
        str(series): int(bool(group[label_col].max()))
        for series, group in metadata.groupby(series_col)
    }
    train_ids = (
        sorted(rows_by_series)
        if args.all_data
        else [str(value) for value in fold["train_series"] if str(value) in rows_by_series]
    )
    val_ids = [] if args.all_data else [str(value) for value in fold["val_series"] if str(value) in rows_by_series]
    if not train_ids or (not args.all_data and not val_ids):
        raise ValueError("MIL train or validation partition is empty")

    image_size = int(mil.get("image_size", 256))
    train_dataset = StudyDataset(train_ids, rows_by_series, labels, image_size=image_size, training=True)
    val_dataset = StudyDataset(val_ids, rows_by_series, labels, image_size=image_size, training=False) if val_ids else None
    positives = sum(labels[series_id] for series_id in train_ids)
    negatives = len(train_ids) - positives
    sample_weights = [0.5 / (positives if labels[sid] else negatives) for sid in train_ids]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ids), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate) if val_dataset else None

    model = StudyMIL(
        embedding_dim=int(mil.get("embedding_dim", 128)),
        pooling=str(mil.get("pooling", "topk")),
        top_k=int(mil.get("top_k", 3)),
        encoder_chunk_size=int(mil.get("encoder_chunk_size", 16)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(mil.get("learning_rate", 3e-4)),
        weight_decay=float(mil.get("weight_decay", 1e-4)),
    )
    # WeightedRandomSampler already yields approximately balanced study classes;
    # applying a second study-level positive weight would over-correct the 8.3%
    # prevalence and strongly bias probabilities upward.
    study_loss = nn.BCEWithLogitsLoss()
    train_slice_rows = [row for series_id in train_ids for row in rows_by_series[series_id]]
    known_slice_rows = [row for row in train_slice_rows if str(row.get("slice_status", "unknown")) != "unknown"]
    positive_slices = sum(str(row["is_positive"]).lower() in {"1", "true", "yes"} for row in known_slice_rows)
    negative_slices = len(known_slice_rows) - positive_slices
    slice_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([min(30.0, negative_slices / max(1, positive_slices))], device=device)
    )
    auxiliary_weight = float(mil.get("slice_loss_weight", 0.25))
    scaler = torch.amp.GradScaler(device.type, enabled=fp16)
    epochs = int(args.epochs or mil.get("epochs", 50))
    patience = int(mil.get("patience", 12))
    output = Path("outputs") / ("final" if args.all_data else f"fold_{args.fold}") / args.run_name
    if output.exists():
        raise FileExistsError(f"MIL output already exists and will not be overwritten: {output}")
    weights_dir = output / "weights"
    weights_dir.mkdir(parents=True)
    history: list[dict] = []
    best_pr = -1.0
    epochs_without_improvement = 0
    preprocessing = {
        "window_level": float(cfg["preprocessing"]["window_level"]),
        "window_width": float(cfg["preprocessing"]["window_width"]),
        "input_mode": str(cfg["preprocessing"]["input_mode"]),
        "boundary_mode": str(cfg["preprocessing"].get("boundary_mode", "repeat")),
        "context_distance_mm": cfg["preprocessing"].get("context_distance_mm"),
        "image_size": image_size,
    }
    provenance = {
        "fold": "all" if args.all_data else args.fold,
        "protocol": "fixed_epoch_final_refit" if args.all_data else "heldout_patient_fold",
        "config_path": str(Path(args.config).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "train_studies": len(train_ids),
        "train_positive_studies": positives,
        "validation_studies": len(val_ids),
        "validation_positive_studies": sum(labels[series_id] for series_id in val_ids),
        "train_slices": len(train_slice_rows),
        "train_known_slices": len(known_slice_rows),
        "train_unknown_slices": len(train_slice_rows) - len(known_slice_rows),
        "train_positive_slices": positive_slices,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "external_pretrained_weights": False,
    }
    (output / "initialization.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _, images, slice_labels, slice_loss_mask, label in train_loader:
            images = images.to(device)
            slice_labels = slice_labels.to(device)
            slice_loss_mask = slice_loss_mask.to(device)
            label = label.to(device)
            mask = torch.ones(images.shape[:2], dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=fp16):
                result = model(images, mask)
                loss = study_loss(result["study_logits"], label)
                if slice_loss_mask.any():
                    loss = loss + auxiliary_weight * slice_loss(
                        result["slice_logits"][slice_loss_mask],
                        slice_labels[slice_loss_mask],
                    )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        predictions, metrics = _evaluate(model, val_loader, device, fp16) if val_loader else ([], {})
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        checkpoint_metadata = provenance | {"epoch": epoch, "validation": metrics or None}
        torch.save(model.checkpoint_payload(preprocessing, checkpoint_metadata), weights_dir / "last.pt")
        if args.all_data:
            # Epoch count was selected using OOF histories before this refit;
            # "best.pt" is simply the final fixed epoch, not an in-sample best.
            torch.save(model.checkpoint_payload(preprocessing, checkpoint_metadata), weights_dir / "best.pt")
        elif metrics["pr_auc"] > best_pr + 1e-6:
            best_pr = metrics["pr_auc"]
            epochs_without_improvement = 0
            torch.save(model.checkpoint_payload(preprocessing, checkpoint_metadata), weights_dir / "best.pt")
            with (output / "heldout_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                fields = ["series_id", "y_true", "study_model_probability", "fold", "prediction_protocol"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    row | {"fold": args.fold, "prediction_protocol": "heldout_patient_fold"}
                    for row in predictions
                ])
        elif not args.all_data:
            epochs_without_improvement += 1
        print(json.dumps(history[-1]))
        if not args.all_data and epochs_without_improvement >= patience:
            break

    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    main()
