"""Offline, fold-specific Ultralytics training entry point."""
from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

from fracture.utils.config import load_config, save_config
from fracture.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--dataset-yaml", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--run-name", default="detector")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume an interrupted Ultralytics run from a local last.pt checkpoint.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["training"].get("seed", 42))
    model_cfg = cfg["model"]
    if model_cfg.get("use_pretrained"):
        weights = model_cfg.get("pretrained_weights_path")
        if not weights or not Path(weights).is_file(): raise ValueError("Pretrained mode requires an explicit local pretrained_weights_path")
    else:
        weights = model_cfg.get("weights")
        if not weights: raise ValueError("Random/local training requires model.weights pointing to a local architecture YAML or weights file; downloads are forbidden")
        if not Path(weights).is_file(): raise FileNotFoundError(weights)
    from ultralytics import YOLO
    from ultralytics.data import utils as ultralytics_data_utils
    import torch, ultralytics

    # Ultralytics checks for Arial while validating every detection dataset and
    # downloads it when absent. Use the installed system font to keep training
    # strictly offline; plots are disabled for this memory-constrained run.
    system_font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not system_font.is_file():
        raise FileNotFoundError(f"Required offline system font is missing: {system_font}")
    ultralytics_data_utils.check_font = lambda _font="Arial.ttf": system_font

    print(f"Python={platform.python_version()} torch={torch.__version__} CUDA={torch.version.cuda} ultralytics={ultralytics.__version__}")
    output = (Path("outputs") / f"fold_{args.fold}").resolve(); output.mkdir(parents=True, exist_ok=True); save_config(cfg, output / "resolved_config.yaml")
    resume_checkpoint = args.resume_from.resolve() if args.resume_from else None
    if resume_checkpoint and not resume_checkpoint.is_file():
        raise FileNotFoundError(resume_checkpoint)

    started = time.time()
    model = YOLO(str(resume_checkpoint or weights))
    train = cfg["training"]
    augmentations = train.get("augmentations", {})
    if resume_checkpoint:
        # The checkpoint contains the original dataset, optimizer, scheduler,
        # epoch, output directory, and augmentation arguments.
        result = model.train(resume=True)
    else:
        result = model.train(data=args.dataset_yaml, epochs=args.epochs or train["epochs"], imgsz=cfg["preprocessing"]["image_size"], batch=args.batch_size or train["batch_size"], workers=args.workers if args.workers is not None else train["workers"], patience=train["patience"], device=train["device"], seed=train.get("seed", 42), project=str(output), name=args.run_name, exist_ok=False, pretrained=False, amp=bool(train.get("amp", False)), plots=bool(train.get("plots", False)), cache=False, **augmentations)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fold", "training_seconds", "annotation_version", "best_model"]); writer.writeheader(); writer.writerow({"fold": args.fold, "training_seconds": time.time()-started, "annotation_version": cfg["data"]["annotation_version"], "best_model": str(getattr(result, "save_dir", ""))})

if __name__ == "__main__": main()
