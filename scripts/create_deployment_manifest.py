"""Create a validated, local-only inference manifest after OOF selection."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _artifact(path: str | None, *, minimum_bytes: int, label: str) -> str | None:
    if not path:
        return None
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    if candidate.stat().st_size < minimum_bytes:
        raise ValueError(f"{label} is empty, truncated, or implausibly small: {candidate}")
    try:
        return str(candidate.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(candidate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True)
    parser.add_argument("--aggregator")
    parser.add_argument("--calibrator")
    parser.add_argument("--study-model")
    parser.add_argument("--output", default="models/deployment.yaml")
    parser.add_argument("--input-mode", choices=["single", "2.5d"], default="2.5d")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--window-level", type=float, default=800.0)
    parser.add_argument("--window-width", type=float, default=1600.0)
    parser.add_argument("--context-distance-mm", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.window_width <= 0:
        raise ValueError("window-width must be positive")

    payload = {
        "detector_weights": _artifact(args.detector, minimum_bytes=100_000, label="Detector checkpoint"),
        "aggregator_path": _artifact(args.aggregator, minimum_bytes=100, label="Aggregator"),
        "calibrator_path": _artifact(args.calibrator, minimum_bytes=100, label="Calibrator"),
        "study_model_path": _artifact(args.study_model, minimum_bytes=100_000, label="StudyMIL checkpoint"),
        "input_mode": args.input_mode,
        "image_size": args.image_size,
        "window_level": args.window_level,
        "window_width": args.window_width,
        "context_distance_mm": args.context_distance_mm,
        "batch_size": args.batch_size,
        "confidence": args.confidence,
        "nms_iou": args.nms_iou,
        "device": args.device,
        "fp16": args.device != "cpu",
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Deployment manifest already exists and will not be overwritten: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
