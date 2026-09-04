"""Offline acceptance smoke for FracturePredictor (no annotations, local artifacts only)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fracture import FracturePredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--study-dir", action="append", required=True)
    parser.add_argument("--aggregator")
    parser.add_argument("--calibrator")
    parser.add_argument("--study-model")
    parser.add_argument("--aggregation-method", choices=["max", "top3_mean", "consecutive", "logistic"])
    parser.add_argument("--calibration-method", default="none")
    parser.add_argument("--input-mode", default="2.5d")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--window-level", type=float, default=800.0)
    parser.add_argument("--window-width", type=float, default=1600.0)
    parser.add_argument("--context-distance-mm", type=float, default=5.0)
    parser.add_argument("--output", default="reports/acceptance_smoke.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    aggregator = args.aggregator if args.aggregator and Path(args.aggregator).is_file() else None
    calibrator = args.calibrator if args.calibrator and Path(args.calibrator).is_file() else None
    method = args.aggregation_method
    calibration = args.calibration_method

    started = time.perf_counter()
    predictor = FracturePredictor(
        args.weights,
        aggregator_path=aggregator,
        calibrator_path=calibrator,
        study_model_path=args.study_model,
        aggregation_method=method,
        calibration_method=calibration,
        input_mode=args.input_mode,
        image_size=args.image_size,
        window_level=args.window_level,
        window_width=args.window_width,
        context_distance_mm=args.context_distance_mm,
        device=args.device,
        fp16=args.device != "cpu",
        confidence=0.01,
        batch_size=2,
    )
    load_s = time.perf_counter() - started
    rows = []
    for study in args.study_dir:
        probability = predictor.predict(study)
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0
        rows.append({"study_dir": study, "fracture_prob": probability, **predictor.last_timings})

    try:
        import torch

        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except ImportError:
        peak = 0

    payload = {
        "model_load_seconds": load_s,
        "aggregation_method": predictor.aggregator.method,
        "calibration_method": calibration,
        "peak_vram_bytes": peak,
        "studies": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
