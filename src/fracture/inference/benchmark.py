"""Offline study inference benchmark."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from fracture import FracturePredictor


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--weights", required=True); parser.add_argument("--study-dir", action="append", required=True); parser.add_argument("--aggregator"); parser.add_argument("--calibrator"); parser.add_argument("--aggregation-method", default="max"); parser.add_argument("--calibration-method", default="none"); parser.add_argument("--input-mode", default="2.5d"); parser.add_argument("--image-size", type=int, default=768); parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--device", default="0"); parser.add_argument("--output", default="reports/runtime_benchmark.json"); args = parser.parse_args()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    except ImportError: pass
    started = time.perf_counter(); predictor = FracturePredictor(args.weights, aggregator_path=args.aggregator, calibrator_path=args.calibrator, aggregation_method=args.aggregation_method, calibration_method=args.calibration_method, input_mode=args.input_mode, image_size=args.image_size, batch_size=args.batch_size, confidence=0.01, device=args.device, fp16=args.device != "cpu"); load_time = time.perf_counter()-started; rows = []
    for study in args.study_dir:
        probability = predictor.predict(study); rows.append({"study_dir": study, "fracture_prob": probability, **predictor.last_timings})
    try:
        import torch
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except ImportError: peak = 0
    totals = np.asarray([float(row["total"]) for row in rows], dtype=float)
    output = {"model_load_seconds": load_time, "mean_study_seconds": float(totals.mean()), "median_study_seconds": float(np.median(totals)), "p95_study_seconds": float(np.percentile(totals, 95)), "peak_vram_bytes": peak, "studies": rows}; path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8"); print(json.dumps(output, indent=2))
if __name__ == "__main__": main()
