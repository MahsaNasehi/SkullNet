"""Offline study inference benchmark."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from fracture import FracturePredictor


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--weights", required=True); parser.add_argument("--study-dir", action="append", required=True); parser.add_argument("--aggregator"); parser.add_argument("--calibrator"); parser.add_argument("--output", default="reports/runtime_benchmark.json"); args = parser.parse_args()
    started = time.perf_counter(); predictor = FracturePredictor(args.weights, aggregator_path=args.aggregator, calibrator_path=args.calibrator); load_time = time.perf_counter()-started; rows = []
    for study in args.study_dir:
        probability = predictor.predict(study); rows.append({"study_dir": study, "fracture_prob": probability, **predictor.last_timings})
    try:
        import torch
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except ImportError: peak = 0
    output = {"model_load_seconds": load_time, "peak_vram_bytes": peak, "studies": rows}; path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(output, indent=2), encoding="utf-8")
if __name__ == "__main__": main()

