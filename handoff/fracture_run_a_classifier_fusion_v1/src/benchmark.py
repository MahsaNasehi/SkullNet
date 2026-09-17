"""Manual-only handoff benchmark; never invoked by the packaging script."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from predictor import FractureFusionPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="append", type=Path, required=True)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--target-series-uid")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    predictor = FractureFusionPredictor(args.package_root, device=args.device,
                                       batch_size=args.batch_size)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    start = time.perf_counter()
    total_slices = 0
    results = []
    for study in args.study:
        probability = predictor.predict_study(study, target_series_uid=args.target_series_uid)
        total_slices += predictor.last_slice_count
        results.append({"study": str(study), "fracture_prob": probability,
                        "slices": predictor.last_slice_count})
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
        allocated = torch.cuda.max_memory_allocated(args.device)
        reserved = torch.cuda.max_memory_reserved(args.device)
    else:
        allocated = reserved = 0
    print(json.dumps({"study_count": len(args.study), "slice_count": total_slices,
                      "wall_clock_seconds": time.perf_counter() - start,
                      "peak_cuda_allocated_bytes": allocated,
                      "peak_cuda_reserved_bytes": reserved,
                      "predictions": results}, indent=2))


if __name__ == "__main__":
    main()
