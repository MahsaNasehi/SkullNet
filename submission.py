"""Thin CSV adapter around FracturePredictor; contains no fracture logic."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from fracture import FracturePredictor


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", required=True); parser.add_argument("--predictions-file-path", required=True); parser.add_argument("--weights", required=True); parser.add_argument("--aggregator"); parser.add_argument("--calibrator")
    args = parser.parse_args(); predictor = FracturePredictor(args.weights, aggregator_path=args.aggregator, calibrator_path=args.calibrator)
    studies = sorted(path for path in Path(args.data_dir).iterdir() if path.is_dir())
    with Path(args.predictions_file_path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["series_id", "fracture_prob"]); writer.writeheader()
        for study in studies: writer.writerow({"series_id": study.name, "fracture_prob": predictor.predict(str(study))})


if __name__ == "__main__": main()
