"""Thin CSV adapter around FracturePredictor; contains no fracture logic."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from fracture import FracturePredictor


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", required=True); parser.add_argument("--predictions-file-path", required=True); parser.add_argument("--weights", required=True); parser.add_argument("--aggregator"); parser.add_argument("--calibrator"); parser.add_argument("--study-model"); parser.add_argument("--input-mode", choices=["single", "2.5d"], default="2.5d"); parser.add_argument("--image-size", type=int, default=768); parser.add_argument("--window-level", type=float, default=800.0); parser.add_argument("--window-width", type=float, default=1600.0); parser.add_argument("--context-distance-mm", type=float, default=5.0); parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--confidence", type=float, default=0.01); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); predictor = FracturePredictor(args.weights, aggregator_path=args.aggregator, calibrator_path=args.calibrator, study_model_path=args.study_model, input_mode=args.input_mode, image_size=args.image_size, window_level=args.window_level, window_width=args.window_width, context_distance_mm=args.context_distance_mm, batch_size=args.batch_size, confidence=args.confidence, device=args.device, fp16=args.device != "cpu")
    studies = sorted(path for path in Path(args.data_dir).iterdir() if path.is_dir())
    with Path(args.predictions_file_path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["series_id", "fracture_prob"]); writer.writeheader()
        for study in studies: writer.writerow({"series_id": study.name, "fracture_prob": predictor.predict(str(study))})


if __name__ == "__main__": main()
