"""Run the held-out patient-level test split once after model selection."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data_prepared/skull_hu800_ww1600/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    YOLO(str(args.weights)).val(
        data=str(args.data.resolve()), split="test", imgsz=args.imgsz,
        batch=args.batch, device=args.device, plots=True
    )


if __name__ == "__main__":
    main()
