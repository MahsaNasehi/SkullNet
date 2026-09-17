"""Run Ultralytics box metrics on the held-out Tier1-only view."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fracture.utils.pretrained import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="YAML created by create_tier1_validation_view")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Validation reports are immutable: {output}")
    weights, data = Path(args.weights).resolve(), Path(args.data).resolve()
    if not weights.is_file() or not data.is_file():
        raise FileNotFoundError(f"Missing weights or data YAML: {weights}, {data}")
    audit_path = data.parent / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("Tier1 validation view is missing audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("created_yolo_label_files") != 0 or "Tier1" not in audit.get("purpose", ""):
        raise ValueError("Data YAML is not an audited Tier1-only validation view")
    os.environ["YOLO_OFFLINE"] = "true"
    from ultralytics import YOLO
    from ultralytics.data import utils as ultralytics_data_utils

    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    ultralytics_data_utils.check_font = lambda _font="Arial.ttf": font
    result = YOLO(str(weights)).val(
        data=str(data), imgsz=args.image_size, batch=args.batch_size,
        device=args.device, plots=False, verbose=False,
    )
    metrics = {
        "weights": str(weights), "weights_sha256": sha256_file(weights),
        "tier1_view": str(data), "tier1_view_sha256": sha256_file(data),
        "view_audit": audit,
        "precision": float(result.results_dict.get("metrics/precision(B)", float("nan"))),
        "recall": float(result.results_dict.get("metrics/recall(B)", float("nan"))),
        "mAP50": float(result.results_dict.get("metrics/mAP50(B)", float("nan"))),
        "mAP50_95": float(result.results_dict.get("metrics/mAP50-95(B)", float("nan"))),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
