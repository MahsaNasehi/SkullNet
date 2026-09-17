"""Detection error matching and review table generation."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); inter = max(0, x2-x1)*max(0, y2-y1)
    return inter / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter, 1e-12)


def classify(gt: list[list[float]], boxes: list[list[float]], scores: list[float], threshold: float = 0.5) -> list[dict]:
    rows = []
    for box, score in zip(boxes, scores, strict=True):
        best = max((iou(box, target) for target in gt), default=0.0)
        rows.append({"error_type": "true_positive" if best >= threshold else "false_positive", "confidence": score, "max_iou": best})
    for target in gt:
        best = max((iou(target, box) for box in boxes), default=0.0)
        if best < threshold: rows.append({"error_type": "false_negative", "confidence": 0.0, "max_iou": best})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--predictions", required=True); parser.add_argument("--output", default="reports/error_analysis.csv"); args = parser.parse_args(); output = []
    with Path(args.predictions).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for result in classify(json.loads(row["gt_boxes"]), json.loads(row["boxes"]), json.loads(row["scores"])): output.append({"series_id": row["series_id"], "sop_uid": row["sop_uid"], "slice_index": row["slice_index"], **result, "review_category": "uncertain", "notes": ""})
    fields = ["series_id", "sop_uid", "slice_index", "error_type", "confidence", "max_iou", "review_category", "notes"]; path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream: writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(output)
if __name__ == "__main__": main()

