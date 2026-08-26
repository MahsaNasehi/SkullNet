"""Select false positives only from verified negatives."""
from __future__ import annotations
import argparse, csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--predictions", required=True); parser.add_argument("--output", default="reports/hard_negatives.csv"); parser.add_argument("--confidence", type=float, default=0.5); args = parser.parse_args(); selected = []
    with Path(args.predictions).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("slice_status") == "negative" and float(row.get("max_confidence", 0)) >= args.confidence: selected.append(row)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(selected[0]) if selected else ["series_id", "sop_uid", "slice_status", "max_confidence"]; writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(selected)
if __name__ == "__main__": main()

