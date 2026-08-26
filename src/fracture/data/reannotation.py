"""Generate review candidates only; never mutate annotations."""
from __future__ import annotations
import argparse, csv
from pathlib import Path

FIELDS = ["series_id", "sop_uid", "slice_index", "candidate_type", "model_confidence", "gt_num_boxes", "max_iou", "reason", "status", "reviewer", "notes"]
LOG_FIELDS = ["series_id", "sop_uid", "slice_index", "original_num_boxes", "corrected_num_boxes", "modification_type", "reason", "reviewer", "timestamp", "status", "notes"]


def candidates_from_rows(rows: list[dict], high_conf: float = 0.7, tiny_area: float = 64, large_area: float = 100000) -> list[dict]:
    output = []
    for row in rows:
        confidence, count = float(row.get("max_confidence", 0)), int(row.get("gt_num_boxes", 0))
        types = []
        if count == 0 and confidence >= high_conf: types.append("possible_missing_box")
        if count > 0 and confidence < 0.05: types.append("possible_false_box_or_missed_gt")
        area = float(row.get("min_box_area", 0)); max_area = float(row.get("max_box_area", 0))
        if count and area < tiny_area: types.append("tiny_box")
        if max_area > large_area: types.append("large_outlier_box")
        if row.get("geometry_issue"): types.append("suspicious_geometry")
        if row.get("overlap_issue"): types.append("overlapping_boxes")
        if row.get("neighbor_inconsistency"): types.append("neighbor_inconsistency")
        for kind in types: output.append({k: row.get(k, "") for k in FIELDS} | {"candidate_type": kind, "model_confidence": confidence, "gt_num_boxes": count, "reason": "OOF/QC disagreement for human review", "status": "pending"})
    return output


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream: writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--output", default="reports/reannotation_candidates.csv"); args = parser.parse_args()
    with Path(args.input).open(newline="", encoding="utf-8") as stream: rows = list(csv.DictReader(stream))
    write_csv(candidates_from_rows(rows), Path(args.output), FIELDS)
if __name__ == "__main__": main()

