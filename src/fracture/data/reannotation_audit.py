"""Validate corrected annotations and quantify the human-reviewed release."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from fracture.data.annotations import parse_annotation
from fracture.data.reannotation import LOG_FIELDS
from fracture.utils.config import load_config, require_path


ACCEPTED_STATUSES = {"accepted", "approved", "complete", "completed"}


def audit_corrections(config: dict, log_path: Path, *, allow_unlogged: bool = False) -> dict:
    original_root = require_path(config, "data", "annotation_root")
    corrected_value = config["data"].get("corrected_annotation_root")
    if not corrected_value:
        raise ValueError("data.corrected_annotation_root is not configured")
    corrected_root = Path(corrected_value)

    log_rows: list[dict[str, str]] = []
    if log_path.is_file():
        with log_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing_columns = set(LOG_FIELDS) - set(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(f"Reannotation log is missing columns: {sorted(missing_columns)}")
            log_rows = list(reader)
    accepted_rows = [
        row
        for row in log_rows
        if str(row.get("status", "")).strip().lower() in ACCEPTED_STATUSES
    ]
    accepted = {
        (str(row.get("series_id", "")), str(row.get("sop_uid", ""))): row
        for row in accepted_rows
    }

    files = sorted(corrected_root.glob("*/*.json"))
    unlogged: list[str] = []
    count_mismatches: list[str] = []
    original_boxes = corrected_boxes = 0
    exact_unchanged = boxes_added = boxes_removed = box_count_changed_files = 0
    corrected_studies: set[str] = set()
    for path in files:
        series_id, sop_uid = path.parent.name, path.stem
        corrected = parse_annotation(path, provenance="corrected")
        original_path = original_root / series_id / path.name
        original = parse_annotation(original_path, provenance="original") if original_path.is_file() else None
        original_values = [tuple(vars(box).values()) for box in (original.boxes if original else ())]
        corrected_values = [tuple(vars(box).values()) for box in corrected.boxes]
        original_boxes += len(original_values)
        corrected_boxes += len(corrected_values)
        exact_unchanged += int(original_values == corrected_values)
        boxes_added += max(0, len(corrected_values) - len(original_values))
        boxes_removed += max(0, len(original_values) - len(corrected_values))
        box_count_changed_files += int(len(original_values) != len(corrected_values))
        corrected_studies.add(series_id)
        if (series_id, sop_uid) not in accepted:
            unlogged.append(str(path))
        else:
            log_row = accepted[(series_id, sop_uid)]
            try:
                logged_original = int(log_row["original_num_boxes"])
                logged_corrected = int(log_row["corrected_num_boxes"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid box counts in provenance log for {series_id}/{sop_uid}") from exc
            if logged_original != len(original_values) or logged_corrected != len(corrected_values):
                count_mismatches.append(f"{series_id}/{sop_uid}")
    if unlogged and not allow_unlogged:
        raise ValueError(
            "Corrected annotations without an accepted provenance-log row: "
            + ", ".join(unlogged[:10])
        )
    if count_mismatches:
        raise ValueError(
            "Corrected annotation counts disagree with the provenance log: "
            + ", ".join(count_mismatches[:10])
        )

    modification_counts = Counter(
        str(row.get("modification_type", "")).strip()
        for row in accepted_rows
        if str(row.get("modification_type", "")).strip()
    )
    reviewed_studies = {
        str(row.get("series_id", ""))
        for row in log_rows
        if str(row.get("series_id", ""))
    }
    return {
        "annotation_version": config["data"].get("annotation_version"),
        "corrected_annotation_files": len(files),
        "corrected_studies": len(corrected_studies),
        "original_boxes_on_corrected_slices": original_boxes,
        "corrected_boxes_on_corrected_slices": corrected_boxes,
        "net_box_change": corrected_boxes - original_boxes,
        "boxes_added_from_count_difference": boxes_added,
        "boxes_removed_from_count_difference": boxes_removed,
        "box_count_changed_files": box_count_changed_files,
        "exactly_unchanged_corrected_files": exact_unchanged,
        "review_log_rows": len(log_rows),
        "accepted_log_rows": len(accepted_rows),
        "reviewed_studies": len(reviewed_studies),
        "modification_type_counts": dict(sorted(modification_counts.items())),
        "unlogged_corrected_files": unlogged,
        "log_count_mismatches": count_mismatches,
        "resolution_policy": "corrected file overrides original; otherwise original is retained",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log", default="reports/reannotation_log.csv")
    parser.add_argument("--output", default="reports/reannotation_summary.json")
    parser.add_argument("--allow-unlogged", action="store_true")
    args = parser.parse_args()
    payload = audit_corrections(load_config(args.config), Path(args.log), allow_unlogged=args.allow_unlogged)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
