"""Dataset audit CLI. Missing JSON remains unknown unless explicitly configured."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path
from .dicom import read_slice
from .annotations import resolve_annotation
from fracture.utils.config import load_config, require_path
from .metadata import load_metadata, slice_label_index


def audit(config: dict) -> tuple[list[dict], dict]:
    data = config["data"]; dicom_root = require_path(config, "data", "dicom_root"); original = require_path(config, "data", "annotation_root")
    corrected = data.get("corrected_annotation_root"); missing_negative = bool(data.get("missing_json_means_negative", False)); metadata_labels = slice_label_index(data)
    rows, errors, widths, heights, areas = [], [], [], [], []
    for series_dir in sorted(x for x in dicom_root.iterdir() if x.is_dir()):
        for path in sorted(x for x in series_dir.rglob("*") if x.is_file() and (x.suffix.lower() == ".dcm" or not x.suffix)):
            try:
                record = read_slice(path, series_dir.name); ann = resolve_annotation(original, corrected, series_dir.name, record.sop_uid, image_shape=record.shape)
                metadata_label = metadata_labels.get((series_dir.name, record.sop_uid))
                status = "positive" if (ann and ann.boxes) or metadata_label is True else "negative" if ann or metadata_label is False or missing_negative else "unknown"
                if ann:
                    for box in ann.boxes:
                        widths.append(box.width); heights.append(box.height); areas.append(box.width * box.height)
                rows.append({"series_id": series_dir.name, "sop_uid": record.sop_uid, "dicom_path": str(path), "annotation_path": str(ann.source_path) if ann else "", "annotation_provenance": ann.provenance if ann else "missing", "slice_status": status, "num_boxes": len(ann.boxes) if ann else 0, "rows": record.shape[0], "columns": record.shape[1], "pixel_spacing": record.pixel_spacing, "slice_thickness": record.slice_thickness, "transfer_syntax_uid": record.transfer_syntax_uid, "decode_error": ""})
            except Exception as exc: errors.append({"series_id": series_dir.name, "dicom_path": str(path), "error": str(exc)})
    counts = Counter(x["slice_status"] for x in rows); studies = {x["series_id"] for x in rows}; positive_studies = {x["series_id"] for x in rows if x["slice_status"] == "positive"}
    uncompressed = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2"}
    transfer_syntaxes = Counter(x["transfer_syntax_uid"] for x in rows)
    image_sizes = Counter(f'{x["rows"]}x{x["columns"]}' for x in rows)
    summary = {"total_studies": len(studies), "total_dicom_slices": len(rows), "positive_studies_from_boxes": len(positive_studies), "positive_slices": counts["positive"], "explicitly_negative_slices": counts["negative"], "unknown_slices": counts["unknown"], "annotated_slices": sum(x["annotation_provenance"] != "missing" for x in rows), "slices_without_annotation_json": sum(x["annotation_provenance"] == "missing" for x in rows), "total_fracture_boxes": sum(x["num_boxes"] for x in rows), "box_width_min_median_max": _min_median_max(widths), "box_height_min_median_max": _min_median_max(heights), "box_area_min_median_max": _min_median_max(areas), "image_size_counts": dict(image_sizes), "transfer_syntax_counts": dict(transfer_syntaxes), "compressed_dicom_count": sum(count for syntax, count in transfer_syntaxes.items() if syntax not in uncompressed), "decoding_or_annotation_errors": len(errors), "missing_json_means_negative": missing_negative, "errors": errors}
    metadata_path = data.get("metadata_path")
    if metadata_path:
        frame = load_metadata(metadata_path); series_col = data.get("series_id_column"); patient_col = data.get("patient_id_column"); label_col = data.get("fracture_label_column")
        series_labels = frame.groupby(series_col)[label_col].max()
        summary.update({"metadata_rows": len(frame), "total_patients": int(frame[patient_col].nunique()) if patient_col else None, "fracture_positive_studies": int(series_labels.sum()), "fracture_negative_studies": int((~series_labels.astype(bool)).sum()), "dicoms_without_metadata": len(rows) - len(metadata_labels), "metadata_without_dicom": max(0, len(metadata_labels) - len(rows))})
    return rows, summary


def _min_median_max(values: list[float]) -> list[float] | None:
    if not values: return None
    import statistics
    return [float(min(values)), float(statistics.median(values)), float(max(values))]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args(); config = load_config(args.config)
    rows, summary = audit(config); report = Path(args.config).resolve().parents[1] / "reports"; report.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else ["series_id", "sop_uid", "slice_status", "num_boxes", "decode_error"]
    with (report / "data_audit.csv").open("w", newline="", encoding="utf-8") as stream: writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    with (report / "data_summary.json").open("w", encoding="utf-8") as stream: json.dump(summary, stream, indent=2)
    with (report / "data_summary.md").open("w", encoding="utf-8") as stream: stream.write("# Data audit summary\n\n" + "\n".join(f"- {k}: {v}" for k, v in summary.items() if k != "errors") + "\n")


if __name__ == "__main__": main()
