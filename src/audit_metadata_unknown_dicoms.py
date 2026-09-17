"""CPU-only header audit for DICOM files absent from organizer metadata."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from dicom_series_guard import canonical_profile, classify_header, discover_headers, orientation_compatible


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DICOM_ROOT = ROOT / "iaaa-contest-bct/Data/training"
DEFAULT_METADATA = ROOT / "iaaa-contest-bct/Data/training_df.pkl"
DEFAULT_CSV = ROOT / "outputs/metadata_unknown_dicom_audit.csv"
DEFAULT_SUMMARY = ROOT / "outputs/metadata_unknown_dicom_audit_summary.json"


def _serialized(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return json.dumps(list(value), separators=(",", ":"))
    return str(value)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_csv.exists() or args.output_summary.exists():
        raise FileExistsError("Refusing to overwrite unknown-DICOM audit outputs")
    metadata = pd.read_pickle(args.metadata).copy()
    metadata["study_id"] = metadata["dicom_series.id"].astype(str)
    metadata["sop_uid"] = metadata["dicom_series.SOPInstanceUID"].astype(str)
    metadata["series_uid"] = metadata["dicom_series.SeriesInstanceUID"].astype(str)
    metadata_keys = set(zip(metadata["study_id"], metadata["sop_uid"]))
    metadata_by_study = {study: group for study, group in metadata.groupby("study_id")}

    rows = []  # type: List[Dict[str, Any]]
    header_errors = []  # type: List[Dict[str, str]]
    scanned_files = 0
    for study_dir in sorted(path for path in args.dicom_root.iterdir() if path.is_dir()):
        try:
            headers = discover_headers(study_dir)
        except Exception as exc:
            header_errors.append({"study_id": study_dir.name, "error": str(exc)})
            continue
        scanned_files += len(headers)
        unknown = [
            header for header in headers
            if (study_dir.name, str(header["SOPInstanceUID"])) not in metadata_keys
        ]
        if not unknown:
            continue
        group = metadata_by_study.get(study_dir.name)
        profile = None
        profile_error = ""
        if group is None:
            profile_error = "Parent study has no organizer metadata"
        else:
            canonical_uids = sorted(set(group["series_uid"].dropna().astype(str)))
            if len(canonical_uids) != 1:
                profile_error = "Expected one canonical SeriesInstanceUID, found %d" % len(canonical_uids)
            else:
                try:
                    profile = canonical_profile(headers, canonical_uids[0], set(group["sop_uid"].astype(str)))
                except Exception as exc:
                    profile_error = str(exc)
        for header in unknown:
            if profile is None:
                classification, reason = "unresolved", profile_error
                matches = None
                geometry = None
            else:
                classification, reason = classify_header(header, profile)
                matches = header.get("SeriesInstanceUID") == profile["canonical_series_uid"]
                geometry = (
                    header.get("Rows") == profile["rows"]
                    and header.get("Columns") == profile["columns"]
                    and orientation_compatible(header.get("ImageOrientationPatient"), profile.get("orientation"))
                )
            rows.append({
                "filesystem_path": header["filesystem_path"],
                "parent_study_directory": study_dir.name,
                "SOPInstanceUID": header["SOPInstanceUID"],
                "SeriesInstanceUID": header["SeriesInstanceUID"],
                "StudyInstanceUID": header["StudyInstanceUID"],
                "Modality": header["Modality"],
                "ImageType": header["ImageType"],
                "Rows": header["Rows"], "Columns": header["Columns"],
                "InstanceNumber": header["InstanceNumber"],
                "ImagePositionPatient": _serialized(header["ImagePositionPatient"]),
                "ImageOrientationPatient": _serialized(header["ImageOrientationPatient"]),
                "PixelSpacing": _serialized(header["PixelSpacing"]),
                "SliceThickness": header["SliceThickness"],
                "SeriesDescription": header["SeriesDescription"],
                "ProtocolName": header["ProtocolName"],
                "canonical_target_SeriesInstanceUID": profile["canonical_series_uid"] if profile else "",
                "series_uid_matches_canonical": matches,
                "geometry_orientation_compatible": geometry,
                "classification": classification,
                "reason": reason,
            })
    if len(rows) != args.expected_unknown:
        raise RuntimeError("Expected %d metadata-unknown DICOMs, found %d" % (args.expected_unknown, len(rows)))
    counts = Counter(str(row["classification"]) for row in rows)
    reasons = Counter(str(row["reason"]) for row in rows)
    summary = {
        "protocol": "metadata_unknown_header_audit_v1",
        "pixel_arrays_decoded": False,
        "scanned_dicom_headers": scanned_files,
        "metadata_unknown_dicoms": len(rows),
        "classification_counts": {name: int(counts.get(name, 0)) for name in (
            "include_target_series", "exclude_foreign_series", "exclude_localizer_or_scout",
            "exclude_incompatible_geometry", "unresolved",
        )},
        "reason_counts": dict(sorted(reasons.items())),
        "studies_with_unknown_dicoms": len({row["parent_study_directory"] for row in rows}),
        "header_read_errors": header_errors,
        "training_policy": "Metadata-unknown slices receive no inferred fracture label and are ineligible for detector supervision.",
        "deployment_policy": "Only include_target_series records are eligible for full-study deployment inference.",
        "csv": str(args.output_csv.resolve()),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-root", type=Path, default=DEFAULT_DICOM_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--expected-unknown", type=int, default=175)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))

