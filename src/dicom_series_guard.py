"""CPU-only DICOM header inspection and deterministic target-series guards."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


HEADER_TAGS = [
    "SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID", "Modality",
    "ImageType", "Rows", "Columns", "InstanceNumber", "ImagePositionPatient",
    "ImageOrientationPatient", "PixelSpacing", "SliceThickness",
    "SeriesDescription", "ProtocolName",
]
LOCALIZER_TERMS = ("LOCALIZER", "SCOUT", "TOPOGRAM", "SURVIEW", "SCANOGRAM")
CLASSIFICATIONS = (
    "include_target_series", "exclude_foreign_series", "exclude_localizer_or_scout",
    "exclude_incompatible_geometry", "unresolved",
)


def _float_tuple(value: Any) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\\".join(str(item) for item in value)
    return str(value)


def read_header(path: Path, parent_study: str) -> Dict[str, Any]:
    """Read only DICOM metadata; pixel data is never decoded."""
    import pydicom
    from pydicom import config as pydicom_config

    # Organizer anonymization deliberately uses stable ID_* strings in UID
    # fields. Suppress only value-representation validation noise; header
    # parsing errors still propagate.
    previous_validation = pydicom_config.settings.reading_validation_mode
    pydicom_config.settings.reading_validation_mode = pydicom_config.IGNORE
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=HEADER_TAGS)
    finally:
        pydicom_config.settings.reading_validation_mode = previous_validation
    return {
        "filesystem_path": str(path.resolve()),
        "parent_study_directory": str(parent_study),
        "SOPInstanceUID": _text(getattr(ds, "SOPInstanceUID", path.stem)),
        "SeriesInstanceUID": _text(getattr(ds, "SeriesInstanceUID", None)),
        "StudyInstanceUID": _text(getattr(ds, "StudyInstanceUID", None)),
        "Modality": _text(getattr(ds, "Modality", None)),
        "ImageType": _text(getattr(ds, "ImageType", None)),
        "Rows": int(ds.Rows) if getattr(ds, "Rows", None) is not None else None,
        "Columns": int(ds.Columns) if getattr(ds, "Columns", None) is not None else None,
        "InstanceNumber": int(ds.InstanceNumber) if getattr(ds, "InstanceNumber", None) is not None else None,
        "ImagePositionPatient": _float_tuple(getattr(ds, "ImagePositionPatient", None)),
        "ImageOrientationPatient": _float_tuple(getattr(ds, "ImageOrientationPatient", None)),
        "PixelSpacing": _float_tuple(getattr(ds, "PixelSpacing", None)),
        "SliceThickness": float(ds.SliceThickness) if getattr(ds, "SliceThickness", None) is not None else None,
        "SeriesDescription": _text(getattr(ds, "SeriesDescription", None)),
        "ProtocolName": _text(getattr(ds, "ProtocolName", None)),
    }


def discover_headers(study_dir: Path) -> List[Dict[str, Any]]:
    paths = sorted(
        path for path in study_dir.rglob("*")
        if path.is_file() and (path.suffix.lower() == ".dcm" or not path.suffix)
    )
    if not paths:
        raise FileNotFoundError("No DICOM files in %s" % study_dir)
    return [read_header(path, study_dir.name) for path in paths]


def is_localizer(header: Dict[str, Any]) -> bool:
    haystack = "|".join(
        str(header.get(name, "")).upper()
        for name in ("ImageType", "SeriesDescription", "ProtocolName")
    )
    return any(term in haystack for term in LOCALIZER_TERMS)


def orientation_compatible(
    candidate: Optional[Sequence[float]], reference: Optional[Sequence[float]], tolerance: float = 1e-3
) -> bool:
    if candidate is None or reference is None or len(candidate) != 6 or len(reference) != 6:
        return False
    return bool(np.allclose(np.asarray(candidate), np.asarray(reference), rtol=0.0, atol=tolerance))


def physical_position(header: Dict[str, Any]) -> Optional[float]:
    orientation = header.get("ImageOrientationPatient")
    position = header.get("ImagePositionPatient")
    if orientation is None or position is None or len(orientation) != 6 or len(position) != 3:
        return None
    row = np.asarray(orientation[:3], dtype=float)
    column = np.asarray(orientation[3:], dtype=float)
    normal = np.cross(row, column)
    if float(np.linalg.norm(normal)) == 0.0:
        return None
    return float(np.dot(normal, np.asarray(position, dtype=float)))


def canonical_profile(
    headers: Sequence[Dict[str, Any]], canonical_series_uid: str, metadata_sops: set
) -> Dict[str, Any]:
    known = [
        header for header in headers
        if header["SOPInstanceUID"] in metadata_sops
        and header.get("SeriesInstanceUID") == canonical_series_uid
        and not is_localizer(header)
    ]
    if not known:
        raise ValueError("No metadata-backed headers in canonical target series %s" % canonical_series_uid)
    shapes = [(header.get("Rows"), header.get("Columns")) for header in known]
    valid_shapes = [shape for shape in shapes if None not in shape]
    if not valid_shapes:
        raise ValueError("Canonical target series has no usable Rows/Columns")
    shape = Counter(valid_shapes).most_common(1)[0][0]
    orientations = [header.get("ImageOrientationPatient") for header in known]
    orientations = [orientation for orientation in orientations if orientation is not None and len(orientation) == 6]
    orientation = tuple(orientations[0]) if orientations else None
    return {
        "canonical_series_uid": canonical_series_uid,
        "rows": int(shape[0]), "columns": int(shape[1]),
        "orientation": orientation,
        "metadata_backed_reference_headers": len(known),
    }


def classify_header(header: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[str, str]:
    if is_localizer(header):
        return "exclude_localizer_or_scout", "ImageType/SeriesDescription/ProtocolName indicates localizer or scout"
    series_uid = str(header.get("SeriesInstanceUID") or "")
    if not series_uid:
        return "unresolved", "SeriesInstanceUID is missing"
    if series_uid != profile["canonical_series_uid"]:
        return "exclude_foreign_series", "SeriesInstanceUID differs from canonical target CT series"
    if str(header.get("Modality") or "").upper() not in {"", "CT"}:
        return "exclude_foreign_series", "Modality is not CT"
    if header.get("Rows") != profile["rows"] or header.get("Columns") != profile["columns"]:
        return "exclude_incompatible_geometry", "Rows/Columns differ from canonical target series"
    reference = profile.get("orientation")
    candidate = header.get("ImageOrientationPatient")
    if reference is None and candidate is None:
        if header.get("InstanceNumber") is None:
            return "unresolved", "Orientation/position and InstanceNumber are unavailable"
        return "include_target_series", "Target series and shape match; explicit InstanceNumber ordering fallback required"
    if not orientation_compatible(candidate, reference):
        return "exclude_incompatible_geometry", "ImageOrientationPatient differs from canonical target series"
    if header.get("ImagePositionPatient") is None:
        return "unresolved", "Target geometry matches but ImagePositionPatient is missing"
    if physical_position(header) is None:
        return "unresolved", "Physical position cannot be derived from orientation and position"
    return "include_target_series", "Canonical target SeriesInstanceUID and compatible geometry/orientation"


def select_and_order_target_headers(
    headers: Sequence[Dict[str, Any]], profile: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """Filter one study and establish a deterministic physical slice order."""
    sop_counts = Counter(str(header.get("SOPInstanceUID") or "") for header in headers)
    duplicates = [sop for sop, count in sop_counts.items() if not sop or count > 1]
    if duplicates:
        raise ValueError("Missing/duplicate SOPInstanceUID values: %s" % duplicates[:10])
    included = []  # type: List[Dict[str, Any]]
    excluded = []  # type: List[Dict[str, Any]]
    for header in headers:
        classification, reason = classify_header(header, profile)
        record = dict(header, classification=classification, reason=reason)
        if classification == "include_target_series":
            included.append(record)
        else:
            excluded.append(record)
    if not included:
        raise ValueError("No valid target-series DICOM slices remain after series guards")
    positions = [physical_position(header) for header in included]
    if all(position is not None for position in positions):
        if len(set(float(position) for position in positions)) != len(positions):
            raise ValueError("Duplicate physical slice positions in target series")
        included = [row for _, row in sorted(zip(positions, included), key=lambda item: float(item[0]))]
        return included, excluded, "orientation_position_projection"
    if all(position is None for position in positions):
        instances = [header.get("InstanceNumber") for header in included]
        if any(value is None for value in instances) or len(set(instances)) != len(instances):
            raise ValueError("Physical ordering unavailable and unique InstanceNumber fallback is unjustified")
        included = [row for _, row in sorted(zip(instances, included), key=lambda item: int(item[0]))]
        return included, excluded, "explicit_unique_instance_number_fallback"
    raise ValueError("Mixed availability of physical slice positions; refusing ambiguous ordering")
