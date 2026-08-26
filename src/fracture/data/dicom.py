"""Robust DICOM discovery, physical ordering, decoding, and HU conversion."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any
import numpy as np
from .windows import to_hu

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SliceRecord:
    path: Path
    series_id: str
    sop_uid: str
    instance_number: int | None
    image_position: tuple[float, ...] | None
    image_orientation: tuple[float, ...] | None
    physical_position: float | None
    pixel_spacing: tuple[float, ...] | None
    slice_thickness: float | None
    shape: tuple[int, int]
    hu: np.ndarray
    transfer_syntax_uid: str | None = None
    photometric_interpretation: str = "MONOCHROME2"


def _tuple(ds: Any, name: str) -> tuple[float, ...] | None:
    value = getattr(ds, name, None)
    return tuple(float(x) for x in value) if value is not None else None


def physical_position(orientation: tuple[float, ...] | None, position: tuple[float, ...] | None) -> float | None:
    if not orientation or len(orientation) != 6 or not position or len(position) != 3:
        return None
    row, col = np.asarray(orientation[:3]), np.asarray(orientation[3:])
    return float(np.dot(np.cross(row, col), np.asarray(position)))


def _pixel_array(ds: Any, path: Path) -> tuple[np.ndarray, bool]:
    try:
        return np.asarray(ds.pixel_array), False
    except Exception as first:
        try:
            import SimpleITK as sitk
            image = sitk.ReadImage(str(path))
            array = sitk.GetArrayFromImage(image)
            # SimpleITK exposes rescaled physical values for CT; do not apply
            # RescaleSlope/Intercept a second time.
            return np.asarray(array[0] if array.ndim == 3 else array), True
        except Exception as second:
            syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "unknown")
            uid = getattr(ds, "SOPInstanceUID", "unknown")
            raise RuntimeError(f"Cannot decode DICOM SOP={uid}, transfer_syntax={syntax}, path={path}: pydicom={first}; SimpleITK={second}") from second


def read_slice(path: str | Path, series_id: str | None = None) -> SliceRecord:
    import pydicom
    from pydicom import config as pydicom_config
    path = Path(path)
    # Organizer anonymization uses values like ``ID_ab12`` in UI elements.
    # They are stable identifiers but not syntactically valid DICOM UIDs.
    # Suppress only this known validation warning; decoding warnings/errors
    # remain visible and fatal where appropriate.
    previous_validation = pydicom_config.settings.reading_validation_mode
    pydicom_config.settings.reading_validation_mode = pydicom_config.IGNORE
    try:
        ds = pydicom.dcmread(path)
        pixels, already_hu = _pixel_array(ds, path)
        if pixels.ndim != 2:
            raise ValueError(f"Expected 2D DICOM pixel data, got {pixels.shape}: {path}")
        orientation, position = _tuple(ds, "ImageOrientationPatient"), _tuple(ds, "ImagePositionPatient")
        syntax = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "")) or None
        hu = pixels.astype(np.float32) if already_hu else to_hu(pixels, getattr(ds, "RescaleSlope", 1), getattr(ds, "RescaleIntercept", 0))
        return SliceRecord(path, series_id or path.parent.name, str(getattr(ds, "SOPInstanceUID", path.stem)), int(ds.InstanceNumber) if hasattr(ds, "InstanceNumber") else None, position, orientation, physical_position(orientation, position), _tuple(ds, "PixelSpacing"), float(ds.SliceThickness) if hasattr(ds, "SliceThickness") else None, tuple(pixels.shape), hu, syntax, str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")))
    finally:
        pydicom_config.settings.reading_validation_mode = previous_validation


def sort_records(records: list[SliceRecord]) -> list[SliceRecord]:
    if records and all(x.physical_position is not None for x in records):
        normals = [np.cross(x.image_orientation[:3], x.image_orientation[3:]) for x in records if x.image_orientation]
        if normals and any(abs(float(np.dot(normals[0], n))) < 0.999 for n in normals[1:]):
            LOGGER.warning("Inconsistent ImageOrientationPatient within series %s", records[0].series_id)
        return sorted(records, key=lambda x: float(x.physical_position))
    if records and all(x.instance_number is not None for x in records):
        LOGGER.warning("Falling back to InstanceNumber ordering for series %s", records[0].series_id)
        return sorted(records, key=lambda x: int(x.instance_number))
    raise ValueError("Cannot establish slice order: physical positions and InstanceNumber are incomplete")


def load_study(study_dir: str | Path) -> list[SliceRecord]:
    root = Path(study_dir)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and (p.suffix.lower() == ".dcm" or not p.suffix))
    if not paths:
        raise FileNotFoundError(f"No DICOM files found in {root}")
    records, failures = [], []
    for path in paths:
        try: records.append(read_slice(path, root.name))
        except Exception as exc: failures.append(f"{path}: {exc}")
    if not records:
        raise RuntimeError("Entire study failed DICOM decoding:\n" + "\n".join(failures))
    if failures:
        raise RuntimeError("Study was only partially decoded; refusing silent omission:\n" + "\n".join(failures))
    return sort_records(records)
