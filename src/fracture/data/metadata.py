"""Configured organizer metadata loading and slice/study indexing."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_metadata(path: str | Path):
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported metadata format: {path}")


def slice_label_index(data_config: dict[str, Any]) -> dict[tuple[str, str], bool]:
    path = data_config.get("metadata_path")
    if not path:
        return {}
    frame = load_metadata(path)
    series_col = data_config.get("series_id_column")
    sop_col = data_config.get("sop_uid_column")
    label_col = data_config.get("fracture_label_column")
    missing = [name for name in (series_col, sop_col, label_col) if not name or name not in frame.columns]
    if missing:
        raise ValueError(f"Metadata columns missing or unconfigured: {missing}; available={frame.columns.tolist()}")
    return {
        (str(series), str(sop)): bool(label)
        for series, sop, label in frame[[series_col, sop_col, label_col]].itertuples(index=False, name=None)
    }


def study_intermediates(data_config: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Build study-level ICH volumes (mL), MLS, fracture label, and triage class.

    When explicit volume columns are absent, approximate mL from per-slice area
    using PixelSpacing and SliceThickness:
        mL += area_px * sx * sy * thickness_mm / 1000
    """
    path = data_config.get("metadata_path")
    if not path:
        return {}
    frame = load_metadata(path)
    series_col = data_config["series_id_column"]
    fracture_col = data_config["fracture_label_column"]
    mls_col = data_config.get("mls_column", "MidlineShiftMM")
    triage_col = data_config.get("triage_class_column", "triage_class")
    spacing0 = data_config.get("pixel_spacing0_column", "dicom_series.PixelSpacing0")
    spacing1 = data_config.get("pixel_spacing1_column", "dicom_series.PixelSpacing1")
    thickness = data_config.get("slice_thickness_column", "dicom_series.SliceThickness")

    area_map = {
        "V_EDH": data_config.get("edh_area_column", "EpiduralHemorrhage_Area"),
        "V_SDH": data_config.get("sdh_area_column", "SubduralHemorrhage_Area"),
        "V_IPH": data_config.get("iph_area_column", "IntraparenchymalHemorrhage_Area"),
        "V_SAH": data_config.get("sah_area_column", "SubarachnoidHemorrhage_Area"),
        "V_IVH": data_config.get("ivh_area_column", "IntraventricularHemorrhage_Area"),
    }
    required = [series_col, fracture_col, mls_col, *area_map.values()]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Metadata columns missing for study intermediates: {missing}")

    work = frame.copy()
    sx = work[spacing0].astype(float) if spacing0 in work.columns else 1.0
    sy = work[spacing1].astype(float) if spacing1 in work.columns else 1.0
    th = work[thickness].astype(float) if thickness in work.columns else 1.0
    th = th.replace(0, 1.0).fillna(1.0)
    factor = sx * sy * th / 1000.0
    for volume_key, area_col in area_map.items():
        work[volume_key] = work[area_col].astype(float) * factor

    aggregations = {key: "sum" for key in area_map}
    aggregations[mls_col] = "max"
    aggregations[fracture_col] = "max"
    if triage_col in work.columns:
        aggregations[triage_col] = "max"
    grouped = work.groupby(series_col).agg(aggregations)

    output: dict[str, dict[str, float]] = {}
    for series_id, row in grouped.iterrows():
        item = {
            "V_EDH": float(row["V_EDH"]),
            "V_SDH": float(row["V_SDH"]),
            "V_IPH": float(row["V_IPH"]),
            "V_SAH": float(row["V_SAH"]),
            "V_IVH": float(row["V_IVH"]),
            "MLS_mm": float(row[mls_col]),
            "y_true_fracture": float(bool(row[fracture_col])),
        }
        if triage_col in grouped.columns:
            item["triage_class"] = float(row[triage_col])
        output[str(series_id)] = item
    return output
