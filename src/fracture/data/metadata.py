"""Configured organizer metadata loading and slice-label indexing."""
from __future__ import annotations
from pathlib import Path
from typing import Any


def load_metadata(path: str | Path):
    import pandas as pd
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv": return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}: return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}: return pd.read_pickle(path)
    raise ValueError(f"Unsupported metadata format: {path}")


def slice_label_index(data_config: dict[str, Any]) -> dict[tuple[str, str], bool]:
    path = data_config.get("metadata_path")
    if not path: return {}
    frame = load_metadata(path)
    series_col = data_config.get("series_id_column")
    sop_col = data_config.get("sop_uid_column")
    label_col = data_config.get("fracture_label_column")
    missing = [name for name in (series_col, sop_col, label_col) if not name or name not in frame.columns]
    if missing: raise ValueError(f"Metadata columns missing or unconfigured: {missing}; available={frame.columns.tolist()}")
    return {(str(series), str(sop)): bool(label) for series, sop, label in frame[[series_col, sop_col, label_col]].itertuples(index=False, name=None)}
