"""Step 1: DICOM discovery → HU → physical order. Set breakpoints in dicom.py."""
from __future__ import annotations

import os
from pathlib import Path

from fracture.data.dicom import load_study, physical_position, read_slice

study = Path(os.environ.get("FRACTURE_STUDY_DIR", "iaaa-contest-bct/Data/training/1011"))
print(f"study_dir={study.resolve()}")

# Breakpoint candidates: read_slice, _pixel_array, physical_position, sort_records, load_study
first = next(p for p in study.rglob("*") if p.is_file() and (p.suffix.lower() == ".dcm" or not p.suffix))
one = read_slice(first, study.name)
print(
    "one_slice:",
    {
        "sop": one.sop_uid,
        "shape": one.shape,
        "hu_min": float(one.hu.min()),
        "hu_max": float(one.hu.max()),
        "physical_position": one.physical_position,
        "photometric": one.photometric_interpretation,
        "transfer_syntax": one.transfer_syntax_uid,
    },
)
print("dot_check:", physical_position(one.image_orientation, one.image_position))

records = load_study(study)
print(f"n_slices={len(records)}")
print("first3_positions:", [r.physical_position for r in records[:3]])
print("last3_positions:", [r.physical_position for r in records[-3:]])
assert all(records[i].physical_position <= records[i + 1].physical_position for i in range(len(records) - 1) if records[i].physical_position is not None and records[i + 1].physical_position is not None)
print("OK: study loaded and ordered")
