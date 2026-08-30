"""Step 3: corrected-over-original annotations → YOLO labels. Breakpoints in annotations.py."""
from __future__ import annotations

import os
from pathlib import Path

from fracture.data.annotations import resolve_annotation, xywh_to_yolo
from fracture.data.dicom import load_study
from fracture.utils.config import load_config

cfg = load_config("configs/fracture_25d_p2.yaml")
data = cfg["data"]
study = Path(os.environ.get("FRACTURE_STUDY_DIR", "iaaa-contest-bct/Data/training/1011"))
records = load_study(study)

n_with, n_boxes = 0, 0
for record in records:
    ann = resolve_annotation(
        data["annotation_root"],
        data.get("corrected_annotation_root"),
        study.name,
        record.sop_uid,
        image_shape=record.shape,
    )
    if ann is None:
        continue
    n_with += 1
    n_boxes += len(ann.boxes)
    if ann.boxes:
        yolo = xywh_to_yolo(ann.boxes[0], record.shape[1], record.shape[0])
        print(
            {
                "sop": record.sop_uid,
                "provenance": ann.provenance,
                "n_boxes": len(ann.boxes),
                "first_xywh": ann.boxes[0],
                "first_yolo": yolo,
            }
        )
        break

print(f"annotated_slices={n_with} total_boxes_seen_until_break={n_boxes}")
print("OK: annotation path")
