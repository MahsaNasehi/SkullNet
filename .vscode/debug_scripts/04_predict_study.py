"""Step 4: end-to-end study predict. Breakpoints in predictor / slice_predictor / detector."""
from __future__ import annotations

import os
from pathlib import Path

from fracture import FracturePredictor

study = os.environ.get("FRACTURE_STUDY_DIR", "iaaa-contest-bct/Data/training/2265")
weights = os.environ.get("FRACTURE_WEIGHTS", "outputs/fold_0/v1_25d_p2/weights/best.pt")
if not Path(weights).is_file():
    raise SystemExit(f"Weights missing: {weights}")

predictor = FracturePredictor(
    weights,
    aggregation_method="max",
    calibration_method="none",
    input_mode="2.5d",
    window_level=500,
    window_width=2500,
    batch_size=2,
    confidence=0.01,
    nms_iou=0.5,
    device=0,
    fp16=True,
)
prob = predictor.predict(study)
print({"study": study, "fracture_prob": prob, "timings": predictor.last_timings})
print("OK: predictor")
