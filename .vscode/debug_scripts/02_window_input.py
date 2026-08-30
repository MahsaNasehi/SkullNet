"""Step 2: bone window + single vs 2.5D tensors. Breakpoints in windows.py."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from fracture.data.dicom import load_study
from fracture.data.windows import bone_window, make_input, to_hu

study = Path(os.environ.get("FRACTURE_STUDY_DIR", "iaaa-contest-bct/Data/training/1011"))
records = load_study(study)
idx = min(5, len(records) - 1)

# Synthetic HU check
toy = to_hu(np.array([[0, 100]], dtype=np.int16), slope=2.0, intercept=-1000.0)
assert toy.tolist() == [[-1000.0, -800.0]]

windows = [
    bone_window(r.hu, level=500, width=2500, monochrome1=(r.photometric_interpretation == "MONOCHROME1"))
    for r in records
]
single = make_input(windows, idx, "single")
stack = make_input(windows, idx, "2.5d")
edge = make_input(windows, 0, "2.5d")

print(f"slice_index={idx}")
print("single_shape", single.shape, "dtype", single.dtype, "unique_channels", np.allclose(single[..., 0], single[..., 1]))
print("25d_shape", stack.shape, "channel_means", [float(stack[..., c].mean()) for c in range(3)])
print("edge_channels_equal_01", np.array_equal(edge[..., 0], edge[..., 1]))
print("OK: windowing + 2.5D")
