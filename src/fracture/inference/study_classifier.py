"""Offline inference wrapper for the compact study-level MIL branch."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from fracture.data.windows import make_hu_input
from fracture.models.study_mil import StudyMIL


def _torch_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        return torch.device(f"cuda:{device}")
    return torch.device(str(device))


def study_tensor(records: list[Any], preprocessing: dict[str, Any]) -> torch.Tensor:
    if not records:
        raise ValueError("Cannot classify an empty CT study")
    hu_images = [record.hu for record in records]
    positions = [record.physical_position for record in records]
    monochrome1 = [record.photometric_interpretation == "MONOCHROME1" for record in records]
    image_size = int(preprocessing.get("image_size", 256))
    context_distance = preprocessing.get("context_distance_mm")
    images = []
    for index in range(len(records)):
        image = make_hu_input(
            hu_images,
            index,
            level=float(preprocessing.get("window_level", 800)),
            width=float(preprocessing.get("window_width", 1600)),
            mode=str(preprocessing.get("input_mode", "2.5d")),
            boundary_mode=str(preprocessing.get("boundary_mode", "repeat")),
            monochrome1=monochrome1,
            physical_positions=positions,
            context_distance_mm=float(context_distance) if context_distance is not None else None,
        )
        resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        images.append(np.moveaxis(resized.astype(np.float32) / 255.0, -1, 0))
    array = np.stack(images, axis=0)
    return torch.from_numpy(array)


class StudyClassifier:
    def __init__(self, checkpoint: str | Path, *, device: Any = 0, fp16: bool = True):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Study-classifier checkpoint not found: {path}")
        if path.stat().st_size < 100_000:
            raise ValueError(f"Study-classifier checkpoint is implausibly small: {path}")
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or not {"state_dict", "model_config", "preprocessing"} <= set(payload):
            raise ValueError(f"Invalid StudyMIL checkpoint payload: {path}")
        self.preprocessing = dict(payload["preprocessing"])
        self.model = StudyMIL(**payload["model_config"])
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.device = _torch_device(device)
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, records: list[Any]) -> float:
        images = study_tensor(records, self.preprocessing).unsqueeze(0).to(self.device)
        mask = torch.ones(images.shape[:2], dtype=torch.bool, device=self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.fp16):
            logit = self.model(images, mask)["study_logits"][0]
        return float(torch.sigmoid(logit.float()).cpu())
