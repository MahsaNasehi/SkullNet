"""Thin direct-evaluator adapter around :class:`FracturePredictor`.

The official evaluator constructs ``Model()`` without training-time arguments.
Artifact paths and the exact HU preprocessing contract therefore live in the
local ``models/deployment.yaml`` manifest created after OOF model selection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fracture import FracturePredictor


def _deployment_kwargs(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Deployment manifest not found: {path}. Create it with "
            "scripts/create_deployment_manifest.py after selecting final OOF artifacts."
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Deployment manifest must contain a mapping: {path}")
    root = path.parent.parent
    for key in ("detector_weights", "aggregator_path", "calibrator_path", "study_model_path"):
        value = payload.get(key)
        if value:
            candidate = Path(str(value))
            payload[key] = str(candidate if candidate.is_absolute() else (root / candidate).resolve())
    return payload


class Model:
    def __init__(
        self,
        deployment_config: str | Path | None = None,
        **predictor_kwargs: Any,
    ):
        if deployment_config is not None and predictor_kwargs:
            raise ValueError("Use either deployment_config or explicit predictor arguments, not both")
        if not predictor_kwargs:
            deployment_config = deployment_config or Path(__file__).parent / "models" / "deployment.yaml"
            predictor_kwargs = _deployment_kwargs(deployment_config)
        self.fracture_predictor = FracturePredictor(**predictor_kwargs)

    def predict(self, study_dir: str) -> dict[str, float]:
        return {"fracture_prob": self.fracture_predictor.predict(study_dir)}
