"""Stable annotation-free fracture prediction API."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .data.dicom import load_study
from .inference.aggregation import StudyAggregator, load_aggregator_bundle
from .inference.calibration import ProbabilityCalibrator
from .inference.slice_predictor import predict_slices
from .models.detector import Detector


class FracturePredictor:
    def __init__(
        self,
        detector_weights: str | Path,
        *,
        aggregator_path: str | Path | None = None,
        calibrator_path: str | Path | None = None,
        aggregation_method: str = "max",
        calibration_method: str = "none",
        input_mode: str = "single",
        image_size: int = 640,
        window_level: float = 500,
        window_width: float = 2500,
        batch_size: int = 8,
        confidence: float = 0.05,
        nms_iou: float = 0.5,
        device: Any = 0,
        fp16: bool = True,
    ):
        self.detector = Detector(detector_weights, confidence, nms_iou, device, fp16, image_size)
        import joblib

        if aggregator_path:
            self.aggregator = load_aggregator_bundle(str(aggregator_path))
            if aggregation_method not in {"max", "top3_mean", "consecutive"}:
                self.aggregator.method = aggregation_method
        else:
            self.aggregator = StudyAggregator(aggregation_method)

        if calibrator_path:
            payload = joblib.load(calibrator_path)
            if isinstance(payload, dict) and "model" in payload:
                self.calibrator = ProbabilityCalibrator(str(payload.get("method", calibration_method)), payload["model"])
            else:
                self.calibrator = ProbabilityCalibrator(calibration_method, payload)
        else:
            self.calibrator = ProbabilityCalibrator(calibration_method)

        self.input_mode = input_mode
        self.window_level = window_level
        self.window_width = window_width
        self.batch_size = batch_size
        self.last_timings: dict[str, float] = {}
        self.last_slice_predictions: list[Any] = []

    def predict(self, study_dir: str) -> float:
        started = time.perf_counter()
        records = load_study(study_dir)
        loaded = time.perf_counter()
        predictions = predict_slices(
            records,
            self.detector,
            self.input_mode,
            self.window_level,
            self.window_width,
            self.batch_size,
        )
        detected = time.perf_counter()
        self.last_slice_predictions = predictions
        raw = self.aggregator.predict(
            [item.max_confidence for item in predictions],
            [item.num_detections for item in predictions],
        )
        aggregated = time.perf_counter()
        probability = float(self.calibrator.predict(raw))
        finished = time.perf_counter()
        self.last_timings = {
            "dicom_loading": loaded - started,
            "preprocessing_and_detector": detected - loaded,
            "aggregation": aggregated - detected,
            "calibration": finished - aggregated,
            "total": finished - started,
        }
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"Invalid fracture probability: {probability}")
        return probability
