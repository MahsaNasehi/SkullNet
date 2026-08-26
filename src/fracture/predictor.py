"""Stable annotation-free fracture prediction API."""
from __future__ import annotations
from pathlib import Path
import time
from typing import Any
from .data.dicom import load_study
from .models.detector import Detector
from .inference.slice_predictor import predict_slices
from .inference.aggregation import StudyAggregator
from .inference.calibration import ProbabilityCalibrator


class FracturePredictor:
    def __init__(self, detector_weights: str | Path, *, aggregator_path: str | Path | None = None, calibrator_path: str | Path | None = None, aggregation_method: str = "max", calibration_method: str = "none", input_mode: str = "single", window_level: float = 500, window_width: float = 2500, batch_size: int = 8, confidence: float = 0.05, nms_iou: float = 0.5, device: Any = 0, fp16: bool = True):
        self.detector = Detector(detector_weights, confidence, nms_iou, device, fp16)
        import joblib
        aggregator_model = joblib.load(aggregator_path) if aggregator_path else None
        calibrator_model = joblib.load(calibrator_path) if calibrator_path else None
        self.aggregator = StudyAggregator(aggregation_method, aggregator_model)
        self.calibrator = ProbabilityCalibrator(calibration_method, calibrator_model)
        self.input_mode, self.window_level, self.window_width, self.batch_size = input_mode, window_level, window_width, batch_size
        self.last_timings: dict[str, float] = {}

    def predict(self, study_dir: str) -> float:
        started = time.perf_counter(); records = load_study(study_dir); loaded = time.perf_counter()
        predictions = predict_slices(records, self.detector, self.input_mode, self.window_level, self.window_width, self.batch_size); detected = time.perf_counter()
        raw = self.aggregator.predict([x.max_confidence for x in predictions], [x.num_detections for x in predictions]); aggregated = time.perf_counter()
        probability = float(self.calibrator.predict(raw)); finished = time.perf_counter()
        self.last_timings = {"dicom_loading": loaded-started, "preprocessing_and_detector": detected-loaded, "aggregation": aggregated-detected, "calibration": finished-aggregated, "total": finished-started}
        if not 0.0 <= probability <= 1.0: raise RuntimeError(f"Invalid fracture probability: {probability}")
        return probability

