"""OOF-only probability calibration."""
from __future__ import annotations
import numpy as np


class ProbabilityCalibrator:
    def __init__(self, method: str = "none", model: object | None = None): self.method, self.model = method, model
    def predict(self, probability: float) -> float:
        p = float(np.clip(probability, 0, 1))
        if self.method == "none": return p
        if self.model is None: raise ValueError(f"Calibration model required for {self.method}")
        if self.method in {"platt", "sigmoid"}: p = float(self.model.predict_proba([[p]])[0, 1])
        elif self.method == "isotonic": p = float(self.model.predict([p])[0])
        else: raise ValueError(f"Unsupported calibration method: {self.method}")
        return float(np.clip(p, 0, 1))


def fit_calibrator(probabilities: list[float], labels: list[int], method: str) -> ProbabilityCalibrator:
    if len(set(labels)) < 2: raise ValueError("Calibration requires both classes")
    if method in {"platt", "sigmoid"}:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression().fit(np.asarray(probabilities).reshape(-1, 1), labels)
    elif method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        model = IsotonicRegression(out_of_bounds="clip").fit(probabilities, labels)
    elif method == "none": return ProbabilityCalibrator()
    else: raise ValueError(method)
    return ProbabilityCalibrator(method, model)

