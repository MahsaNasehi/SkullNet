"""Study-level detector feature extraction and aggregation."""
from __future__ import annotations

from typing import Any

import numpy as np

# Stable feature order used when fitting and applying logistic aggregators.
FEATURE_NAMES: tuple[str, ...] = (
    "max_confidence",
    "second_confidence",
    "top2_mean",
    "top3_mean",
    "top5_mean",
    "mean_confidence",
    "total_detections",
    "detected_slices",
    "num_slices",
    "count_ge_0_1",
    "fraction_ge_0_1",
    "longest_run_ge_0_1",
    "count_ge_0_3",
    "fraction_ge_0_3",
    "longest_run_ge_0_3",
    "count_ge_0_5",
    "fraction_ge_0_5",
    "longest_run_ge_0_5",
)


def longest_run(values: list[float], threshold: float) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        best = max(best, current)
    return best


def aggregation_features(
    scores: list[float],
    detection_counts: list[int] | None = None,
    thresholds: tuple[float, ...] = (0.1, 0.3, 0.5),
) -> dict[str, float]:
    array = np.asarray(scores, dtype=float)
    if array.size == 0:
        array = np.asarray([0.0])
    ordered = np.sort(array)[::-1]
    result = {
        "max_confidence": float(ordered[0]),
        "second_confidence": float(ordered[min(1, len(ordered) - 1)]),
        "top2_mean": float(ordered[:2].mean()),
        "top3_mean": float(ordered[:3].mean()),
        "top5_mean": float(ordered[:5].mean()),
        "mean_confidence": float(array.mean()),
        "total_detections": float(sum(detection_counts or [int(x > 0) for x in scores])),
        "detected_slices": float(sum(x > 0 for x in scores)),
        "num_slices": float(len(scores)),
    }
    for threshold in thresholds:
        key = str(threshold).replace(".", "_")
        count = int(np.sum(array >= threshold))
        result[f"count_ge_{key}"] = float(count)
        result[f"fraction_ge_{key}"] = float(count / len(array))
        result[f"longest_run_ge_{key}"] = float(longest_run(scores, threshold))
    return result


def feature_vector(features: dict[str, float], columns: tuple[str, ...] | list[str] = FEATURE_NAMES) -> np.ndarray:
    return np.asarray([[float(features[name]) for name in columns]], dtype=float)


class StudyAggregator:
    def __init__(
        self,
        method: str = "max",
        model: object | None = None,
        *,
        feature_names: tuple[str, ...] | list[str] | None = None,
        min_run: int = 2,
        run_threshold: float = 0.1,
    ):
        self.method = method
        self.model = model
        self.feature_names = tuple(feature_names or FEATURE_NAMES)
        self.min_run = int(min_run)
        self.run_threshold = float(run_threshold)

    def predict(self, scores: list[float], detection_counts: list[int] | None = None) -> float:
        features = aggregation_features(scores, detection_counts)
        if self.method == "max":
            return float(np.clip(features["max_confidence"], 0, 1))
        if self.method == "top3_mean":
            return float(np.clip(features["top3_mean"], 0, 1))
        if self.method == "consecutive":
            # Down-weight isolated spikes (sutures/vessels) unless a short run exists.
            peak = features["max_confidence"]
            run = features.get(f"longest_run_ge_{str(self.run_threshold).replace('.', '_')}", features["longest_run_ge_0_1"])
            if run >= self.min_run:
                return float(np.clip(peak, 0, 1))
            return float(np.clip(0.5 * peak, 0, 1))
        if self.method in {"logistic", "logreg", "lightgbm", "sklearn"}:
            if self.model is None:
                raise ValueError(f"Aggregator model required for method {self.method}")
            x = feature_vector(features, self.feature_names)
            if hasattr(self.model, "predict_proba"):
                probability = float(self.model.predict_proba(x)[0, 1])
            else:
                probability = float(self.model.predict(x)[0])
            return float(np.clip(probability, 0, 1))
        raise ValueError(f"Unsupported aggregation method: {self.method}")


def load_aggregator_bundle(path: str) -> StudyAggregator:
    import joblib

    payload: Any = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        method = str(payload.get("selected_on_oof") or payload.get("method", "logistic"))
        return StudyAggregator(
            method=method,
            model=payload["model"],
            feature_names=tuple(payload.get("feature_names", FEATURE_NAMES)),
            min_run=int(payload.get("min_run", 2)),
            run_threshold=float(payload.get("run_threshold", 0.1)),
        )
    return StudyAggregator(method="logistic", model=payload)
