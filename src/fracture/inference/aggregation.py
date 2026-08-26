"""Study-level detector feature extraction and aggregation."""
from __future__ import annotations
import numpy as np


def longest_run(values: list[float], threshold: float) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value >= threshold else 0; best = max(best, current)
    return best


def aggregation_features(scores: list[float], detection_counts: list[int] | None = None, thresholds: tuple[float, ...] = (0.1, 0.3, 0.5)) -> dict[str, float]:
    array = np.asarray(scores, dtype=float)
    if array.size == 0: array = np.asarray([0.0])
    ordered = np.sort(array)[::-1]
    result = {
        "max_confidence": float(ordered[0]), "second_confidence": float(ordered[min(1, len(ordered)-1)]),
        "top2_mean": float(ordered[:2].mean()), "top3_mean": float(ordered[:3].mean()), "top5_mean": float(ordered[:5].mean()),
        "mean_confidence": float(array.mean()), "total_detections": float(sum(detection_counts or [int(x > 0) for x in scores])),
        "detected_slices": float(sum(x > 0 for x in scores)), "num_slices": float(len(scores)),
    }
    for threshold in thresholds:
        key = str(threshold).replace(".", "_"); count = int(np.sum(array >= threshold))
        result[f"count_ge_{key}"] = float(count); result[f"fraction_ge_{key}"] = float(count / len(array)); result[f"longest_run_ge_{key}"] = float(longest_run(scores, threshold))
    return result


class StudyAggregator:
    def __init__(self, method: str = "max", model: object | None = None): self.method, self.model = method, model
    def predict(self, scores: list[float], detection_counts: list[int] | None = None) -> float:
        features = aggregation_features(scores, detection_counts)
        if self.method == "max": return float(np.clip(features["max_confidence"], 0, 1))
        if self.model is None: raise ValueError(f"Aggregator model required for method {self.method}")
        columns = list(features); x = np.asarray([[features[c] for c in columns]])
        return float(np.clip(self.model.predict_proba(x)[0, 1], 0, 1))

