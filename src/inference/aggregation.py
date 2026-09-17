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
    "median_spacing_mm",
    "longest_run_mm_ge_0_1",
    "longest_run_mm_ge_0_3",
    "longest_run_mm_ge_0_5",
)


SPATIAL_FEATURE_NAMES: tuple[str, ...] = (
    "spatial_strongest_single",
    "spatial_best_track_length",
    "spatial_best_track_span_mm",
    "spatial_best_track_mean",
    "spatial_best_track_min",
    "spatial_coherent_top3_mean",
)

SPATIAL_COMPACT_FEATURE_NAMES: tuple[str, ...] = (
    "max_confidence",
    "top3_mean",
    "count_ge_0_1",
    "longest_run_mm_ge_0_1",
    *SPATIAL_FEATURE_NAMES,
)


def longest_run(values: list[float], threshold: float) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        best = max(best, current)
    return best


def longest_run_mm(values: list[float], positions: list[float | None], threshold: float) -> float:
    """Physical span of the longest above-threshold contiguous slice run."""
    if len(values) != len(positions) or not values or any(position is None for position in positions):
        return 0.0
    best = 0.0
    start: int | None = None
    for index, value in enumerate(values + [float("-inf")]):
        if value >= threshold and start is None:
            start = index
        elif value < threshold and start is not None:
            end = index - 1
            best = max(best, abs(float(positions[end]) - float(positions[start])))
            start = None
    return float(best)


def _normalised_box(
    box: list[float], image_shape: tuple[int, int] | list[int] | None
) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"Expected xyxy box with four coordinates, got {box}")
    if image_shape is None:
        height = width = 1.0
    else:
        height, width = map(float, image_shape)
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image shape: {image_shape}")
    x0, y0, x1, y1 = map(float, box)
    return x0 / width, y0 / height, x1 / width, y1 / height


def _box_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return float(intersection / union) if union else 0.0


def spatial_track_features(
    boxes_by_slice: list[list[list[float]]] | None,
    box_scores_by_slice: list[list[float]] | None,
    physical_positions: list[float | None] | None = None,
    image_shapes: list[tuple[int, int] | list[int]] | None = None,
    *,
    minimum_score: float = 0.05,
    maximum_center_distance: float = 0.12,
    minimum_iou: float = 0.02,
    maximum_gap_factor: float = 1.75,
) -> dict[str, float]:
    """Summarize coherent detections across physically adjacent CT slices.

    All proposals are considered, coordinates are normalized by image size,
    and a strong single-slice feature remains available for genuinely sparse
    fractures. This avoids choosing a track solely because it contains the
    globally strongest isolated box.
    """
    empty = {name: 0.0 for name in SPATIAL_FEATURE_NAMES}
    if boxes_by_slice is None or box_scores_by_slice is None:
        return empty
    if len(boxes_by_slice) != len(box_scores_by_slice):
        raise ValueError("boxes_by_slice and box_scores_by_slice must have equal length")
    length = len(boxes_by_slice)
    if physical_positions is not None and len(physical_positions) != length:
        raise ValueError("physical_positions must align with boxes_by_slice")
    if image_shapes is not None and len(image_shapes) != length:
        raise ValueError("image_shapes must align with boxes_by_slice")

    positions = physical_positions or [None] * length
    valid_positions = np.asarray([float(value) for value in positions if value is not None], dtype=float)
    spacings = np.abs(np.diff(valid_positions)) if valid_positions.size > 1 else np.asarray([])
    positive_spacings = spacings[spacings > 0]
    median_spacing = float(np.median(positive_spacings)) if positive_spacings.size else 0.0
    maximum_gap = median_spacing * float(maximum_gap_factor) if median_spacing else float("inf")

    all_states: list[dict[str, Any]] = []
    previous_states: list[dict[str, Any]] = []
    strongest_single = 0.0
    for index, (boxes, scores) in enumerate(zip(boxes_by_slice, box_scores_by_slice, strict=True)):
        if len(boxes) != len(scores):
            raise ValueError(f"Slice {index} has {len(boxes)} boxes but {len(scores)} scores")
        shape = image_shapes[index] if image_shapes is not None else None
        current_states: list[dict[str, Any]] = []
        for box, raw_score in zip(boxes, scores, strict=True):
            score = float(raw_score)
            if score < minimum_score:
                continue
            strongest_single = max(strongest_single, score)
            normalised = _normalised_box(box, shape)
            centre = ((normalised[0] + normalised[2]) / 2.0, (normalised[1] + normalised[3]) / 2.0)
            compatible: list[dict[str, Any]] = []
            for state in previous_states:
                if state["slice_index"] != index - 1:
                    continue
                if positions[index] is not None and state["position"] is not None:
                    if abs(float(positions[index]) - float(state["position"])) > maximum_gap:
                        continue
                previous_centre = state["centre"]
                distance = float(np.hypot(centre[0] - previous_centre[0], centre[1] - previous_centre[1]))
                if distance <= maximum_center_distance or _box_iou(normalised, state["box"]) >= minimum_iou:
                    compatible.append(state)
            predecessor = max(
                compatible,
                key=lambda item: (item["length"], item["score_sum"], item["minimum_score"]),
                default=None,
            )
            if predecessor is None:
                state = {
                    "length": 1,
                    "score_sum": score,
                    "minimum_score": score,
                    "scores": [score],
                    "start_position": positions[index],
                }
            else:
                state = {
                    "length": int(predecessor["length"]) + 1,
                    "score_sum": float(predecessor["score_sum"]) + score,
                    "minimum_score": min(float(predecessor["minimum_score"]), score),
                    "scores": list(predecessor["scores"]) + [score],
                    "start_position": predecessor["start_position"],
                }
            state.update({
                "slice_index": index,
                "position": positions[index],
                "box": normalised,
                "centre": centre,
            })
            current_states.append(state)
            all_states.append(state)
        previous_states = current_states

    multi_slice = [state for state in all_states if int(state["length"]) >= 2]
    if not multi_slice:
        return empty | {"spatial_strongest_single": float(strongest_single)}
    best = max(
        multi_slice,
        key=lambda item: (
            float(item["score_sum"]) / int(item["length"]),
            int(item["length"]),
            float(item["minimum_score"]),
        ),
    )
    ordered_scores = sorted(map(float, best["scores"]), reverse=True)
    start, end = best["start_position"], best["position"]
    span = abs(float(end) - float(start)) if start is not None and end is not None else 0.0
    return {
        "spatial_strongest_single": float(strongest_single),
        "spatial_best_track_length": float(best["length"]),
        "spatial_best_track_span_mm": float(span),
        "spatial_best_track_mean": float(best["score_sum"] / best["length"]),
        "spatial_best_track_min": float(best["minimum_score"]),
        "spatial_coherent_top3_mean": float(np.mean(ordered_scores[:3])),
    }


def aggregation_features(
    scores: list[float],
    detection_counts: list[int] | None = None,
    physical_positions: list[float | None] | None = None,
    thresholds: tuple[float, ...] = (0.1, 0.3, 0.5),
    extra_features: dict[str, float] | None = None,
    boxes_by_slice: list[list[list[float]]] | None = None,
    box_scores_by_slice: list[list[float]] | None = None,
    image_shapes: list[tuple[int, int] | list[int]] | None = None,
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
    positions = physical_positions or []
    valid_positions = np.asarray([float(value) for value in positions if value is not None], dtype=float)
    spacing = np.abs(np.diff(valid_positions)) if valid_positions.size > 1 else np.asarray([], dtype=float)
    result["median_spacing_mm"] = float(np.median(spacing)) if spacing.size else 0.0
    for threshold in (0.1, 0.3, 0.5):
        key = str(threshold).replace(".", "_")
        result[f"longest_run_mm_ge_{key}"] = longest_run_mm(scores, positions, threshold)
    if boxes_by_slice is not None or box_scores_by_slice is not None:
        result.update(
            spatial_track_features(
                boxes_by_slice,
                box_scores_by_slice,
                physical_positions,
                image_shapes,
            )
        )
    if extra_features:
        result.update({str(name): float(value) for name, value in extra_features.items()})
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

    def predict(
        self,
        scores: list[float],
        detection_counts: list[int] | None = None,
        physical_positions: list[float | None] | None = None,
        extra_features: dict[str, float] | None = None,
        boxes_by_slice: list[list[list[float]]] | None = None,
        box_scores_by_slice: list[list[float]] | None = None,
        image_shapes: list[tuple[int, int] | list[int]] | None = None,
    ) -> float:
        features = aggregation_features(
            scores,
            detection_counts,
            physical_positions,
            extra_features=extra_features,
            boxes_by_slice=boxes_by_slice,
            box_scores_by_slice=box_scores_by_slice,
            image_shapes=image_shapes,
        )
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
        if self.method in {"logistic", "logreg", "lightgbm", "sklearn", "spatial_logistic"}:
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
