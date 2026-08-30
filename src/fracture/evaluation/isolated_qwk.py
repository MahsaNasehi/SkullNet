"""Isolated fracture QWK: GT ICH/MLS + predicted fracture_prob only."""
from __future__ import annotations

from typing import Any

from fracture.evaluation.qwk import quadratic_weighted_kappa
from fracture.evaluation.triage import triage_from_intermediates


def predicted_triage(intermediates: dict[str, float], fracture_prob: float) -> int:
    return triage_from_intermediates(
        intermediates["V_EDH"],
        intermediates["V_SDH"],
        intermediates["V_IPH"],
        intermediates["V_SAH"],
        intermediates["V_IVH"],
        float(fracture_prob),
        intermediates["MLS_mm"],
    )


def ground_truth_triage(intermediates: dict[str, float]) -> int:
    if "triage_class" in intermediates and intermediates["triage_class"] is not None:
        return int(intermediates["triage_class"])
    return predicted_triage(intermediates, intermediates["y_true_fracture"])


def isolated_fracture_qwk(
    series_ids: list[str],
    fracture_probabilities: list[float],
    study_table: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if len(series_ids) != len(fracture_probabilities):
        raise ValueError("series_ids and fracture_probabilities must have equal length")
    y_true: list[int] = []
    y_pred: list[int] = []
    missing: list[str] = []
    for series_id, probability in zip(series_ids, fracture_probabilities, strict=True):
        intermediates = study_table.get(str(series_id))
        if intermediates is None:
            missing.append(str(series_id))
            continue
        y_true.append(ground_truth_triage(intermediates))
        y_pred.append(predicted_triage(intermediates, probability))
    if missing:
        raise KeyError(f"Missing study intermediates for {len(missing)} series; examples={missing[:5]}")
    if not y_true:
        raise ValueError("No studies available for isolated fracture QWK")
    return {
        "isolated_fracture_qwk": quadratic_weighted_kappa(y_true, y_pred, n_classes=3),
        "n_studies": len(y_true),
        "gt_triage_counts": {str(i): y_true.count(i) for i in range(3)},
        "pred_triage_counts": {str(i): y_pred.count(i) for i in range(3)},
    }
