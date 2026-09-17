"""Frozen-grid rescue gate; tau is selected using source-fold data only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fracture_macro_f1_oof import binary_report, oracle_triage_report


TAU_GRID = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
OFFICIAL_THRESHOLD = 0.5


def gated_probability(detector: np.ndarray, meta: np.ndarray, tau_low: float) -> np.ndarray:
    if tau_low not in TAU_GRID:
        raise ValueError("tau_low is outside the frozen gate grid")
    d, m = np.asarray(detector, float), np.asarray(meta, float)
    if d.shape != m.shape or not np.isfinite(d).all() or not np.isfinite(m).all():
        raise ValueError("Gate requires paired finite probabilities")
    if ((d < 0) | (d > 1)).any() or ((m < 0) | (m > 1)).any():
        raise ValueError("Gate inputs must be in [0,1]")
    eligible = (d >= tau_low) & (d < OFFICIAL_THRESHOLD)
    return np.where(eligible, m, d)


def choose_tau(source_inner: pd.DataFrame) -> tuple[float, list[dict]]:
    """Choose gate on inner-cross-fitted source scores, never outer held-out."""
    candidates = []
    for tau in TAU_GRID:
        scores = gated_probability(source_inner["detector_fixed"],
                                   source_inner["inner_meta_probability"], tau)
        triage = oracle_triage_report(source_inner, scores)
        binary = binary_report(source_inner["fracture_true"].astype(bool), scores)
        candidates.append({"tau_low": tau,
                           "oracle_other_heads_macro_f1": triage["pooled_macro_f1"],
                           "fp": binary["fp"], "sensitivity": binary["sensitivity"],
                           "specificity": binary["specificity"],
                           "pr_auc": binary["pr_auc"]})
    # Frozen lexicographic rule: primary Macro-F1, then lower FP, higher
    # specificity/sensitivity/PR-AUC; largest tau wins a remaining tie.
    winner = max(candidates, key=lambda item: (
        item["oracle_other_heads_macro_f1"], -item["fp"],
        item["specificity"], item["sensitivity"], item["pr_auc"], item["tau_low"]))
    return float(winner["tau_low"]), candidates
