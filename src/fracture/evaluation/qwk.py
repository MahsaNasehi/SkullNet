"""Quadratic weighted kappa without a runtime sklearn dependency."""
from __future__ import annotations
import numpy as np


def quadratic_weighted_kappa(y_true: list[int], y_pred: list[int], n_classes: int = 3) -> float:
    if len(y_true) != len(y_pred) or not y_true: raise ValueError("Non-empty equally sized inputs required")
    observed = np.zeros((n_classes, n_classes), dtype=float)
    for a, b in zip(y_true, y_pred, strict=True): observed[int(a), int(b)] += 1
    hist_a, hist_b = observed.sum(1), observed.sum(0); expected = np.outer(hist_a, hist_b) / observed.sum()
    weights = np.fromfunction(lambda i, j: ((i-j)/(n_classes-1))**2, (n_classes, n_classes))
    denominator = float((weights * expected).sum())
    return 1.0 if denominator == 0 else float(1 - (weights * observed).sum() / denominator)
