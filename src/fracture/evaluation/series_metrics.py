"""Study metrics for OOF predictions."""
from __future__ import annotations
import numpy as np


def evaluate(y_true: list[int], probability: list[float], threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
    y, p = np.asarray(y_true), np.clip(np.asarray(probability, dtype=float), 1e-7, 1-1e-7); pred = p >= threshold
    tp, tn, fp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum(), ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
    result = {
        "auroc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "threshold": float(threshold),
        "sensitivity_at_threshold": float(tp/(tp+fn)) if tp+fn else 0.0,
        "specificity_at_threshold": float(tn/(tn+fp)) if tn+fp else 0.0,
        "precision_at_threshold": float(tp/(tp+fp)) if tp+fp else 0.0,
        "f1_at_threshold": float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    if abs(float(threshold) - 0.5) < 1e-12:
        for metric in ("sensitivity", "specificity", "precision", "f1"):
            result[f"{metric}_at_0_5"] = result[f"{metric}_at_threshold"]
    return result
