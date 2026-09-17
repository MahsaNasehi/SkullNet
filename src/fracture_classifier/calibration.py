"""Source-fold-only L2 logistic meta-calibration for existing OOF scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


EPS = 1e-5
C_GRID = (0.1, 1.0, 10.0)


def logit_features(detector: np.ndarray, classifier: np.ndarray) -> np.ndarray:
    a, b = np.asarray(detector, float), np.asarray(classifier, float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired finite one-dimensional detector/classifier scores required")
    if ((a < 0) | (a > 1)).any() or ((b < 0) | (b > 1)).any():
        raise ValueError("Meta-features must be probabilities")
    clipped = np.column_stack((np.clip(a, EPS, 1 - EPS), np.clip(b, EPS, 1 - EPS)))
    return np.log(clipped / (1 - clipped))


def fit_logistic(table: pd.DataFrame, c: float) -> LogisticRegression:
    if c not in C_GRID:
        raise ValueError("C must be on the frozen grid")
    truth = table["fracture_true"].astype(int).to_numpy()
    if set(truth) != {0, 1}:
        raise RuntimeError("Both fracture classes required to fit calibration")
    model = LogisticRegression(C=c, penalty="l2", class_weight=None,
                               solver="lbfgs", max_iter=1000)
    model.fit(logit_features(table["detector_fixed"], table["classifier_top5"]), truth)
    return model


def predict_logistic(model: LogisticRegression, table: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(logit_features(table["detector_fixed"],
                                              table["classifier_top5"]))[:, 1]


def inner_crossfit(source: pd.DataFrame, c: float) -> pd.DataFrame:
    """For each of four source folds, train on the other three source folds."""
    source_folds = sorted(source["fold"].unique().tolist())
    if len(source_folds) != 4:
        raise RuntimeError("Exactly four source folds required for inner C/gate selection")
    parts = []
    for validation_fold in source_folds:
        training = source[source["fold"] != validation_fold]
        validation = source[source["fold"] == validation_fold].copy()
        if set(training["fold"]) & {validation_fold} or len(validation) == 0:
            raise RuntimeError("Inner held-out fold entered logistic fitting")
        model = fit_logistic(training, c)
        validation["inner_meta_probability"] = predict_logistic(model, validation)
        parts.append(validation)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != len(source) or set(result["study_id"]) != set(source["study_id"]):
        raise RuntimeError("Incomplete source-only inner cross-fitting")
    return result


def choose_c(source: pd.DataFrame) -> tuple[float, pd.DataFrame, list[dict]]:
    """Select C by minimum inner-cross-fitted source-fold log loss only."""
    candidates = []
    crossfits = {}
    for c in C_GRID:
        inner = inner_crossfit(source, c)
        loss = float(log_loss(inner["fracture_true"].astype(int),
                              inner["inner_meta_probability"], labels=[0, 1]))
        candidates.append({"C": c, "source_inner_log_loss": loss})
        crossfits[c] = inner
    selected = min(candidates, key=lambda item: (item["source_inner_log_loss"], item["C"]))["C"]
    return selected, crossfits[selected], candidates
