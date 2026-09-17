"""Official IAAA triage rules and three-class Macro-F1 reporting.

The rule implementation is a strict wrapper of the organizer-provided
``triage_from_intermediates`` function.  The fracture threshold inside the
official rule is always 0.5.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


TRIAGE_REQUIRED_KEYS = {
    "V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH", "fracture_prob", "MLS_mm"
}
TRIAGE_LABELS = (0, 1, 2)
FRACTURE_PRESENCE_THRESHOLD = 0.5


def validate_intermediates(intermediates: Mapping[str, Any]) -> Dict[str, float]:
    """Validate exactly the seven keys required by the official function."""
    missing = TRIAGE_REQUIRED_KEYS - intermediates.keys()
    extra = intermediates.keys() - TRIAGE_REQUIRED_KEYS
    if missing:
        raise ValueError(
            "Missing keys in intermediates: %s. Expected keys: %s."
            % (sorted(missing), sorted(TRIAGE_REQUIRED_KEYS))
        )
    if extra:
        raise ValueError(
            "Unexpected keys in intermediates: %s. Expected keys: %s."
            % (sorted(extra), sorted(TRIAGE_REQUIRED_KEYS))
        )
    cleaned = {}  # type: Dict[str, float]
    for key in TRIAGE_REQUIRED_KEYS:
        try:
            cleaned[key] = float(intermediates[key])
        except Exception as exc:
            raise TypeError(
                "Value for key %r must be convertible to float, got type %s."
                % (key, type(intermediates[key]).__name__)
            ) from exc
    return cleaned


def triage_from_intermediates(intermediates: Mapping[str, Any]) -> int:
    """Return 0=non-urgent, 1=urgent, 2=critical using official rule order."""
    vals = validate_intermediates(intermediates)
    v_edh = max(0.0, vals["V_EDH"])
    v_sdh = max(0.0, vals["V_SDH"])
    v_iph = max(0.0, vals["V_IPH"])
    v_sah = max(0.0, vals["V_SAH"])
    v_ivh = max(0.0, vals["V_IVH"])
    mls_mm = max(0.0, vals["MLS_mm"])
    fracture_present = vals["fracture_prob"] >= FRACTURE_PRESENCE_THRESHOLD

    total_vol = v_edh + v_sdh + v_iph + v_sah + v_ivh
    has_ich = total_vol >= 0.1
    mls_present = mls_mm >= 1.0

    if mls_mm >= 5.0 and (has_ich or fracture_present):
        return 2
    if v_edh >= 30.0:
        return 2
    if v_sdh >= 70.0:
        return 2
    if v_iph >= 70.0:
        return 2
    if total_vol >= 60.0:
        return 2
    if has_ich and mls_mm >= 3.0 and total_vol >= 40.0:
        return 2
    if fracture_present and total_vol >= 15.0:
        return 2
    if mls_mm >= 5.0 and not (has_ich or fracture_present):
        return 1
    if has_ich:
        return 1
    if 3.0 <= mls_mm < 5.0:
        return 1
    if fracture_present and total_vol < 15.0:
        return 1
    if total_vol >= 0.1 and mls_present:
        return 1
    return 0


def triage_macro_f1_report(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    """Return pooled Macro-F1, classwise metrics, confusion matrix and accuracy."""
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    if truth.ndim != 1 or predicted.ndim != 1 or len(truth) != len(predicted) or not len(truth):
        raise ValueError("y_true and y_pred must be non-empty, equal-length one-dimensional arrays")
    allowed = set(TRIAGE_LABELS)
    if not set(truth.tolist()) <= allowed or not set(predicted.tolist()) <= allowed:
        raise ValueError("Triage labels must be 0, 1, or 2")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(TRIAGE_LABELS), zero_division=0
    )
    return {
        "pooled_macro_f1": float(np.mean(f1)),
        "micro_f1": float(accuracy_score(truth, predicted)),
        "classwise": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(TRIAGE_LABELS)
        },
        "confusion_matrix_labels": list(TRIAGE_LABELS),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(TRIAGE_LABELS)).tolist(),
        "accuracy": float(accuracy_score(truth, predicted)),
        "n_studies": int(len(truth)),
    }


def triage_many(rows: Iterable[Mapping[str, Any]]) -> list:
    return [triage_from_intermediates(row) for row in rows]
