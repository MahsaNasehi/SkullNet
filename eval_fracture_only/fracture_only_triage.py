"""Fracture-only isolation of the official IAAA triage rule.

Analogous to ICH-only isolation (zero MLS + fracture), this zeros every
ICH volume and MLS so that ``triage_from_intermediates`` can fire only via
``fracture_prob``.  The official function itself is left unchanged.

With V_* = 0 and MLS_mm = 0 the official rule reduces to:
  fracture_prob >= 0.5  ->  class 1 (Urgent; fracture with total_vol < 15)
  otherwise             ->  class 0 (Non-urgent)
Class 2 is unreachable without ICH or MLS evidence.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from triage_macro_f1 import (
    FRACTURE_PRESENCE_THRESHOLD,
    TRIAGE_REQUIRED_KEYS,
    triage_from_intermediates,
    validate_intermediates,
)

ZERO_NON_FRACTURE = {
    "V_EDH": 0.0,
    "V_SDH": 0.0,
    "V_IPH": 0.0,
    "V_SAH": 0.0,
    "V_IVH": 0.0,
    "MLS_mm": 0.0,
}


def fracture_only_intermediates(fracture_prob: float) -> Dict[str, float]:
    """Build a seven-key intermediate dict driven solely by fracture_prob."""
    return {
        **ZERO_NON_FRACTURE,
        "fracture_prob": float(fracture_prob),
    }


def fracture_only_triage(fracture_prob: float) -> int:
    """Official triage with ICH volumes and MLS forced to zero."""
    return triage_from_intermediates(fracture_only_intermediates(fracture_prob))


def isolate_fracture_head(intermediates: Mapping[str, Any]) -> Dict[str, float]:
    """Keep fracture_prob; zero every other official intermediate."""
    vals = validate_intermediates(intermediates)
    return fracture_only_intermediates(vals["fracture_prob"])


__all__ = [
    "FRACTURE_PRESENCE_THRESHOLD",
    "TRIAGE_REQUIRED_KEYS",
    "ZERO_NON_FRACTURE",
    "fracture_only_intermediates",
    "fracture_only_triage",
    "isolate_fracture_head",
    "triage_from_intermediates",
    "validate_intermediates",
]
