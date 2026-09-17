"""Standalone fracture triage rules — no ICH, no MLS, no official function call.

These rules are *derived* from the competition rule set by deleting every
branch that needs hemorrhage volume or midline shift, then keeping only the
fracture-driven decisions that remain well-defined on their own.

Official branches that mention fracture
---------------------------------------
Critical:
  1) MLS >= 5 AND (has_ich OR fracture)     -> needs MLS / ICH  -> DROP
  5) fracture AND total_vol >= 15           -> needs ICH vol    -> DROP
Urgent:
  9) fracture AND total_vol < 15            -> with vol ignored,
                                               reduces to: fracture -> Urgent

Standalone rule set (only input: fracture_prob)
-----------------------------------------------
  if fracture_prob >= 0.5:  return 1   # Urgent — fracture present
  else:                     return 0   # Non-urgent

Class 2 (Critical) is intentionally absent: under the official clinical
logic, fracture alone never makes a study Critical without hemorrhage.

This is exactly what you want for measuring *your* fracture head:
GT class = rule(SkullFracture in {0,1})
Pred class = rule(model fracture_prob)
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

# Same cutoff the official competition rule uses for fracture presence.
FRACTURE_PRESENCE_THRESHOLD = 0.5

# Reachable labels under these standalone rules.
STANDALONE_LABELS = (0, 1)

RULE_DERIVATION = {
    "source": "IAAA official triage_from_intermediates",
    "dropped_because_needs_MLS_or_ICH": [
        "MLS >= 5 and (has_ich or fracture) -> Critical",
        "V_EDH/SDH/IPH/total volume Critical thresholds",
        "has_ich and MLS >= 3 and total_vol >= 40 -> Critical",
        "fracture and total_vol >= 15 -> Critical",
        "MLS >= 5 without ich/fracture -> Urgent",
        "has_ich -> Urgent",
        "3 <= MLS < 5 -> Urgent",
        "total_vol >= 0.1 and mls_present -> Urgent",
    ],
    "kept_and_simplified": [
        {
            "official": "fracture_present and total_vol < 15 -> Urgent (1)",
            "standalone": "fracture_present -> Urgent (1)",
            "reason": "total_vol is an ICH quantity; ignoring it leaves fracture alone.",
        }
    ],
    "standalone_rules": [
        "if fracture_prob >= 0.5: return 1  # Urgent",
        "return 0  # Non-urgent",
    ],
}


def fracture_present(fracture_prob: float, threshold: float = FRACTURE_PRESENCE_THRESHOLD) -> bool:
    """Official-style fracture presence bit."""
    return float(fracture_prob) >= float(threshold)


def standalone_fracture_triage(
    fracture_prob: float,
    threshold: float = FRACTURE_PRESENCE_THRESHOLD,
) -> int:
    """Triage class from fracture probability alone.

    Parameters
    ----------
    fracture_prob:
        Model probability in [0, 1], or GT {0, 1}.
    threshold:
        Presence cutoff (competition default = 0.5).

    Returns
    -------
    int
        0 = Non-urgent, 1 = Urgent. Never returns 2.
    """
    if fracture_present(fracture_prob, threshold=threshold):
        return 1
    return 0


def triage_many(
    fracture_probs: Sequence[float],
    threshold: float = FRACTURE_PRESENCE_THRESHOLD,
) -> list:
    return [standalone_fracture_triage(p, threshold=threshold) for p in fracture_probs]


def explain_rules() -> Dict[str, Any]:
    """Human-readable derivation for reports / README."""
    return {
        "name": "standalone_fracture_triage",
        "inputs": ["fracture_prob"],
        "outputs": {"0": "Non-urgent", "1": "Urgent"},
        "threshold": FRACTURE_PRESENCE_THRESHOLD,
        "depends_on_ICH": False,
        "depends_on_MLS": False,
        "calls_official_triage": False,
        "derivation": RULE_DERIVATION,
    }


__all__ = [
    "FRACTURE_PRESENCE_THRESHOLD",
    "STANDALONE_LABELS",
    "RULE_DERIVATION",
    "explain_rules",
    "fracture_present",
    "standalone_fracture_triage",
    "triage_many",
]
