"""Official triage rules and exact fracture threshold behavior.

Ported line-for-line from the organizer HTML
(``iaaa-competition-2026-brain-ct-triage-challenge.html``, the reference
``triage_from_intermediates``-equivalent implementation around lines
480-571). The previous version of this module returned Critical whenever
``MLS_mm >= MLS_CRITICAL`` regardless of hemorrhage/fracture evidence; the
official rule requires MLS-driven Critical to be accompanied by
intracranial hemorrhage (``has_ich``) or fracture, and otherwise downgrades
a high-MLS study to Urgent (rule 6 below). See
``FRACTURE_QWK_99_REVIEW_FA.md`` section 4 for the audit that found this
discrepancy: it silently reclassified nine fracture-negative,
(near-)zero-ICH studies (271016, 271656, 272468, 2732, 4167, 4445, 674, 880,
9248) from Urgent to Critical.
"""
from __future__ import annotations

EPS_VOLUME, EPS_MLS = 0.1, 1.0
MLS_CRITICAL, MLS_URGENT_LOW = 5.0, 3.0
EDH_CRIT, SDH_CRIT, IPH_CRIT, TOTAL_VOL_CRIT = 30.0, 70.0, 70.0, 60.0
COMBO_MLS, COMBO_VOL, FRAC_VOL_CRIT = 3.0, 40.0, 15.0
FRACTURE_PRESENCE_THRESHOLD = 0.5


def triage_from_intermediates(
    V_EDH: float,
    V_SDH: float,
    V_IPH: float,
    V_SAH: float,
    V_IVH: float,
    fracture_prob: float,
    MLS_mm: float,
) -> int:
    """Return 0=routine, 1=urgent, 2=critical using the official rule order.

    Every branch below is numbered to match the organizer's reference
    implementation exactly; do not reorder or merge branches. Later
    "Urgent" rules (6, 9, 10) are only reached when no "Critical" rule
    (1-5) already returned, which is why MLS-driven Critical (rule 1)
    requires ``has_ich or fracture_present`` while MLS-driven Urgent
    (rule 6) requires the opposite.
    """
    V_EDH, V_SDH, V_IPH, V_SAH, V_IVH = (
        max(0.0, float(x)) for x in (V_EDH, V_SDH, V_IPH, V_SAH, V_IVH)
    )
    MLS_mm = max(0.0, float(MLS_mm))
    fracture_present = float(fracture_prob) >= FRACTURE_PRESENCE_THRESHOLD

    total_vol = V_EDH + V_SDH + V_IPH + V_SAH + V_IVH
    has_ich = total_vol >= EPS_VOLUME
    mls_present = MLS_mm >= EPS_MLS

    # --- Critical (2) ---
    # 1) MLS-driven critical *only if* there is ICH or fracture.
    if MLS_mm >= MLS_CRITICAL and (has_ich or fracture_present):
        return 2
    # 2) Large single-compartment hemorrhages.
    if V_EDH >= EDH_CRIT or V_SDH >= SDH_CRIT or V_IPH >= IPH_CRIT:
        return 2
    # 3) Large total hemorrhage burden.
    if total_vol >= TOTAL_VOL_CRIT:
        return 2
    # 4) Combined hemorrhage + MLS rule.
    if has_ich and MLS_mm >= COMBO_MLS and total_vol >= COMBO_VOL:
        return 2
    # 5) Fracture with substantial hemorrhage.
    if fracture_present and total_vol >= FRAC_VOL_CRIT:
        return 2

    # --- Urgent (1) ---
    # 6) MLS in the critical range but with neither hemorrhage nor fracture.
    if MLS_mm >= MLS_CRITICAL and not (has_ich or fracture_present):
        return 1
    # 7) Any meaningful hemorrhage.
    if has_ich:
        return 1
    # 8) Moderate MLS range.
    if MLS_URGENT_LOW <= MLS_mm < MLS_CRITICAL:
        return 1
    # 9) Fracture with small (sub-Critical) hemorrhage.
    if fracture_present and total_vol < FRAC_VOL_CRIT:
        return 1
    # 10) Small hemorrhage with any non-trivial MLS (unreachable given rule
    #     7 above, kept only to mirror the official implementation exactly).
    if total_vol >= EPS_VOLUME and mls_present:
        return 1

    return 0
