"""Official triage rules and exact fracture threshold behavior."""
from __future__ import annotations

EPS_VOLUME, EPS_MLS = 0.1, 1.0
MLS_CRITICAL, MLS_URGENT_LOW = 5.0, 3.0
EDH_CRIT, SDH_CRIT, IPH_CRIT, TOTAL_VOL_CRIT = 30.0, 70.0, 70.0, 60.0
COMBO_MLS, COMBO_VOL, FRAC_VOL_CRIT = 3.0, 40.0, 15.0
FRACTURE_PRESENCE_THRESHOLD = 0.5


def triage_from_intermediates(V_EDH: float, V_SDH: float, V_IPH: float, V_SAH: float, V_IVH: float, fracture_prob: float, MLS_mm: float) -> int:
    """Return 0=routine, 1=urgent, 2=critical using published constants."""
    volumes = [max(0.0, float(x)) for x in (V_EDH, V_SDH, V_IPH, V_SAH, V_IVH)]
    total = sum(volumes); fracture = float(fracture_prob) >= FRACTURE_PRESENCE_THRESHOLD; mls = max(0.0, float(MLS_mm))
    if mls >= MLS_CRITICAL or volumes[0] >= EDH_CRIT or volumes[1] >= SDH_CRIT or volumes[2] >= IPH_CRIT or total >= TOTAL_VOL_CRIT: return 2
    if mls >= COMBO_MLS and total >= COMBO_VOL: return 2
    if fracture and (mls >= MLS_CRITICAL or total >= FRAC_VOL_CRIT): return 2
    if mls >= MLS_URGENT_LOW or total >= EPS_VOLUME or fracture: return 1
    return 0

