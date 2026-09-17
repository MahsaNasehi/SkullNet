"""Synthetic and authoritative CPU checks for OOF-only gated rescue."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fracture_classifier.calibration import C_GRID, choose_c, logit_features
from fracture_classifier.compare_fusion_v2 import (category, crossfit_v2,
                                                    fn_and_distribution_audit,
                                                    load_verified_oof,
                                                    patient_bootstrap,
                                                    system_report)
from fracture_classifier.gated_fusion import OFFICIAL_THRESHOLD, TAU_GRID, choose_tau, gated_probability


def test_exact_run_a_baseline_and_cohort_without_fixed_test():
    table, provenance = load_verified_oof()
    report = provenance["run_a_detector_only"]
    assert provenance["baseline_reproduced_exactly"] is True
    assert [report["fracture"][key] for key in ("tp", "fp", "fn", "tn")] == [7, 3, 17, 142]
    assert report["fracture"]["pr_auc"] == pytest.approx(0.5355486380673358)
    assert report["fracture"]["roc_auc"] == pytest.approx(0.7298850574712643)
    assert report["triage"]["pooled_macro_f1"] == pytest.approx(0.9696273781380164)
    assert len(table) == 169 and table.patient_id.nunique() == 155
    assert table.groupby("patient_id").fold.nunique().max() == 1
    assert table.fracture_true.sum() == 24


def test_logit_clipping_and_frozen_official_gate():
    features = logit_features(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    assert features.shape == (2, 2) and np.isfinite(features).all()
    assert OFFICIAL_THRESHOLD == 0.5 and TAU_GRID == (
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
    detector = np.array([0.8, 0.49, 0.1, 0.01])
    meta = np.array([0.2, 0.8, 0.9, 0.9])
    assert gated_probability(detector, meta, 0.10).tolist() == pytest.approx([0.8, 0.8, 0.9, 0.01])
    with pytest.raises(ValueError):
        gated_probability(detector, meta, 0.07)
    assert category(0.4, 0.1) == "near_threshold"
    assert category(0.2, 0.8) == "detector_low_classifier_high"
    assert category(0.2, 0.1) == "both_low"


def _synthetic_five_folds() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for index in range(8):
            positive = index == 0
            row = {"fold": fold, "study_id": f"{fold}-{index}",
                   "patient_id": f"p-{fold}-{index}", "fracture_true": positive,
                   "detector_fixed": 0.38 + 0.02 * fold if positive else 0.05 + 0.01 * index,
                   "classifier_top5": 0.7 if positive else 0.15 + 0.01 * index,
                   "MLS_mm": 0.0}
            row.update({key: 0.0 for key in ("V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH")})
            rows.append(row)
    return pd.DataFrame(rows)


def test_c_and_tau_selection_use_only_four_source_folds():
    table = _synthetic_five_folds()
    source = table[table.fold != 4].copy()
    c, inner, candidates = choose_c(source)
    tau, gate_candidates = choose_tau(inner)
    assert c in C_GRID and tau in TAU_GRID
    assert len(candidates) == 3 and len(gate_candidates) == 9
    assert set(inner.fold) == {0, 1, 2, 3} and not set(inner.study_id) & set(table[table.fold == 4].study_id)
    changed = table.copy()
    changed.loc[changed.fold == 4, "fracture_true"] = ~changed.loc[changed.fold == 4, "fracture_true"]
    changed.loc[changed.fold == 4, "classifier_top5"] = 0.99
    c_again, inner_again, _ = choose_c(changed[changed.fold != 4])
    tau_again, _ = choose_tau(inner_again)
    assert c_again == c and tau_again == tau
    assert inner_again.inner_meta_probability.tolist() == pytest.approx(inner.inner_meta_probability.tolist())


def test_actual_fn_audit_and_outer_fold_selection():
    table, _ = load_verified_oof()
    fn, distribution, impact, review, summary = fn_and_distribution_audit(table)
    assert len(fn) == len(impact) == 17
    assert len(distribution) == 6 and len(review) >= 17
    assert sum(summary["category_counts"].values()) == 17
    assert set(fn.diagnostic_category) <= {"near_threshold", "detector_low_classifier_high", "both_low"}
    evaluated, selections = crossfit_v2(table)
    assert len(evaluated) == 169 and len(selections) == 5
    for row in selections:
        assert row["heldout_fold"] not in row["source_folds"]
        assert len(row["source_folds"]) == 4
        assert row["selected_C"] in C_GRID and row["selected_tau_low"] in TAU_GRID


def test_metrics_and_patient_bootstrap_determinism():
    table, _ = load_verified_oof()
    report = system_report(table, "detector_probability")
    assert report["binary_fracture"]["tp"] == 7
    assert report["oracle_other_heads_triage"]["confusion_matrix"] == [
        [22, 1, 0], [0, 50, 1], [0, 3, 92]]
    assert report["binary_fracture"]["balanced_accuracy"] == pytest.approx(
        (0.2916666666666667 + 0.9793103448275862) / 2)
    first = patient_bootstrap(table, {"fixed_f0": "f0_fixed"}, 20, 20260916)
    second = patient_bootstrap(table, {"fixed_f0": "f0_fixed"}, 20, 20260916)
    assert first == second
    assert "delta_95_ci" in first["fixed_f0"]["oracle_other_heads_macro_f1"]
