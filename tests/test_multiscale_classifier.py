"""Fast CPU/static checks; no training, CUDA inference or fixed-test data."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fracture_classifier.compare import aggregate, load_detector_baseline
from fracture_classifier.data import detector_checkpoints, fold_rows
from fracture_classifier.feature_hooks import SPECS, verify_backbone_graph
from fracture_classifier.multiscale_compare import P5_BEST_SLICE_AP, load_p5_matched, paired_bootstrap
from fracture_classifier.multiscale_model import MultiScaleFractureClassifier
from fracture_classifier.multiscale_training import audited_fold, seed_all
from fracture_classifier.training import ReviewedSlices
from fracture_classifier import inference as p5_inference
from fracture_classifier import multiscale_inference, multiscale_training


def test_all_real_run_a_checkpoint_graphs_and_feature_specs():
    assert [(spec.name, spec.layer, spec.stride, spec.channels) for spec in SPECS] == [
        ("P3", 4, 8, 256), ("P4", 6, 16, 256), ("P5", 10, 32, 512)]
    for path in detector_checkpoints().values():
        model = torch.load(path, map_location="cpu", weights_only=False)["model"]
        verify_backbone_graph(model)
    model.model[4] = nn.Identity()
    with pytest.raises(RuntimeError, match="P3"):
        verify_backbone_graph(model)


def test_real_fold0_synthetic_768_cpu_forward_and_frozen_backbone():
    model = MultiScaleFractureClassifier.from_detector(detector_checkpoints()[0])
    assert model.pooling_mode == "gap"
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.branches.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    model.train()
    assert not model.backbone.training
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 768, 768))
        shape_768 = {name: tuple(feature.shape) for name, feature in model.hooks.features.items()}
        probabilities = model.probabilities(torch.zeros(1, 3, 128, 128))
    assert logits.shape == probabilities.shape == (1,)
    assert shape_768 == {"P3": (1, 256, 96, 96), "P4": (1, 256, 48, 48),
                         "P5": (1, 512, 24, 24)}
    assert 0 <= probabilities.item() <= 1
    assert {name: tuple(feature.shape) for name, feature in model.hooks.features.items()} == {
        "P3": (1, 256, 16, 16), "P4": (1, 256, 8, 8), "P5": (1, 512, 4, 4)}
    # The 768 forward is also independently checked in the manual audit command.
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())


def test_actual_fold_counts_train_only_weight_and_existing_p5_protocol():
    expected = [
        (136, 33, 177, 3341, 46, 854, 18.875706214689266),
        (136, 33, 177, 3364, 46, 831, 19.005649717514125),
        (137, 32, 178, 3371, 45, 824, 18.93820224719101),
        (133, 36, 180, 3353, 43, 842, 18.627777777777776),
        (134, 35, 180, 3351, 43, 844, 18.616666666666667),
    ]
    for fold, target in enumerate(expected):
        train, val, weight, protocol = audited_fold(fold)
        observed = (train.study_id.nunique(), val.study_id.nunique(),
                    int((train.fracture_label == 1).sum()), int((train.fracture_label == 0).sum()),
                    int((val.fracture_label == 1).sum()), int((val.fracture_label == 0).sum()))
        assert observed == target[:6]
        assert weight == pytest.approx(target[6])
        assert protocol["seed"] == 42 + fold
        assert not set(train.patient_id) & set(val.patient_id)
        assert train.patient_id.nunique() == 124 and val.patient_id.nunique() == 31
        assert train.is_reviewed.all() and val.is_reviewed.all()


def test_unreviewed_json_ignored_and_patient_leakage_rejected():
    table = pd.DataFrame([
        {"fold": 0, "study_id": "s0", "patient_id": "p0", "is_reviewed": True, "fracture_label": 1},
        {"fold": 0, "study_id": "s0", "patient_id": "p0", "is_reviewed": True, "fracture_label": 0},
        {"fold": 1, "study_id": "s1", "patient_id": "p1", "is_reviewed": True, "fracture_label": 1},
        {"fold": 1, "study_id": "s1", "patient_id": "p1", "is_reviewed": True, "fracture_label": 0},
        {"fold": 1, "study_id": "s1", "patient_id": "p1", "is_reviewed": False, "fracture_label": pd.NA},
    ])
    train, val, weight = fold_rows(table, 0)
    assert len(train) == len(val) == 2 and weight == 1
    assert len(ReviewedSlices(train)) == 2
    with pytest.raises(ValueError, match="Unknown/unreviewed"):
        ReviewedSlices(table)
    leaked = table.copy()
    leaked.loc[leaked.fold == 1, "patient_id"] = "p0"
    with pytest.raises(RuntimeError, match="Patient leakage"):
        fold_rows(leaked, 0)


def test_deterministic_seed_and_fixed_top5_aggregation():
    seed_all(42)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    seed_all(42)
    second = (random.random(), np.random.random(), torch.rand(1).item())
    assert first == second
    assert aggregate([0.1, 0.9, 0.4, 0.8, 0.5, 0.7], "top5_mean") == pytest.approx(0.66)
    assert np.mean(P5_BEST_SLICE_AP) == pytest.approx(0.13509429196073868)


def test_preprocessing_reuses_exact_existing_p5_functions_and_requires_complete_oof(tmp_path, monkeypatch):
    assert multiscale_training.ReviewedSlices is ReviewedSlices
    assert multiscale_inference.image_tensor is p5_inference.image_tensor
    assert multiscale_inference.make_hu_input is p5_inference.make_hu_input
    assert multiscale_inference.read_slice is p5_inference.read_slice
    monkeypatch.setattr("fracture_classifier.multiscale_compare.OOF_OUTPUT", tmp_path)
    from fracture_classifier.multiscale_compare import load_multiscale_studies
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_multiscale_studies(pd.DataFrame())


def test_patient_clustered_bootstrap_is_paired_and_deterministic():
    rows = []
    for patient in range(10):
        for study in range(2 if patient == 0 else 1):
            positive = patient in (0, 1, 2)
            row = {"study_id": f"{patient}-{study}", "patient_id": str(patient),
                   "fracture_true": positive, "detector_probability": 0.4 if positive else 0.1,
                   "p5_fusion": 0.45 if positive else 0.2,
                   "multi_fusion": 0.6 if positive else 0.2, "MLS_mm": 0.0}
            row.update({key: 0.0 for key in ("V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH")})
            rows.append(row)
    table = pd.DataFrame(rows)
    first = paired_bootstrap(table, 20, 20260916)
    second = paired_bootstrap(table, 20, 20260916)
    assert first == second
    assert set(first) == {"vs_run_a", "vs_existing_p5_fusion"}
    assert set(first["vs_run_a"]) == {"sensitivity", "specificity", "pr_auc",
                                        "fracture_f1", "oracle_other_heads_macro_f1"}


def test_historical_run_a_and_p5_oof_pairing_reproduces_without_inference():
    matched = load_p5_matched(load_detector_baseline())
    assert len(matched) == 169 and matched.patient_id.nunique() == 155
    assert matched.groupby("patient_id").fold.nunique().max() == 1
