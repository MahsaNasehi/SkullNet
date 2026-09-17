"""Synthetic CPU-only safety tests; no real model inference or GPU training."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fracture_classifier.compare import ALPHAS, AGGREGATORS, aggregate, crossfit_classifier_fusion, fusion
from fracture_classifier.data import (EXPERIMENT, FINAL_DEPLOYMENT, detector_checkpoints,
                                      fold_rows, reviewed_binary_label, sha256)
from fracture_classifier.model import FractureSliceClassifier
from fracture_classifier.training import ReviewedSlices


def test_reviewed_positive_requires_valid_box_and_negative_requires_json(tmp_path):
    annotation, label = tmp_path / "sop.json", tmp_path / "sop.txt"
    with pytest.raises(FileNotFoundError):
        reviewed_binary_label(annotation, 0, label)
    annotation.write_text(json.dumps({"boxes_xywh": [[1, 2, 3, 4]]}))
    label.write_text("0 0.1 0.2 0.3 0.4\n")
    assert reviewed_binary_label(annotation, 1, label) == 1
    with pytest.raises(RuntimeError):
        reviewed_binary_label(annotation, 0, label)
    annotation.write_text(json.dumps({"boxes_xywh": []}))
    label.write_text("")
    assert reviewed_binary_label(annotation, 0, label) == 0


def test_unknown_is_not_negative_training_data():
    rows = pd.DataFrame([{"is_reviewed": False, "fracture_label": None, "image_path": "ignored"}])
    with pytest.raises(ValueError, match="Unknown"):
        ReviewedSlices(rows)


def test_fold_patient_leakage_and_train_only_pos_weight():
    rows = pd.DataFrame([
        {"fold": 0, "patient_id": "a", "is_reviewed": True, "fracture_label": 1},
        {"fold": 1, "patient_id": "b", "is_reviewed": True, "fracture_label": 1},
        {"fold": 2, "patient_id": "c", "is_reviewed": True, "fracture_label": 0},
        {"fold": 3, "patient_id": "d", "is_reviewed": True, "fracture_label": 0},
        {"fold": 4, "patient_id": "e", "is_reviewed": True, "fracture_label": 0},
        {"fold": 4, "patient_id": "e", "is_reviewed": False, "fracture_label": None},
    ])
    train, val, weight = fold_rows(rows, 0)
    assert len(train) == 4 and len(val) == 1 and weight == 3.0
    assert set(train.patient_id).isdisjoint(set(val.patient_id))
    rows.loc[rows.fold == 1, "patient_id"] = "a"
    with pytest.raises(RuntimeError, match="leakage"):
        fold_rows(rows, 0)


def test_stage1_backbone_frozen_one_logit_and_probability_range():
    backbone = nn.Sequential(*([nn.Identity()] * 10), nn.Conv2d(3, 512, 1))
    model = FractureSliceClassifier(backbone)
    model.train()
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    assert not model.backbone.training
    x = torch.rand(2, 3, 8, 8)
    logits = model(x)
    assert logits.shape == (2,)
    probability = model.probabilities(x)
    assert probability.shape == (2,) and torch.all((probability >= 0) & (probability <= 1))
    model.unfreeze_last_stage()
    assert all(p.requires_grad for p in model.backbone[10].parameters())
    assert all(not p.requires_grad for p in model.backbone[:10].parameters())


def test_aggregation_short_studies_and_exact_fusion():
    assert all(aggregate([0.8], method) == pytest.approx(0.8) for method in AGGREGATORS)
    assert aggregate([0.2, 0.8], "top5_mean") == pytest.approx(0.5)
    assert aggregate([0.2, 0.8], "top10_percent_mean") == pytest.approx(0.8)
    assert np.allclose(fusion(np.array([0.2]), np.array([0.8]), 0.25), [0.65])
    with pytest.raises(ValueError):
        fusion(np.array([0.2]), np.array([0.8]), 0.1)
    assert ALPHAS == (0.25, 0.50, 0.75)


def test_crossfit_selection_excludes_evaluated_fold(monkeypatch):
    rows = []
    for fold in range(5):
        for positive in (False, True):
            score = 0.8 if positive else 0.2
            row = {"fold": fold, "study_id": f"{fold}_{int(positive)}", "patient_id": f"p{fold}_{int(positive)}",
                   "fracture_true": positive, "V_EDH": 0.0, "V_SDH": 0.0, "V_IPH": 0.0,
                   "V_SAH": 0.0, "V_IVH": 0.0, "MLS_mm": 0.0, "detector_probability": score}
            for method in AGGREGATORS:
                row[f"classifier_{method}"] = score
            rows.append(row)
    table = pd.DataFrame(rows)
    seen_training_folds = []

    def detector_selector(training):
        seen_training_folds.append(set(training.fold))
        training["max"] = training["detector_probability"]
        return "max", 0.5, {}

    monkeypatch.setattr("fracture_classifier.compare.select_aggregator_and_threshold", detector_selector)
    heldout, choices = crossfit_classifier_fusion(table)
    assert len(heldout) == 10
    for fold, choice in enumerate(choices):
        assert set(choice["selection_source_folds"]) == set(range(5)) - {fold}
        assert seen_training_folds[fold] == set(range(5)) - {fold}
        assert choice["fusion_alpha"] in ALPHAS
    assert np.isfinite(heldout.fusion_probability).all()


def test_protected_checkpoint_paths_are_disjoint_from_classifier_outputs():
    assert FINAL_DEPLOYMENT.name == "final_deployment.pt"
    assert all(path.name == "best.pt" for path in detector_checkpoints().values())
    for fold in range(5):
        target = EXPERIMENT.parent / f"run_a_fracture_classifier_fold{fold}"
        assert target != FINAL_DEPLOYMENT.parent.parent
        assert all(target not in path.parents for path in detector_checkpoints().values())
        assert target not in FINAL_DEPLOYMENT.parents


def test_existing_audit_proves_development_only_and_protected_hashes():
    audit = json.loads((EXPERIMENT / "slice_label_audit.json").read_text())
    labels = pd.read_csv(EXPERIMENT / "slice_label_manifest.csv")
    assert audit["studies"] == 169 and audit["patients"] == 155
    assert not audit["patient_leakage"]
    assert len(labels[labels.is_reviewed]) == 4418
    assert not labels.loc[labels.is_reviewed, "image_path"].str.contains("/images/test/").any()
    assert sha256(FINAL_DEPLOYMENT) == audit["full169_deployment_sha256"]
    for fold, path in detector_checkpoints().items():
        assert sha256(path) == audit["detector_sha256"][str(fold)]


def test_official_threshold_is_not_tunable_in_classifier_experiment():
    source = Path(__file__).parents[1] / "src/fracture_classifier/compare.py"
    assert '"official_threshold": 0.5' in source.read_text()
