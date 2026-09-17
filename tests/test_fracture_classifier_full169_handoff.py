"""Fast CPU-only checks; synthetic prediction contract, no DICOM inference."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from fracture_classifier import build_handoff, train_full169
from fracture_classifier.data import FINAL_DEPLOYMENT, sha256
from fracture_classifier.model import FractureSliceClassifier


def _runtime():
    detector_stub = types.ModuleType("detector_runtime")
    detector_stub.Detector = object
    detector_stub.aggregate_top10_percent_mean = lambda scores: float(max(scores))
    detector_stub.rescale_for_macro_f1 = lambda score, threshold: float(score)
    detector_stub.make_hu_input = lambda *args, **kwargs: np.zeros((768, 768, 3), dtype=np.uint8)
    detector_stub.read_slice = lambda *args, **kwargs: None
    detector_stub.sort_records = lambda records: records
    classifier_stub = types.ModuleType("classifier_model")
    classifier_stub.FractureSliceClassifier = object
    path = Path(__file__).resolve().parents[1] / "src/fracture_classifier/handoff_runtime.py"
    with patch.dict(sys.modules, {"detector_runtime": detector_stub,
                                  "classifier_model": classifier_stub}):
        spec = importlib.util.spec_from_file_location("synthetic_handoff_runtime", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def test_full169_cohort_and_fixed_schedule_cpu():
    reviewed, audit = train_full169.audit_training_inputs()
    assert len(reviewed) == 4418
    assert {key: audit[key] for key in train_full169.EXPECTED_COUNTS} == {
        "positive": 223, "negative": 4195, "ignored": 24,
        "studies": 169, "patients": 155}
    assert set(reviewed["fracture_label"]) == {0, 1}
    assert audit["oof_best_epochs"] == [3, 7, 9, 9, 10]
    assert train_full169.FIXED_EPOCHS == 9
    assert audit["pos_weight"] == pytest.approx(4195 / 223)
    assert sha256(FINAL_DEPLOYMENT) == train_full169.EXPECTED_DETECTOR_SHA256


def test_head_only_architecture_and_freeze():
    model = FractureSliceClassifier.from_detector(FINAL_DEPLOYMENT)
    model.freeze_backbone()
    assert sum(parameter.numel() for parameter in model.backbone.parameters()) > 0
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert isinstance(model.head[0], torch.nn.LayerNorm)
    assert isinstance(model.head[1], torch.nn.Linear)
    assert (model.head[1].in_features, model.head[1].out_features) == (512, 128)
    assert model.head[3].p == 0.10
    assert model.head[4].out_features == 1


def test_frozen_detector_rule_and_five_oof_selections():
    helper = build_handoff._source_rules()
    compile(helper, "detector_runtime.py", "exec")
    assert "def aggregate_top10_percent_mean" in helper
    assert "def rescale_for_macro_f1" in helper
    assert "class Model:" not in helper
    report = json.loads((build_handoff.ROOT / "outputs/run_a_fracture_classifier_oof/comparison/report.json").read_text())
    assert len(report["selection"]) == 5
    assert all(item["classifier_aggregator"] == "top5_mean" and item["fusion_alpha"] == 0.75
               for item in report["selection"])


def test_top5_fusion_range_and_official_threshold():
    runtime = _runtime()
    assert runtime.top5_mean([0.1, 0.5, 0.7, 0.3, 0.9, 0.2]) == pytest.approx(0.52)
    assert runtime.fuse_probabilities(0.8, 0.4) == pytest.approx(0.7)
    assert runtime.fuse_probabilities(0, 0) == 0
    assert runtime.fuse_probabilities(1, 1) == 1
    with pytest.raises(ValueError):
        runtime.fuse_probabilities(1.2, 0.5)
    assert build_handoff.RAW_THRESHOLD == 0.39717610677083337
    assert '"official_fracture_threshold": 0.5' in inspect_source(build_handoff)


def inspect_source(module) -> str:
    return Path(module.__file__).read_text()


def test_synthetic_one_study_predicts_one_probability():
    runtime = _runtime()
    predictor = runtime.FractureFusionPredictor.__new__(runtime.FractureFusionPredictor)
    predictor.config = {"detector_raw_operating_point": 0.39717610677083337}
    predictor.device = "cpu"
    predictor.batch_size = 8
    predictor.detector = types.SimpleNamespace(predict_batch=lambda images: [
        {"max_confidence": 0.8}, {"max_confidence": 0.2}])
    predictor.classifier = types.SimpleNamespace(probabilities=lambda tensor: torch.tensor([0.4, 0.6]))
    record = types.SimpleNamespace(hu=np.zeros((4, 4), dtype=np.float32), physical_position=0.0,
                                   photometric_interpretation="MONOCHROME2")
    with patch.object(runtime, "load_target_records", return_value=[record, record]):
        result = predictor.predict_study("synthetic_study")
    assert isinstance(result, float)
    assert result == pytest.approx(0.75 * 0.8 + 0.25 * 0.5)
    assert predictor.last_slice_count == 2


def test_no_overwrite_guards(tmp_path):
    with patch.object(train_full169, "OUTPUT", tmp_path):
        with pytest.raises(FileExistsError):
            train_full169.train("cpu")
    with patch.object(build_handoff, "DESTINATION", tmp_path):
        with pytest.raises(FileExistsError):
            build_handoff._validate()


def test_package_refuses_untrained_classifier(tmp_path):
    with patch.object(build_handoff, "OUTPUT", tmp_path / "untrained"), patch.object(
            build_handoff, "DESTINATION", tmp_path / "new_handoff"):
        with pytest.raises(RuntimeError, match="finalized"):
            build_handoff._validate()


def test_synthetic_finalization_is_byte_identical_and_no_overwrite(tmp_path):
    manifest_sha = sha256(train_full169.EXPERIMENT / "slice_label_manifest.csv")
    history = [{"epoch_one_based": epoch, "train_loss": 1.0 / epoch}
               for epoch in range(1, 10)]
    (tmp_path / "protocol.json").write_text(json.dumps({
        "fixed_epochs": 9, "validation_used": False,
        "checkpoint_selection": "fixed final epoch 9 only",
        "source_detector_sha256": train_full169.EXPECTED_DETECTOR_SHA256,
        "slice_label_manifest_sha256": manifest_sha}))
    (tmp_path / "classifier_training_history.json").write_text(json.dumps(history))
    last = tmp_path / "last_classifier.pt"
    torch.save({"model_state": {}, "stage": 1, "epoch_one_based": 9,
                "fixed_epochs": 9, "training_history": history,
                "source_detector_sha256": train_full169.EXPECTED_DETECTOR_SHA256,
                "slice_label_manifest_sha256": manifest_sha}, last)
    (tmp_path / "training_finished.json").write_text(json.dumps({
        "epochs": 9, "last_classifier_sha256": sha256(last),
        "frozen_backbone_unchanged": True}))
    with patch.object(train_full169, "OUTPUT", tmp_path):
        report = train_full169.finalize()
        assert report["final_classifier_sha256"] == sha256(last)
        assert (tmp_path / "final_classifier.pt").read_bytes() == last.read_bytes()
        with pytest.raises(FileExistsError):
            train_full169.finalize()
