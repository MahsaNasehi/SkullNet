"""CPU-only submission checks; no YOLO load or real DICOM inference."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submit"
EXPECTED_SHA = "109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640"


def _module():
    spec = importlib.util.spec_from_file_location("full169_submit_under_test", SUBMIT / "model.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_active_checkpoint_and_selection_provenance():
    source = ROOT / "outputs/yolo26s_p2_hu800_ww1600_run_a_full169/weights/final_deployment.pt"
    assert _sha(SUBMIT / "models/best.pt") == _sha(source) == EXPECTED_SHA
    assert _sha(ROOT / "submit_backup_before_full169_20260916/models/best.pt") != EXPECTED_SHA
    report = json.loads((SUBMIT / "SELECTION_REPORT.json").read_text())
    assert report["selected_checkpoint"]["sha256"] == EXPECTED_SHA
    assert report["selected_checkpoint"]["fold"] is None
    assert report["selected_checkpoint"]["fixed_training_epochs"] == 59
    assert "absent" in report["fracture_classifier_status"]


def test_defaults_and_submission_api_with_synthetic_predictions():
    module = _module()
    assert inspect.signature(module.Model.predict).parameters["study_dir"]
    assert (module.WINDOW_LEVEL, module.WINDOW_WIDTH, module.IMAGE_SIZE,
            module.CONTEXT_DISTANCE_MM, module.CONFIDENCE, module.NMS_IOU) == (
                800.0, 1600.0, 768, 5.0, 0.01, 0.5)
    assert module.AGGREGATION_METHOD == "top10_percent_mean"
    assert module.OFFICIAL_FRACTURE_THRESHOLD == 0.5
    model = module.Model.__new__(module.Model)
    model.detector = object()
    model.target_series_uid = None
    model.aggregation_method = "top10_percent_mean"
    model.optimize_macro_f1 = True
    model.raw_macro_f1_threshold = module.RAW_MACRO_F1_THRESHOLD
    model.batch_size = 1
    synthetic = [SimpleNamespace(max_confidence=module.RAW_MACRO_F1_THRESHOLD)]
    with patch.object(module, "load_study", return_value=[object()]), patch.object(
            module, "predict_slices", return_value=synthetic):
        outputs = model.predict("synthetic-study")
    assert list(outputs) == ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH",
                             "fracture_prob", "MLS_mm"]
    assert outputs["fracture_prob"] == pytest.approx(0.5)
    assert all(value == 0.0 for key, value in outputs.items() if key != "fracture_prob")
    assert all(isinstance(value, float) and 0.0 <= value <= 1.0 for value in outputs.values())
    for raw in (0.0, 0.2, 0.8, 1.0):
        assert 0.0 <= module.rescale_for_macro_f1(raw) <= 1.0


def test_target_series_all_slices_and_multiseries_guard(tmp_path):
    module = _module()
    paths = [tmp_path / "a1.dcm", tmp_path / "a2.dcm", tmp_path / "b1.dcm"]
    for path in paths:
        path.write_bytes(b"synthetic")
    series = {"a1.dcm": "A", "a2.dcm": "A", "b1.dcm": "B"}

    def header(path, stop_before_pixels=True):
        assert stop_before_pixels is True
        return SimpleNamespace(SeriesInstanceUID=series[Path(path).name])

    def record(path, study_id):
        name = Path(path).name
        return SimpleNamespace(sop_uid=name, physical_position=float(
            {"a1.dcm": 2, "a2.dcm": 1, "b1.dcm": 3}[name]), instance_number=None,
            series_id=study_id, hu=np.zeros((2, 2)))

    import pydicom
    with patch.object(pydicom, "dcmread", side_effect=header), patch.object(
            module, "read_slice", side_effect=record):
        with pytest.raises(RuntimeError, match="Multiple DICOM series"):
            module.load_study(tmp_path)
        selected = module.load_study(tmp_path, target_series_uid="A")
    assert [item.sop_uid for item in selected] == ["a2.dcm", "a1.dcm"]
    assert len(selected) == 2
