import numpy as np
import pytest
from fracture.data.windows import to_hu, bone_window, make_input
from fracture.data.annotations import Box, xywh_to_yolo, resolve_annotation
from fracture.inference.aggregation import aggregation_features, longest_run, StudyAggregator, FEATURE_NAMES
from fracture.inference.calibration import ProbabilityCalibrator, fit_calibrator
from fracture.evaluation.triage import triage_from_intermediates
from fracture.evaluation.predict_fold import _best_f1_threshold
from fracture.evaluation.qwk import quadratic_weighted_kappa
from fracture.evaluation.isolated_qwk import isolated_fracture_qwk
from fracture.evaluation.series_metrics import evaluate
from fracture.data.splits import validate_no_leakage, make_folds


def test_hu_and_window_and_monochrome1():
    hu = to_hu(np.array([[0, 100]], dtype=np.int16), 2, -1000)
    assert hu.tolist() == [[-1000.0, -800.0]]
    normal = bone_window(hu)
    inverse = bone_window(hu, monochrome1=True)
    assert np.array_equal(normal + inverse, np.full_like(normal, 255))


def test_25d_order_and_boundaries():
    images = [np.full((2, 2), i, np.uint8) for i in range(3)]
    assert make_input(images, 1, "2.5d")[0, 0].tolist() == [0, 1, 2]
    assert make_input(images, 0, "2.5d")[0, 0].tolist() == [0, 0, 1]
    assert make_input(images, 0, "2.5d", boundary_mode="mirror")[0, 0].tolist() == [1, 0, 1]


def test_yolo_conversion():
    assert xywh_to_yolo(Box(10, 20, 30, 40), 100, 100) == (0.25, 0.4, 0.3, 0.4)
    with pytest.raises(ValueError):
        xywh_to_yolo(Box(-1, 0, 1, 1), 100, 100)


def test_aggregation():
    scores = [0.4, 0.5, 0.1, 0.6]
    assert longest_run(scores, 0.3) == 2
    assert aggregation_features(scores)["top3_mean"] == pytest.approx(0.5)
    assert StudyAggregator().predict(scores) == 0.6
    assert list(aggregation_features(scores))[:3] == list(FEATURE_NAMES)[:3]


def test_consecutive_downweights_isolated_spike():
    spike = [0.01, 0.80, 0.01]
    run = [0.01, 0.40, 0.45, 0.42, 0.01]
    agg = StudyAggregator("consecutive", min_run=2, run_threshold=0.1)
    assert agg.predict(spike) == pytest.approx(0.40)
    assert agg.predict(run) == pytest.approx(0.45)


def test_calibration_range():
    assert ProbabilityCalibrator().predict(2) == 1.0
    calibrator = fit_calibrator([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], "platt")
    assert 0.0 <= calibrator.predict(0.5) <= 1.0


def test_exact_fracture_threshold():
    args = (0, 0, 0, 0, 0)
    assert triage_from_intermediates(*args, 0.4999, 0) == 0
    assert triage_from_intermediates(*args, 0.5, 0) == 1
    assert triage_from_intermediates(*args, 0.5001, 0) == 1


def test_leakage_detection():
    with pytest.raises(ValueError):
        validate_no_leakage([{"series_id": "a"}], [{"series_id": "a"}])


def test_make_folds_patient_grouped():
    rows = [
        {"series_id": f"s{i}", "patient_id": f"p{i // 2}", "label": i % 2}
        for i in range(20)
    ]
    folds = make_folds(rows, n_splits=5, seed=0, label_key="label", patient_key="patient_id")
    assert folds["num_folds"] == 5
    for fold in folds["folds"]:
        validate_no_leakage(
            [{"series_id": s, "patient_id": next(r["patient_id"] for r in rows if r["series_id"] == s)} for s in fold["train_series"]],
            [{"series_id": s, "patient_id": next(r["patient_id"] for r in rows if r["series_id"] == s)} for s in fold["val_series"]],
            patient_key="patient_id",
        )


def test_exploratory_study_threshold():
    threshold, metrics = _best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9])
    assert threshold == pytest.approx(0.5)
    assert metrics["f1_at_threshold"] == pytest.approx(1.0)


def test_arbitrary_threshold_metrics_are_not_mislabeled_as_point_five():
    metrics = evaluate([0, 1], [0.2, 0.4], threshold=0.3)
    assert metrics["threshold"] == pytest.approx(0.3)
    assert metrics["sensitivity_at_threshold"] == pytest.approx(1.0)
    assert "sensitivity_at_0_5" not in metrics
    point_five = evaluate([0, 1], [0.2, 0.8], threshold=0.5)
    assert point_five["sensitivity_at_0_5"] == point_five["sensitivity_at_threshold"]


def test_quadratic_weighted_kappa_perfect():
    assert quadratic_weighted_kappa([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)


def test_isolated_fracture_qwk_changes_with_fracture_prob():
    table = {
        "a": {"V_EDH": 0.0, "V_SDH": 0.0, "V_IPH": 0.0, "V_SAH": 0.0, "V_IVH": 0.0, "MLS_mm": 0.0, "y_true_fracture": 1.0, "triage_class": 1},
        "b": {"V_EDH": 0.0, "V_SDH": 0.0, "V_IPH": 0.0, "V_SAH": 0.0, "V_IVH": 0.0, "MLS_mm": 0.0, "y_true_fracture": 0.0, "triage_class": 0},
    }
    good = isolated_fracture_qwk(["a", "b"], [0.9, 0.1], table)
    bad = isolated_fracture_qwk(["a", "b"], [0.1, 0.9], table)
    assert good["isolated_fracture_qwk"] > bad["isolated_fracture_qwk"]


def test_missing_annotation_returns_none(tmp_path):
    assert resolve_annotation(tmp_path / "orig", tmp_path / "corr", "series", "sop") is None
