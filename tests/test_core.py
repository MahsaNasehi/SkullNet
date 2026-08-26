import numpy as np
import pytest
from fracture.data.windows import to_hu, bone_window, make_input
from fracture.data.annotations import Box, xywh_to_yolo
from fracture.inference.aggregation import aggregation_features, longest_run, StudyAggregator
from fracture.inference.calibration import ProbabilityCalibrator
from fracture.evaluation.triage import triage_from_intermediates
from fracture.data.splits import validate_no_leakage


def test_hu_and_window_and_monochrome1():
    hu = to_hu(np.array([[0, 100]], dtype=np.int16), 2, -1000)
    assert hu.tolist() == [[-1000.0, -800.0]]
    normal = bone_window(hu); inverse = bone_window(hu, monochrome1=True)
    assert np.array_equal(normal + inverse, np.full_like(normal, 255))


def test_25d_order_and_boundaries():
    images = [np.full((2, 2), i, np.uint8) for i in range(3)]
    assert make_input(images, 1, "2.5d")[0, 0].tolist() == [0, 1, 2]
    assert make_input(images, 0, "2.5d")[0, 0].tolist() == [0, 0, 1]


def test_yolo_conversion():
    assert xywh_to_yolo(Box(10, 20, 30, 40), 100, 100) == (0.25, 0.4, 0.3, 0.4)
    with pytest.raises(ValueError): xywh_to_yolo(Box(-1, 0, 1, 1), 100, 100)


def test_aggregation():
    scores = [0.4, 0.5, 0.1, 0.6]
    assert longest_run(scores, 0.3) == 2
    assert aggregation_features(scores)["top3_mean"] == pytest.approx(0.5)
    assert StudyAggregator().predict(scores) == 0.6


def test_calibration_range():
    assert ProbabilityCalibrator().predict(2) == 1.0


def test_exact_fracture_threshold():
    args = (0, 0, 0, 0, 0)
    assert triage_from_intermediates(*args, 0.4999, 0) == 0
    assert triage_from_intermediates(*args, 0.5, 0) == 1
    assert triage_from_intermediates(*args, 0.5001, 0) == 1


def test_leakage_detection():
    with pytest.raises(ValueError): validate_no_leakage([{"series_id": "a"}], [{"series_id": "a"}])
