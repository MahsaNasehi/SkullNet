"""Fast CPU-only tests for the apparent FULL169 evaluator; no model inference."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.windows import make_hu_input as historical_hu_input
from fracture_classifier import evaluate_full169 as evaluator


def test_frozen_rule_top5_fusion_and_probability_range():
    detector, classifier, fusion = evaluator.probability_triplet(
        [evaluator.submission_runtime().RAW_MACRO_F1_THRESHOLD] * 6,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
    assert detector == pytest.approx(0.5)
    assert classifier == pytest.approx((0.9 + 0.5 + 0.4 + 0.3 + 0.2) / 5)
    assert fusion == pytest.approx(0.75 * detector + 0.25 * classifier)
    assert 0 <= fusion <= 1
    with pytest.raises(RuntimeError, match="coverage differs"):
        evaluator.probability_triplet([0.1], [0.2, 0.3])


def test_current_submission_hu_context_matches_historical_oof_helper():
    images = [np.array([[0, 400], [800, 1600]], dtype=np.float32) + 10 * index
              for index in range(4)]
    kwargs = dict(level=800.0, width=1600.0, mode="2.5d", boundary_mode="repeat",
                  monochrome1=[False, True, False, False],
                  physical_positions=[0.0, 2.5, 5.0, 7.5], context_distance_mm=5.0)
    current = evaluator.submission_runtime().make_hu_input(images, 1, **kwargs)
    historical = historical_hu_input(images, 1, **kwargs)
    assert np.array_equal(current, historical)


def test_binary_metric_confusion_layout():
    truth = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.6, 0.2, 0.9, 0.4, 0.8])
    result = evaluator.binary_metrics(truth, scores)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (2, 1, 1, 2)
    assert result["confusion_matrix_actual_rows_predicted_columns"] == [[2, 1], [1, 2]]
    assert result["sensitivity"] == result["specificity"] == pytest.approx(2 / 3)
    assert result["balanced_accuracy"] == pytest.approx(2 / 3)


def test_cache_signature_and_slice_coverage_rejected(tmp_path):
    path = tmp_path / "one.json"
    item = {"study_id": "s1", "patient_id": "p1", "series_uid": "uid"}
    payload = {**item, "signature": {"imgsz": 768}, "sop_uids": ["a", "b"],
               "slice_count": 2,
               "detector_probability": 0.3, "classifier_probability": 0.4,
               "fusion_probability": 0.325}
    path.write_text(json.dumps(payload))
    assert evaluator.cache_matches(path, item, {"imgsz": 768}, ["a", "b"]) == payload
    with pytest.raises(RuntimeError, match="provenance/coverage"):
        evaluator.cache_matches(path, item, {"imgsz": 1024}, ["a", "b"])
    with pytest.raises(RuntimeError, match="provenance/coverage"):
        evaluator.cache_matches(path, item, {"imgsz": 768}, ["a"])


def test_synthetic_final_outputs_are_explicitly_apparent(tmp_path):
    plan = [{"study_id": str(index), "patient_id": str(index % 155)} for index in range(169)]
    metadata = pd.DataFrame({
        "study_id": [item["study_id"] for item in plan],
        "patient_id": [item["patient_id"] for item in plan],
        "SkullFracture": [1 if index < 24 else 0 for index in range(169)],
    })
    predictions = [{"study_id": item["study_id"], "patient_id": item["patient_id"],
                    "slice_count": 1, "detector_probability": 0.7 if index < 20 else 0.1,
                    "classifier_probability": 0.7 if index < 22 else 0.1,
                    "fusion_probability": 0.7 if index < 21 else 0.1}
                   for index, item in enumerate(plan)]
    report = evaluator.finalize_predictions(predictions, plan, metadata,
                                            {"synthetic": True}, tmp_path / "evaluation")
    assert report["label"] == "apparent_training_cohort_performance"
    assert report["cohort"] == {"studies": 169, "patients": 155,
                                "positive": 24, "negative": 145}
    assert (tmp_path / "evaluation/report.json").is_file()
    assert (tmp_path / "evaluation/study_predictions.csv").is_file()
    assert (tmp_path / "evaluation/confusion_matrix.csv").is_file()
    assert (tmp_path / "evaluation/confusion_matrix.png").is_file()
    assert report["unbiased_5fold_oof_reference"]["tp"] == 8
    with pytest.raises(FileExistsError):
        evaluator.finalize_predictions(predictions, plan, metadata,
                                       {"synthetic": True}, tmp_path / "evaluation")
