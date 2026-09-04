from pathlib import Path

import numpy as np
import pytest

from fracture.data.prepare_yolo import _choose_rendered_variants, _jittered_window, _sample_training_rows, _slice_status
from fracture.data.reannotation import candidates_from_rows
from fracture.data.reannotation_audit import audit_corrections
from fracture.data.windows import context_indices, make_hu_input, window_bounds
from fracture.evaluation.generate_oof import _cross_fitted_calibration, _nested_oof_probabilities, _validate_oof_rows
from fracture.inference.aggregation import aggregation_features, longest_run_mm
from fracture.models.detector import Detector
from fracture.models.study_mil import StudyMIL, pool_slice_logits
from fracture.training.hard_negative_mining import select_hard_negatives
from fracture.training.train_study_mil import StudyDataset


def test_window_contract_and_physical_25d_context():
    assert window_bounds(800, 1600) == (0.0, 1600.0)
    positions = [0.0, 1.0, 4.0, 5.0, 10.0]
    assert context_indices(2, 5, physical_positions=positions, context_distance_mm=5.0) == (0, 2, 4)
    hu = [np.full((2, 2), value, dtype=np.float32) for value in (0, 100, 200, 300, 400)]
    image = make_hu_input(
        hu,
        2,
        level=200,
        width=400,
        mode="2.5d",
        physical_positions=positions,
        context_distance_mm=5.0,
    )
    assert image[0, 0].tolist() == [0, 128, 255]


def test_window_jitter_is_deterministic_and_variants_replace_only_duplicates():
    prep = {
        "window_level": 800,
        "window_width": 1600,
        "training_window_jitter": {"level_delta": 150, "width_fraction": 0.2, "minimum_width": 1000},
    }
    assert _jittered_window(prep, 42, "slice", 1) == _jittered_window(prep, 42, "slice", 1)
    source = {"series_id": "s", "sop_uid": "u", "image_path": "fixed.png", "window_variant_paths": "a.png;b.png"}
    selected = _choose_rendered_variants([source, source, source, source])
    assert [row["image_path"] for row in selected] == ["fixed.png", "a.png", "b.png", "a.png"]


def test_partition_sampling_keeps_priority_negatives_and_jitters_duplicates():
    rows = [
        {"series_id": "s", "sop_uid": "p", "slice_index": 2, "is_positive": True, "is_hard_negative": False, "image_path": "p.png", "window_variant_paths": "p_j.png"},
        {"series_id": "s", "sop_uid": "a", "slice_index": 1, "is_positive": False, "is_hard_negative": False, "image_path": "a.png", "window_variant_paths": "a_j.png"},
        {"series_id": "n", "sop_uid": "h", "slice_index": 0, "is_positive": False, "is_hard_negative": True, "image_path": "h.png", "window_variant_paths": "h1.png;h2.png"},
        {"series_id": "n", "sop_uid": "r", "slice_index": 1, "is_positive": False, "is_hard_negative": False, "image_path": "r.png", "window_variant_paths": ""},
    ]
    config = {"training": {"negative_downsample_ratio": 1, "positive_oversample_factor": 2, "adjacent_positive_oversample_factor": 2, "hard_negative_oversample_factor": 3}}
    sampled = _sample_training_rows(rows, config, 42)
    paths = [row["image_path"] for row in sampled]
    assert {"p.png", "p_j.png", "a.png", "a_j.png", "h.png", "h1.png", "h2.png"} <= set(paths)


def test_unknown_slice_is_not_silently_converted_to_detector_negative():
    assert _slice_status(has_boxes=False, has_annotation=False, metadata_label=None, missing_json_means_negative=False) == "unknown"
    assert _slice_status(has_boxes=False, has_annotation=False, metadata_label=False, missing_json_means_negative=False) == "negative"
    assert _slice_status(has_boxes=True, has_annotation=True, metadata_label=True, missing_json_means_negative=False) == "positive"


def test_physical_run_features_are_spacing_aware():
    scores = [0.2, 0.3, 0.0, 0.8]
    positions = [0.0, 2.5, 5.0, 10.0]
    assert longest_run_mm(scores, positions, 0.1) == pytest.approx(2.5)
    features = aggregation_features(scores, physical_positions=positions)
    assert features["median_spacing_mm"] == pytest.approx(2.5)
    assert features["longest_run_mm_ge_0_1"] == pytest.approx(2.5)


def test_hard_negative_mining_excludes_unknown_and_positive_studies():
    rows = [
        {"series_id": "neg", "sop_uid": "a", "slice_status": "negative", "study_y_true": "0", "fold": "0", "prediction_protocol": "heldout_patient_fold", "max_confidence": "0.9"},
        {"series_id": "neg", "sop_uid": "b", "slice_status": "unknown", "study_y_true": "0", "fold": "0", "prediction_protocol": "heldout_patient_fold", "max_confidence": "0.99"},
        {"series_id": "pos", "sop_uid": "c", "slice_status": "negative", "study_y_true": "1", "fold": "0", "prediction_protocol": "heldout_patient_fold", "max_confidence": "0.95"},
    ]
    selected = select_hard_negatives(rows, confidence=0.1)
    assert [(row["series_id"], row["sop_uid"]) for row in selected] == [("neg", "a")]


def test_reannotation_queue_covers_missing_missed_and_neighbor_gap():
    common = {"series_id": "s", "fold": "0", "prediction_protocol": "heldout_patient_fold", "study_y_true": "1", "image_height": "512", "image_width": "512"}
    rows = [
        common | {"sop_uid": "a", "slice_index": "0", "slice_status": "positive", "gt_num_boxes": "1", "gt_boxes": "[[10,10,20,20]]", "boxes": "[]", "max_confidence": "0.01"},
        common | {"sop_uid": "b", "slice_index": "1", "slice_status": "unknown", "gt_num_boxes": "0", "gt_boxes": "[]", "boxes": "[[10,10,20,20]]", "max_confidence": "0.8"},
        common | {"sop_uid": "c", "slice_index": "2", "slice_status": "positive", "gt_num_boxes": "1", "gt_boxes": "[[10,10,20,20]]", "boxes": "[[10,10,20,20]]", "max_confidence": "0.9"},
    ]
    kinds = {row["candidate_type"] for row in candidates_from_rows(rows)}
    assert {"missed_gt", "possible_missing_box", "neighbor_annotation_gap"} <= kinds


def test_correction_audit_requires_accepted_provenance(tmp_path: Path):
    original = tmp_path / "original" / "s"
    corrected = tmp_path / "corrected" / "s"
    original.mkdir(parents=True)
    corrected.mkdir(parents=True)
    (original / "u.json").write_text('{"boxes_xywh": [[1, 2, 3, 4]]}')
    (corrected / "u.json").write_text('{"boxes_xywh": [[1, 2, 5, 4]]}')
    config = {"data": {"annotation_root": str(original.parent), "corrected_annotation_root": str(corrected.parent), "annotation_version": "v2"}}
    log = tmp_path / "log.csv"
    header = "series_id,sop_uid,slice_index,original_num_boxes,corrected_num_boxes,modification_type,reason,reviewer,timestamp,status,notes\n"
    log.write_text(header + "s,u,0,1,1,box_resized,review,r,2026-01-01,accepted,\n")
    summary = audit_corrections(config, log)
    assert summary["accepted_log_rows"] == 1
    assert summary["modification_type_counts"] == {"box_resized": 1}
    log.write_text(header + "s,u,0,1,1,box_resized,review,r,2026-01-01,pending,\n")
    with pytest.raises(ValueError, match="without an accepted"):
        audit_corrections(config, log)


def test_oof_protocol_rejects_duplicate_study_and_crossfit_is_bounded():
    with pytest.raises(ValueError, match="duplicate"):
        _validate_oof_rows(
            [{"series_id": "s", "fold": "0"}, {"series_id": "s", "fold": "1"}],
            {0, 1},
        )
    scores = np.asarray([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])
    y = np.asarray([0, 0, 1, 1, 0, 1])
    folds = np.asarray([0, 0, 0, 0, 1, 1])
    calibrated = _cross_fitted_calibration(scores, y, folds, "platt")
    assert calibrated.shape == scores.shape
    assert np.all((0 <= calibrated) & (calibrated <= 1))
    x = np.asarray([[0.1], [0.2], [0.3], [0.9]])
    fallback = _nested_oof_probabilities(
        x,
        np.asarray([0, 0, 0, 1]),
        np.zeros(4, dtype=int),
        ("max_confidence",),
    )
    assert np.array_equal(fallback, x[:, 0])


def test_study_mil_shapes_pooling_and_checkpoint_payload():
    import torch

    logits = torch.tensor([[0.1, 0.9, 0.5]])
    mask = torch.tensor([[True, True, False]])
    assert pool_slice_logits(logits, mask, method="topk", top_k=2).item() == pytest.approx(0.5)
    model = StudyMIL(embedding_dim=16, encoder_chunk_size=2).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 3, 32, 32), torch.ones(1, 3, dtype=torch.bool))
    assert output["study_logits"].shape == (1,)
    assert output["slice_logits"].shape == (1, 3)
    payload = model.checkpoint_payload({"window_level": 800}, {"fold": 0})
    assert payload["training_metadata"]["fold"] == 0
    assert payload["external_pretrained_weights"] is False


def test_study_dataset_preserves_unknown_slice_but_masks_auxiliary_loss(tmp_path: Path):
    import cv2

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    known, unknown = tmp_path / "known.png", tmp_path / "unknown.png"
    assert cv2.imwrite(str(known), image)
    assert cv2.imwrite(str(unknown), image)
    rows = {
        "s": [
            {"image_path": str(known), "window_variant_paths": "", "slice_index": "0", "is_positive": "True", "slice_status": "positive"},
            {"image_path": str(unknown), "window_variant_paths": "", "slice_index": "1", "is_positive": "False", "slice_status": "unknown"},
        ]
    }
    _, images, labels, loss_mask, study_label = StudyDataset(
        ["s"], rows, {"s": 1}, image_size=8, training=False
    )[0]
    assert images.shape[0] == 2
    assert labels.tolist() == [1.0, 0.0]
    assert loss_mask.tolist() == [True, False]
    assert study_label.item() == 1.0


def test_detector_rejects_empty_checkpoint_before_importing_ultralytics(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="empty, truncated"):
        Detector(checkpoint)
