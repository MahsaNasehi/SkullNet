"""Manual FULL169 fracture evaluation on its 169 TRAINING Studies (apparent only).

Preflight is CPU-only. Evaluation is manual and resumable per Study; no fixed-test
DICOM is eligible. Ground-truth fracture labels are attached only after all
prediction caches are complete. This is never an unbiased validation score.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from cache_full_study_oof import (DATASET, DEFAULT_DICOM_ROOT, DEFAULT_MANIFEST,
                                  DEFAULT_METADATA, enforce_vram_guard, metadata_index)
from dicom_series_guard import canonical_profile, discover_headers, select_and_order_target_headers
from fracture_classifier.data import FINAL_DEPLOYMENT, ROOT, sha256
from fracture_classifier.model import FractureSliceClassifier
from fracture_classifier.training import image_tensor
from oof_cohort import derive_explicit_validation_cohort


CLASSIFIER_DIR = ROOT / "outputs/run_a_fracture_classifier_full169"
CLASSIFIER = CLASSIFIER_DIR / "final_classifier.pt"
OUTPUT = CLASSIFIER_DIR / "evaluation"
OOF_REPORT = ROOT / "outputs/run_a_fracture_classifier_oof/comparison/report.json"
SUBMISSION_CODE = ROOT / "submit/model.py"
DETECTOR_SHA = "109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640"
CLASSIFIER_SHA = "ec0d34c7164453a8a80e476762080829bb4f8b219fd5fc8067380af891a50485"
EXPECTED = {"studies": 169, "patients": 155, "positive": 24, "negative": 145}
SCOPE = "apparent_training_cohort_performance"


def submission_runtime():
    """Import the actual detector deployment helpers without constructing YOLO."""
    name = "full169_fracture_submission_runtime"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SUBMISSION_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import current submission detector implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need their defining module registered
    spec.loader.exec_module(module)
    return module


def verify_final_weights() -> dict:
    detector_hash, classifier_hash = sha256(FINAL_DEPLOYMENT), sha256(CLASSIFIER)
    if (detector_hash, classifier_hash) != (DETECTOR_SHA, CLASSIFIER_SHA):
        raise RuntimeError("Final FULL169 checkpoint SHA256 mismatch")
    completion = json.loads((CLASSIFIER_DIR / "training_completed.json").read_text())
    protocol = json.loads((CLASSIFIER_DIR / "protocol.json").read_text())
    if (completion.get("status") != "completed_fixed_nine_epoch_classifier_refit"
            or completion.get("final_epoch_one_based") != 9
            or completion.get("final_classifier_sha256") != classifier_hash
            or completion.get("last_classifier_sha256") != classifier_hash
            or sha256(CLASSIFIER_DIR / "last_classifier.pt") != classifier_hash
            or completion.get("source_detector_sha256") != detector_hash
            or completion.get("training_cohort_studies") != 169
            or completion.get("training_cohort_patients") != 155
            or protocol.get("fixed_epochs") != 9
            or protocol.get("validation_used") is not False
            or protocol.get("source_detector_sha256") != detector_hash):
        raise RuntimeError("FULL169 classifier completion/protocol is not validated")
    current = submission_runtime()
    if (current.AGGREGATION_METHOD != "top10_percent_mean"
            or current.RAW_MACRO_F1_THRESHOLD != 0.39717610677083337
            or current.OFFICIAL_FRACTURE_THRESHOLD != 0.5
            or (current.WINDOW_LEVEL, current.WINDOW_WIDTH, current.IMAGE_SIZE,
                current.CONTEXT_DISTANCE_MM, current.CONFIDENCE, current.NMS_IOU) !=
               (800.0, 1600.0, 768, 5.0, 0.01, 0.5)):
        raise RuntimeError("Current submission detector protocol differs from frozen Run A deployment")
    return {"detector_sha256": detector_hash, "classifier_sha256": classifier_hash,
            "submission_model_py_sha256": sha256(SUBMISSION_CODE)}


def development_plan() -> tuple[list[dict], pd.DataFrame, dict]:
    """Select explicit development Studies; use no fracture annotation here."""
    manifest = pd.read_csv(DEFAULT_MANIFEST, dtype={
        "study_id": str, "patient_id": str, "sop_uid": str, "image_path": str})
    vals = [DATASET / f"kfold/fold_{fold}_val.txt" for fold in range(5)]
    cohort = derive_explicit_validation_cohort(manifest, vals)
    expected_patient = {}
    for fold in cohort["folds"]:
        for study, patient in fold["study_to_patient"].items():
            if study in expected_patient and expected_patient[study] != patient:
                raise RuntimeError("Study maps to multiple patients")
            expected_patient[str(study)] = str(patient)
    if len(expected_patient) != 169 or len(set(expected_patient.values())) != 155:
        raise RuntimeError("FULL169 explicit cohort size changed")
    fixed_rows = manifest[manifest["split"] == "test"]
    if (set(expected_patient) & set(fixed_rows["study_id"].astype(str))
            or set(expected_patient.values()) & set(fixed_rows["patient_id"].astype(str))):
        raise RuntimeError("Fixed-test Study/patient overlaps development evaluation")
    if cohort["per_fold_study_counts"] != [33, 33, 32, 36, 35]:
        raise RuntimeError("FULL169 Fold Study counts changed")
    metadata, _ = metadata_index(DEFAULT_METADATA)
    selected = metadata[metadata["study_id"].isin(expected_patient)].copy()
    if set(selected["study_id"]) != set(expected_patient):
        raise RuntimeError("Metadata does not cover all explicit FULL169 Studies")
    plan = []
    for study in sorted(expected_patient):
        rows = selected[selected["study_id"] == study]
        patients = set(rows["patient_id"].astype(str))
        uids = set(rows["dicom_series.SeriesInstanceUID"].astype(str))
        if patients != {expected_patient[study]} or len(uids) != 1:
            raise RuntimeError(f"Metadata patient/target series mismatch for Study {study}")
        if not (DEFAULT_DICOM_ROOT / study).is_dir():
            raise FileNotFoundError(DEFAULT_DICOM_ROOT / study)
        plan.append({"study_id": study, "patient_id": expected_patient[study],
                     "series_uid": next(iter(uids)),
                     "metadata_sops": set(rows["sop_uid"].astype(str))})
    return plan, selected, cohort


def study_truth_after_predictions(metadata: pd.DataFrame, plan: list[dict]) -> pd.DataFrame:
    """Only called after prediction caches are complete (or CPU-only preflight)."""
    values = pd.to_numeric(metadata["SkullFracture"], errors="raise")
    if values.isna().any() or not set(values.unique()) <= {0, 1}:
        raise RuntimeError("FULL169 Study ground truth is not binary and complete")
    truth = metadata.assign(_fracture=values.astype(int)).groupby("study_id", as_index=False).agg(
        ground_truth=("_fracture", "max"), patient_id=("patient_id", "first"))
    expected = {item["study_id"]: item["patient_id"] for item in plan}
    if (len(truth) != 169 or set(truth["study_id"]) != set(expected)
            or any(expected[row.study_id] != row.patient_id for row in truth.itertuples(index=False))):
        raise RuntimeError("Ground-truth Study/patient coverage differs from inference cohort")
    observed = {"studies": len(truth), "patients": truth["patient_id"].nunique(),
                "positive": int(truth["ground_truth"].sum()),
                "negative": int((truth["ground_truth"] == 0).sum())}
    if observed != EXPECTED:
        raise RuntimeError(f"Unexpected FULL169 Study truth counts: {observed}")
    return truth


def top5_mean(scores: list[float]) -> float:
    if not scores or not all(math.isfinite(x) and 0 <= x <= 1 for x in scores):
        raise ValueError("Invalid classifier slice probabilities")
    return float(np.mean(sorted(scores, reverse=True)[:5]))


def probability_triplet(detector_scores: list[float], classifier_scores: list[float]) -> tuple[float, float, float]:
    current = submission_runtime()
    if len(detector_scores) != len(classifier_scores) or not detector_scores:
        raise RuntimeError("Detector/classifier slice coverage differs")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in detector_scores):
        raise ValueError("Invalid detector slice confidences")
    detector_raw = current.aggregate_top10_percent_mean(detector_scores)
    detector = current.rescale_for_macro_f1(detector_raw, current.RAW_MACRO_F1_THRESHOLD)
    classifier = top5_mean(classifier_scores)
    fused = float(np.clip(0.75 * detector + 0.25 * classifier, 0.0, 1.0))
    return float(detector), classifier, fused


def binary_metrics(truth: np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(truth, dtype=int)
    p = np.asarray(probability, dtype=float)
    if (len(y) != len(p) or not len(y) or set(y) != {0, 1}
            or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
        raise ValueError("Binary metric inputs require two classes and finite probabilities")
    predicted = (p >= 0.5).astype(int)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": sensitivity, "recall": sensitivity,
            "specificity": specificity, "precision": precision,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy": (tp + tn) / len(y),
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "confusion_matrix_actual_rows_predicted_columns": [[tn, fp], [fn, tp]]}


def oof_reference() -> dict:
    report = json.loads(OOF_REPORT.read_text())
    metric = report["systems"]["weighted_fusion"]
    fracture = metric["fracture"]
    if ([fracture[key] for key in ("tp", "fp", "fn", "tn")]
            != [8, 4, 16, 141]
            or abs(metric["triage"]["pooled_macro_f1"] - 0.9696273781) > 1e-8):
        raise RuntimeError("Existing OOF fusion reference differs from recorded evidence")
    return {"tp": fracture["tp"], "tn": fracture["tn"], "fp": fracture["fp"],
            "fn": fracture["fn"], "sensitivity": fracture["sensitivity"],
            "specificity": fracture["specificity"], "precision": fracture["precision"],
            "f1": fracture["f1"], "pr_auc": fracture["pr_auc"],
            "roc_auc": fracture["roc_auc"],
            "confusion_matrix_actual_rows_predicted_columns": [[141, 4], [16, 8]],
            "oracle_other_heads_macro_f1": metric["triage"]["pooled_macro_f1"]}


def infer_one(plan_item: dict, included: list[dict], order: str,
              detector, classifier, device: str, batch: int) -> dict:
    current = submission_runtime()
    study = plan_item["study_id"]
    sops = [str(item["SOPInstanceUID"]) for item in included]
    if not plan_item["metadata_sops"] <= set(sops) or len(sops) != len(set(sops)):
        raise RuntimeError(f"Target-series full-slice coverage failed for Study {study}")
    records = [current.read_slice(item["filesystem_path"], study) for item in included]
    if [str(record.sop_uid) for record in records] != sops:
        raise RuntimeError("Decoded DICOM sequence differs from selected target-series SOPs")
    hu = [record.hu for record in records]
    positions = [record.physical_position for record in records]
    mono = [record.photometric_interpretation == "MONOCHROME1" for record in records]
    detector_scores, classifier_scores = [], []
    with torch.inference_mode():
        for start in range(0, len(records), batch):
            images = [current.make_hu_input(
                hu, index, level=800.0, width=1600.0, mode="2.5d",
                boundary_mode="repeat", monochrome1=mono,
                physical_positions=positions, context_distance_mm=5.0)
                for index in range(start, min(start + batch, len(records)))]
            predictions = detector.predict_batch(images)
            if len(predictions) != len(images):
                raise RuntimeError("Detector returned incomplete slice batch")
            detector_scores.extend(float(item["max_confidence"]) for item in predictions)
            tensor = torch.stack([image_tensor(image) for image in images]).to(device)
            classifier_scores.extend(float(value) for value in classifier.probabilities(tensor).cpu().tolist())
            del images, predictions, tensor
    detector_prob, classifier_prob, fusion_prob = probability_triplet(detector_scores, classifier_scores)
    return {"study_id": study, "patient_id": plan_item["patient_id"],
            "series_uid": plan_item["series_uid"], "sop_uids": sops,
            "slice_count": len(sops), "slice_ordering_method": order,
            "detector_probability": detector_prob,
            "classifier_probability": classifier_prob,
            "fusion_probability": fusion_prob}


def cache_matches(path: Path, item: dict, signature: dict, expected_sops: list[str]) -> dict | None:
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    if (cached.get("signature") != signature
            or cached.get("study_id") != item["study_id"]
            or cached.get("patient_id") != item["patient_id"]
            or cached.get("series_uid") != item["series_uid"]
            or cached.get("sop_uids") != expected_sops):
        raise RuntimeError(f"Existing Study cache provenance/coverage mismatch: {path}")
    probability_triplet_valid = [cached.get(key) for key in (
        "detector_probability", "classifier_probability", "fusion_probability")]
    if not all(isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= 1
               for value in probability_triplet_valid):
        raise RuntimeError(f"Existing Study cache has invalid probabilities: {path}")
    if cached.get("slice_count") != len(expected_sops):
        raise RuntimeError(f"Existing Study cache slice count mismatch: {path}")
    detector, classifier, fusion = probability_triplet_valid
    if abs(float(fusion) - float(np.clip(0.75 * detector + 0.25 * classifier, 0, 1))) > 1e-7:
        raise RuntimeError(f"Existing Study cache fusion equation mismatch: {path}")
    return cached


def save_json_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def save_confusion_matrix(metrics: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(metrics["confusion_matrix_actual_rows_predicted_columns"], dtype=int)
    frame = pd.DataFrame(matrix, index=["Actual Negative", "Actual Positive"],
                         columns=["Predicted Negative", "Predicted Positive"])
    frame.to_csv(output / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=16)
    ax.set_xticks([0, 1], ["Predicted Negative", "Predicted Positive"])
    ax.set_yticks([0, 1], ["Actual Negative", "Actual Positive"])
    ax.set_xlabel("Predicted fracture presence")
    ax.set_ylabel("Actual fracture presence")
    ax.set_title("FULL169 fusion — apparent training-cohort performance")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def finalize_predictions(predictions: list[dict], plan: list[dict], metadata: pd.DataFrame,
                         signature: dict, output: Path) -> dict:
    if len(predictions) != 169 or len({item["study_id"] for item in predictions}) != 169:
        raise RuntimeError("Incomplete/duplicate FULL169 predictions")
    truth = study_truth_after_predictions(metadata, plan)
    table = pd.DataFrame(predictions).merge(truth, on=["study_id", "patient_id"], validate="one_to_one")
    if len(table) != 169 or set(table["study_id"]) != {item["study_id"] for item in plan}:
        raise RuntimeError("Prediction/truth merge did not preserve exact FULL169 cohort")
    table = table.sort_values("study_id").reset_index(drop=True)
    y = table["ground_truth"].to_numpy(dtype=int)
    systems = {name: binary_metrics(y, table[column].to_numpy(dtype=float)) for name, column in (
        ("detector_only", "detector_probability"),
        ("classifier_only", "classifier_probability"),
        ("fusion", "fusion_probability"))}
    detector_pred = table["detector_probability"].to_numpy(float) >= 0.5
    fusion_pred = table["fusion_probability"].to_numpy(float) >= 0.5
    table["detector_prediction"] = detector_pred.astype(int)
    table["classifier_prediction"] = (table["classifier_probability"].to_numpy(float) >= 0.5).astype(int)
    table["fusion_prediction"] = fusion_pred.astype(int)
    report = {"label": SCOPE,
              "warning": "These 169 Studies trained both FULL169 models. Results are in-sample/apparent, NOT unbiased validation, OOF, fixed-test, or final competition Macro-F1.",
              "cohort": EXPECTED, "protocol": signature,
              "systems": systems,
              "fusion_change_vs_detector": {
                  "detector_false_negatives_rescued": int(np.sum((y == 1) & ~detector_pred & fusion_pred)),
                  "new_false_positives_introduced": int(np.sum((y == 0) & ~detector_pred & fusion_pred)),
                  "previous_false_positives_removed": int(np.sum((y == 0) & detector_pred & ~fusion_pred))},
              "unbiased_5fold_oof_reference": oof_reference()}
    for path in (output / "report.json", output / "study_predictions.csv",
                 output / "confusion_matrix.csv", output / "confusion_matrix.png"):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite completed evaluation artifact: {path}")
    output.mkdir(parents=True, exist_ok=True)
    table[["study_id", "patient_id", "ground_truth", "detector_probability",
           "classifier_probability", "fusion_probability", "detector_prediction",
           "classifier_prediction", "fusion_prediction", "slice_count"]].to_csv(
               output / "study_predictions.csv", index=False)
    save_confusion_matrix(systems["fusion"], output)
    save_json_once(output / "report.json", report)
    return report


def evaluate(device: str, batch: int, output: Path, force: bool = False) -> dict:
    if batch <= 0:
        raise ValueError("Batch size must be positive")
    if (output / "report.json").exists():
        raise FileExistsError("Completed FULL169 evaluation exists; refusing overwrite")
    hashes = verify_final_weights()
    plan, metadata, _ = development_plan()
    enforce_vram_guard(device.replace("cuda:", ""), force)
    current = submission_runtime()
    signature = {"scope": SCOPE, **hashes, "study_count": 169,
                 "detector_aggregation": "top10_percent_mean",
                 "detector_raw_operating_point": current.RAW_MACRO_F1_THRESHOLD,
                 "classifier_aggregation": "top5_mean", "alpha_detector": 0.75,
                 "alpha_classifier": 0.25, "official_fracture_threshold": 0.5,
                 "imgsz": 768, "conf": 0.01, "nms_iou": 0.5,
                 "window_level": 800.0, "window_width": 1600.0,
                 "context_distance_mm": 5.0, "batch": batch}
    detector = classifier = None
    predictions = []
    existing_cache = output / "study_cache"
    if existing_cache.exists():
        extra = {path.stem for path in existing_cache.glob("*.json")} - {
            item["study_id"] for item in plan}
        if extra:
            raise RuntimeError(f"Unexpected Study cache files outside FULL169 cohort: {sorted(extra)[:5]}")
    for index, item in enumerate(plan, 1):
        study = item["study_id"]
        headers = discover_headers(DEFAULT_DICOM_ROOT / study)
        profile = canonical_profile(headers, item["series_uid"], item["metadata_sops"])
        included, _, order = select_and_order_target_headers(headers, profile)
        expected_sops = [str(header["SOPInstanceUID"]) for header in included]
        cache_path = output / "study_cache" / f"{study}.json"
        cached = cache_matches(cache_path, item, signature, expected_sops)
        if cached is None:
            if detector is None:
                enforce_vram_guard(device.replace("cuda:", ""), force)
                detector_device = 0 if device == "cuda:0" else device
                detector = current.Detector(FINAL_DEPLOYMENT, confidence=0.01, iou=0.5,
                                            device=detector_device, fp16=device != "cpu", image_size=768)
                checkpoint = torch.load(CLASSIFIER, map_location="cpu", weights_only=False)
                if (checkpoint.get("epoch_one_based") != 9 or checkpoint.get("stage") != 1
                        or checkpoint.get("fixed_epochs") != 9
                        or checkpoint.get("source_detector_sha256") != DETECTOR_SHA
                        or checkpoint.get("slice_label_manifest_sha256") != json.loads(
                            (CLASSIFIER_DIR / "protocol.json").read_text())["slice_label_manifest_sha256"]):
                    raise RuntimeError("Final classifier checkpoint metadata mismatch")
                classifier = FractureSliceClassifier.from_detector(FINAL_DEPLOYMENT)
                classifier.load_state_dict(checkpoint["model_state"], strict=True)
                classifier.to(device).eval()
            result = infer_one(item, included, order, detector, classifier, device, batch)
            result["signature"] = signature
            save_json_once(cache_path, result)
            cached = result
        predictions.append(cached)
        print(f"FULL169 apparent evaluation {index}/169 Study={study} slices={cached['slice_count']}", flush=True)
    return finalize_predictions(predictions, plan, metadata, signature, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="CPU-only provenance/cohort audit")
    mode.add_argument("--evaluate", action="store_true", help="Manual full-study inference")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--force", action="store_true", help="Override 8-GiB free-VRAM guard")
    args = parser.parse_args()
    if args.preflight:
        hashes = verify_final_weights()
        plan, metadata, cohort = development_plan()
        truth = study_truth_after_predictions(metadata, plan)  # CPU audit, no inference
        print(json.dumps({"scope": SCOPE, "cohort": EXPECTED,
                          "per_fold_studies": cohort["per_fold_study_counts"],
                          "positive": int(truth["ground_truth"].sum()),
                          "fixed_test_study_patient_overlap": 0,
                          "checkpoints": hashes, "oof_reference": oof_reference()}, indent=2))
    else:
        report = evaluate(args.device, args.batch, args.output, args.force)
        print(json.dumps({"output": str(args.output), "scope": report["label"],
                          "systems": report["systems"]}, indent=2))


if __name__ == "__main__":
    main()
