"""Diagnose fixed-test fracture FN/FP cases at proposal, localization and pooling levels.

This is diagnostic post-hoc analysis, not model/protocol selection evidence. It
reruns only the currently misclassified studies at a low proposal confidence,
matches predictions to YOLO ground-truth boxes, and writes auditable tables and
overlays for all five fold models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data_prepared/skull_hu800_ww1600"
DEFAULT_PREDICTIONS = ROOT / "outputs/oof_fracture_threshold_v1/fixed_test_secondary_predictions.csv"
DEFAULT_WEIGHTS = [
    ROOT / f"outputs/yolo26s_p2_hu800_ww1600_fold{fold}/weights/best.pt" for fold in range(5)
]
DEFAULT_OUTPUT = ROOT / "outputs/fixed_test_error_audit_v1"


def yolo_labels_to_xyxy(path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label in {path}: {line}")
        _, xc, yc, bw, bh = map(float, fields)
        boxes.append([(xc - bw / 2) * width, (yc - bh / 2) * height,
                      (xc + bw / 2) * width, (yc + bh / 2) * height])
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=float).reshape(-1, 4)
    second = np.asarray(second, dtype=float).reshape(-1, 4)
    if not len(first) or not len(second):
        return np.zeros((len(first), len(second)), dtype=float)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area_first = np.clip(first[:, 2:] - first[:, :2], 0, None).prod(axis=1)
    area_second = np.clip(second[:, 2:] - second[:, :2], 0, None).prod(axis=1)
    union = area_first[:, None] + area_second[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def proposal_summary(gt: np.ndarray, predicted: np.ndarray, confidence: np.ndarray) -> tuple[dict, list[dict]]:
    ious = box_iou(gt, predicted)
    details = []
    for gt_index in range(len(gt)):
        if len(predicted):
            best_index = int(np.argmax(ious[gt_index]))
            best_iou = float(ious[gt_index, best_index])
            best_confidence = float(confidence[best_index])
            matched50 = confidence[ious[gt_index] >= 0.5]
            max_confidence_iou50 = float(matched50.max()) if len(matched50) else 0.0
        else:
            best_index, best_iou, best_confidence, max_confidence_iou50 = -1, 0.0, 0.0, 0.0
        details.append({
            "gt_box_index": gt_index, "best_prediction_index": best_index,
            "best_iou": best_iou, "best_prediction_confidence": best_confidence,
            "max_confidence_at_iou50": max_confidence_iou50,
        })
    best_per_gt = np.array([row["best_iou"] for row in details], dtype=float)
    matched_scores = np.array([row["max_confidence_at_iou50"] for row in details], dtype=float)
    summary = {
        "proposals": int(len(predicted)),
        "max_proposal_confidence": float(confidence.max()) if len(confidence) else 0.0,
        "max_iou_to_any_gt": float(ious.max()) if ious.size else 0.0,
        "gt_boxes_matched_iou30": int((best_per_gt >= 0.3).sum()),
        "gt_boxes_matched_iou50": int((best_per_gt >= 0.5).sum()),
        "max_matched_confidence_iou50": float(matched_scores.max()) if len(matched_scores) else 0.0,
    }
    return summary, details


def draw_overlay(image_path: Path, gt: np.ndarray, predicted: np.ndarray, confidence: np.ndarray,
                 selected_predictions: set[int], destination: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Could not read {image_path}")
    for index, (x1, y1, x2, y2) in enumerate(gt):
        cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"GT{index}", (round(x1), max(14, round(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    for index in sorted(selected_predictions):
        x1, y1, x2, y2 = predicted[index]
        cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 0, 255), 1)
        cv2.putText(image, f"P {confidence[index]:.3f}", (round(x1), min(image.shape[0] - 4, round(y2) + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not write {destination}")


def classify_fn(study_rows: pd.DataFrame, threshold: float) -> str:
    summaries = study_rows.groupby("fold").agg(
        matched30=("gt_boxes_matched_iou30", "sum"),
        matched50=("gt_boxes_matched_iou50", "sum"),
        matched_conf50=("max_matched_confidence_iou50", "max"),
        top3=("max_proposal_confidence", lambda s: float(np.sort(s.to_numpy())[-3:].mean())),
    )
    localized_folds = int((summaries["matched50"] > 0).sum())
    weak_localized_folds = int((summaries["matched30"] > 0).sum())
    confident_matched_folds = int((summaries["matched_conf50"] >= threshold).sum())
    voting_folds = int((summaries["top3"] >= threshold).sum())
    if weak_localized_folds == 0:
        return "proposal_failure_no_iou30_in_any_fold"
    if localized_folds == 0:
        return "localization_failure_iou30_exists_but_no_iou50"
    if 0 < localized_folds < 5:
        return "fold_instability_some_models_localize"
    if confident_matched_folds < 3:
        return "localized_but_confidence_too_low"
    if voting_folds < 3:
        return "study_aggregation_failure"
    return "localized_and_confident_check_vote_consistency"


def run(args: argparse.Namespace) -> dict[str, object]:
    if (args.output / "audit_report.json").exists():
        raise FileExistsError(f"Refusing to overwrite prior test audit: {args.output}")
    if len(args.weights) != 5:
        raise ValueError("Exactly five fold weights are required")
    predictions = pd.read_csv(args.predictions, dtype={"patient_id": str, "study_id": str})
    truth = predictions["fracture_true"].astype(bool)
    decision = predictions["fracture_predicted_frozen_oof_threshold"].astype(bool)
    errors = predictions.loc[truth != decision].copy()
    if errors.empty:
        raise ValueError("No errors found to audit")
    threshold_values = predictions["fold0_vote_at_frozen_threshold"].notna()  # schema guard
    if not threshold_values.all():
        raise ValueError("Incomplete frozen-threshold test predictions")

    threshold_report = json.loads((args.predictions.parent / "threshold_report.json").read_text())
    threshold = float(threshold_report["final_frozen_threshold_from_all_oof"])
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str, "study_id": str})
    selected = manifest[(manifest["split"] == "test") & manifest["study_id"].isin(errors["study_id"])].copy()
    if set(selected["study_id"]) != set(errors["study_id"]):
        raise ValueError("Misclassified studies do not match fixed-test manifest")

    slice_rows: list[dict] = []
    gt_rows: list[dict] = []
    for fold, weight in enumerate(args.weights):
        if not weight.is_file():
            raise FileNotFoundError(weight)
        model = YOLO(str(weight))
        for study_id, study in selected.groupby("study_id", sort=True):
            study = study.sort_values("sop_uid")
            paths = study["image_path"].tolist()
            print(f"Fold {fold}: audit study {study_id} ({len(paths)} slices)", flush=True)
            results = model.predict(paths, imgsz=args.imgsz, device=args.device, batch=args.batch,
                                    conf=args.proposal_conf, iou=args.nms_iou, max_det=args.max_det,
                                    verbose=False, stream=False)
            cached = []
            for (_, row), result in zip(study.iterrows(), results):
                image = cv2.imread(str(row["image_path"]))
                if image is None:
                    raise OSError(f"Could not read {row['image_path']}")
                height, width = image.shape[:2]
                gt = yolo_labels_to_xyxy(Path(row["label_path"]), width, height)
                predicted = result.boxes.xyxy.detach().cpu().numpy().astype(float)
                confidence = result.boxes.conf.detach().cpu().numpy().astype(float)
                summary, details = proposal_summary(gt, predicted, confidence)
                base = {"study_id": study_id, "patient_id": row["patient_id"], "fold": fold,
                        "sop_uid": row["sop_uid"], "image_path": row["image_path"],
                        "gt_boxes": len(gt), **summary}
                slice_rows.append(base)
                for detail in details:
                    gt_rows.append({**base, **detail})
                cached.append((row, gt, predicted, confidence, details, summary))

            # Positive errors: every GT slice. Negative error: three strongest slices per fold.
            if bool(errors.set_index("study_id").loc[study_id, "fracture_true"]):
                render_indices = [i for i, item in enumerate(cached) if len(item[1])]
            else:
                strengths = np.array([item[5]["max_proposal_confidence"] for item in cached])
                render_indices = np.argsort(strengths)[-min(3, len(strengths)):].tolist()
            for index in render_indices:
                row, gt, predicted, confidence, details, _ = cached[index]
                chosen = set(np.argsort(confidence)[-min(args.overlay_topk, len(confidence)):].tolist())
                chosen.update(d["best_prediction_index"] for d in details if d["best_prediction_index"] >= 0)
                destination = args.output / "overlays" / f"study_{study_id}" / f"fold{fold}_{row['sop_uid']}.png"
                draw_overlay(Path(row["image_path"]), gt, predicted, confidence, chosen, destination)
        del model

    slices = pd.DataFrame(slice_rows)
    gt_table = pd.DataFrame(gt_rows)
    summaries = slices.groupby(["study_id", "patient_id", "fold"], as_index=False).agg(
        slices=("sop_uid", "size"), gt_boxes=("gt_boxes", "sum"), proposals=("proposals", "sum"),
        max_proposal_confidence=("max_proposal_confidence", "max"),
        max_iou_to_any_gt=("max_iou_to_any_gt", "max"),
        gt_boxes_matched_iou30=("gt_boxes_matched_iou30", "sum"),
        gt_boxes_matched_iou50=("gt_boxes_matched_iou50", "sum"),
        max_matched_confidence_iou50=("max_matched_confidence_iou50", "max"),
    )
    top3 = slices.groupby(["study_id", "fold"])["max_proposal_confidence"].apply(
        lambda s: float(np.sort(s.to_numpy())[-min(3, len(s)):].mean())
    ).rename("recomputed_top3_mean").reset_index()
    summaries = summaries.merge(top3, on=["study_id", "fold"], validate="one_to_one")

    classifications = {}
    for study_id in errors["study_id"]:
        error = errors.set_index("study_id").loc[study_id]
        if bool(error["fracture_true"]):
            classifications[study_id] = classify_fn(slices[slices["study_id"] == study_id], threshold)
        else:
            classifications[study_id] = "false_positive_requires_visual_mimic_review"
    report = {
        "scope": "post-hoc diagnostic audit of three FN and one FP on already-inspected fixed test",
        "warning": "Do not claim performance improvement or select a final protocol on these four cases.",
        "imgsz": args.imgsz, "proposal_conf": args.proposal_conf, "nms_iou": args.nms_iou,
        "frozen_oof_threshold": threshold,
        "error_studies": errors[["study_id", "fracture_true", "positive_votes"]].to_dict("records"),
        "automatic_failure_classification": classifications,
        "definitions": {
            "proposal_or_localization": "GT matching uses post-NMS predictions at proposal_conf and IoU 0.30/0.50",
            "green_overlay": "ground-truth fracture box", "red_overlay": "model proposal with confidence",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    slices.to_csv(args.output / "slice_model_audit.csv", index=False)
    gt_table.to_csv(args.output / "gt_box_model_audit.csv", index=False)
    summaries.to_csv(args.output / "study_model_summary.csv", index=False)
    (args.output / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--manifest", type=Path, default=DATASET / "manifest.csv")
    parser.add_argument("--weights", nargs=5, type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--proposal-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--overlay-topk", type=int, default=10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
