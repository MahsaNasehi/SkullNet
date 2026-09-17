"""Evaluate detector proposal recall and study metrics on patient-level OOF folds.

The evaluator is shared by resolution/HNM ablations. Each model predicts only
its held-out fold. Fixed test data are explicitly rejected and never read.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from ultralytics import YOLO

from audit_fracture_test_errors import box_iou, yolo_labels_to_xyxy
from select_oof_fracture_threshold import cross_fitted_predictions, decision_metrics, select_threshold
from train_yolo26s_p2 import sha256


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data_prepared/skull_hu800_ww1600"
DEFAULT_WEIGHTS = [
    ROOT / f"outputs/yolo26s_p2_hu800_ww1600_fold{fold}/weights/best.pt" for fold in range(5)
]
DEFAULT_VAL_LISTS = [DATASET / f"kfold/fold_{fold}_val.txt" for fold in range(5)]
DEFAULT_OUTPUT = ROOT / "outputs/oof_proposal_recall_run_a_768"
QUANTILES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]


def box_size_category(box: np.ndarray) -> str:
    """COCO area bins in original-image pixels; invariant to inference imgsz."""
    width = max(0.0, float(box[2] - box[0]))
    height = max(0.0, float(box[3] - box[1]))
    area = width * height
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def geometry_diagnostics(gt: np.ndarray, proposal: np.ndarray | None) -> dict[str, object]:
    """Describe the best-IoU proposal without changing primary IoU recall."""
    gx1, gy1, gx2, gy2 = map(float, gt)
    gw, gh = max(gx2 - gx1, 0.0), max(gy2 - gy1, 0.0)
    if proposal is None:
        return {
            "best_proposal_area_over_gt_area": None,
            "intersection_over_gt_area": 0.0,
            "intersection_over_proposal_area": None,
            "proposal_center_inside_gt": False,
            "gt_center_inside_proposal": False,
            "normalized_center_distance_by_gt_diagonal": None,
            "proposal_width_over_gt_width": None,
            "proposal_height_over_gt_height": None,
        }
    px1, py1, px2, py2 = map(float, proposal)
    pw, ph = max(px2 - px1, 0.0), max(py2 - py1, 0.0)
    intersection = max(0.0, min(gx2, px2) - max(gx1, px1)) * max(0.0, min(gy2, py2) - max(gy1, py1))
    gt_area, proposal_area = gw * gh, pw * ph
    gcx, gcy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    gt_diagonal = float(np.hypot(gw, gh))
    return {
        "best_proposal_area_over_gt_area": proposal_area / gt_area if gt_area else None,
        "intersection_over_gt_area": intersection / gt_area if gt_area else None,
        "intersection_over_proposal_area": intersection / proposal_area if proposal_area else None,
        "proposal_center_inside_gt": bool(gx1 <= pcx <= gx2 and gy1 <= pcy <= gy2),
        "gt_center_inside_proposal": bool(px1 <= gcx <= px2 and py1 <= gcy <= py2),
        "normalized_center_distance_by_gt_diagonal": (
            float(np.hypot(pcx - gcx, pcy - gcy) / gt_diagonal) if gt_diagonal else None
        ),
        "proposal_width_over_gt_width": pw / gw if gw else None,
        "proposal_height_over_gt_height": ph / gh if gh else None,
    }


def confidence_distribution(values: pd.Series) -> dict[str, object]:
    array = values.to_numpy(dtype=float)
    if not len(array):
        return {"count": 0}
    return {
        "count": len(array), "mean": float(array.mean()), "std": float(array.std()),
        "quantiles": {str(q): float(np.quantile(array, q)) for q in QUANTILES},
    }


def proposal_metrics(gt: pd.DataFrame) -> dict[str, object]:
    def summarize(frame: pd.DataFrame) -> dict[str, object]:
        total = len(frame)
        recalled30 = frame["best_iou"] >= 0.3
        recalled50 = frame["best_iou"] >= 0.5
        return {
            "gt_boxes": total,
            "gt_on_slices_with_zero_proposals": int((frame["proposals"] == 0).sum()),
            "gt_without_iou30_proposal": int((~recalled30).sum()),
            "gt_without_iou50_proposal": int((~recalled50).sum()),
            "proposal_recall_iou30": float(recalled30.mean()) if total else None,
            "proposal_recall_iou50": float(recalled50.mean()) if total else None,
            "best_iou_distribution": confidence_distribution(frame["best_iou"]),
            "confidence_of_localization_best_proposal": confidence_distribution(
                frame["best_iou_proposal_confidence"]
            ),
            "max_confidence_among_iou30_matches": confidence_distribution(
                frame.loc[recalled30, "max_confidence_iou30"]
            ),
            "max_confidence_among_iou50_matches": confidence_distribution(
                frame.loc[recalled50, "max_confidence_iou50"]
            ),
        }

    return {
        "overall": summarize(gt),
        "by_original_pixel_area_coco_bins": {
            category: summarize(gt[gt["size_category"] == category])
            for category in ("small", "medium", "large")
        },
    }


def geometry_summary(frame: pd.DataFrame) -> dict[str, object]:
    with_proposal = frame[frame["proposals"] > 0]
    columns = [
        "best_proposal_area_over_gt_area", "intersection_over_gt_area",
        "intersection_over_proposal_area", "normalized_center_distance_by_gt_diagonal",
        "proposal_width_over_gt_width", "proposal_height_over_gt_height",
        "max_proposal_confidence",
    ]
    return {
        "gt_boxes": len(frame), "gt_with_any_slice_proposal": len(with_proposal),
        "proposal_center_inside_gt_rate": (
            float(with_proposal["proposal_center_inside_gt"].mean()) if len(with_proposal) else None
        ),
        "gt_center_inside_proposal_rate": (
            float(with_proposal["gt_center_inside_proposal"].mean()) if len(with_proposal) else None
        ),
        "distributions_for_best_iou_proposal": {
            column: confidence_distribution(with_proposal[column].dropna()) for column in columns
        },
    }


def validate_oof_lists(manifest: pd.DataFrame, lists: list[Path]) -> list[list[str]]:
    manifest = manifest.copy()
    manifest["resolved_image_path"] = manifest["image_path"].map(lambda x: str(Path(x).resolve()))
    development = manifest[manifest["split"] != "test"]
    test_paths = set(manifest.loc[manifest["split"] == "test", "resolved_image_path"])
    fold_images = []
    all_images = []
    for path in lists:
        if not path.is_file():
            raise FileNotFoundError(path)
        images = [str(Path(line).resolve()) for line in path.read_text().splitlines() if line.strip()]
        if not images or len(images) != len(set(images)):
            raise ValueError(f"Empty or duplicated OOF list: {path}")
        if set(images) & test_paths:
            raise ValueError(f"Fixed-test image found in OOF list: {path}")
        fold_images.append(images)
        all_images.extend(images)
    if len(all_images) != len(set(all_images)):
        raise ValueError("An image occurs in more than one held-out fold")
    if set(all_images) != set(development["resolved_image_path"]):
        raise ValueError("OOF lists do not exactly cover development images")
    image_fold = {image: fold for fold, images in enumerate(fold_images) for image in images}
    check = development[["patient_id", "resolved_image_path"]].copy()
    check["fold"] = check["resolved_image_path"].map(image_fold)
    if check.groupby("patient_id")["fold"].nunique().max() != 1:
        raise ValueError("Patient leakage across held-out folds")
    return fold_images


def cuda_device(device_argument: str) -> int | None:
    if not torch.cuda.is_available() or str(device_argument).lower() == "cpu":
        return None
    first = str(device_argument).split(",")[0]
    return int(first)


def run(args: argparse.Namespace) -> dict[str, object]:
    report_path = args.output / "metrics.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite OOF evaluation: {report_path}")
    if len(args.weights) != 5 or len(args.val_lists) != 5:
        raise ValueError("Exactly five weights and five held-out lists are required")
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str, "study_id": str, "sop_uid": str})
    fold_images = validate_oof_lists(manifest, args.val_lists)
    manifest["resolved_image_path"] = manifest["image_path"].map(lambda x: str(Path(x).resolve()))
    row_by_image = manifest.set_index("resolved_image_path")

    slice_rows: list[dict] = []
    gt_rows: list[dict] = []
    runtime_rows: list[dict] = []
    device = cuda_device(args.device)
    if device is not None:
        # This Torch build requires an initialized CUDA context before peak
        # counters can be reset reliably in a fresh process.
        torch.cuda.set_device(device)
        torch.cuda.current_device()
    evaluation_start = time.perf_counter()
    for fold, (weight, images) in enumerate(zip(args.weights, fold_images)):
        if not weight.is_file():
            raise FileNotFoundError(weight)
        if device is not None:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        load_start = time.perf_counter()
        model = YOLO(str(weight))
        model_load_seconds = time.perf_counter() - load_start
        fold_start = time.perf_counter()
        predict_seconds = 0.0
        for start in range(0, len(images), args.batch):
            paths = images[start:start + args.batch]
            if device is not None:
                torch.cuda.synchronize(device)
            predict_start = time.perf_counter()
            results = model.predict(paths, imgsz=args.imgsz, device=args.device, batch=args.batch,
                                    conf=args.proposal_conf, iou=args.nms_iou,
                                    max_det=args.max_det, stream=False, verbose=False)
            if device is not None:
                torch.cuda.synchronize(device)
            predict_seconds += time.perf_counter() - predict_start
            if len(results) != len(paths):
                raise RuntimeError(f"Prediction count mismatch in fold {fold}")
            for image_path, result in zip(paths, results):
                row = row_by_image.loc[image_path]
                image = cv2.imread(image_path)
                if image is None:
                    raise OSError(f"Could not read {image_path}")
                height, width = image.shape[:2]
                gt = yolo_labels_to_xyxy(Path(row["label_path"]), width, height)
                predicted = result.boxes.xyxy.detach().cpu().numpy().astype(float)
                confidence = result.boxes.conf.detach().cpu().numpy().astype(float)
                ious = box_iou(gt, predicted)
                slice_base = {
                    "fold": fold, "patient_id": row["patient_id"], "study_id": row["study_id"],
                    "sop_uid": row["sop_uid"], "image_path": image_path, "gt_boxes": len(gt),
                    "proposals": len(predicted),
                    "max_proposal_confidence": float(confidence.max()) if len(confidence) else 0.0,
                }
                slice_rows.append(slice_base)
                for gt_index, box in enumerate(gt):
                    if len(predicted):
                        best_index = int(np.argmax(ious[gt_index]))
                        best_iou = float(ious[gt_index, best_index])
                        best_iou_confidence = float(confidence[best_index])
                        geometry = geometry_diagnostics(box, predicted[best_index])
                        matched30 = confidence[ious[gt_index] >= 0.3]
                        matched50 = confidence[ious[gt_index] >= 0.5]
                    else:
                        best_iou, best_iou_confidence = 0.0, 0.0
                        geometry = geometry_diagnostics(box, None)
                        matched30 = matched50 = np.array([], dtype=float)
                    gt_rows.append({
                        **slice_base, "gt_box_index": gt_index,
                        "gt_x1": float(box[0]), "gt_y1": float(box[1]),
                        "gt_x2": float(box[2]), "gt_y2": float(box[3]),
                        "gt_width_px": float(box[2] - box[0]), "gt_height_px": float(box[3] - box[1]),
                        "gt_area_px": float((box[2] - box[0]) * (box[3] - box[1])),
                        "size_category": box_size_category(box), "best_iou": best_iou,
                        "best_iou_proposal_confidence": best_iou_confidence,
                        "max_confidence_iou30": float(matched30.max()) if len(matched30) else 0.0,
                        "max_confidence_iou50": float(matched50.max()) if len(matched50) else 0.0,
                        **geometry,
                    })
        if device is not None:
            torch.cuda.synchronize(device)
        fold_seconds = time.perf_counter() - fold_start
        runtime_rows.append({
            "fold": fold, "images": len(images), "model_load_seconds": model_load_seconds,
            "model_predict_wall_seconds": predict_seconds,
            "fold_evaluation_wall_seconds": fold_seconds,
            "model_predict_images_per_second": len(images) / predict_seconds,
            "cuda_peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30 if device is not None else None
            ),
            "cuda_peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / 2**30 if device is not None else None
            ),
        })
        print(f"Fold {fold}: {len(images)} images, predict={predict_seconds:.1f}s, "
              f"evaluation={fold_seconds:.1f}s", flush=True)
        del model

    slices = pd.DataFrame(slice_rows)
    gt = pd.DataFrame(gt_rows)
    runtimes = pd.DataFrame(runtime_rows)
    studies = slices.groupby(["patient_id", "study_id", "fold"], as_index=False).agg(
        fracture_true=("gt_boxes", lambda x: bool((x > 0).any())),
        slices=("sop_uid", "size"), positive_slices=("gt_boxes", lambda x: int((x > 0).sum())),
        max_score=("max_proposal_confidence", "max"),
        top3_mean=("max_proposal_confidence", lambda x: float(np.sort(x.to_numpy())[-min(3, len(x)):].mean())),
    )
    study_gt = gt.groupby("study_id").agg(
        gt_boxes=("gt_box_index", "size"),
        proposal_recall_iou30=("best_iou", lambda x: float((x >= 0.3).mean())),
        proposal_recall_iou50=("best_iou", lambda x: float((x >= 0.5).mean())),
    ).reset_index()
    studies = studies.merge(study_gt, on="study_id", how="left", validate="one_to_one")
    studies[["gt_boxes", "proposal_recall_iou30", "proposal_recall_iou50"]] = studies[
        ["gt_boxes", "proposal_recall_iou30", "proposal_recall_iou50"]
    ].fillna(0)

    y = studies["fracture_true"].to_numpy(bool)
    scores = studies["top3_mean"].to_numpy(float)
    fixed_metrics = decision_metrics(y, scores >= 0.5, scores)
    fixed_metrics["threshold"] = 0.5
    selected_threshold, apparent_metrics, _ = select_threshold(y, scores)
    crossfit, crossfit_thresholds = cross_fitted_predictions(studies)
    crossfit_metrics = decision_metrics(
        crossfit["fracture_true"].to_numpy(bool), crossfit["crossfit_predicted"].to_numpy(bool),
        crossfit["top3_mean"].to_numpy(float)
    )
    macro_positive = studies[studies["fracture_true"]]
    report = {
        "run_name": args.run_name,
        "scope": "development-only patient-level OOF; fixed test rejected and unread",
        "protocol": {
            "imgsz": args.imgsz, "proposal_conf": args.proposal_conf, "nms_iou": args.nms_iou,
            "max_det": args.max_det, "batch": args.batch,
            "proposal_recall_definition": "GT covered by any post-NMS proposal at the stated IoU",
            "size_bins_original_pixels": {"small": "area < 32^2", "medium": "32^2 <= area < 96^2",
                                                  "large": "area >= 96^2"},
            "input_rasterization": (
                "HU-windowed 2.5D PNG is stored at original 512x512; YOLO resizes/letterboxes it "
                "directly to imgsz. No stored 768 raster and no DICOM->768->1024 chain."
            ),
        },
        "weights": [{"fold": fold, "path": str(path.resolve()), "sha256": sha256(path)}
                    for fold, path in enumerate(args.weights)],
        "heldout_lists": [{"fold": fold, "path": str(path.resolve()), "sha256": sha256(path)}
                          for fold, path in enumerate(args.val_lists)],
        "images": len(slices), "studies": len(studies), "patients": int(studies["patient_id"].nunique()),
        "positive_studies": int(y.sum()), "gt_boxes": len(gt),
        "proposal_metrics": proposal_metrics(gt),
        "secondary_geometry_diagnostics": {
            "all_gt": geometry_summary(gt),
            "large_gt": geometry_summary(gt[gt["size_category"] == "large"]),
            "warning": "Secondary annotation-extent diagnostic; never replaces primary IoU proposal recall.",
        },
        "macro_positive_study_proposal_recall": {
            "iou30": float(macro_positive["proposal_recall_iou30"].mean()),
            "iou50": float(macro_positive["proposal_recall_iou50"].mean()),
        },
        "study_ranking": {"pr_auc": float(average_precision_score(y, scores)),
                          "roc_auc": float(roc_auc_score(y, scores))},
        "study_metrics_at_fixed_0_5": fixed_metrics,
        "all_oof_selected_threshold_apparent_only": {
            "threshold": selected_threshold, "metrics": apparent_metrics,
            "warning": "Threshold selected and evaluated on the same pooled OOF; optimistic, not unbiased performance.",
        },
        "cross_fitted_study_metrics": crossfit_metrics,
        "cross_fitted_thresholds": crossfit_thresholds,
        "runtime_per_fold": runtime_rows,
        "runtime_summary": {
            "total_evaluator_wall_seconds": time.perf_counter() - evaluation_start,
            "total_model_predict_wall_seconds": float(runtimes["model_predict_wall_seconds"].sum()),
            "aggregate_model_predict_images_per_second": float(
                runtimes["images"].sum() / runtimes["model_predict_wall_seconds"].sum()
            ),
            "max_cuda_peak_allocated_gib": (
                float(runtimes["cuda_peak_allocated_gib"].max()) if device is not None else None
            ),
            "max_cuda_peak_reserved_gib": (
                float(runtimes["cuda_peak_reserved_gib"].max()) if device is not None else None
            ),
            "note": "Observed one-model held-out inference; a sequential five-model ensemble is approximately 5x compute.",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    slices.to_csv(args.output / "slice_predictions.csv", index=False)
    gt.to_csv(args.output / "gt_proposal_matches.csv", index=False)
    studies.to_csv(args.output / "study_predictions.csv", index=False)
    crossfit.to_csv(args.output / "study_crossfit_predictions.csv", index=False)
    runtimes.to_csv(args.output / "runtime_by_fold.csv", index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="A_768_no_hnm")
    parser.add_argument("--weights", nargs=5, type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--val-lists", nargs=5, type=Path, default=DEFAULT_VAL_LISTS)
    parser.add_argument("--manifest", type=Path, default=DATASET / "manifest.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--proposal-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
