"""Evaluate one fold at study level and compute official triage QWK.

The detector only predicts fracture. Therefore the official-rule QWK reported
here is *isolated fracture QWK*: predicted fracture is combined with ground-
truth ICH volumes and ground-truth MLS. This is not a full submission QWK.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from ultralytics import YOLO

from evaluation.triage import triage_from_intermediates


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data_prepared/skull_hu800_ww1600"


def _qwk(y_true: list[int], y_pred: list[int], labels: list[int]) -> float:
    return float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic"))


def _predict_triage(row, score_column: str, threshold: float) -> int:
    # The organizer still receives a fracture bit/probability with cutoff 0.5.
    # A custom raw-score operating point must be applied BEFORE that rule.
    present = float(getattr(row, score_column) >= threshold)
    return triage_from_intermediates(
        row.V_EDH, row.V_SDH, row.V_IPH, row.V_SAH, row.V_IVH, present, row.MLS_mm
    )


def _study_ground_truth(metadata_path: Path, studies: set[str]) -> pd.DataFrame:
    frame = pd.read_pickle(metadata_path).copy()
    frame["study_id"] = frame["dicom_series.id"].astype(str)
    frame = frame[frame["study_id"].isin(studies)].copy()
    if set(frame["study_id"]) != studies:
        raise ValueError(f"Metadata missing studies: {sorted(studies - set(frame['study_id']))}")
    sx = pd.to_numeric(frame["dicom_series.PixelSpacing0"], errors="raise")
    sy = pd.to_numeric(frame["dicom_series.PixelSpacing1"], errors="raise")
    thickness = pd.to_numeric(frame["dicom_series.SliceThickness"], errors="coerce").fillna(1.0)
    factor = sx * sy * thickness.replace(0, 1.0) / 1000.0
    area_columns = {
        "V_EDH": "EpiduralHemorrhage_Area",
        "V_SDH": "SubduralHemorrhage_Area",
        "V_IPH": "IntraparenchymalHemorrhage_Area",
        "V_SAH": "SubarachnoidHemorrhage_Area",
        "V_IVH": "IntraventricularHemorrhage_Area",
    }
    for output, source in area_columns.items():
        frame[output] = pd.to_numeric(frame[source], errors="raise") * factor
    aggregation = {name: "sum" for name in area_columns}
    aggregation.update({"MidlineShiftMM": "max", "SkullFracture": "max", "triage_class": "max"})
    result = frame.groupby("study_id", as_index=False).agg(aggregation)
    result = result.rename(columns={"MidlineShiftMM": "MLS_mm", "SkullFracture": "fracture_true"})
    result["fracture_true"] = result["fracture_true"].astype(bool)

    # Guard that our port of the organizer rule reproduces provided labels.
    derived = [
        triage_from_intermediates(
            row.V_EDH, row.V_SDH, row.V_IPH, row.V_SAH, row.V_IVH,
            float(row.fracture_true), row.MLS_mm,
        )
        for row in result.itertuples(index=False)
    ]
    mismatches = int(np.sum(np.asarray(derived) != result["triage_class"].to_numpy()))
    if mismatches:
        raise RuntimeError(f"Official triage rule disagrees with metadata for {mismatches} studies")
    return result


def _binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[False, True]).ravel()
    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "accuracy": float((tp + tn) / len(y_true)),
        "binary_qwk": _qwk(y_true.astype(int).tolist(), predicted.astype(int).tolist(), [0, 1]),
        "study_pr_auc": float(average_precision_score(y_true, scores)),
        "study_roc_auc": float(roc_auc_score(y_true, scores)),
    }


def _detector_box_metrics(run_dir: Path) -> dict[str, object]:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return {"results_csv": str(results_path), "available": False}
    results = pd.read_csv(results_path)
    results.columns = results.columns.str.strip()
    target = "metrics/mAP50-95(B)"
    best = results.loc[results[target].idxmax()]
    return {
        "results_csv": str(results_path.resolve()),
        "available": True,
        "best_epoch_by_map50_95": int(best["epoch"]),
        "precision_at_best_epoch": float(best["metrics/precision(B)"]),
        "recall_at_best_epoch": float(best["metrics/recall(B)"]),
        "map50_at_best_epoch": float(best["metrics/mAP50(B)"]),
        "map50_95_at_best_epoch": float(best[target]),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.batch < 1 or not 0 <= args.threshold <= 1 or not 0 <= args.proposal_conf <= 1:
        raise ValueError("Require positive batch size and confidence/threshold in [0, 1]")
    fold_dir = DATASET / "kfold"
    val_list = fold_dir / f"fold_{args.fold}_val.txt"
    weights = args.weights or ROOT / f"outputs/yolo26s_p2_hu800_ww1600_fold{args.fold}/weights/best.pt"
    if not val_list.is_file() or not weights.is_file():
        raise FileNotFoundError(f"Missing fold list or weights: {val_list}, {weights}")
    images = [line.strip() for line in val_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = pd.read_csv(DATASET / "manifest.csv", dtype={"patient_id": str, "study_id": str})
    image_to_study = dict(zip(manifest["image_path"].astype(str), manifest["study_id"].astype(str)))
    if any(path not in image_to_study for path in images):
        raise ValueError("Fold image list contains paths absent from manifest")

    slice_rows: list[dict[str, object]] = []
    model = YOLO(str(weights))
    prediction_count = 0
    # Ultralytics interprets one large Python list as a single batch during
    # warmup, regardless of ``batch=``. Chunk explicitly to keep VRAM bounded.
    for start in range(0, len(images), args.batch):
        batch_paths = images[start : start + args.batch]
        batch_predictions = model.predict(
            source=batch_paths, imgsz=args.imgsz, device=args.device,
            conf=args.proposal_conf, iou=0.7, stream=False, verbose=False,
        )
        if len(batch_predictions) != len(batch_paths):
            raise RuntimeError(
                f"Prediction count mismatch in batch: {len(batch_paths)} images vs "
                f"{len(batch_predictions)} outputs"
            )
        for path, prediction in zip(batch_paths, batch_predictions):
            prediction_count += 1
            confidences = (
                prediction.boxes.conf.detach().cpu().numpy()
                if prediction.boxes is not None else np.empty(0)
            )
            slice_rows.append(
                {
                    "image_path": path,
                    "study_id": image_to_study[path],
                    "slice_score": float(confidences.max()) if confidences.size else 0.0,
                    "detections": int(confidences.size),
                }
            )
    if prediction_count != len(images):
        raise RuntimeError(
            f"Prediction count mismatch: {len(images)} images vs {prediction_count} outputs"
        )
    slices = pd.DataFrame(slice_rows)

    def top3(values: pd.Series) -> float:
        scores = np.sort(values.to_numpy(dtype=float))[::-1]
        return float(scores[: min(3, len(scores))].mean())

    studies = (
        slices.groupby("study_id", as_index=False)
        .agg(max_score=("slice_score", "max"), top3_mean=("slice_score", top3), slices=("slice_score", "size"))
    )
    truth = _study_ground_truth(
        ROOT / "iaaa-contest-bct/Data/training_df.pkl", set(studies["study_id"])
    )
    table = truth.merge(studies, on="study_id", validate="one_to_one")
    score_column = args.aggregator
    scores = table[score_column].to_numpy(dtype=float)
    y_fracture = table["fracture_true"].to_numpy(dtype=bool)
    binary = _binary_metrics(y_fracture, scores, args.threshold)

    y_triage_true: list[int] = []
    y_triage_pred: list[int] = []
    y_triage_negative: list[int] = []
    for row in table.itertuples(index=False):
        common = (row.V_EDH, row.V_SDH, row.V_IPH, row.V_SAH, row.V_IVH)
        y_triage_true.append(triage_from_intermediates(*common, float(row.fracture_true), row.MLS_mm))
        y_triage_pred.append(_predict_triage(row, score_column, args.threshold))
        y_triage_negative.append(triage_from_intermediates(*common, 0.0, row.MLS_mm))

    triage_matrix = confusion_matrix(y_triage_true, y_triage_pred, labels=[0, 1, 2])
    triage_counts_true = np.bincount(y_triage_true, minlength=3)
    triage_counts_pred = np.bincount(y_triage_pred, minlength=3)

    # Exploratory only: selecting and evaluating a threshold on the same fold is optimistic.
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    best_threshold, best_qwk = args.threshold, -np.inf
    for threshold in candidates:
        predictions_at_threshold = []
        for row in table.itertuples(index=False):
            common = (row.V_EDH, row.V_SDH, row.V_IPH, row.V_SAH, row.V_IVH)
            fracture_bit = float(getattr(row, score_column) >= threshold)
            predictions_at_threshold.append(triage_from_intermediates(*common, fracture_bit, row.MLS_mm))
        value = _qwk(y_triage_true, predictions_at_threshold, [0, 1, 2])
        if value > best_qwk:
            best_threshold, best_qwk = float(threshold), value

    output = args.output or ROOT / f"outputs/yolo26s_p2_hu800_ww1600_fold{args.fold}/study_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    table["fracture_predicted"] = scores >= args.threshold
    table["triage_true"] = y_triage_true
    table["triage_predicted"] = y_triage_pred
    slices.to_csv(output / "slice_predictions.csv", index=False)
    table.to_csv(output / "study_predictions.csv", index=False)
    report: dict[str, object] = {
        "fold": args.fold,
        "weights": str(weights.resolve()),
        "evaluation_partition": "patient-held-out validation fold; fixed test was not used",
        "studies": len(table),
        "positive_studies": int(y_fracture.sum()),
        "aggregator": score_column,
        "proposal_confidence_floor": args.proposal_conf,
        "triage_raw_score_threshold": args.threshold,
        "detector_box_metrics_from_training": _detector_box_metrics(weights.parents[1]),
        "fracture_study_metrics": binary,
        "official_triage_qwk_isolated_fracture": _qwk(y_triage_true, y_triage_pred, [0, 1, 2]),
        "triage_confusion_matrix_rows_true_cols_pred_0_1_2": triage_matrix.tolist(),
        "triage_true_counts_0_1_2": triage_counts_true.tolist(),
        "triage_predicted_counts_0_1_2": triage_counts_pred.tolist(),
        "triage_errors": int(np.sum(np.asarray(y_triage_true) != np.asarray(y_triage_pred))),
        "official_triage_qwk_always_negative_fracture": _qwk(y_triage_true, y_triage_negative, [0, 1, 2]),
        "official_triage_qwk_oracle_fracture": 1.0,
        "exploratory_same_fold_best_threshold": best_threshold,
        "exploratory_same_fold_best_qwk": best_qwk,
        "qwk_scope_warning": "Uses ground-truth ICH volumes and MLS; not full-system/submission QWK.",
        "threshold_warning": "Exploratory best threshold is selected on this same validation fold and is optimistic.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--aggregator", choices=("top3_mean", "max_score"), default="top3_mean")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--proposal-conf", type=float, default=0.001)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
