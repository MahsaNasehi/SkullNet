"""Evaluate fracture-only binary QWK once on the fixed patient-held-out test.

Primary prediction is fixed in advance: average the five K-fold models' maximum
box confidence on each slice, take the mean of the top three slice scores per
study, then threshold at 0.5. ICH and MLS are never read by this evaluator.
Per-model results are diagnostics only and must not be used to select a model
after seeing this test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data_prepared/skull_hu800_ww1600"
DEFAULT_WEIGHTS = [
    ROOT / f"outputs/yolo26s_p2_hu800_ww1600_fold{fold}/weights/best.pt"
    for fold in range(5)
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def top3_mean(values: pd.Series) -> float:
    scores = np.sort(values.to_numpy(dtype=float))[::-1]
    return float(scores[: min(3, len(scores))].mean())


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[False, True]).ravel()
    return {
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "accuracy": float((tp + tn) / len(y_true)),
        "binary_qwk": float(cohen_kappa_score(
            y_true.astype(int), predicted.astype(int), labels=[0, 1], weights="quadratic"
        )),
        "study_pr_auc": float(average_precision_score(y_true, scores)),
        "study_roc_auc": float(roc_auc_score(y_true, scores)),
    }


def patient_bootstrap_qwk(table: pd.DataFrame, score_column: str, threshold: float,
                          repeats: int, seed: int) -> dict[str, object]:
    """Stratified patient-cluster bootstrap; preserves all studies per sampled patient."""
    patient_truth = table.groupby("patient_id")["fracture_true"].max()
    positive = patient_truth[patient_truth].index.to_numpy()
    negative = patient_truth[~patient_truth].index.to_numpy()
    if not len(positive) or not len(negative):
        return {"repeats": 0, "reason": "both positive and negative patients are required"}
    groups = {patient: group for patient, group in table.groupby("patient_id", sort=False)}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = np.r_[rng.choice(positive, len(positive), replace=True),
                        rng.choice(negative, len(negative), replace=True)]
        parts = [groups[patient] for patient in sampled]
        frame = pd.concat(parts, ignore_index=True)
        y = frame["fracture_true"].to_numpy(dtype=bool)
        p = frame[score_column].to_numpy(dtype=float) >= threshold
        value = cohen_kappa_score(y.astype(int), p.astype(int), labels=[0, 1], weights="quadratic")
        if np.isfinite(value):
            values.append(float(value))
    return {
        "method": "stratified patient-cluster bootstrap",
        "seed": seed, "requested_repeats": repeats, "valid_repeats": len(values),
        "qwk_percentile_95_ci": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
    }


def validate_fixed_test(manifest: pd.DataFrame, images: list[str]) -> pd.DataFrame:
    if not images or len(images) != len(set(images)):
        raise ValueError("Fixed test list is empty or contains duplicate images")
    manifest = manifest.copy()
    manifest["image_path"] = manifest["image_path"].map(lambda p: str(Path(p).resolve()))
    test = manifest[manifest["image_path"].isin(images)].copy()
    if len(test) != len(images) or set(test["image_path"]) != set(images):
        raise ValueError("Fixed test images do not match the manifest")
    if set(test["split"]) != {"test"}:
        raise ValueError("Fixed test list contains non-test manifest rows")
    development_patients = set(manifest.loc[manifest["split"] != "test", "patient_id"])
    test_patients = set(test["patient_id"])
    if development_patients & test_patients:
        raise ValueError("Patient leakage between development and fixed test")
    test["fracture_slice_true"] = test["num_boxes"].astype(int) > 0

    # Independent consistency guard: annotations and metadata must give the same study label.
    annotation_truth = test.groupby("study_id")["fracture_slice_true"].max()
    metadata = pd.read_pickle(ROOT / "iaaa-contest-bct/Data/training_df.pkl").copy()
    metadata["study_id"] = metadata["dicom_series.id"].astype(str)
    metadata_truth = metadata[metadata["study_id"].isin(annotation_truth.index)].groupby("study_id")["SkullFracture"].max().astype(bool)
    metadata_truth = metadata_truth.reindex(annotation_truth.index)
    if metadata_truth.isna().any() or not np.array_equal(annotation_truth.to_numpy(), metadata_truth.to_numpy()):
        raise ValueError("Fracture ground truth disagrees between box annotations and metadata")
    return test


def predict_slice_scores(weights: Path, images: list[str], args: argparse.Namespace) -> np.ndarray:
    model = YOLO(str(weights))
    scores: list[float] = []
    for start in range(0, len(images), args.batch):
        paths = images[start : start + args.batch]
        results = model.predict(source=paths, imgsz=args.imgsz, device=args.device,
                                conf=args.proposal_conf, iou=0.7, stream=False, verbose=False)
        if len(results) != len(paths):
            raise RuntimeError(f"Prediction count mismatch for {weights}")
        for result in results:
            confidence = result.boxes.conf.detach().cpu().numpy()
            scores.append(float(confidence.max()) if confidence.size else 0.0)
    del model
    return np.asarray(scores, dtype=float)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if (args.output / "metrics.json").exists():
        raise FileExistsError(
            f"Fixed-test result already exists: {args.output / 'metrics.json'}. "
            "Refusing to overwrite or repeatedly inspect the held-out test."
        )
    if len(args.weights) != 5 or len(set(map(str, args.weights))) != 5:
        raise ValueError("Primary protocol requires exactly five distinct K-fold weights")
    if args.threshold != 0.5:
        raise ValueError("Fixed test protocol uses the predeclared threshold 0.5")
    for weight in args.weights:
        if not weight.is_file():
            raise FileNotFoundError(weight)
    test_list = DATASET / "kfold/fixed_test.txt"
    images = [str(Path(line.strip()).resolve()) for line in test_list.read_text().splitlines() if line.strip()]
    manifest = pd.read_csv(DATASET / "manifest.csv", dtype={"patient_id": str, "study_id": str})
    test = validate_fixed_test(manifest, images)
    row_by_image = test.set_index("image_path")

    slices = pd.DataFrame({
        "image_path": images,
        "patient_id": [row_by_image.loc[p, "patient_id"] for p in images],
        "study_id": [row_by_image.loc[p, "study_id"] for p in images],
        "fracture_slice_true": [bool(row_by_image.loc[p, "fracture_slice_true"]) for p in images],
    })
    weight_audit = []
    model_columns = []
    for index, weight in enumerate(args.weights):
        column = f"fold{index}_slice_score"
        print(f"[{index + 1}/5] Predicting fixed test with {weight}", flush=True)
        slices[column] = predict_slice_scores(weight, images, args)
        model_columns.append(column)
        weight_audit.append({"fold": index, "path": str(weight.resolve()), "sha256": sha256(weight)})

    # Primary ensemble: probability averaging on the same slice, then top-3 study pooling.
    slices["ensemble_mean_slice_score"] = slices[model_columns].mean(axis=1)
    studies = slices.groupby(["patient_id", "study_id"], as_index=False).agg(
        fracture_true=("fracture_slice_true", "max"),
        slices=("image_path", "size"),
        positive_slices=("fracture_slice_true", "sum"),
        ensemble_top3_mean=("ensemble_mean_slice_score", top3_mean),
        ensemble_max=("ensemble_mean_slice_score", "max"),
    )
    diagnostics = {}
    for index, column in enumerate(model_columns):
        per_study = slices.groupby("study_id")[column].agg(top3_mean).reindex(studies["study_id"])
        score_column = f"fold{index}_top3_mean"
        studies[score_column] = per_study.to_numpy()
        diagnostics[f"fold{index}"] = binary_metrics(
            studies["fracture_true"].to_numpy(dtype=bool), studies[score_column].to_numpy(), args.threshold
        )
    y_true = studies["fracture_true"].to_numpy(dtype=bool)
    primary_scores = studies["ensemble_top3_mean"].to_numpy(dtype=float)
    primary = binary_metrics(y_true, primary_scores, args.threshold)
    studies["fracture_predicted"] = primary_scores >= args.threshold

    args.output.mkdir(parents=True, exist_ok=True)
    slices.to_csv(args.output / "slice_predictions.csv", index=False)
    studies.to_csv(args.output / "study_predictions.csv", index=False)
    report = {
        "metric_scope": "fracture-only binary study classification; ICH and MLS are not used",
        "evaluation_partition": "fixed patient-held-out test used once after model/protocol selection",
        "primary_protocol": "mean of five fold slice scores -> study top3_mean -> fixed threshold 0.5",
        "test_list": str(test_list.resolve()),
        "test_list_sha256": sha256(test_list),
        "images": len(slices), "patients": int(studies["patient_id"].nunique()),
        "studies": len(studies), "positive_patients": int(studies.groupby("patient_id")["fracture_true"].max().sum()),
        "positive_studies": int(y_true.sum()), "weights": weight_audit,
        "proposal_confidence_floor": args.proposal_conf,
        "primary_ensemble_metrics": primary,
        "primary_qwk_interpretation": "For two classes, quadratic weighted kappa equals ordinary Cohen kappa.",
        "patient_bootstrap": patient_bootstrap_qwk(studies, "ensemble_top3_mean", args.threshold,
                                                    args.bootstrap_repeats, args.seed),
        "per_fold_diagnostics_do_not_select_on_test": diagnostics,
        "test_selection_warning": "Do not choose a fold, threshold, aggregator, or new model from these test diagnostics.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs=5, type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fixed_test_fracture_kfold_ensemble")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--proposal-conf", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
