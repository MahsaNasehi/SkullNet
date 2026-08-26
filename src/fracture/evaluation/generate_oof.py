"""Merge held-out fold predictions and fit OOF-only study components."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import joblib, numpy as np
from fracture.inference.aggregation import aggregation_features
from fracture.inference.calibration import fit_calibrator
from fracture.evaluation.series_metrics import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--fold-csv", action="append", required=True); parser.add_argument("--output-dir", default="reports"); parser.add_argument("--models-dir", default="models"); parser.add_argument("--calibration", choices=["none", "platt", "isotonic"], default="platt"); args = parser.parse_args()
    rows = []
    for filename in args.fold_csv:
        with Path(filename).open(newline="", encoding="utf-8") as stream: rows.extend(csv.DictReader(stream))
    required = {"series_id", "y_true", "fold", "slice_scores"}
    if not rows or not required <= set(rows[0]): raise ValueError(f"Each fold CSV needs columns {sorted(required)}")
    feature_rows = []
    for row in rows:
        scores = [float(x) for x in row["slice_scores"].split(";") if x]
        feature_rows.append(row | aggregation_features(scores))
    names = list(aggregation_features([0.0])); x = np.asarray([[float(r[n]) for n in names] for r in feature_rows]); y = np.asarray([int(r["y_true"]) for r in feature_rows])
    from sklearn.linear_model import LogisticRegression
    aggregator = LogisticRegression(max_iter=2000).fit(x, y); raw = aggregator.predict_proba(x)[:, 1]
    # Calibration here consumes merged held-out predictions; for strict nesting, supply
    # separately cross-fitted aggregator probabilities on small datasets.
    calibrator = fit_calibrator(raw.tolist(), y.tolist(), args.calibration); calibrated = [calibrator.predict(float(p)) for p in raw]
    for row, p, c in zip(feature_rows, raw, calibrated, strict=True): row["raw_probability"], row["calibrated_probability"] = float(p), float(c)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / "oof_fracture_predictions.csv").open("w", newline="", encoding="utf-8") as stream: writer = csv.DictWriter(stream, fieldnames=list(feature_rows[0])); writer.writeheader(); writer.writerows(feature_rows)
    models = Path(args.models_dir); models.mkdir(parents=True, exist_ok=True); joblib.dump(aggregator, models / "aggregator.joblib")
    if args.calibration != "none": joblib.dump(calibrator.model, models / "calibrator.joblib")
    metrics = evaluate(y.tolist(), calibrated); (output / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
if __name__ == "__main__": main()

