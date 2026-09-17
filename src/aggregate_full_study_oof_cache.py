"""CPU-only cross-fitted selection from immutable full-study slice caches."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fracture_macro_f1_oof import (
    INTERMEDIATE_COLUMNS,
    binary_report,
    monotonic_operating_point_map,
    oracle_triage_report,
    select_operating_point,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_AGGREGATORS = (
    "max", "top2_mean", "top3_mean", "top5_mean", "top10_percent_mean", "consecutive"
)


def apply_higher_confidence_floor(slice_payload: Dict[str, Any], new_floor: float) -> Dict[str, Any]:
    """Filter cached post-NMS boxes by confidence; never changes/reconstructs NMS."""
    source_floor = float(slice_payload.get("source_confidence_floor", 0.0))
    if new_floor < source_floor:
        raise ValueError("A permissive cache cannot reconstruct scores below its source confidence floor")
    scores = [float(value) for value in slice_payload.get("scores", [])]
    boxes = list(slice_payload.get("boxes", []))
    if len(scores) != len(boxes):
        raise ValueError("Cached boxes/scores length mismatch")
    retained = [(box, score) for box, score in zip(boxes, scores) if score >= new_floor]
    return {
        **slice_payload,
        "boxes": [item[0] for item in retained],
        "scores": [item[1] for item in retained],
        "max_confidence": max((item[1] for item in retained), default=0.0),
        "num_detections": len(retained),
        "applied_confidence_floor": float(new_floor),
        "nms_reconstructed": False,
    }


def aggregate_scores(scores: Sequence[float], method: str) -> float:
    values = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    if not len(values):
        return 0.0
    ordered = np.sort(values)[::-1]
    if method == "max":
        return float(ordered[0])
    if method in {"top2_mean", "top3_mean", "top5_mean"}:
        k = int(method[3:method.index("_")])
        return float(ordered[: min(k, len(ordered))].mean())
    if method == "top10_percent_mean":
        k = max(1, int(math.ceil(0.10 * len(ordered))))
        return float(ordered[:k].mean())
    if method == "consecutive":
        if len(values) == 1:
            return float(0.5 * values[0])
        # Strongest adjacent two-slice mean. An isolated spike is diluted by
        # its neighbour, while physically consecutive evidence is retained.
        return float(max((values[index] + values[index + 1]) / 2.0 for index in range(len(values) - 1)))
    raise ValueError("Unsupported aggregator %r; frozen set=%s" % (method, CANDIDATE_AGGREGATORS))


def load_cache(cache_dir: Path) -> pd.DataFrame:
    rows = []  # type: List[Dict[str, Any]]
    for path in sorted(cache_dir.glob("fold_*/studies/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"fold", "series_id", "patient_id", "prediction_protocol", "slices"}
        if required - set(payload):
            raise ValueError("Incomplete cache %s: %s" % (path, sorted(required - set(payload))))
        if payload["prediction_protocol"] != "heldout_patient_full_study":
            raise ValueError("Non-OOF cache protocol in %s" % path)
        slices = payload["slices"]
        scores = [float(item["max_confidence"]) for item in slices]
        row = {
            "fold": int(payload["fold"]),
            "study_id": str(payload["series_id"]),
            "patient_id": str(payload["patient_id"]),
            "included_target_series_slices": len(slices),
            "included_metadata_unknown_slices": int(payload.get("included_metadata_unknown_slices", 0)),
            "score_generation_protocol": str(payload.get("score_generation_protocol", "")),
            "study_inclusion_protocol": str(payload.get("study_inclusion_protocol", "")),
            "inference_signature_json": json.dumps(payload.get("inference_signature", {}), sort_keys=True),
        }  # type: Dict[str, Any]
        for method in CANDIDATE_AGGREGATORS:
            row[method] = aggregate_scores(scores, method)
        rows.append(row)
    if not rows:
        raise ValueError("No study cache files found under %s" % cache_dir)
    table = pd.DataFrame(rows)
    if table["study_id"].duplicated().any():
        raise ValueError("A study occurs in more than one cache/fold")
    if int(table.groupby("patient_id")["fold"].nunique().max()) != 1:
        raise ValueError("Patient leakage across cached folds")
    if table["score_generation_protocol"].nunique() != 1:
        raise ValueError("Mixed score-generation protocols in one cache")
    if table["study_inclusion_protocol"].nunique() != 1:
        raise ValueError("Mixed study-inclusion protocols in one cache")
    return table


def attach_truth(cache: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_pickle(metadata_path)
    series_col = "dicom_series.id"
    metadata[series_col] = metadata[series_col].astype(str)
    metadata["patient_id"] = metadata["dicom_series.PatientID"].astype(str)
    sx = pd.to_numeric(metadata["dicom_series.PixelSpacing0"], errors="raise")
    sy = pd.to_numeric(metadata["dicom_series.PixelSpacing1"], errors="raise")
    thickness = pd.to_numeric(metadata["dicom_series.SliceThickness"], errors="coerce").fillna(1.0)
    factor = sx * sy * thickness.replace(0, 1.0) / 1000.0
    areas = {
        "V_EDH": "EpiduralHemorrhage_Area",
        "V_SDH": "SubduralHemorrhage_Area",
        "V_IPH": "IntraparenchymalHemorrhage_Area",
        "V_SAH": "SubarachnoidHemorrhage_Area",
        "V_IVH": "IntraventricularHemorrhage_Area",
    }
    for target, source in areas.items():
        metadata[target] = pd.to_numeric(metadata[source], errors="raise") * factor
    aggregation = {name: "sum" for name in areas}
    aggregation.update({"MidlineShiftMM": "max", "SkullFracture": "max", "patient_id": "first"})
    truth = metadata.groupby(series_col, as_index=False).agg(aggregation).rename(
        columns={series_col: "study_id", "MidlineShiftMM": "MLS_mm", "SkullFracture": "fracture_true"}
    )
    if set(cache["study_id"]) - set(truth["study_id"]):
        raise ValueError("Cached studies missing from metadata")
    merged = cache.merge(truth, on="study_id", suffixes=("_cache", "_metadata"), validate="one_to_one")
    if (merged["patient_id_cache"].astype(str) != merged["patient_id_metadata"].astype(str)).any():
        raise ValueError("Cache/metadata patient mismatch")
    merged["patient_id"] = merged.pop("patient_id_cache")
    merged = merged.drop(columns=["patient_id_metadata"])
    merged["fracture_true"] = merged["fracture_true"].astype(bool)
    return merged


def select_aggregator_and_threshold(train: pd.DataFrame) -> Tuple[str, float, Dict[str, Any]]:
    candidates = []
    truth = train["fracture_true"].astype(bool).to_numpy()
    for method in CANDIDATE_AGGREGATORS:
        threshold, detail = select_operating_point(train, method)
        mapped = monotonic_operating_point_map(train[method].to_numpy(dtype=float), threshold)
        binary = binary_report(truth, mapped)
        pr_auc = float(average_precision_score(truth, train[method].to_numpy(dtype=float)))
        candidates.append({
            "aggregator": method,
            "raw_operating_point": threshold,
            "oracle_other_heads_macro_f1": detail["selection_oracle_metrics"]["pooled_macro_f1"],
            "specificity": binary["specificity"],
            "sensitivity": binary["sensitivity"],
            "pr_auc": pr_auc,
            "fp": binary["fp"],
        })
    ranked = sorted(
        candidates,
        key=lambda row: (
            -row["oracle_other_heads_macro_f1"],
            row["fp"],
            -(row["specificity"] if row["specificity"] is not None else -1.0),
            -(row["sensitivity"] if row["sensitivity"] is not None else -1.0),
            -row["pr_auc"],
            CANDIDATE_AGGREGATORS.index(row["aggregator"]),
        ),
    )
    selected = ranked[0]
    return str(selected["aggregator"]), float(selected["raw_operating_point"]), {
        "selection_primary": "oracle_other_heads_macro_f1",
        "guardrail_order": ["lower_fp", "specificity", "sensitivity", "pr_auc"],
        "candidate_table": candidates,
    }


def crossfit_cache_table(table: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    observed = sorted(table["fold"].unique().tolist())
    if observed != [0, 1, 2, 3, 4]:
        raise ValueError("Complete five-fold cache required; observed=%s" % observed)
    parts = []
    selections = []
    for fold in observed:
        train = table[table["fold"] != fold]
        heldout = table[table["fold"] == fold].copy()
        method, threshold, detail = select_aggregator_and_threshold(train)
        heldout["selected_aggregator"] = method
        heldout["selected_raw_operating_point"] = threshold
        heldout["crossfit_fracture_prob"] = monotonic_operating_point_map(
            heldout[method].to_numpy(dtype=float), threshold
        )
        parts.append(heldout)
        selections.append({
            "heldout_fold": int(fold), "selected_aggregator": method,
            "selected_raw_operating_point": threshold, **detail,
        })
    crossfit = pd.concat(parts, ignore_index=True).sort_values(["fold", "study_id"])
    probabilities = crossfit["crossfit_fracture_prob"].to_numpy(dtype=float)
    report = {
        "evaluation_partition": "full-study held-out-patient OOF only",
        "score_generation_protocol": str(table["score_generation_protocol"].iloc[0]),
        "study_inclusion_protocol": str(table["study_inclusion_protocol"].iloc[0]),
        "nms_reconstruction_policy": "Confidence floors may only be raised within one cache; different NMS requires separate inference/cache.",
        "candidate_aggregators_frozen": list(CANDIDATE_AGGREGATORS),
        "primary_selection_metric": "oracle_other_heads_macro_f1",
        "official_downstream_fracture_threshold": 0.5,
        "cross_fitted": {
            **oracle_triage_report(crossfit, probabilities),
            "binary_guardrails": binary_report(crossfit["fracture_true"].astype(bool), probabilities),
        },
        "fold_selections": selections,
    }
    return crossfit, report


def run(args: argparse.Namespace) -> Dict[str, Any]:
    report_path = args.output / "report.json"
    if report_path.exists():
        raise FileExistsError("Refusing to overwrite %s" % report_path)
    table = attach_truth(load_cache(args.cache_dir), args.metadata)
    crossfit, report = crossfit_cache_table(table)
    args.output.mkdir(parents=True, exist_ok=True)
    crossfit.to_csv(args.output / "cross_fitted_predictions.csv", index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "iaaa-contest-bct/Data/training_df.pkl")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
