"""Select leakage-safe hard negatives from held-out slice predictions."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def select_hard_negatives(
    rows: list[dict[str, str]],
    *,
    confidence: float = 0.10,
    per_study_limit: int = 10,
    include_positive_studies: bool = False,
) -> list[dict[str, str]]:
    if rows and not {"slice_status", "fold", "prediction_protocol"} <= set(rows[0]):
        raise ValueError(
            "Prediction CSV lacks the held-out slice protocol fields. Regenerate it with "
            "fracture.evaluation.predict_fold; unknown/missing-JSON slices must never be mined as negatives."
        )
    eligible: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source in rows:
        if str(source.get("prediction_protocol", "")) != "heldout_patient_fold":
            raise ValueError("Hard negatives must come from held-out patient-fold predictions")
        if str(source.get("slice_status", "")) != "negative":
            continue
        if not include_positive_studies and int(float(source.get("study_y_true", 0) or 0)) != 0:
            continue
        score = float(source.get("max_confidence", 0) or 0)
        if score < confidence:
            continue
        row = dict(source)
        row["hard_negative_reason"] = "OOF false positive on metadata-verified negative slice"
        row["hard_negative_confidence_threshold"] = f"{confidence:.8g}"
        eligible[str(row.get("series_id", ""))].append(row)

    selected: list[dict[str, str]] = []
    for series_id in sorted(eligible):
        ranked = sorted(
            eligible[series_id],
            key=lambda row: float(row.get("max_confidence", 0) or 0),
            reverse=True,
        )
        selected.extend(ranked[:per_study_limit] if per_study_limit > 0 else ranked)

    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in selected:
        key = (str(row.get("series_id", "")), str(row.get("sop_uid", "")))
        previous = unique.get(key)
        if previous is None or float(row["max_confidence"]) > float(previous["max_confidence"]):
            unique[key] = row
    return sorted(
        unique.values(),
        key=lambda row: (
            -float(row.get("max_confidence", 0) or 0),
            str(row.get("series_id", "")),
            str(row.get("sop_uid", "")),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="append", required=True)
    parser.add_argument("--output", default="reports/hard_negatives_oof.csv")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--per-study-limit", type=int, default=10)
    parser.add_argument("--include-positive-studies", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for prediction_file in args.predictions:
        with Path(prediction_file).open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    selected = select_hard_negatives(
        rows,
        confidence=args.confidence,
        per_study_limit=args.per_study_limit,
        include_positive_studies=args.include_positive_studies,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(selected[0]) if selected else [
        "series_id", "sop_uid", "fold", "study_y_true", "slice_status",
        "max_confidence", "hard_negative_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    print(f"Selected {len(selected)} OOF hard negatives -> {path}")


if __name__ == "__main__":
    main()
