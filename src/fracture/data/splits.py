"""Leakage-safe study/patient folds."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def validate_no_leakage(train: list[dict[str, Any]], val: list[dict[str, Any]], patient_key: str | None = None) -> None:
    train_series, val_series = {str(x["series_id"]) for x in train}, {str(x["series_id"]) for x in val}
    overlap = train_series & val_series
    if overlap: raise ValueError(f"Series leakage detected: {sorted(overlap)[:10]}")
    if patient_key:
        a = {str(x[patient_key]) for x in train if x.get(patient_key) is not None}
        b = {str(x[patient_key]) for x in val if x.get(patient_key) is not None}
        if a & b: raise ValueError(f"Patient leakage detected: {sorted(a & b)[:10]}")


def make_folds(rows: list[dict[str, Any]], n_splits: int = 5, seed: int = 42, label_key: str = "label", patient_key: str | None = None) -> dict[str, Any]:
    if n_splits < 2: raise ValueError("n_splits must be at least 2")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows: groups.setdefault(str(row.get(patient_key) if patient_key and row.get(patient_key) is not None else row["series_id"]), []).append(row)
    if len(groups) < n_splits: raise ValueError(f"Only {len(groups)} groups for {n_splits} folds")
    import random
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = {}
    for group, members in groups.items(): buckets.setdefault(str(members[0].get(label_key, "unknown")), []).append(group)
    assignment: dict[str, int] = {}; sizes = [0] * n_splits
    for label in sorted(buckets):
        items = buckets[label]; rng.shuffle(items)
        for group in items:
            fold = min(range(n_splits), key=lambda i: sizes[i]); assignment[group] = fold; sizes[fold] += len(groups[group])
    result = {"seed": seed, "num_folds": n_splits, "folds": []}
    for fold in range(n_splits):
        val = [r for g, members in groups.items() if assignment[g] == fold for r in members]
        train = [r for g, members in groups.items() if assignment[g] != fold for r in members]
        validate_no_leakage(train, val, patient_key)
        result["folds"].append({"fold": fold, "train_series": sorted({r["series_id"] for r in train}), "val_series": sorted({r["series_id"] for r in val})})
    return result


def save_folds(folds: dict[str, Any], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream: json.dump(folds, stream, indent=2)

