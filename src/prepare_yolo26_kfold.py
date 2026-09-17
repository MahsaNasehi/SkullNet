"""Create five balanced, patient-disjoint folds while preserving the fixed test set."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data_prepared" / "skull_hu800_ww1600"


def _patient_table(pool: pd.DataFrame) -> list[dict[str, object]]:
    table = (
        pool.groupby("patient_id", as_index=False)
        .agg(
            studies=("study_id", "nunique"),
            slices=("num_boxes", "size"),
            positive_slices=("num_boxes", lambda values: int((values > 0).sum())),
            boxes=("num_boxes", "sum"),
        )
    )
    table["positive"] = table["positive_slices"] > 0
    return table.to_dict("records")


def _balanced_patient_folds(patients: list[dict[str, object]], folds: int, seed: int) -> list[list[dict[str, object]]]:
    """Greedily balance positive patients, positive slices/boxes, and total slices.

    Every patient is an indivisible group. Balancing box load as well as the
    binary label is useful here because one positive patient can contribute one
    annotated slice while another contributes dozens.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    positive = [row for row in patients if bool(row["positive"])]
    negative = [row for row in patients if not bool(row["positive"])]
    if len(positive) < folds:
        raise ValueError(f"Only {len(positive)} positive patients for {folds} folds")
    rng = random.Random(seed)
    assignments: list[list[dict[str, object]]] = [[] for _ in range(folds)]

    positive_caps = [len(positive) // folds + int(i < len(positive) % folds) for i in range(folds)]
    rng.shuffle(positive_caps)
    rng.shuffle(positive)
    positive.sort(
        key=lambda row: (int(row["boxes"]), int(row["positive_slices"]), int(row["slices"])),
        reverse=True,
    )
    for row in positive:
        eligible = [
            i for i in range(folds)
            if sum(bool(item["positive"]) for item in assignments[i]) < positive_caps[i]
        ]
        target = min(
            eligible,
            key=lambda i: (
                sum(int(item["boxes"]) for item in assignments[i]),
                sum(int(item["positive_slices"]) for item in assignments[i]),
                len(assignments[i]),
            ),
        )
        assignments[target].append(row)

    patient_caps = [len(patients) // folds + int(i < len(patients) % folds) for i in range(folds)]
    rng.shuffle(negative)
    negative.sort(key=lambda row: int(row["slices"]), reverse=True)
    for row in negative:
        eligible = [i for i in range(folds) if len(assignments[i]) < patient_caps[i]]
        target = min(
            eligible,
            key=lambda i: (sum(int(item["slices"]) for item in assignments[i]), len(assignments[i])),
        )
        assignments[target].append(row)
    return assignments


def build(dataset_root: Path, folds: int, seed: int) -> Path:
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str, "study_id": str, "sop_uid": str})
    required = {"split", "patient_id", "study_id", "image_path", "num_boxes"}
    if not required <= set(manifest.columns):
        raise ValueError(f"Manifest is missing: {sorted(required - set(manifest.columns))}")

    # The original test patients remain completely untouched. Only the original
    # train+val pool participates in cross-validation.
    pool = manifest[manifest["split"].isin(["train", "val"])].copy()
    test = manifest[manifest["split"] == "test"].copy()
    pool_patients = set(pool["patient_id"])
    test_patients = set(test["patient_id"])
    if pool_patients & test_patients:
        raise RuntimeError("Patient leakage exists between trainval pool and fixed test")

    assignments = _balanced_patient_folds(_patient_table(pool), folds, seed)
    output = dataset_root / "kfold"
    output.mkdir(parents=True, exist_ok=True)
    patient_fold: dict[str, int] = {
        str(row["patient_id"]): fold
        for fold, rows in enumerate(assignments)
        for row in rows
    }
    if set(patient_fold) != pool_patients:
        raise RuntimeError("K-fold assignment does not cover every trainval patient exactly once")

    test_list = output / "fixed_test.txt"
    test_list.write_text("\n".join(test["image_path"].astype(str)) + "\n", encoding="utf-8")
    audit: dict[str, object] = {
        "protocol": "fixed patient-held-out test plus balanced 5-fold patient-grouped cross-validation",
        "seed": seed,
        "folds": folds,
        "fixed_test": {
            "patients": len(test_patients),
            "studies": int(test["study_id"].nunique()),
            "slices": len(test),
            "positive_slices": int((test["num_boxes"] > 0).sum()),
            "boxes": int(test["num_boxes"].sum()),
        },
        "validation_folds": [],
    }
    assignment_rows: list[dict[str, object]] = []
    for patient in sorted(test_patients):
        assignment_rows.append({"patient_id": patient, "partition": "fixed_test", "validation_fold": -1})

    for fold in range(folds):
        val_patients = {patient for patient, assigned in patient_fold.items() if assigned == fold}
        train_patients = pool_patients - val_patients
        if train_patients & val_patients or train_patients & test_patients or val_patients & test_patients:
            raise RuntimeError(f"Patient leakage detected in fold {fold}")
        train_rows = pool[pool["patient_id"].isin(train_patients)]
        val_rows = pool[pool["patient_id"].isin(val_patients)]
        train_txt = output / f"fold_{fold}_train.txt"
        val_txt = output / f"fold_{fold}_val.txt"
        train_txt.write_text("\n".join(train_rows["image_path"].astype(str)) + "\n", encoding="utf-8")
        val_txt.write_text("\n".join(val_rows["image_path"].astype(str)) + "\n", encoding="utf-8")
        yaml_path = output / f"fold_{fold}.yaml"
        yaml_path.write_text(
            f"path: {dataset_root}\n"
            f"train: {train_txt}\nval: {val_txt}\ntest: {test_list}\n"
            "names:\n  0: skull_fracture\n",
            encoding="utf-8",
        )
        fold_stats = {
            "fold": fold,
            "train_patients": len(train_patients),
            "validation_patients": len(val_patients),
            "validation_positive_patients": sum(
                bool(row["positive"]) for row in assignments[fold]
            ),
            "train_slices": len(train_rows),
            "validation_slices": len(val_rows),
            "validation_positive_slices": int((val_rows["num_boxes"] > 0).sum()),
            "validation_boxes": int(val_rows["num_boxes"].sum()),
            "patient_overlap": 0,
        }
        audit["validation_folds"].append(fold_stats)
        for patient in sorted(val_patients):
            assignment_rows.append(
                {"patient_id": patient, "partition": "trainval", "validation_fold": fold}
            )

    with (output / "patient_fold_assignments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["patient_id", "partition", "validation_fold"])
        writer.writeheader()
        writer.writerows(assignment_rows)
    (output / "kfold_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(build(args.dataset_root, args.folds, args.seed))


if __name__ == "__main__":
    main()
