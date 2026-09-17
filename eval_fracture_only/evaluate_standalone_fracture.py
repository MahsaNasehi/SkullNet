"""Evaluate ONLY the fracture model with standalone fracture rules.

No ICH volumes. No MLS. No call to official triage_from_intermediates.
Classes come from ``standalone_fracture_triage``:

    fracture_prob >= 0.5 -> 1 (Urgent)
    otherwise            -> 0 (Non-urgent)

Primary metrics = accuracy of *your* fracture head against SkullFracture GT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
HERE = Path(__file__).resolve().parent
for path in (SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_full_study_oof_cache import attach_truth, load_cache  # noqa: E402
from fracture_macro_f1_oof import monotonic_operating_point_map  # noqa: E402

from standalone_fracture_rules import (  # noqa: E402
    FRACTURE_PRESENCE_THRESHOLD,
    explain_rules,
    standalone_fracture_triage,
    triage_many,
)

SUBMIT_AGGREGATOR = "top10_percent_mean"
SUBMIT_RAW_OPERATING_POINT = 0.39717610677083337
DEFAULT_CACHE = ROOT / "outputs/full_study_oof_run_a_768_deployment_score_cache_v2_169"
DEFAULT_METADATA = ROOT / "iaaa-contest-bct/Data/training_df.pkl"
DEFAULT_OUTPUT = HERE / "results_standalone"


def model_only_report(
    fracture_true: Sequence[bool],
    fracture_prob: Sequence[float],
    threshold: float = FRACTURE_PRESENCE_THRESHOLD,
) -> Dict[str, Any]:
    truth_bits = np.asarray(fracture_true, dtype=bool)
    probs = np.asarray(fracture_prob, dtype=float)
    if len(truth_bits) != len(probs) or not len(truth_bits):
        raise ValueError("fracture_true / fracture_prob must be non-empty and aligned")

    y_true = triage_many(truth_bits.astype(float), threshold=threshold)
    y_pred = triage_many(probs, threshold=threshold)
    pred_bits = probs >= threshold

    labels = [0, 1]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(truth_bits, pred_bits, labels=[False, True]).ravel()

    out: Dict[str, Any] = {
        "rule": "standalone_fracture_triage",
        "threshold": float(threshold),
        "n_studies": int(len(truth_bits)),
        "n_positive": int(truth_bits.sum()),
        "n_negative": int((~truth_bits).sum()),
        # Direct model accuracy (binary fracture detection)
        "model_accuracy": float(accuracy_score(truth_bits, pred_bits)),
        "model_balanced_accuracy": float(
            ((tp / (tp + fn)) if tp + fn else 0.0) + ((tn / (tn + fp)) if tn + fp else 0.0)
        )
        / 2.0,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "positive_f1": float(f1_score(truth_bits, pred_bits, zero_division=0)),
        "macro_f1_binary": float(
            f1_score(truth_bits.astype(int), pred_bits.astype(int), average="macro", zero_division=0)
        ),
        "tp_fp_fn_tn": [int(tp), int(fp), int(fn), int(tn)],
        "binary_qwk": float(
            cohen_kappa_score(
                truth_bits.astype(int), pred_bits.astype(int), labels=[0, 1], weights="quadratic"
            )
        ),
        # Standalone triage Macro-F1 (isomorphic to binary macro-F1 here)
        "standalone_triage_macro_f1": float(np.mean(f1)),
        "standalone_triage_accuracy": float(accuracy_score(y_true, y_pred)),
        "classwise": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "gt_class_counts": {str(c): y_true.count(c) for c in labels},
        "pred_class_counts": {str(c): y_pred.count(c) for c in labels},
    }
    if len(np.unique(truth_bits)) == 2:
        out["pr_auc"] = float(average_precision_score(truth_bits, probs))
        out["roc_auc"] = float(roc_auc_score(truth_bits, probs))
    else:
        out["pr_auc"] = None
        out["roc_auc"] = None
    return out


def _short(block: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "model_accuracy",
        "macro_f1_binary",
        "standalone_triage_macro_f1",
        "sensitivity",
        "specificity",
        "precision",
        "positive_f1",
        "pr_auc",
        "roc_auc",
        "tp_fp_fn_tn",
        "binary_qwk",
    )
    return {k: block[k] for k in keys}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output)
    report_path = output / "report.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError("Refusing to overwrite %s (pass --overwrite)" % report_path)

    table = attach_truth(load_cache(Path(args.cache_dir)), Path(args.metadata))
    truth = table["fracture_true"].astype(bool).to_numpy()
    raw = table[SUBMIT_AGGREGATOR].to_numpy(dtype=float)
    mapped = monotonic_operating_point_map(raw, SUBMIT_RAW_OPERATING_POINT)

    # Cross-fit: choose raw operating point by standalone Macro-F1 on other folds
    parts = []
    fold_selections = []
    for fold in sorted(table["fold"].unique()):
        train = table[table["fold"] != fold]
        heldout = table[table["fold"] == fold].copy()
        best_thr, best_score = 0.5, -1.0
        scores = train[SUBMIT_AGGREGATOR].to_numpy(dtype=float)
        candidates = np.unique(np.r_[0.0, 0.5, (np.unique(scores)[:-1] + np.unique(scores)[1:]) / 2.0, 1.0])
        for thr in candidates:
            mapped_train = monotonic_operating_point_map(scores, float(thr))
            score = model_only_report(
                train["fracture_true"].astype(bool), mapped_train
            )["standalone_triage_macro_f1"]
            if score > best_score or (score == best_score and abs(thr - 0.5) < abs(best_thr - 0.5)):
                best_thr, best_score = float(thr), float(score)
        heldout["selected_raw_operating_point"] = best_thr
        heldout["crossfit_fracture_prob"] = monotonic_operating_point_map(
            heldout[SUBMIT_AGGREGATOR].to_numpy(dtype=float), best_thr
        )
        parts.append(heldout)
        fold_selections.append(
            {
                "heldout_fold": int(fold),
                "selected_raw_operating_point": best_thr,
                "selection_standalone_macro_f1": best_score,
                "selection_studies": int(len(train)),
            }
        )
    crossfit = pd.concat(parts, ignore_index=True).sort_values(["fold", "study_id"])

    submit_report = model_only_report(truth, mapped)
    raw_report = model_only_report(truth, raw)
    crossfit_report = model_only_report(
        crossfit["fracture_true"].astype(bool),
        crossfit["crossfit_fracture_prob"].to_numpy(dtype=float),
    )
    baselines = {
        "always_no_fracture": model_only_report(truth, np.zeros(len(table))),
        "always_fracture": model_only_report(truth, np.ones(len(table))),
    }

    report = {
        "evaluation_name": "standalone_fracture_model_only",
        "purpose": (
            "Measure only the fracture model. Rules accept fracture_prob alone; "
            "ICH and MLS are not inputs and are never used."
        ),
        "rules": explain_rules(),
        "model_source": {
            "cache_dir": str(Path(args.cache_dir).resolve()),
            "aggregator": SUBMIT_AGGREGATOR,
            "submit_raw_operating_point": SUBMIT_RAW_OPERATING_POINT,
            "submit_artifact": "submit/models/best.pt + submit/model.py mapping",
        },
        "n_studies": int(len(table)),
        "n_fracture_positive": int(truth.sum()),
        "n_fracture_negative": int((~truth).sum()),
        "primary_metrics": [
            "model_accuracy",
            "macro_f1_binary",
            "standalone_triage_macro_f1",
            "sensitivity",
            "specificity",
            "pr_auc",
            "roc_auc",
        ],
        "submit_deployment_apparent": {
            "warning": "Apparent all-OOF with packaged submit threshold; not unbiased.",
            "summary": _short(submit_report),
            "detail": submit_report,
        },
        "raw_score_at_official_0.5": {
            "summary": _short(raw_report),
            "detail": raw_report,
        },
        "unbiased_crossfit": {
            "description": (
                "Raw operating point chosen per fold by standalone Macro-F1 on "
                "other folds; aggregator fixed to submit top10_percent_mean."
            ),
            "summary": _short(crossfit_report),
            "detail": crossfit_report,
            "fold_selections": fold_selections,
        },
        "baselines": {name: _short(block) for name, block in baselines.items()},
    }

    output.mkdir(parents=True, exist_ok=True)
    pred = table[["fold", "study_id", "patient_id", "fracture_true", SUBMIT_AGGREGATOR]].copy()
    pred["submit_mapped_fracture_prob"] = mapped
    pred["gt_class"] = [standalone_fracture_triage(1.0 if t else 0.0) for t in truth]
    pred["pred_class"] = [standalone_fracture_triage(float(p)) for p in mapped]
    pred.to_csv(output / "submit_predictions.csv", index=False)
    crossfit.to_csv(output / "crossfit_predictions.csv", index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary = "\n".join(
        [
            "# Standalone fracture-only rules — model accuracy",
            "",
            "Rules use **only** `fracture_prob` (no ICH, no MLS, no official triage call).",
            "",
            "```",
            "if fracture_prob >= 0.5: return 1  # Urgent",
            "else:                    return 0  # Non-urgent",
            "```",
            "",
            f"- Studies: **{report['n_studies']}** "
            f"(pos={report['n_fracture_positive']}, neg={report['n_fracture_negative']})",
            "",
            "## Submit deployment (apparent)",
            "",
            f"- Accuracy: **{submit_report['model_accuracy']:.4f}**",
            f"- Macro-F1 (binary / standalone triage): "
            f"**{submit_report['macro_f1_binary']:.4f}**",
            f"- Sensitivity / Specificity: "
            f"**{submit_report['sensitivity']:.4f}** / **{submit_report['specificity']:.4f}**",
            f"- Precision / Pos-F1: "
            f"**{submit_report['precision']:.4f}** / **{submit_report['positive_f1']:.4f}**",
            f"- PR-AUC / ROC-AUC: "
            f"**{submit_report['pr_auc']:.4f}** / **{submit_report['roc_auc']:.4f}**",
            f"- TP/FP/FN/TN: `{submit_report['tp_fp_fn_tn']}`",
            "",
            "## Unbiased cross-fit",
            "",
            f"- Accuracy: **{crossfit_report['model_accuracy']:.4f}**",
            f"- Macro-F1: **{crossfit_report['macro_f1_binary']:.4f}**",
            f"- Sensitivity / Specificity: "
            f"**{crossfit_report['sensitivity']:.4f}** / **{crossfit_report['specificity']:.4f}**",
            f"- TP/FP/FN/TN: `{crossfit_report['tp_fp_fn_tn']}`",
            "",
        ]
    )
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "rules": result["rules"]["derivation"]["standalone_rules"],
                "submit_deployment": result["submit_deployment_apparent"]["summary"],
                "crossfit": result["unbiased_crossfit"]["summary"],
                "wrote": str(Path(args.output).resolve()),
            },
            indent=2,
        )
    )
