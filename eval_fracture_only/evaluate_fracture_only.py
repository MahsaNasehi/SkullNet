"""Evaluate the fracture head in isolation under official triage rules.

Problem this solves
-------------------
Oracle-other-heads Macro-F1 keeps GT ICH volumes and GT MLS while swapping
only ``fracture_prob``.  Those oracle heads can already force Urgent/Critical,
so fracture false-negatives / false-positives are often invisible in the
triage class.

Method (mirrors ICH-only isolation)
-----------------------------------
For both ground truth and predictions:

    intermediates = {
        "V_EDH": 0, "V_SDH": 0, "V_IPH": 0, "V_SAH": 0, "V_IVH": 0,
        "MLS_mm": 0,
        "fracture_prob": <gt 0/1  or  model probability>,
    }
    class = triage_from_intermediates(intermediates)   # official, unchanged

Primary score for this folder is the two-class Macro-F1 over {0, 1}, because
class 2 is unreachable when ICH and MLS are zero.  Binary fracture metrics at
the official 0.5 cutoff are reported alongside.  Oracle-other-heads Macro-F1
is kept only as a contrast metric to quantify masking.

Default inputs match the packaged submit artifact:
  cache  = outputs/full_study_oof_run_a_768_deployment_score_cache_v2_169
  aggregator = top10_percent_mean
  raw operating point = submit RAW_MACRO_F1_THRESHOLD (0.397176...)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_full_study_oof_cache import (  # noqa: E402
    CANDIDATE_AGGREGATORS,
    attach_truth,
    load_cache,
)
from fracture_macro_f1_oof import (  # noqa: E402
    binary_report,
    monotonic_operating_point_map,
    oracle_triage_report,
    threshold_candidates,
)
from triage_macro_f1 import triage_macro_f1_report  # noqa: E402

from fracture_only_triage import (  # noqa: E402
    FRACTURE_PRESENCE_THRESHOLD,
    fracture_only_triage,
)

# Matches submit/model.py deployment post-processing.
SUBMIT_AGGREGATOR = "top10_percent_mean"
SUBMIT_RAW_OPERATING_POINT = 0.39717610677083337
DEFAULT_CACHE = ROOT / "outputs/full_study_oof_run_a_768_deployment_score_cache_v2_169"
DEFAULT_METADATA = ROOT / "iaaa-contest-bct/Data/training_df.pkl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def two_class_macro_f1_report(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    """Macro-F1 over {0, 1} only — the reachable fracture-only triage classes."""
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    if truth.ndim != 1 or predicted.ndim != 1 or len(truth) != len(predicted) or not len(truth):
        raise ValueError("y_true and y_pred must be non-empty equal-length 1-D arrays")
    allowed = {0, 1}
    if not set(truth.tolist()) <= allowed or not set(predicted.tolist()) <= allowed:
        raise ValueError(
            "Fracture-only triage labels must be 0 or 1; got truth=%s pred=%s"
            % (sorted(set(truth.tolist())), sorted(set(predicted.tolist())))
        )
    labels = [0, 1]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    return {
        "pooled_macro_f1": float(np.mean(f1)),
        "micro_f1": float(accuracy_score(truth, predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "classwise": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "n_studies": int(len(truth)),
    }


def fracture_only_triage_report(
    fracture_true: Sequence[bool],
    fracture_prob: Sequence[float],
) -> Dict[str, Any]:
    """GT vs pred triage when only fracture_prob is allowed to drive the rule."""
    if len(fracture_true) != len(fracture_prob):
        raise ValueError("fracture_true and fracture_prob length mismatch")
    y_true = [fracture_only_triage(1.0 if bool(flag) else 0.0) for flag in fracture_true]
    y_pred = [fracture_only_triage(float(probability)) for probability in fracture_prob]
    binary = binary_report(fracture_true, fracture_prob)
    return {
        "metric_name": "fracture_only_official_triage",
        "method": (
            "Official triage_from_intermediates with V_EDH=V_SDH=V_IPH=V_SAH=V_IVH=MLS_mm=0 "
            "for both GT and predictions; only fracture_prob varies."
        ),
        "note_class_2_unreachable": (
            "With ICH and MLS zeroed, official rules never return Critical (2); "
            "fracture_present maps to Urgent (1), else Non-urgent (0)."
        ),
        "official_fracture_presence_threshold": FRACTURE_PRESENCE_THRESHOLD,
        "primary_two_class_macro_f1": two_class_macro_f1_report(y_true, y_pred),
        "three_class_macro_f1_for_reference": triage_macro_f1_report(y_true, y_pred),
        "binary_fracture_metrics": binary,
        "gt_triage_counts": {str(label): y_true.count(label) for label in (0, 1, 2)},
        "pred_triage_counts": {str(label): y_pred.count(label) for label in (0, 1, 2)},
    }


def select_operating_point_fracture_only(
    table: pd.DataFrame,
    score_column: str,
) -> Tuple[float, Dict[str, Any]]:
    """Pick raw threshold by fracture-only two-class Macro-F1 (not oracle heads)."""
    truth = table["fracture_true"].astype(bool).to_numpy()
    scores = table[score_column].to_numpy(dtype=float)
    candidates = []
    for threshold in threshold_candidates(scores):
        mapped = monotonic_operating_point_map(scores, float(threshold))
        report = fracture_only_triage_report(truth, mapped)
        binary = report["binary_fracture_metrics"]
        candidates.append((float(threshold), report, binary))
    selected = max(
        candidates,
        key=lambda item: (
            item[1]["primary_two_class_macro_f1"]["pooled_macro_f1"],
            -(item[2]["fp"] if item[2]["fp"] is not None else 10**9),
            item[2]["specificity"] if item[2]["specificity"] is not None else -1.0,
            item[2]["sensitivity"] if item[2]["sensitivity"] is not None else -1.0,
            -abs(item[0] - 0.5),
        ),
    )
    return selected[0], {
        "selection_metric": "fracture_only_two_class_macro_f1",
        "selection_report": selected[1],
        "selection_binary_guardrails": selected[2],
    }


def cross_fitted_fracture_only(
    table: pd.DataFrame,
    score_column: str,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    parts = []
    selections = []
    for fold in sorted(table["fold"].unique()):
        train = table[table["fold"] != fold]
        heldout = table[table["fold"] == fold].copy()
        threshold, details = select_operating_point_fracture_only(train, score_column)
        heldout["selected_raw_operating_point"] = threshold
        heldout["crossfit_fracture_prob"] = monotonic_operating_point_map(
            heldout[score_column].to_numpy(dtype=float), threshold
        )
        parts.append(heldout)
        selections.append(
            {
                "heldout_fold": int(fold),
                "selected_raw_operating_point": threshold,
                "selection_studies": int(len(train)),
                **details,
            }
        )
    return pd.concat(parts, ignore_index=True).sort_values(["fold", "study_id"]), selections


def select_aggregator_fracture_only(train: pd.DataFrame) -> Tuple[str, float, Dict[str, Any]]:
    ranked = []
    truth = train["fracture_true"].astype(bool).to_numpy()
    for method in CANDIDATE_AGGREGATORS:
        threshold, detail = select_operating_point_fracture_only(train, method)
        mapped = monotonic_operating_point_map(train[method].to_numpy(dtype=float), threshold)
        binary = binary_report(truth, mapped)
        pr_auc = float(average_precision_score(truth, train[method].to_numpy(dtype=float)))
        ranked.append(
            {
                "aggregator": method,
                "raw_operating_point": threshold,
                "fracture_only_two_class_macro_f1": detail["selection_report"][
                    "primary_two_class_macro_f1"
                ]["pooled_macro_f1"],
                "fp": binary["fp"],
                "specificity": binary["specificity"],
                "sensitivity": binary["sensitivity"],
                "pr_auc": pr_auc,
                "detail": detail,
            }
        )
    best = max(
        ranked,
        key=lambda row: (
            row["fracture_only_two_class_macro_f1"],
            -row["fp"],
            row["specificity"] if row["specificity"] is not None else -1.0,
            row["sensitivity"] if row["sensitivity"] is not None else -1.0,
            row["pr_auc"],
            -CANDIDATE_AGGREGATORS.index(row["aggregator"]),
        ),
    )
    return str(best["aggregator"]), float(best["raw_operating_point"]), {
        "selection_primary": "fracture_only_two_class_macro_f1",
        "candidate_table": [
            {k: v for k, v in row.items() if k != "detail"} for row in ranked
        ],
        "selected_detail": best["detail"],
    }


def cross_fitted_aggregator_and_threshold(
    table: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    parts = []
    selections = []
    for fold in sorted(table["fold"].unique()):
        train = table[table["fold"] != fold]
        heldout = table[table["fold"] == fold].copy()
        method, threshold, detail = select_aggregator_fracture_only(train)
        heldout["selected_aggregator"] = method
        heldout["selected_raw_operating_point"] = threshold
        heldout["crossfit_fracture_prob"] = monotonic_operating_point_map(
            heldout[method].to_numpy(dtype=float), threshold
        )
        parts.append(heldout)
        selections.append(
            {
                "heldout_fold": int(fold),
                "selected_aggregator": method,
                "selected_raw_operating_point": threshold,
                **detail,
            }
        )
    return pd.concat(parts, ignore_index=True).sort_values(["fold", "study_id"]), selections


def _summary_block(report: Dict[str, Any]) -> Dict[str, Any]:
    primary = report["primary_two_class_macro_f1"]
    binary = report["binary_fracture_metrics"]
    return {
        "fracture_only_two_class_macro_f1": primary["pooled_macro_f1"],
        "accuracy": primary["accuracy"],
        "sensitivity": binary["sensitivity"],
        "specificity": binary["specificity"],
        "precision": binary["precision"],
        "pr_auc": binary.get("pr_auc"),
        "roc_auc": binary.get("roc_auc"),
        "tp_fp_fn_tn": [binary["tp"], binary["fp"], binary["fn"], binary["tn"]],
        "binary_qwk": binary["binary_qwk_diagnostic"],
        "three_class_macro_f1_reference": report["three_class_macro_f1_for_reference"][
            "pooled_macro_f1"
        ],
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output)
    report_path = output / "report.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError("Refusing to overwrite %s (pass --overwrite)" % report_path)

    table = attach_truth(load_cache(Path(args.cache_dir)), Path(args.metadata))
    if sorted(table["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Expected five folds 0..4")

    truth = table["fracture_true"].astype(bool).to_numpy()
    submit_raw = table[SUBMIT_AGGREGATOR].to_numpy(dtype=float)
    submit_mapped = monotonic_operating_point_map(submit_raw, SUBMIT_RAW_OPERATING_POINT)

    # --- Baselines ---
    baselines = {
        "fracture_prob_all_zero": fracture_only_triage_report(truth, np.zeros(len(table))),
        "fracture_prob_all_one": fracture_only_triage_report(truth, np.ones(len(table))),
        "raw_top10_percent_mean_at_0.5": fracture_only_triage_report(truth, submit_raw),
    }

    # --- Submit deployment setting (apparent on all OOF) ---
    submit_deployment = fracture_only_triage_report(truth, submit_mapped)
    submit_oracle_contrast = oracle_triage_report(table, submit_mapped)

    # --- Cross-fit: fixed submit aggregator, fracture-only threshold selection ---
    crossfit_fixed_agg, selections_fixed = cross_fitted_fracture_only(table, SUBMIT_AGGREGATOR)
    crossfit_fixed_probs = crossfit_fixed_agg["crossfit_fracture_prob"].to_numpy(dtype=float)
    crossfit_fixed_report = fracture_only_triage_report(
        crossfit_fixed_agg["fracture_true"].astype(bool), crossfit_fixed_probs
    )
    crossfit_fixed_oracle = oracle_triage_report(crossfit_fixed_agg, crossfit_fixed_probs)

    # --- Cross-fit: choose aggregator + threshold by fracture-only metric ---
    crossfit_full, selections_full = cross_fitted_aggregator_and_threshold(table)
    crossfit_full_probs = crossfit_full["crossfit_fracture_prob"].to_numpy(dtype=float)
    crossfit_full_report = fracture_only_triage_report(
        crossfit_full["fracture_true"].astype(bool), crossfit_full_probs
    )
    crossfit_full_oracle = oracle_triage_report(crossfit_full, crossfit_full_probs)

    # --- Apparent all-OOF fracture-only retune of the submit aggregator ---
    apparent_threshold, apparent_details = select_operating_point_fracture_only(
        table, SUBMIT_AGGREGATOR
    )
    apparent_mapped = monotonic_operating_point_map(submit_raw, apparent_threshold)
    apparent_report = fracture_only_triage_report(truth, apparent_mapped)

    report = {
        "evaluation_name": "fracture_only_official_triage",
        "evaluation_partition": "patient-grouped full-study OOF (169); fixed test excluded",
        "model_source": {
            "cache_dir": str(Path(args.cache_dir).resolve()),
            "matches_submit_run": "Run A 768 YOLO26s-P2 deployment_score cache v2_169",
            "submit_artifact": "submit/models/best.pt (Fold 2 packaged for deployment)",
            "submit_aggregator": SUBMIT_AGGREGATOR,
            "submit_raw_operating_point": SUBMIT_RAW_OPERATING_POINT,
            "official_fracture_cutoff_after_mapping": 0.5,
        },
        "why_this_exists": (
            "Oracle-other-heads Macro-F1 can stay high even when the fracture head "
            "is weak, because GT ICH/MLS already determine many triage classes. "
            "This evaluation zeros ICH and MLS for both GT and predictions so only "
            "fracture_prob can change the official triage class."
        ),
        "primary_metric": "fracture_only_two_class_macro_f1",
        "n_studies": int(len(table)),
        "n_fracture_positive": int(truth.sum()),
        "n_fracture_negative": int((~truth).sum()),
        "baselines_summary": {name: _summary_block(block) for name, block in baselines.items()},
        "submit_deployment_apparent_all_oof": {
            "warning": (
                "Uses the packaged submit aggregator + raw operating point on all "
                "OOF studies. Apparent/deployment diagnostic, not unbiased."
            ),
            "summary": _summary_block(submit_deployment),
            "detail": submit_deployment,
            "oracle_other_heads_contrast": {
                "warning": "Same fracture scores, but GT ICH/MLS kept — shows masking.",
                "pooled_macro_f1": submit_oracle_contrast["pooled_macro_f1"],
                "classwise": submit_oracle_contrast["classwise"],
                "confusion_matrix": submit_oracle_contrast["confusion_matrix"],
            },
            "masking_gap_oracle_minus_fracture_only_two_class": (
                float(submit_oracle_contrast["pooled_macro_f1"])
                - float(submit_deployment["primary_two_class_macro_f1"]["pooled_macro_f1"])
            ),
        },
        "unbiased_crossfit_fixed_submit_aggregator": {
            "description": (
                "Aggregator fixed to submit top10_percent_mean; raw operating point "
                "chosen each fold by fracture-only two-class Macro-F1 on the other folds."
            ),
            "summary": _summary_block(crossfit_fixed_report),
            "detail": crossfit_fixed_report,
            "oracle_other_heads_contrast_macro_f1": crossfit_fixed_oracle["pooled_macro_f1"],
            "fold_selections": [
                {
                    "heldout_fold": item["heldout_fold"],
                    "selected_raw_operating_point": item["selected_raw_operating_point"],
                    "selection_studies": item["selection_studies"],
                    "selection_macro_f1": item["selection_report"]["primary_two_class_macro_f1"][
                        "pooled_macro_f1"
                    ],
                }
                for item in selections_fixed
            ],
        },
        "unbiased_crossfit_aggregator_and_threshold": {
            "description": (
                "Both aggregator and raw operating point chosen each fold by "
                "fracture-only two-class Macro-F1 on the other folds."
            ),
            "summary": _summary_block(crossfit_full_report),
            "detail": crossfit_full_report,
            "oracle_other_heads_contrast_macro_f1": crossfit_full_oracle["pooled_macro_f1"],
            "fold_selections": [
                {
                    "heldout_fold": item["heldout_fold"],
                    "selected_aggregator": item["selected_aggregator"],
                    "selected_raw_operating_point": item["selected_raw_operating_point"],
                }
                for item in selections_full
            ],
        },
        "apparent_fracture_only_retune_submit_aggregator": {
            "warning": "Selected on all OOF; deployment parameter only, not unbiased.",
            "raw_operating_point": apparent_threshold,
            "summary": _summary_block(apparent_report),
            "selection": {
                "selection_metric": apparent_details["selection_metric"],
                "selection_macro_f1": apparent_details["selection_report"][
                    "primary_two_class_macro_f1"
                ]["pooled_macro_f1"],
            },
        },
        "baselines_detail": baselines,
    }

    output.mkdir(parents=True, exist_ok=True)
    crossfit_fixed_agg.to_csv(output / "crossfit_fixed_aggregator_predictions.csv", index=False)
    crossfit_full.to_csv(output / "crossfit_full_predictions.csv", index=False)
    pred_table = table[
        ["fold", "study_id", "patient_id", "fracture_true", SUBMIT_AGGREGATOR]
    ].copy()
    pred_table["submit_mapped_fracture_prob"] = submit_mapped
    pred_table["submit_fracture_only_gt_class"] = [
        fracture_only_triage(1.0 if flag else 0.0) for flag in truth
    ]
    pred_table["submit_fracture_only_pred_class"] = [
        fracture_only_triage(float(p)) for p in submit_mapped
    ]
    pred_table.to_csv(output / "submit_deployment_predictions.csv", index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary_path = output / "SUMMARY.md"
    summary_path.write_text(_render_summary_md(report), encoding="utf-8")
    return report


def _render_summary_md(report: Dict[str, Any]) -> str:
    submit = report["submit_deployment_apparent_all_oof"]["summary"]
    fixed = report["unbiased_crossfit_fixed_submit_aggregator"]["summary"]
    full = report["unbiased_crossfit_aggregator_and_threshold"]["summary"]
    gap = report["submit_deployment_apparent_all_oof"][
        "masking_gap_oracle_minus_fracture_only_two_class"
    ]
    lines = [
        "# Fracture-only official triage evaluation",
        "",
        "Official `triage_from_intermediates` with **ICH volumes and MLS forced to 0**",
        "for both GT and predictions. Only `fracture_prob` drives the class.",
        "",
        f"- Studies: **{report['n_studies']}** "
        f"(pos={report['n_fracture_positive']}, neg={report['n_fracture_negative']})",
        f"- Primary metric: **{report['primary_metric']}** (classes {{0,1}})",
        "",
        "## Submit deployment (apparent all-OOF)",
        "",
        f"- Fracture-only two-class Macro-F1: **{submit['fracture_only_two_class_macro_f1']:.4f}**",
        f"- Sensitivity / Specificity: "
        f"**{submit['sensitivity']:.4f}** / **{submit['specificity']:.4f}**",
        f"- Precision / PR-AUC: **{submit['precision']:.4f}** / **{submit['pr_auc']:.4f}**",
        f"- TP/FP/FN/TN: `{submit['tp_fp_fn_tn']}`",
        f"- Oracle-other-heads Macro-F1 (contrast): "
        f"**{report['submit_deployment_apparent_all_oof']['oracle_other_heads_contrast']['pooled_macro_f1']:.4f}**",
        f"- Masking gap (oracle − fracture-only): **{gap:.4f}**",
        "",
        "## Unbiased cross-fit (fixed submit aggregator)",
        "",
        f"- Fracture-only two-class Macro-F1: **{fixed['fracture_only_two_class_macro_f1']:.4f}**",
        f"- Sensitivity / Specificity: "
        f"**{fixed['sensitivity']:.4f}** / **{fixed['specificity']:.4f}**",
        f"- TP/FP/FN/TN: `{fixed['tp_fp_fn_tn']}`",
        "",
        "## Unbiased cross-fit (aggregator + threshold)",
        "",
        f"- Fracture-only two-class Macro-F1: **{full['fracture_only_two_class_macro_f1']:.4f}**",
        f"- Sensitivity / Specificity: "
        f"**{full['sensitivity']:.4f}** / **{full['specificity']:.4f}**",
        f"- TP/FP/FN/TN: `{full['tp_fp_fn_tn']}`",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "primary_metric": result["primary_metric"],
        "submit_deployment": result["submit_deployment_apparent_all_oof"]["summary"],
        "crossfit_fixed_aggregator": result["unbiased_crossfit_fixed_submit_aggregator"]["summary"],
        "crossfit_full": result["unbiased_crossfit_aggregator_and_threshold"]["summary"],
        "masking_gap": result["submit_deployment_apparent_all_oof"][
            "masking_gap_oracle_minus_fracture_only_two_class"
        ],
        "wrote": str(DEFAULT_OUTPUT),
    }, indent=2))
