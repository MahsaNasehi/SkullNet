"""Paired, development-only comparison of frozen Run A (768) and Run B (1024)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A = ROOT / "outputs/oof_proposal_recall_run_a_768"
DEFAULT_B = ROOT / "outputs/oof_proposal_recall_run_b_1024"
DEFAULT_OUTPUT = ROOT / "outputs/run_a_vs_b_1024_paired"
# These three cost measurements were explicitly frozen with authoritative Run A.
# A later schema-only rerun reproduced all predictions/metrics but naturally had
# slightly different wall-clock throughput, which must not silently replace them.
RUN_A_FROZEN_COST = {
    "aggregate_model_predict_images_per_second": 75.18,
    "max_cuda_peak_allocated_gib": 1.948,
    "max_cuda_peak_reserved_gib": 2.240,
}
GT_KEYS = ["fold", "patient_id", "study_id", "sop_uid", "gt_box_index"]
STUDY_KEYS = ["fold", "patient_id", "study_id"]


def transitions(frame: pd.DataFrame, threshold: float) -> dict[str, int]:
    a = frame["best_iou_A"] >= threshold
    b = frame["best_iou_B"] >= threshold
    return {"miss_A_to_hit_B": int((~a & b).sum()), "hit_A_to_miss_B": int((a & ~b).sum()),
            "hit_both": int((a & b).sum()), "miss_both": int((~a & ~b).sum())}


def percentile_ci(values: list[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def paired_patient_bootstrap_gt(frame: pd.DataFrame, repeats: int, seed: int) -> dict[str, object]:
    groups = {patient: group for patient, group in frame.groupby("patient_id", sort=False)}
    patients = np.array(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    values = {"delta_recall_iou30": [], "delta_recall_iou50": []}
    for _ in range(repeats):
        sample = pd.concat([groups[p] for p in rng.choice(patients, len(patients), replace=True)], ignore_index=True)
        for threshold, key in ((0.3, "delta_recall_iou30"), (0.5, "delta_recall_iou50")):
            values[key].append(float((sample["best_iou_B"] >= threshold).mean()
                                     - (sample["best_iou_A"] >= threshold).mean()))
    return {key: percentile_ci(value) for key, value in values.items()}


def binary_values(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[False, True]).ravel()
    return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "qwk": cohen_kappa_score(y.astype(int), pred.astype(int), labels=[0, 1], weights="quadratic"),
            "fp": float(fp)}


def paired_patient_bootstrap_study(frame: pd.DataFrame, repeats: int, seed: int) -> dict[str, object]:
    patient_truth = frame.groupby("patient_id")["fracture_true"].max().astype(bool)
    positive, negative = patient_truth[patient_truth].index.to_numpy(), patient_truth[~patient_truth].index.to_numpy()
    groups = {patient: group for patient, group in frame.groupby("patient_id", sort=False)}
    rng = np.random.default_rng(seed)
    keys = ["delta_pr_auc", "delta_roc_auc", "delta_fixed_sensitivity", "delta_fixed_specificity",
            "delta_fixed_precision", "delta_fixed_qwk", "delta_crossfit_sensitivity",
            "delta_crossfit_specificity", "delta_crossfit_precision", "delta_crossfit_qwk",
            "delta_crossfit_fp", "delta_macro_study_recall_iou30", "delta_macro_study_recall_iou50"]
    samples = {key: [] for key in keys}
    for _ in range(repeats):
        chosen = np.r_[rng.choice(positive, len(positive), replace=True),
                       rng.choice(negative, len(negative), replace=True)]
        sample = pd.concat([groups[p] for p in chosen], ignore_index=True)
        y = sample["fracture_true"].to_numpy(bool)
        samples["delta_pr_auc"].append(float(average_precision_score(y, sample["top3_mean_B"])
                                                   - average_precision_score(y, sample["top3_mean_A"])))
        samples["delta_roc_auc"].append(float(roc_auc_score(y, sample["top3_mean_B"])
                                                    - roc_auc_score(y, sample["top3_mean_A"])))
        fixed_a = binary_values(y, sample["top3_mean_A"].to_numpy(float) >= 0.5)
        fixed_b = binary_values(y, sample["top3_mean_B"].to_numpy(float) >= 0.5)
        ma = binary_values(y, sample["crossfit_predicted_A"].to_numpy(bool))
        mb = binary_values(y, sample["crossfit_predicted_B"].to_numpy(bool))
        for metric in ("sensitivity", "specificity", "precision", "qwk"):
            samples[f"delta_fixed_{metric}"].append(float(fixed_b[metric] - fixed_a[metric]))
            samples[f"delta_crossfit_{metric}"].append(float(mb[metric] - ma[metric]))
        samples["delta_crossfit_fp"].append(float(mb["fp"] - ma["fp"]))
        positive_studies = sample[sample["fracture_true"]]
        for threshold in (30, 50):
            samples[f"delta_macro_study_recall_iou{threshold}"].append(float(
                positive_studies[f"proposal_recall_iou{threshold}_B"].mean()
                - positive_studies[f"proposal_recall_iou{threshold}_A"].mean()
            ))
    return {key: percentile_ci(values) for key, values in samples.items()}


def metric_delta(a: dict, b: dict, key: str) -> float:
    return float(b[key] - a[key])


def training_fold_metrics(prefix: str) -> list[dict[str, object]]:
    rows = []
    for fold in range(5):
        run_dir = ROOT / "outputs" / f"{prefix}{fold}"
        results = pd.read_csv(run_dir / "results.csv")
        results.columns = results.columns.str.strip()
        best = results.loc[results["metrics/mAP50-95(B)"].idxmax()]
        completion_path = run_dir / "training_completed.json"
        completion = json.loads(completion_path.read_text()) if completion_path.is_file() else {}
        rows.append({
            "fold": fold, "epochs": len(results), "best_epoch_by_map50_95": int(best["epoch"]),
            "best_map50": float(best["metrics/mAP50(B)"]),
            "best_map50_95": float(best["metrics/mAP50-95(B)"]),
            "ultralytics_cumulative_time_seconds": (
                float(results.iloc[-1]["time"]) if "time" in results.columns else None
            ),
            "completion_marker": completion.get("status") == "completed",
            "init_sha256": (json.loads((run_dir / "run_protocol.json").read_text()).get(
                "initialized_checkpoint_sha256") if (run_dir / "run_protocol.json").is_file() else None),
        })
    return rows


def write_markdown_summary(report: dict[str, object], path: Path) -> None:
    """Write the complete human-readable A/B handoff from the machine report."""
    a, b = report["run_a"], report["run_b"]
    lines = [
        "# مقایسه paired و OOF: Run A 768 در برابر Run B 1024",
        "",
        "این گزارش فقط از development OOF ساخته شده است؛ fixed test خوانده یا در تصمیم استفاده نشده است.",
        "",
        "## Proposal recall (primary)", "",
        "| Size | GT | A IoU30 | B IoU30 | Delta | 95% CI Delta | A IoU50 | B IoU50 | Delta | 95% CI Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in ("overall", "small", "medium", "large"):
        row = report["proposal_recall_A_vs_B_by_size"][size]
        ci = report["paired_patient_bootstrap_proposal_delta_95ci"][size]
        lines.append(
            f"| {size} | {row['gt_boxes']} | {row['A_recall_iou30']:.5f} | {row['B_recall_iou30']:.5f} | "
            f"{row['delta_iou30']:+.5f} | [{ci['delta_recall_iou30'][0]:+.5f}, {ci['delta_recall_iou30'][1]:+.5f}] | "
            f"{row['A_recall_iou50']:.5f} | {row['B_recall_iou50']:.5f} | {row['delta_iou50']:+.5f} | "
            f"[{ci['delta_recall_iou50'][0]:+.5f}, {ci['delta_recall_iou50'][1]:+.5f}] |"
        )
    p = report["primary_deltas_B_minus_A"]
    lines += [
        "", "Macro-positive-Study proposal recall:", "",
        "| IoU | A | B | Delta | paired patient-bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    sci = report["paired_patient_bootstrap_study_delta_95ci"]
    for threshold in (30, 50):
        key = f"iou{threshold}"
        delta_key = f"macro_study_recall_{key}"
        lines.append(
            f"| {threshold/100:.2f} | {a['macro_positive_study_proposal_recall'][key]:.5f} | "
            f"{b['macro_positive_study_proposal_recall'][key]:.5f} | {p[delta_key]:+.5f} | "
            f"[{sci['delta_' + delta_key][0]:+.5f}, {sci['delta_' + delta_key][1]:+.5f}] |"
        )
    lines += ["", "## GT transition counts", ""]
    for size in ("overall", "small", "medium", "large"):
        lines += [f"### {size}", "", "| IoU | miss A → hit B | hit A → miss B | hit both | miss both |",
                  "|---|---:|---:|---:|---:|"]
        for threshold in (30, 50):
            t = report["transitions"][size][f"iou{threshold}"]
            lines.append(f"| {threshold/100:.2f} | {t['miss_A_to_hit_B']} | {t['hit_A_to_miss_B']} | {t['hit_both']} | {t['miss_both']} |")
        lines.append("")
    lines += ["## Study-level OOF", "",
              "| Metric | Run A | Run B | Delta B−A | paired patient-bootstrap 95% CI |",
              "|---|---:|---:|---:|---:|"]
    metric_rows = [
        ("PR-AUC", a["study_ranking"]["pr_auc"], b["study_ranking"]["pr_auc"], "delta_pr_auc"),
        ("ROC-AUC", a["study_ranking"]["roc_auc"], b["study_ranking"]["roc_auc"], "delta_roc_auc"),
    ]
    for label, av, bv, ci_key in metric_rows:
        lines.append(f"| {label} | {av:.5f} | {bv:.5f} | {bv-av:+.5f} | [{sci[ci_key][0]:+.5f}, {sci[ci_key][1]:+.5f}] |")
    for section, title, prefix in (("study_metrics_at_fixed_0_5", "@0.5", "fixed"),
                                   ("cross_fitted_study_metrics", "cross-fitted", "crossfit")):
        lines += ["", f"### {title}", "", "| Metric | Run A | Run B | Delta B−A | 95% CI Delta |",
                  "|---|---:|---:|---:|---:|"]
        for metric, ci_metric in (("sensitivity", "sensitivity"), ("specificity", "specificity"),
                                  ("precision", "precision"), ("binary_qwk", "qwk")):
            av, bv = a[section][metric], b[section][metric]
            ci = sci[f"delta_{prefix}_{ci_metric}"]
            lines.append(f"| {metric} | {av:.5f} | {bv:.5f} | {bv-av:+.5f} | [{ci[0]:+.5f}, {ci[1]:+.5f}] |")
        lines.append("")
        lines.append(f"Confusion TP/FP/FN/TN: A=`{a[section]['tp']}/{a[section]['fp']}/{a[section]['fn']}/{a[section]['tn']}`، "
                     f"B=`{b[section]['tp']}/{b[section]['fp']}/{b[section]['fn']}/{b[section]['tn']}`.")
    apparent = b.get("all_oof_selected_threshold_apparent_only", {})
    lines += ["", "Run B apparent/deployment threshold انتخاب‌شده روی کل OOF: "
              f"`{apparent.get('threshold', 'NA')}`. این عدد apparent/optimistic است و performance unbiased محسوب نمی‌شود.",
              "", "## هزینه inference", ""]
    cost = report["inference_cost_A_vs_B"]
    lines += [
        "| Metric | Run A authoritative | Run B | Delta/ratio |", "|---|---:|---:|---:|",
        f"| throughput image/s | {cost['A_authoritative_frozen']['aggregate_model_predict_images_per_second']:.3f} | "
        f"{cost['B']['aggregate_model_predict_images_per_second']:.3f} | ×{cost['throughput_ratio_B_over_A']:.3f} |",
        f"| peak CUDA allocated GiB | {cost['A_authoritative_frozen']['max_cuda_peak_allocated_gib']:.3f} | "
        f"{cost['B']['max_cuda_peak_allocated_gib']:.3f} | {cost['peak_allocated_delta_gib']:+.3f} |",
        "", "## محدودیت آماری", "",
        "CIها paired و patient-clustered هستند. برای cross-fitted metrics، decisionهای cross-fit فریز و resample شده‌اند؛ "
        "عدم‌قطعیت ناشی از refit دوباره threshold داخل هر bootstrap در CI لحاظ نشده است. sample کوچک است و delta کوچک قطعی تلقی نمی‌شود.",
        "", "## Verdict", "", f"`{report['verdict']}`", "",
        "این verdict فقط با evidence تجمیعی OOF و rule ازپیش‌ثبت‌شده ساخته شده است.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    if (args.output / "comparison.json").exists():
        raise FileExistsError(f"Refusing overwrite: {args.output}")
    metrics_a = json.loads((args.run_a / "metrics.json").read_text())
    metrics_b = json.loads((args.run_b / "metrics.json").read_text())
    if metrics_a["scope"] != metrics_b["scope"] or metrics_a["gt_boxes"] != metrics_b["gt_boxes"]:
        raise ValueError("A/B evaluation scope or GT count differs")

    ga = pd.read_csv(args.run_a / "gt_proposal_matches.csv", dtype={"patient_id": str, "study_id": str,
                                                                     "sop_uid": str})
    gb = pd.read_csv(args.run_b / "gt_proposal_matches.csv", dtype={"patient_id": str, "study_id": str,
                                                                     "sop_uid": str})
    keep = GT_KEYS + ["size_category", "best_iou", "max_confidence_iou30", "max_confidence_iou50"]
    paired = ga[keep].merge(gb[keep], on=GT_KEYS, suffixes=("_A", "_B"), validate="one_to_one")
    if len(paired) != len(ga) or len(paired) != len(gb) or not paired["size_category_A"].eq(paired["size_category_B"]).all():
        raise ValueError("GT pairing or size category mismatch")
    paired["size_category"] = paired.pop("size_category_A")
    paired = paired.drop(columns="size_category_B")
    for threshold in (30, 50):
        paired[f"covered_A_{threshold}"] = paired["best_iou_A"] >= threshold / 100
        paired[f"covered_B_{threshold}"] = paired["best_iou_B"] >= threshold / 100
    paired = paired.rename(columns={
        "max_confidence_iou30_A": "best_matching_conf_A_30",
        "max_confidence_iou30_B": "best_matching_conf_B_30",
        "max_confidence_iou50_A": "best_matching_conf_A_50",
        "max_confidence_iou50_B": "best_matching_conf_B_50",
    })

    transition_report = {}
    bootstrap_gt = {}
    proposal_by_size = {}
    for category in ("overall", "small", "medium", "large"):
        subset = paired if category == "overall" else paired[paired["size_category"] == category]
        transition_report[category] = {"gt_boxes": len(subset), "iou30": transitions(subset, 0.3),
                                       "iou50": transitions(subset, 0.5)}
        bootstrap_gt[category] = paired_patient_bootstrap_gt(subset, args.bootstrap_repeats, args.seed)
        proposal_by_size[category] = {
            "gt_boxes": len(subset),
            "A_recall_iou30": float((subset["best_iou_A"] >= 0.3).mean()),
            "B_recall_iou30": float((subset["best_iou_B"] >= 0.3).mean()),
            "delta_iou30": float((subset["best_iou_B"] >= 0.3).mean()
                                 - (subset["best_iou_A"] >= 0.3).mean()),
            "A_recall_iou50": float((subset["best_iou_A"] >= 0.5).mean()),
            "B_recall_iou50": float((subset["best_iou_B"] >= 0.5).mean()),
            "delta_iou50": float((subset["best_iou_B"] >= 0.5).mean()
                                 - (subset["best_iou_A"] >= 0.5).mean()),
        }

    sa = pd.read_csv(args.run_a / "study_predictions.csv", dtype={"patient_id": str, "study_id": str})
    sb = pd.read_csv(args.run_b / "study_predictions.csv", dtype={"patient_id": str, "study_id": str})
    ca = pd.read_csv(args.run_a / "study_crossfit_predictions.csv", dtype={"patient_id": str, "study_id": str})
    cb = pd.read_csv(args.run_b / "study_crossfit_predictions.csv", dtype={"patient_id": str, "study_id": str})
    study = sa[STUDY_KEYS + ["fracture_true", "top3_mean", "proposal_recall_iou30", "proposal_recall_iou50"]].merge(
        sb[STUDY_KEYS + ["fracture_true", "top3_mean", "proposal_recall_iou30", "proposal_recall_iou50"]],
        on=STUDY_KEYS, suffixes=("_A", "_B"), validate="one_to_one")
    if not study["fracture_true_A"].eq(study["fracture_true_B"]).all():
        raise ValueError("Study truth mismatch")
    study["fracture_true"] = study.pop("fracture_true_A")
    study = study.drop(columns="fracture_true_B")
    for label, crossfit in (("A", ca), ("B", cb)):
        study = study.merge(crossfit[STUDY_KEYS + ["crossfit_predicted"]].rename(
            columns={"crossfit_predicted": f"crossfit_predicted_{label}"}), on=STUDY_KEYS, validate="one_to_one")

    pa, pb = metrics_a["proposal_metrics"]["overall"], metrics_b["proposal_metrics"]["overall"]
    fixed_a = metrics_a["study_metrics_at_fixed_0_5"]
    fixed_b = metrics_b["study_metrics_at_fixed_0_5"]
    cfa, cfb = metrics_a["cross_fitted_study_metrics"], metrics_b["cross_fitted_study_metrics"]
    study_delta = {
        "pr_auc": metric_delta(metrics_a["study_ranking"], metrics_b["study_ranking"], "pr_auc"),
        "roc_auc": metric_delta(metrics_a["study_ranking"], metrics_b["study_ranking"], "roc_auc"),
        "fixed_sensitivity": metric_delta(fixed_a, fixed_b, "sensitivity"),
        "fixed_specificity": metric_delta(fixed_a, fixed_b, "specificity"),
        "fixed_precision": metric_delta(fixed_a, fixed_b, "precision"),
        "fixed_qwk": metric_delta(fixed_a, fixed_b, "binary_qwk"),
        "fixed_tp": int(fixed_b["tp"] - fixed_a["tp"]),
        "fixed_fp": int(fixed_b["fp"] - fixed_a["fp"]),
        "fixed_fn": int(fixed_b["fn"] - fixed_a["fn"]),
        "fixed_tn": int(fixed_b["tn"] - fixed_a["tn"]),
        "crossfit_sensitivity": metric_delta(cfa, cfb, "sensitivity"),
        "crossfit_specificity": metric_delta(cfa, cfb, "specificity"),
        "crossfit_precision": metric_delta(cfa, cfb, "precision"),
        "crossfit_qwk": metric_delta(cfa, cfb, "binary_qwk"),
        "crossfit_tp": int(cfb["tp"] - cfa["tp"]),
        "crossfit_fp": int(cfb["fp"] - cfa["fp"]),
        "crossfit_fn": int(cfb["fn"] - cfa["fn"]),
        "crossfit_tn": int(cfb["tn"] - cfa["tn"]),
    }
    primary_delta = {"overall_recall_iou30": metric_delta(pa, pb, "proposal_recall_iou30"),
                     "overall_recall_iou50": metric_delta(pa, pb, "proposal_recall_iou50"),
                     "macro_study_recall_iou30": metric_delta(
                         metrics_a["macro_positive_study_proposal_recall"],
                         metrics_b["macro_positive_study_proposal_recall"], "iou30"),
                     "macro_study_recall_iou50": metric_delta(
                         metrics_a["macro_positive_study_proposal_recall"],
                         metrics_b["macro_positive_study_proposal_recall"], "iou50")}
    bootstrap_study = paired_patient_bootstrap_study(study, args.bootstrap_repeats, args.seed)
    ci30, ci50 = bootstrap_gt["overall"]["delta_recall_iou30"], bootstrap_gt["overall"]["delta_recall_iou50"]
    supported_gain = ci30[0] > 0 or ci50[0] > 0
    primary_nonnegative = primary_delta["overall_recall_iou30"] > 0 and primary_delta["overall_recall_iou50"] > 0
    downstream_safe = (study_delta["pr_auc"] >= -0.01 and study_delta["crossfit_qwk"] >= -0.05
                       and study_delta["crossfit_sensitivity"] >= -0.02 and study_delta["crossfit_fp"] <= 2)
    supported_harm = ci30[1] < 0 or ci50[1] < 0
    if primary_nonnegative and supported_gain and downstream_safe:
        verdict = "KEEP_1024"
    elif supported_harm or (primary_delta["overall_recall_iou30"] <= 0
                            and primary_delta["overall_recall_iou50"] <= 0):
        verdict = "REJECT_1024"
    else:
        verdict = "INCONCLUSIVE_1024"
    report = {
        "scope": "paired development-only OOF A/B; fixed test never read",
        "decision_priority": ["overall_and_macro_study_proposal_recall", "size_recall", "study_pr_auc",
                              "sensitivity_with_specificity", "cross_fitted_qwk", "fp_count", "runtime_vram", "map"],
        "predeclared_verdict_rule": {
            "keep": "both overall recalls improve, >=1 paired 95% CI lower bound >0, and downstream safety",
            "downstream_safety": "PR-AUC delta>=-0.01, crossfit QWK>=-0.05, sensitivity>=-0.02, FP delta<=2",
            "reject": ">=1 recall CI upper bound <0, or both overall recall point deltas <=0",
            "otherwise": "inconclusive",
        },
        "run_a": metrics_a, "run_b": metrics_b,
        "primary_deltas_B_minus_A": primary_delta,
        "proposal_recall_A_vs_B_by_size": proposal_by_size,
        "study_deltas_B_minus_A": study_delta,
        "transitions": transition_report,
        "paired_patient_bootstrap_proposal_delta_95ci": bootstrap_gt,
        "paired_patient_bootstrap_study_delta_95ci": bootstrap_study,
        "inference_cost_A_vs_B": {
            "A_authoritative_frozen": RUN_A_FROZEN_COST,
            "A_schema_rerun_observed": metrics_a["runtime_summary"],
            "B": metrics_b["runtime_summary"],
            "throughput_ratio_B_over_A": (
                metrics_b["runtime_summary"]["aggregate_model_predict_images_per_second"]
                / RUN_A_FROZEN_COST["aggregate_model_predict_images_per_second"]
            ),
            "peak_allocated_delta_gib": (
                metrics_b["runtime_summary"]["max_cuda_peak_allocated_gib"]
                - RUN_A_FROZEN_COST["max_cuda_peak_allocated_gib"]
            ),
            "note": "A cost uses the user-frozen authoritative 75.18 img/s and 1.948/2.240 GiB; schema-rerun timing is retained separately.",
        },
        "training_detector_metrics_supplementary": {
            "A": training_fold_metrics("yolo26s_p2_hu800_ww1600_fold"),
            "B": training_fold_metrics("yolo26s_p2_hu800_ww1600_run_b_1024_fold"),
        },
        "verdict": verdict,
        "bootstrap_note": "Patient-clustered paired bootstrap; frozen crossfit decisions, threshold refit uncertainty excluded.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output / "paired_gt.csv", index=False)
    study.to_csv(args.output / "paired_studies.csv", index=False)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown_summary(report, args.output / "COMPARISON_SUMMARY_FA.md")
    print(json.dumps({"primary_deltas": primary_delta, "study_deltas": study_delta,
                      "proposal_delta_ci": bootstrap_gt["overall"], "study_delta_ci": bootstrap_study,
                      "verdict": verdict}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, default=DEFAULT_A)
    parser.add_argument("--run-b", type=Path, default=DEFAULT_B)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
