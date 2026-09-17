"""CPU-only, manual packaging after the fixed-epoch FULL169 refit is finalized."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from fracture_classifier.data import FINAL_DEPLOYMENT, sha256
from fracture_classifier.train_full169 import (EXPECTED_DETECTOR_SHA256, FIXED_EPOCHS,
                                               OUTPUT)


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "handoff/fracture_run_a_classifier_fusion_v1"
RAW_THRESHOLD = 0.39717610677083337
OOF_DETECTOR = {"oracle_other_heads_macro_f1": 0.9696273781,
                "study_pr_auc": 0.5355486381, "sensitivity": 0.2916667}
OOF_FUSION = {"oracle_other_heads_macro_f1": 0.9696273781,
              "study_pr_auc": 0.5592862949, "sensitivity": 0.3333333}


def _source_rules() -> str:
    source = (ROOT / "submit/model.py").read_text()
    required = ("AGGREGATION_METHOD = \"top10_percent_mean\"",
                f"RAW_MACRO_F1_THRESHOLD = {RAW_THRESHOLD}",
                "OFFICIAL_FRACTURE_THRESHOLD = 0.5",
                "CONFIDENCE = 0.01", "NMS_IOU = 0.5", "IMAGE_SIZE = 768")
    if not all(item in source for item in required):
        raise RuntimeError("Established Run A deployment rule has changed")
    helper = "from __future__ import annotations" + source.split(
        "from __future__ import annotations", 1)[1].split("\nclass Model:", 1)[0]
    if "def rescale_for_macro_f1(" not in helper or "def make_hu_input(" not in helper:
        raise RuntimeError("Detector runtime helper extraction failed")
    return ('"""Frozen Run A fracture preprocessing and detector helpers.\n\n'
            'Copied from the selected 768 submission helper; no final Model API.\n"""\n' + helper)


def _validate() -> dict:
    if DESTINATION.exists():
        raise FileExistsError(f"Handoff exists; refusing overwrite: {DESTINATION}")
    final = OUTPUT / "final_classifier.pt"
    last = OUTPUT / "last_classifier.pt"
    completion = OUTPUT / "training_completed.json"
    if not all(path.is_file() for path in (final, last, completion)):
        raise RuntimeError("FULL169 classifier must finish and be finalized before packaging")
    report = json.loads(completion.read_text())
    classifier_sha = sha256(final)
    if (report.get("final_epoch_one_based") != FIXED_EPOCHS
            or report.get("validation_used") is not False
            or report.get("final_classifier_sha256") != classifier_sha
            or report.get("last_classifier_sha256") != classifier_sha
            or sha256(last) != classifier_sha
            or report.get("source_detector_sha256") != EXPECTED_DETECTOR_SHA256
            or sha256(FINAL_DEPLOYMENT) != EXPECTED_DETECTOR_SHA256):
        raise RuntimeError("Final checkpoint provenance or byte identity failed")
    # Source-of-truth OOF experiment: top5_mean and alpha 0.75 on every held-out Fold.
    comparison = json.loads((ROOT / "outputs/run_a_fracture_classifier_oof/comparison/report.json").read_text())
    folds = comparison.get("selection")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("Missing five-fold fusion selection evidence")
    for item in folds:
        if item.get("classifier_aggregator") != "top5_mean" or float(item.get("fusion_alpha", -1)) != 0.75:
            raise RuntimeError("OOF classifier/fusion rule differs from proposed handoff")
    _source_rules()
    return {"classifier_sha": classifier_sha, "completion": report,
            "fusion_report_sha": sha256(ROOT / "outputs/run_a_fracture_classifier_oof/comparison/report.json")}


def build() -> Path:
    audit = _validate()
    src_dir = DESTINATION / "src"
    model_dir = DESTINATION / "models"
    config_dir = DESTINATION / "config"
    for path in (src_dir, model_dir, config_dir):
        path.mkdir(parents=True)
    shutil.copy2(FINAL_DEPLOYMENT, model_dir / "fracture_detector.pt")
    shutil.copy2(OUTPUT / "final_classifier.pt", model_dir / "fracture_classifier.pt")
    shutil.copy2(ROOT / "src/fracture_classifier/model.py", src_dir / "classifier_model.py")
    shutil.copy2(ROOT / "src/fracture_classifier/handoff_runtime.py", src_dir / "predictor.py")
    shutil.copy2(ROOT / "src/fracture_classifier/handoff_benchmark.py", src_dir / "benchmark.py")
    (src_dir / "detector_runtime.py").write_text(_source_rules())
    config = {
        "detector_architecture": "YOLO26s-P2 Run A FULL169",
        "detector_sha256": EXPECTED_DETECTOR_SHA256,
        "classifier_architecture": "Run A P5 C2PSA layer 10 512ch -> GAP -> LN512 -> Linear512x128 -> SiLU -> Dropout0.10 -> Linear128x1",
        "classifier_sha256": audit["classifier_sha"],
        "window_level": 800.0, "window_width": 1600.0,
        "context_distance_mm": 5.0, "imgsz": 768,
        "detector_conf": 0.01, "nms_iou": 0.5,
        "detector_aggregation": "top10_percent_mean",
        "detector_raw_operating_point": RAW_THRESHOLD,
        "detector_calibration": "Run A monotonic piecewise linear raw threshold to official 0.5",
        "classifier_aggregation": "top5_mean",
        "alpha_detector": 0.75, "alpha_classifier": 0.25,
        "official_fracture_threshold": 0.5,
        "classifier_fixed_epochs": FIXED_EPOCHS,
    }
    (config_dir / "fracture_fusion.json").write_text(json.dumps(config, indent=2) + "\n")
    provenance = {
        "source_detector": str(FINAL_DEPLOYMENT),
        "source_classifier": str(OUTPUT / "final_classifier.pt"),
        "fusion_oof_report_sha256": audit["fusion_report_sha"],
        "detector_only_oof": OOF_DETECTOR,
        "fusion_oof": OOF_FUSION,
        "warning": "The fusion improved fracture-specific OOF metrics but did not improve oracle-other-heads triage Macro-F1. It should therefore be treated as a fracture-recall-oriented candidate, not as proven improvement in the official competition primary metric.",
        "detector_transform_limit": "The full-OOF Run A deployment top10-percent/raw-threshold mapping is apparent-selected. Fusion OOF used fold-specific cross-fitted detector probabilities; its metric is not direct validation of this final global mapping plus FULL169 classifier.",
        "channel_limit": "The completed classifier experiment trained from RGB-converted PNGs but inferred direct 2.5D HU-channel arrays. This asymmetric channel convention is deliberately preserved for protocol fidelity, not claimed optimal.",
        "evaluation_scope": "Oracle-other-heads uses ground-truth ICH/MLS; it is not final submission Macro-F1. Fixed test was not used for handoff selection.",
    }
    (DESTINATION / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    readme = ("# Fracture-only fusion handoff\n\n"
              "This component returns only `fracture_prob`; the integrating team supplies ICH volumes and MLS to official triage.\n\n"
              "Use `PYTHONPATH=src python -c 'from predictor import FractureFusionPredictor; "
              "print(FractureFusionPredictor().predict_study(\"/path/to/one/target-series-study\"))'` from this directory. "
              "Pass `target_series_uid` when the study directory contains multiple series; ambiguous series fail closed.\n\n"
              "See `PROVENANCE.json` for OOF scope and limitations. No fixed-test evidence is claimed. "
              "The model uses two separate forward passes; batch_size is configurable.\n")
    (DESTINATION / "README.md").write_text(readme)
    files = sorted(path for path in DESTINATION.rglob("*") if path.is_file())
    (DESTINATION / "SHA256SUMS.txt").write_text("".join(
        f"{sha256(path)}  {path.relative_to(DESTINATION)}\n" for path in files))
    if (sha256(model_dir / "fracture_detector.pt") != EXPECTED_DETECTOR_SHA256
            or sha256(model_dir / "fracture_classifier.pt") != audit["classifier_sha"]):
        raise RuntimeError("Copied model checksum mismatch")
    return DESTINATION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    print(build())


if __name__ == "__main__":
    main()
