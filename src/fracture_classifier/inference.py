"""Manual full-study OOF classifier inference on explicit held-out Studies only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch

from cache_full_study_oof import enforce_vram_guard
from data.dicom import read_slice
from data.windows import make_hu_input
from dicom_series_guard import discover_headers
from fracture_classifier.data import CACHE, EXPERIMENT, FINAL_DEPLOYMENT, ROOT, detector_checkpoints, sha256
from fracture_classifier.model import FractureSliceClassifier
from fracture_classifier.training import image_tensor


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def infer_fold(fold: int, device_name: str, batch: int, stage: str = "stage1") -> dict:
    if fold not in range(5) or batch <= 0:
        raise ValueError("Invalid Fold or batch")
    source = detector_checkpoints()[fold]
    fold_dir = EXPERIMENT.parent / f"run_a_fracture_classifier_fold{fold}"
    protocol = json.loads((fold_dir / "protocol.json").read_text())
    completion_path = fold_dir / "COMPLETE.json"
    if not completion_path.is_file():
        raise RuntimeError(f"Classifier training Fold {fold} is not complete")
    completion = json.loads(completion_path.read_text())
    if stage not in ("stage1", "stage2"):
        raise ValueError("Classifier stage must be stage1 or stage2")
    if stage == "stage2" and not protocol["stage2"]["enabled"]:
        raise RuntimeError("Stage 2 was not enabled for this classifier Fold")
    if stage == "stage2" and not completion.get("stage2_completed"):
        raise RuntimeError("Stage 2 has not completed for this classifier Fold")
    classifier_path = fold_dir / ("best_classifier.pt" if stage == "stage1" else "best_stage2_classifier.pt")
    if not classifier_path.is_file() or sha256(source) != protocol["source_detector_sha256"]:
        raise RuntimeError("Classifier or source detector checkpoint missing/drifted")
    if sha256(FINAL_DEPLOYMENT) != protocol["full169_deployment_sha256_at_start"]:
        raise RuntimeError("Protected FULL169 deployment checkpoint changed")
    if sha256(EXPERIMENT / "slice_label_manifest.csv") != protocol["slice_label_manifest_sha256"]:
        raise RuntimeError("Classifier slice-label manifest changed")
    marker = json.loads((CACHE / f"fold_{fold}/COMPLETE.json").read_text())
    cache_signature = marker.get("inference_signature", {})
    if (cache_signature.get("weights_sha256") != sha256(source)
            or cache_signature.get("score_generation_protocol") != "deployment_score"
            or (cache_signature.get("imgsz"), cache_signature.get("conf"),
                cache_signature.get("nms_iou")) != (768, 0.01, 0.5)):
        raise RuntimeError("Run A cache/checkpoint score-protocol provenance mismatch")
    expected = set(map(str, marker["expected_study_ids"]))
    if len(expected) != [33, 33, 32, 36, 35][fold]:
        raise RuntimeError("Unexpected held-out Study count")
    source_studies = sorted((CACHE / f"fold_{fold}/studies").glob("*.json"))
    if {path.stem for path in source_studies} != expected:
        raise RuntimeError("Full-study Run A cache cohort is not exact")
    labels = pd.read_csv(EXPERIMENT / "slice_label_manifest.csv",
                         dtype={"study_id": str, "patient_id": str, "sop_uid": str})
    assigned = labels[labels["fold"] == fold][["study_id", "patient_id"]].drop_duplicates()
    if assigned["study_id"].duplicated().any() or set(assigned["study_id"]) != expected:
        raise RuntimeError("Classifier manifest Study/patient assignment is not exact")
    expected_patient = dict(zip(assigned["study_id"], assigned["patient_id"]))
    # Check resource availability before constructing the model.
    guard_device = device_name.split(":", 1)[1] if device_name.startswith("cuda:") else device_name
    enforce_vram_guard(guard_device, force=False)
    saved = torch.load(classifier_path, map_location="cpu", weights_only=False)
    if saved["source_detector_sha256"] != protocol["source_detector_sha256"] or saved["stage"] != (1 if stage == "stage1" else 2):
        raise RuntimeError("Classifier/source-detector hash mismatch")
    model = FractureSliceClassifier.from_detector(source)
    model.load_state_dict(saved["model_state"], strict=True)
    model.to(device_name).eval()
    output = EXPERIMENT / ("slice_predictions" if stage == "stage1" else "slice_predictions_stage2") / f"fold_{fold}"
    signature = {"fold": fold, "stage": stage, "classifier_sha256": sha256(classifier_path),
                 "source_detector_sha256": sha256(source), "imgsz": 768,
                 "context_distance_mm": 5.0, "window_level": 800.0, "window_width": 1600.0,
                 "source_full_study_cache": str(CACHE)}
    completed = 0
    with torch.inference_mode():
        for source_path in source_studies:
            source_payload = json.loads(source_path.read_text())
            study, patient = str(source_payload["series_id"]), str(source_payload["patient_id"])
            if int(source_payload["fold"]) != fold or study not in expected or patient != expected_patient[study]:
                raise RuntimeError("Unexpected Study reached classifier inference")
            sops = [str(row["sop_uid"]) for row in source_payload["slices"]]
            if not sops or len(sops) != len(set(sops)):
                raise RuntimeError("Empty/duplicate cached SOP list")
            target = output / "studies" / f"{study}.json"
            if target.is_file():
                previous = json.loads(target.read_text())
                if previous.get("signature") != signature or previous.get("sop_uids") != sops:
                    raise RuntimeError("Existing classifier Study cache protocol/coverage mismatch")
                completed += 1
                continue
            headers = discover_headers(ROOT / "iaaa-contest-bct/Data/training" / study)
            by_sop = {}
            for header in headers:
                uid = str(header["SOPInstanceUID"])
                if uid in by_sop:
                    raise RuntimeError(f"Duplicate DICOM SOP in Study {study}")
                by_sop[uid] = header
            if not set(sops) <= set(by_sop):
                raise RuntimeError(f"Cached SOP missing on disk for Study {study}")
            records = [read_slice(Path(by_sop[sop]["filesystem_path"]), study) for sop in sops]
            if [str(record.sop_uid) for record in records] != sops:
                raise RuntimeError("DICOM SOP sequence differs from Run A cache")
            hu = [record.hu for record in records]
            positions = [record.physical_position for record in records]
            mono = [record.photometric_interpretation == "MONOCHROME1" for record in records]
            probabilities = []
            for start in range(0, len(records), batch):
                images = [make_hu_input(
                    hu, index, level=800.0, width=1600.0, mode="2.5d",
                    boundary_mode="repeat", monochrome1=mono,
                    physical_positions=positions, context_distance_mm=5.0,
                ) for index in range(start, min(start + batch, len(records)))]
                tensor = torch.stack([image_tensor(image) for image in images]).to(device_name)
                probabilities.extend(model.probabilities(tensor).cpu().tolist())
            if len(probabilities) != len(sops) or not all(0.0 <= p <= 1.0 for p in probabilities):
                raise RuntimeError("Invalid classifier Study predictions")
            _write_json(target, {"fold": fold, "study_id": study, "patient_id": patient,
                                 "signature": signature, "sop_uids": sops,
                                 "slice_probabilities": probabilities})
            completed += 1
    _write_json(output / "COMPLETE.json", {"fold": fold, "studies": completed,
                                           "expected_study_ids": sorted(expected), "signature": signature})
    return {"fold": fold, "studies": completed, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
    args = parser.parse_args()
    for fold in sorted(set(args.fold)):
        print(json.dumps(infer_fold(fold, args.device, args.batch, args.stage)), flush=True)


if __name__ == "__main__":
    main()
