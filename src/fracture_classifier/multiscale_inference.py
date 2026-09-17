"""Manual held-out full-study inference for the frozen P3+P4+P5 classifier."""
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
from fracture_classifier.multiscale_model import MultiScaleFractureClassifier
from fracture_classifier.multiscale_training import OUTPUT_PREFIX
from fracture_classifier.training import image_tensor


OOF_OUTPUT = ROOT / "outputs/run_a_fracture_multiscale_oof"
FOLD_COUNTS = (33, 33, 32, 36, 35)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def infer_fold(fold: int, device_name: str = "cuda:0", batch: int = 8) -> dict:
    if fold not in range(5) or batch <= 0:
        raise ValueError("Invalid Fold or batch")
    fold_dir = ROOT / f"outputs/{OUTPUT_PREFIX}{fold}"
    protocol = json.loads((fold_dir / "protocol.json").read_text())
    completion = json.loads((fold_dir / "COMPLETE.json").read_text())
    if (int(protocol["fold"]) != fold or completion.get("stage1_completed") is not True
            or completion.get("epochs_completed") != 10):
        raise RuntimeError("Multi-scale Fold training did not complete")
    source = detector_checkpoints()[fold]
    classifier = fold_dir / "best_classifier.pt"
    label_manifest = EXPERIMENT / "slice_label_manifest.csv"
    if (not classifier.is_file() or sha256(source) != protocol["source_detector_sha256"]
            or sha256(label_manifest) != protocol["slice_label_manifest_sha256"]
            or sha256(FINAL_DEPLOYMENT) != protocol["full169_deployment_sha256_at_start"]):
        raise RuntimeError("Run A detector/classifier/label/FULL169 provenance changed")
    marker = json.loads((CACHE / f"fold_{fold}/COMPLETE.json").read_text())
    source_signature = marker.get("inference_signature", {})
    if (source_signature.get("weights_sha256") != sha256(source)
            or source_signature.get("score_generation_protocol") != "deployment_score"
            or (source_signature.get("imgsz"), source_signature.get("conf"),
                source_signature.get("nms_iou")) != (768, 0.01, 0.5)):
        raise RuntimeError("Run A full-study cache protocol does not match historical detector")
    expected = set(map(str, marker["expected_study_ids"]))
    source_studies = sorted((CACHE / f"fold_{fold}/studies").glob("*.json"))
    if len(expected) != FOLD_COUNTS[fold] or {path.stem for path in source_studies} != expected:
        raise RuntimeError("Run A explicit held-out Study cohort is incomplete")
    labels = pd.read_csv(label_manifest, dtype={"study_id": str, "patient_id": str, "sop_uid": str})
    assigned = labels[labels["fold"] == fold][["study_id", "patient_id"]].drop_duplicates()
    if assigned.study_id.duplicated().any() or set(assigned.study_id) != expected:
        raise RuntimeError("Multi-scale held-out Study/patient assignment changed")
    expected_patient = dict(zip(assigned.study_id, assigned.patient_id))
    if labels.groupby("patient_id").fold.nunique().max() != 1:
        raise RuntimeError("Cross-fold patient leakage in classifier manifest")
    output = OOF_OUTPUT / "slice_predictions" / f"fold_{fold}"
    existing = {path.stem for path in (output / "studies").glob("*.json")}
    if existing - expected:
        raise RuntimeError("Unexpected Study found in multi-scale output cache")
    guard_device = device_name.split(":", 1)[1] if device_name.startswith("cuda:") else device_name
    enforce_vram_guard(guard_device, force=False)
    saved = torch.load(classifier, map_location="cpu", weights_only=False)
    if (saved["source_detector_sha256"] != protocol["source_detector_sha256"]
            or saved["slice_label_manifest_sha256"] != protocol["slice_label_manifest_sha256"]
            or saved["stage"] != 1 or not 1 <= int(saved["epoch_one_based"]) <= 10):
        raise RuntimeError("Multi-scale best checkpoint metadata mismatch")
    model = MultiScaleFractureClassifier.from_detector(source)
    model.load_state_dict(saved["model_state"], strict=True)
    model.to(device_name).eval()
    signature = {"fold": fold, "stage": "frozen_multiscale_gap",
                 "classifier_sha256": sha256(classifier),
                 "source_detector_sha256": sha256(source),
                 "slice_label_manifest_sha256": sha256(label_manifest),
                 "imgsz": 768, "context_distance_mm": 5.0,
                 "window_level": 800.0, "window_width": 1600.0,
                 "source_full_study_cache": str(CACHE)}
    completed = 0
    with torch.inference_mode():
        for source_path in source_studies:
            source_payload = json.loads(source_path.read_text())
            study, patient = str(source_payload["series_id"]), str(source_payload["patient_id"])
            if int(source_payload["fold"]) != fold or study not in expected or patient != expected_patient[study]:
                raise RuntimeError("Unexpected Study/patient reached multi-scale inference")
            sops = [str(row["sop_uid"]) for row in source_payload["slices"]]
            if not sops or len(sops) != len(set(sops)):
                raise RuntimeError("Empty/duplicate SOP list in full-study source cache")
            target = output / "studies" / f"{study}.json"
            if target.is_file():
                previous = json.loads(target.read_text())
                if (previous.get("signature") != signature or previous.get("sop_uids") != sops
                        or len(previous.get("slice_probabilities", [])) != len(sops)):
                    raise RuntimeError("Existing multi-scale Study cache protocol/coverage mismatch")
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
                raise RuntimeError(f"Run A cached SOP missing from source Study {study}")
            records = [read_slice(Path(by_sop[sop]["filesystem_path"]), study) for sop in sops]
            if [str(record.sop_uid) for record in records] != sops:
                raise RuntimeError("Physical DICOM SOP ordering differs from Run A cache")
            hu = [record.hu for record in records]
            positions = [record.physical_position for record in records]
            mono = [record.photometric_interpretation == "MONOCHROME1" for record in records]
            probabilities = []
            for start in range(0, len(records), batch):
                images = [make_hu_input(hu, index, level=800.0, width=1600.0,
                                        mode="2.5d", boundary_mode="repeat",
                                        monochrome1=mono, physical_positions=positions,
                                        context_distance_mm=5.0)
                          for index in range(start, min(start + batch, len(records)))]
                tensor = torch.stack([image_tensor(image) for image in images]).to(device_name)
                probabilities.extend(model.probabilities(tensor).cpu().tolist())
            if len(probabilities) != len(sops) or not all(0 <= p <= 1 for p in probabilities):
                raise RuntimeError("Invalid multi-scale slice probabilities")
            _write_json(target, {"fold": fold, "study_id": study, "patient_id": patient,
                                 "signature": signature, "sop_uids": sops,
                                 "slice_probabilities": probabilities})
            completed += 1
    _write_json(output / "COMPLETE.json", {"fold": fold, "studies": completed,
                                            "expected_study_ids": sorted(expected),
                                            "signature": signature})
    return {"fold": fold, "studies": completed, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    for fold in sorted(set(args.fold)):
        print(json.dumps(infer_fold(fold, args.device, args.batch)), flush=True)


if __name__ == "__main__":
    main()
