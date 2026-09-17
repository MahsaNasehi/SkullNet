"""Template copied as handoff/src/predictor.py after FULL169 classifier finalization.

This is a fracture-only component: predict_study returns one probability, not
the competition's final seven-key triage output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from classifier_model import FractureSliceClassifier
from detector_runtime import (Detector, aggregate_top10_percent_mean, make_hu_input,
                              read_slice, rescale_for_macro_f1, sort_records)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def top5_mean(scores: list[float]) -> float:
    if not scores or not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in scores):
        raise ValueError("Expected non-empty finite classifier slice probabilities")
    return float(np.mean(sorted(scores, reverse=True)[:5]))


def fuse_probabilities(detector_probability: float, classifier_probability: float) -> float:
    if not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in
               (detector_probability, classifier_probability)):
        raise ValueError("Fusion inputs must be probabilities")
    return float(np.clip(0.75 * detector_probability + 0.25 * classifier_probability, 0.0, 1.0))


def load_target_records(study_dir: str | Path, target_series_uid: str | None = None):
    """Include every valid DICOM in one target series, regardless of metadata/JSON."""
    import pydicom

    root = Path(study_dir)
    paths = sorted(path for path in root.rglob("*") if path.is_file()
                   and (path.suffix.lower() == ".dcm" or not path.suffix))
    if not paths:
        raise FileNotFoundError(f"No DICOM files in {root}")
    by_series: dict[str, list[Path]] = {}
    for path in paths:
        header = pydicom.dcmread(path, stop_before_pixels=True)
        series_uid = str(getattr(header, "SeriesInstanceUID", ""))
        if not series_uid:
            raise RuntimeError(f"DICOM lacks SeriesInstanceUID: {path}")
        by_series.setdefault(series_uid, []).append(path)
    if target_series_uid is None:
        if len(by_series) != 1:
            raise RuntimeError("Multiple DICOM series present; pass target_series_uid explicitly")
        target_series_uid = next(iter(by_series))
    if target_series_uid not in by_series:
        raise RuntimeError(f"Target SeriesInstanceUID not present: {target_series_uid}")
    records = sort_records([read_slice(path, root.name) for path in by_series[target_series_uid]])
    sops = [str(record.sop_uid) for record in records]
    if len(sops) != len(set(sops)):
        raise RuntimeError("Duplicate SOPInstanceUID in target DICOM series")
    return records


def _classifier_tensor(image: np.ndarray) -> torch.Tensor:
    # Preserve the completed OOF classifier inference's channel convention:
    # make_hu_input array directly, without BGR->RGB reversal. The reviewed
    # PNG training path reversed channels; this asymmetry is documented in the
    # provenance report and must not be silently changed in the handoff.
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected three-channel 2.5D HU input")
    if image.shape[:2] != (768, 768):
        image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)


class FractureFusionPredictor:
    def __init__(self, package_root: str | Path | None = None, *, device: str | None = None,
                 batch_size: int = 8):
        self.root = Path(package_root) if package_root else Path(__file__).resolve().parents[1]
        self.config = json.loads((self.root / "config/fracture_fusion.json").read_text())
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cfg = self.config
        if (cfg["imgsz"], cfg["window_level"], cfg["window_width"], cfg["context_distance_mm"],
            cfg["detector_conf"], cfg["nms_iou"], cfg["classifier_aggregation"],
            cfg["alpha_detector"], cfg["alpha_classifier"], cfg["official_fracture_threshold"]) != (
                768, 800.0, 1600.0, 5.0, 0.01, 0.5, "top5_mean", 0.75, 0.25, 0.5):
            raise RuntimeError("Handoff protocol configuration drift")
        self.detector_path = self.root / "models/fracture_detector.pt"
        self.classifier_path = self.root / "models/fracture_classifier.pt"
        if sha256(self.detector_path) != cfg["detector_sha256"] or sha256(self.classifier_path) != cfg["classifier_sha256"]:
            raise RuntimeError("Handoff checkpoint checksum mismatch")
        detector_device = 0 if self.device == "cuda:0" else self.device
        self.detector = Detector(self.detector_path, confidence=0.01, iou=0.5,
                                 device=detector_device, fp16=self.device != "cpu", image_size=768)
        checkpoint = torch.load(self.classifier_path, map_location="cpu", weights_only=False)
        if checkpoint.get("source_detector_sha256") != cfg["detector_sha256"] or checkpoint.get("epoch_one_based") != 9:
            raise RuntimeError("Classifier checkpoint is not the validated FULL169 final epoch")
        self.classifier = FractureSliceClassifier.from_detector(self.detector_path)
        self.classifier.load_state_dict(checkpoint["model_state"], strict=True)
        self.classifier.to(self.device).eval()
        self.last_slice_count = 0

    def predict_study(self, dicom_study: str | Path, *, target_series_uid: str | None = None) -> float:
        records = load_target_records(dicom_study, target_series_uid)
        self.last_slice_count = len(records)
        hu = [record.hu for record in records]
        positions = [record.physical_position for record in records]
        mono = [record.photometric_interpretation == "MONOCHROME1" for record in records]
        detector_scores, classifier_scores = [], []
        with torch.inference_mode():
            for start in range(0, len(records), self.batch_size):
                stop = min(start + self.batch_size, len(records))
                images = [make_hu_input(
                    hu, index, level=800.0, width=1600.0, mode="2.5d",
                    boundary_mode="repeat", monochrome1=mono,
                    physical_positions=positions, context_distance_mm=5.0,
                ) for index in range(start, stop)]
                detections = self.detector.predict_batch(images)
                detector_scores.extend(float(item["max_confidence"]) for item in detections)
                tensor = torch.stack([_classifier_tensor(image) for image in images]).to(self.device)
                classifier_scores.extend(self.classifier.probabilities(tensor).cpu().tolist())
                del tensor, images, detections
        if len(detector_scores) != len(records) or len(classifier_scores) != len(records):
            raise RuntimeError("Incomplete full-study fracture inference")
        detector_raw = aggregate_top10_percent_mean(detector_scores)
        detector_probability = rescale_for_macro_f1(detector_raw, cfg_threshold(self.config))
        classifier_probability = top5_mean(classifier_scores)
        return fuse_probabilities(detector_probability, classifier_probability)


def cfg_threshold(config: dict) -> float:
    return float(config["detector_raw_operating_point"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--target-series-uid")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    predictor = FractureFusionPredictor(args.package_root, device=args.device, batch_size=args.batch_size)
    probability = predictor.predict_study(args.study, target_series_uid=args.target_series_uid)
    print(json.dumps({"fracture_prob": probability, "slices": predictor.last_slice_count}))


if __name__ == "__main__":
    main()
