"""Official Ultralytics checkpoint caching and initialization verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fracture.utils.config import load_config


ALLOWED_INITIALIZATION_MODES = {"random", "local_transfer", "pretrained"}
OFFICIAL_ULTRALYTICS_REPO = "ultralytics/assets"
OFFICIAL_ULTRALYTICS_RELEASE = "v8.4.0"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_offline() -> bool:
    return os.environ.get("YOLO_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _official_download(model_name: str, destination: Path) -> Path:
    if _is_offline():
        raise RuntimeError(
            f"YOLO_OFFLINE is enabled and {destination} is missing; cache {model_name} during development first"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset

    downloaded = Path(
        attempt_download_asset(
            destination,
            repo=OFFICIAL_ULTRALYTICS_REPO,
            release=OFFICIAL_ULTRALYTICS_RELEASE,
        )
    )
    if downloaded.resolve() != destination.resolve() and downloaded.is_file():
        destination.write_bytes(downloaded.read_bytes())
    return destination


def resolve_pretrained_checkpoint(
    *,
    model_name: str,
    destination: str | Path,
    download_if_missing: bool,
    downloader: Callable[[str, Path], Path] | None = None,
    ultralytics_version: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve one persistent official checkpoint and record verifiable provenance."""
    if Path(model_name).name != model_name or not model_name.endswith(".pt"):
        raise ValueError(f"pretrained_model must be a plain .pt filename, got {model_name!r}")
    path = Path(destination).expanduser().resolve()
    metadata_path = path.with_suffix(".metadata.json")
    downloaded_now = False
    if not path.is_file():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Pretrained checkpoint is not cached: {path}. Enable download_if_missing during development."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = (downloader or _official_download)(model_name, path)
        if Path(resolved).resolve() != path:
            raise RuntimeError(f"Downloader returned unexpected path: {resolved}; expected {path}")
        downloaded_now = True
    if path.stat().st_size < 100_000:
        raise ValueError(f"Checkpoint is implausibly small ({path.stat().st_size} bytes): {path}")

    digest = sha256_file(path)
    existing: dict[str, Any] = {}
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = existing.get("sha256")
        if expected and expected != digest:
            raise ValueError(f"Cached checkpoint SHA256 mismatch: expected {expected}, got {digest}")

    if ultralytics_version is None:
        try:
            import ultralytics

            ultralytics_version = ultralytics.__version__
        except ImportError:
            ultralytics_version = "unavailable"
    timestamp = (
        datetime.now(timezone.utc).isoformat()
        if downloaded_now
        else existing.get("download_timestamp")
    )
    metadata = {
        "name": model_name,
        "local_path": str(path),
        "source": "official_ultralytics_assets",
        "source_repository": OFFICIAL_ULTRALYTICS_REPO,
        "source_release": OFFICIAL_ULTRALYTICS_RELEASE,
        "pretraining_dataset": "COCO",
        "sha256": digest,
        "file_size_bytes": path.stat().st_size,
        "ultralytics_version": ultralytics_version,
        "download_timestamp": timestamp,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path, metadata


@dataclass(frozen=True)
class InitializationPlan:
    mode: str
    checkpoint_path: Path | None
    checkpoint_metadata: dict[str, Any] | None


def resolve_initialization(
    model_config: dict[str, Any],
    *,
    downloader: Callable[[str, Path], Path] | None = None,
    ultralytics_version: str | None = None,
) -> InitializationPlan:
    """Resolve exactly one of random, local-transfer, or official-pretrained initialization."""
    initialization = model_config.get("initialization") or {}
    mode = str(initialization.get("mode", "")).strip().lower()
    if not mode:  # Backward compatibility for historical configs.
        mode = "local_transfer" if model_config.get("initial_weights_path") else "random"
    if mode not in ALLOWED_INITIALIZATION_MODES:
        raise ValueError(
            f"model.initialization.mode must be one of {sorted(ALLOWED_INITIALIZATION_MODES)}, got {mode!r}"
        )
    local_path = model_config.get("initial_weights_path")
    pretrained_path = initialization.get("pretrained_path")

    if mode == "random":
        if local_path or pretrained_path:
            raise ValueError("random initialization cannot specify local or pretrained checkpoint paths")
        return InitializationPlan(mode, None, None)

    if mode == "local_transfer":
        if pretrained_path:
            raise ValueError("local_transfer cannot specify initialization.pretrained_path")
        if not local_path:
            raise ValueError("local_transfer requires model.initial_weights_path")
        path = Path(local_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local transfer checkpoint not found: {path}")
        metadata = {
            "name": path.name,
            "local_path": str(path),
            "source": "local_fracture_checkpoint",
            "pretraining_dataset": "IAAA_fold0_training_partition",
            "sha256": sha256_file(path),
            "file_size_bytes": path.stat().st_size,
        }
        return InitializationPlan(mode, path, metadata)

    if local_path:
        raise ValueError("pretrained initialization cannot also specify model.initial_weights_path")
    if initialization.get("pretrained_source") != "ultralytics_coco":
        raise ValueError("pretrained initialization currently supports only pretrained_source=ultralytics_coco")
    model_name = str(initialization.get("pretrained_model") or "")
    if not model_name or not pretrained_path:
        raise ValueError("pretrained initialization requires pretrained_model and pretrained_path")
    path, metadata = resolve_pretrained_checkpoint(
        model_name=model_name,
        destination=pretrained_path,
        download_if_missing=bool(initialization.get("download_if_missing", False)),
        downloader=downloader,
        ultralytics_version=ultralytics_version,
    )
    return InitializationPlan(mode, path, metadata)


def verify_transfer(model: Any, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load compatible tensors and verify the live target still has four P2-P5 scales."""
    import torch
    from ultralytics.nn.tasks import load_checkpoint

    target = model.model
    before = {name: tensor.detach().cpu().clone() for name, tensor in target.state_dict().items()}
    source, _ = load_checkpoint(checkpoint_path, device="cpu")
    source_state = source.state_dict()
    compatible = {
        name
        for name, tensor in target.state_dict().items()
        if name in source_state and tuple(tensor.shape) == tuple(source_state[name].shape)
    }
    model.load(str(checkpoint_path))
    after = {name: tensor.detach().cpu() for name, tensor in target.state_dict().items()}
    changed = {name for name in compatible if not torch.equal(before[name], after[name])}
    exact_matches = {name for name in compatible if torch.equal(after[name], source_state[name].detach().cpu())}
    backbone_key = "model.0.conv.weight"
    backbone_verified = backbone_key in exact_matches and backbone_key in changed

    strides = verify_p2_architecture(model)
    parameter_names = set(dict(target.named_parameters()))
    transferred_parameters = sum(after[name].numel() for name in exact_matches if name in parameter_names)
    target_parameters = sum(parameter.numel() for parameter in target.parameters())
    report = {
        "target_items": len(after),
        "compatible_items": len(compatible),
        "transferred_items": len(exact_matches),
        "changed_compatible_items": len(changed),
        "newly_initialized_items": len(after) - len(exact_matches),
        "transfer_fraction": len(exact_matches) / len(after),
        "target_parameters": target_parameters,
        "transferred_parameters": transferred_parameters,
        "newly_initialized_parameters": target_parameters - transferred_parameters,
        "parameter_transfer_fraction": transferred_parameters / target_parameters,
        "backbone_tensor_verified": backbone_verified,
        "backbone_tensor_key": backbone_key,
        "detection_strides": strides,
        "detection_scales": len(strides),
        "p2_stride4_active": any(abs(value - 4.0) < 1e-6 for value in strides),
    }
    if not backbone_verified:
        raise RuntimeError("Pretrained transfer sanity check failed for model.0.conv.weight")
    if report["transfer_fraction"] < 0.2:
        raise RuntimeError(f"Only a trivial fraction of target tensors transferred: {report}")
    return report


def verify_p2_architecture(model: Any) -> list[float]:
    """Fail unless the live detector exposes the required P2-P5 prediction scales."""
    target = model.model
    stride = getattr(target, "stride", None)
    if stride is None:
        raise RuntimeError("Target model does not expose detection strides")
    strides = [float(value) for value in stride.detach().cpu().tolist()]
    if len(strides) != 4 or not any(abs(value - 4.0) < 1e-6 for value in strides):
        raise RuntimeError(f"P2 architecture verification failed: strides={strides}")
    if strides != [4.0, 8.0, 16.0, 32.0]:
        raise RuntimeError(f"Unexpected P2-P5 stride ordering: {strides}")
    return strides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify-transfer", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    plan = resolve_initialization(cfg["model"])
    payload = {"mode": plan.mode, "checkpoint_path": str(plan.checkpoint_path) if plan.checkpoint_path else None}
    if plan.checkpoint_metadata:
        payload["checkpoint_metadata"] = plan.checkpoint_metadata
    if args.verify_transfer:
        os.environ["YOLO_OFFLINE"] = "1"
        from ultralytics import YOLO

        model = YOLO(cfg["model"]["weights"])
        payload["detection_strides"] = verify_p2_architecture(model)
        payload["p2_stride4_active"] = True
        payload["transfer"] = verify_transfer(model, plan.checkpoint_path) if plan.checkpoint_path else None
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
