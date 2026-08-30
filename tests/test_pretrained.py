import json
from pathlib import Path

import pytest

from fracture.utils.pretrained import (
    resolve_initialization,
    resolve_pretrained_checkpoint,
    sha256_file,
    verify_p2_architecture,
)


def test_sha256_file(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"fracture")
    assert sha256_file(path) == "c7935748bd98de23e6ed5e5bed9bfa043ff6bc0053e304ba2410f8751b22dd05"


def test_pretrained_download_once_and_cache_reuse(tmp_path):
    calls = []

    def downloader(name: str, destination: Path) -> Path:
        calls.append(name)
        destination.write_bytes(b"x" * 100_001)
        return destination

    destination = tmp_path / "cache" / "yolo11s.pt"
    path, metadata = resolve_pretrained_checkpoint(
        model_name="yolo11s.pt",
        destination=destination,
        download_if_missing=True,
        downloader=downloader,
        ultralytics_version="test",
    )
    cached, second_metadata = resolve_pretrained_checkpoint(
        model_name="yolo11s.pt",
        destination=destination,
        download_if_missing=False,
        downloader=downloader,
        ultralytics_version="test",
    )
    assert path == cached == destination.resolve()
    assert calls == ["yolo11s.pt"]
    assert metadata["sha256"] == second_metadata["sha256"] == sha256_file(destination)
    saved = json.loads(destination.with_suffix(".metadata.json").read_text())
    assert saved["pretraining_dataset"] == "COCO"
    assert saved["source"] == "official_ultralytics_assets"


def test_missing_pretrained_fails_when_download_disabled(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_pretrained_checkpoint(
            model_name="yolo11s.pt",
            destination=tmp_path / "yolo11s.pt",
            download_if_missing=False,
        )


def test_offline_mode_never_attempts_download(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="YOLO_OFFLINE"):
        resolve_pretrained_checkpoint(
            model_name="yolo11s.pt",
            destination=tmp_path / "yolo11s.pt",
            download_if_missing=True,
        )


def test_initialization_modes_are_mutually_exclusive(tmp_path):
    checkpoint = tmp_path / "local.pt"
    checkpoint.write_bytes(b"local")
    with pytest.raises(ValueError, match="cannot also specify"):
        resolve_initialization({
            "initial_weights_path": str(checkpoint),
            "initialization": {
                "mode": "pretrained",
                "pretrained_source": "ultralytics_coco",
                "pretrained_model": "yolo11s.pt",
                "pretrained_path": str(tmp_path / "official.pt"),
            },
        })
    assert resolve_initialization({"initialization": {"mode": "random"}}).mode == "random"


def test_live_target_has_p2_through_p5_strides():
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    model = YOLO("configs/yolo11s_p2.yaml")
    assert verify_p2_architecture(model) == [4.0, 8.0, 16.0, 32.0]


def test_fold0_ab_configs_only_change_initialization():
    import yaml

    with Path("configs/fracture_25d_p2_ab_local_transfer.yaml").open() as stream:
        local = yaml.safe_load(stream)
    with Path("configs/fracture_25d_p2_pretrained.yaml").open() as stream:
        pretrained = yaml.safe_load(stream)
    local_model = local.pop("model")
    pretrained_model = pretrained.pop("model")
    assert local == pretrained
    assert local_model["weights"] == pretrained_model["weights"] == "configs/yolo11s_p2.yaml"
    assert local_model["initialization"]["mode"] == "local_transfer"
    assert pretrained_model["initialization"]["mode"] == "pretrained"
