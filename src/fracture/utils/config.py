"""Strict configuration loading without hidden defaults or downloads."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(deepcopy(config), stream, sort_keys=False)


def require_path(config: dict[str, Any], section: str, key: str) -> Path:
    value = config.get(section, {}).get(key)
    if not value:
        raise ValueError(f"Required configuration value is missing: {section}.{key}")
    return Path(value).expanduser().resolve()

