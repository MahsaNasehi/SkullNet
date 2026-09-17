"""Select the best 5-fold detector checkpoint from held-out study metrics."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Matches OOF selection priority in generate_oof / final_metrics.json.
DEFAULT_PRIORITY: tuple[str, ...] = (
    "pr_auc",
    "sensitivity_at_0_5",
    "auroc",
    "brier",
    "isolated_fracture_qwk",
)

# Lower is better for these keys; everything else is higher-is-better.
_LOWER_IS_BETTER = frozenset({"brier", "log_loss"})

DEFAULT_RUN_NAME = "v5_hu800_ww1600_original"
DEFAULT_METRICS_GLOB = "reports/fold*_v5_hu800_ww1600_original_metrics.json"


@dataclass(frozen=True)
class FoldCandidate:
    fold: int
    metrics_path: str
    weights_path: str
    pr_auc: float
    auroc: float
    sensitivity_at_0_5: float
    specificity_at_0_5: float
    f1_at_0_5: float
    brier: float
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _sort_key(metrics: dict[str, Any], priority: tuple[str, ...]) -> tuple[float, ...]:
    key: list[float] = []
    for name in priority:
        value = float(metrics.get(name, float("-inf") if name not in _LOWER_IS_BETTER else float("inf")))
        key.append(-value if name in _LOWER_IS_BETTER else value)
    return tuple(key)


def default_weights_path(repo_root: Path, fold: int, run_name: str = DEFAULT_RUN_NAME) -> Path:
    return repo_root / "outputs" / f"fold_{fold}" / run_name / "weights" / "best.pt"


def discover_fold_candidates(
    repo_root: str | Path,
    *,
    metrics_glob: str = DEFAULT_METRICS_GLOB,
    run_name: str = DEFAULT_RUN_NAME,
    require_weights: bool = True,
) -> list[FoldCandidate]:
    root = Path(repo_root).resolve()
    candidates: list[FoldCandidate] = []
    for metrics_path in sorted(root.glob(metrics_glob)):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        fold = int(metrics["fold"])
        weights = default_weights_path(root, fold, run_name)
        if require_weights and not weights.is_file():
            raise FileNotFoundError(f"Missing detector weights for fold {fold}: {weights}")
        candidates.append(
            FoldCandidate(
                fold=fold,
                metrics_path=str(metrics_path),
                weights_path=str(weights),
                pr_auc=float(metrics.get("pr_auc", 0.0)),
                auroc=float(metrics.get("auroc", 0.0)),
                sensitivity_at_0_5=float(metrics.get("sensitivity_at_0_5", 0.0)),
                specificity_at_0_5=float(metrics.get("specificity_at_0_5", 0.0)),
                f1_at_0_5=float(metrics.get("f1_at_0_5", 0.0)),
                brier=float(metrics.get("brier", 1.0)),
                metrics=metrics,
            )
        )
    if not candidates:
        raise FileNotFoundError(f"No fold metrics matched {root / metrics_glob}")
    return candidates


def select_best_fold(
    repo_root: str | Path,
    *,
    fold: int | None = None,
    metrics_glob: str = DEFAULT_METRICS_GLOB,
    run_name: str = DEFAULT_RUN_NAME,
    priority: tuple[str, ...] = DEFAULT_PRIORITY,
) -> FoldCandidate:
    """Return one fold's detector: explicit fold, or best by OOF priority."""
    candidates = discover_fold_candidates(repo_root, metrics_glob=metrics_glob, run_name=run_name)
    if fold is not None:
        selected = next((item for item in candidates if item.fold == fold), None)
        if selected is None:
            available = ", ".join(str(item.fold) for item in candidates)
            raise ValueError(f"Fold {fold} not found in metrics. Available: {available}")
        return selected
    return max(candidates, key=lambda item: _sort_key(item.metrics, priority))


def resolve_inference_artifacts(
    repo_root: str | Path,
    *,
    fold: int | None = None,
    weights: str | Path | None = None,
    aggregator: str | Path | None = None,
    calibrator: str | Path | None = None,
    prefer_calibration: bool = False,
) -> dict[str, Any]:
    """Resolve detector + OOF aggregator paths for convenient inference."""
    root = Path(repo_root).resolve()
    if weights is not None:
        weights_path = Path(weights).resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Detector weights not found: {weights_path}")
        selected = None
        fold_id = fold
    else:
        selected = select_best_fold(root, fold=fold)
        weights_path = Path(selected.weights_path)
        fold_id = selected.fold

    if aggregator is not None:
        aggregator_path = Path(aggregator).resolve()
    else:
        preferred = [
            root / "models" / "oof_original_no_calibration" / "aggregator.joblib",
            root / "reports" / "oof_original" / "models" / "aggregator.joblib",
            root / "models" / "aggregator.joblib",
        ]
        aggregator_path = next((path for path in preferred if path.is_file()), None)

    if calibrator is not None:
        calibrator_path = Path(calibrator).resolve()
    elif prefer_calibration:
        preferred_cal = [
            root / "reports" / "oof_original" / "models" / "calibrator.joblib",
            root / "models" / "calibrator.joblib",
        ]
        calibrator_path = next((path for path in preferred_cal if path.is_file()), None)
    else:
        # Default: no calibrator. Platt OOF calibration collapsed sensitivity at 0.5.
        calibrator_path = None

    return {
        "repo_root": str(root),
        "fold": fold_id,
        "selected": selected.to_dict() if selected is not None else None,
        "detector_weights": str(weights_path),
        "aggregator_path": str(aggregator_path) if aggregator_path else None,
        "calibrator_path": str(calibrator_path) if calibrator_path else None,
        "aggregation_method": "top3_mean",
        "calibration_method": "none" if calibrator_path is None else "platt",
        "input_mode": "2.5d",
        "image_size": 768,
        "window_level": 800.0,
        "window_width": 1600.0,
        "context_distance_mm": 5.0,
        "batch_size": 2,
        "confidence": 0.01,
        "nms_iou": 0.5,
    }
