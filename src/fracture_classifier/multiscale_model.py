"""Frozen Run A P3+P4+P5 GAP slice classifier; Stage 1 only."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from fracture_classifier.feature_hooks import SPECS, FrozenBackboneHooks, verify_backbone_graph


class MultiScaleFractureClassifier(nn.Module):
    def __init__(self, backbone: nn.Sequential):
        super().__init__()
        # The experiment exposes no pooling option: future GAP+top-k work can
        # replace _pool in a separate ablation without changing feature hooks.
        self.pooling_mode = "gap"
        self.backbone = backbone.float()
        self.hooks = FrozenBackboneHooks(self.backbone)
        self.branches = nn.ModuleDict({
            spec.name: nn.Sequential(nn.LayerNorm(spec.channels),
                                     nn.Linear(spec.channels, 128), nn.SiLU())
            for spec in SPECS
        })
        self.head = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 128),
                                  nn.SiLU(), nn.Dropout(0.10), nn.Linear(128, 1))
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @classmethod
    def from_detector(cls, checkpoint_path: Path) -> "MultiScaleFractureClassifier":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        detector = checkpoint.get("model")
        if detector is None:
            raise RuntimeError("Run A checkpoint has no detector model")
        verify_backbone_graph(detector)
        return cls(nn.Sequential(*list(detector.model[:11])))

    def train(self, mode: bool = True) -> "MultiScaleFractureClassifier":
        super().train(mode)
        self.backbone.eval()  # Frozen BatchNorm statistics must never drift.
        return self

    def _pool(self, feature: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode != "gap":
            raise RuntimeError("Only GAP pooling is implemented in this experiment")
        return F.adaptive_avg_pool2d(feature, 1).flatten(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.hooks.extract(self.backbone, images)
        pooled = [self.branches[spec.name](self._pool(features[spec.name]))
                  for spec in SPECS]
        logits = self.head(torch.cat(pooled, dim=1)).flatten()
        if logits.shape != (images.shape[0],):
            raise RuntimeError("Multi-scale classifier must return one logit per slice")
        return logits

    def probabilities(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(images))
