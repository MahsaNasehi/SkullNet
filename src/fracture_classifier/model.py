"""A detached slice classifier using the historical YOLO26s-P2 P5 backbone feature."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


FEATURE_LAYER = 10  # final backbone C2PSA, immediately before the detector neck
FEATURE_CHANNELS = 512  # Run A YOLO26s-P2 scale-s output channels


class FractureSliceClassifier(nn.Module):
    def __init__(self, backbone: nn.Sequential, feature_channels: int = FEATURE_CHANNELS):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_channels), nn.Linear(feature_channels, 128),
            nn.SiLU(), nn.Dropout(0.10), nn.Linear(128, 1),
        )
        self.stage = 1
        self.freeze_backbone()

    @classmethod
    def from_detector(cls, checkpoint_path: Path) -> "FractureSliceClassifier":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        detector = checkpoint.get("model")
        if detector is None or len(detector.model) <= FEATURE_LAYER:
            raise RuntimeError("Run A detector checkpoint has no expected backbone")
        layers = list(detector.model[:FEATURE_LAYER + 1])
        if any(layer.f != -1 for layer in layers):
            raise RuntimeError("Selected detector backbone is not a sequential feature path")
        if type(layers[-1]).__name__ != "C2PSA" or layers[-1].cv2.conv.out_channels != FEATURE_CHANNELS:
            raise RuntimeError("Run A P5 feature architecture changed")
        # Ultralytics strips Run A checkpoints to FP16 for deployment. Head-only
        # AdamW training uses FP32 inputs/parameters; cast the detached copy only.
        return cls(nn.Sequential(*layers).float())

    def freeze_backbone(self) -> None:
        self.stage = 1
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.head.requires_grad_(True)

    def unfreeze_last_stage(self) -> None:
        self.stage = 2
        self.backbone.requires_grad_(False)
        self.backbone[FEATURE_LAYER].requires_grad_(True)
        self.head.requires_grad_(True)

    def train(self, mode: bool = True) -> "FractureSliceClassifier":
        super().train(mode)
        # Frozen BatchNorm statistics must never drift during head-only training.
        if self.stage == 1:
            self.backbone.eval()
        else:
            for block in self.backbone[:FEATURE_LAYER]:
                block.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.stage == 1:
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)
        return self.head(self.pool(features).flatten(1)).flatten()

    def probabilities(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(images))
