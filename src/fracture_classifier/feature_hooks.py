"""Verified Run A *backbone* P3/P4/P5 hooks for the frozen ablation.

The neck P3/P4/P5 are deliberately not used: retaining backbone layer 10
keeps the P5 input identical to the historical single-scale classifier.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    layer: int
    module_type: str
    stride: int
    channels: int


SPECS = (
    FeatureSpec("P3", 4, "C3k2", 8, 256),
    FeatureSpec("P4", 6, "C3k2", 16, 256),
    FeatureSpec("P5", 10, "C2PSA", 32, 512),
)


def verify_backbone_graph(detector: nn.Module) -> None:
    """Fail closed if checkpoint graph is not the audited YOLO26s-P2 scale-s."""
    modules = getattr(detector, "model", None)
    if modules is None or len(modules) <= 29:
        raise RuntimeError("Run A detector graph is incomplete")
    if any(getattr(layer, "f", None) != -1 for layer in modules[:11]):
        raise RuntimeError("P3/P4/P5 backbone path is not sequential")
    for spec in SPECS:
        layer = modules[spec.layer]
        actual_channels = getattr(getattr(getattr(layer, "cv2", None), "conv", None), "out_channels", None)
        if type(layer).__name__ != spec.module_type or actual_channels != spec.channels:
            raise RuntimeError(f"{spec.name} layer/type/channels differ from audited Run A graph")
    detect = modules[29]
    if type(detect).__name__ != "Detect" or list(detect.f) != [19, 22, 25, 28]:
        raise RuntimeError("YOLO26s-P2 detector pyramid graph changed")
    if [int(value) for value in detect.stride] != [4, 8, 16, 32]:
        raise RuntimeError("YOLO26s-P2 detector strides changed")


class FrozenBackboneHooks:
    """Capture exactly one validated feature from each audited backbone layer."""

    def __init__(self, backbone: nn.Sequential):
        if len(backbone) != 11:
            raise RuntimeError("Expected exactly Run A backbone layers 0..10")
        self.features: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.input_shape: tuple[int, int, int] | None = None
        self.handles = []
        for spec in SPECS:
            layer = backbone[spec.layer]
            actual_channels = getattr(getattr(getattr(layer, "cv2", None), "conv", None), "out_channels", None)
            if type(layer).__name__ != spec.module_type or actual_channels != spec.channels:
                raise RuntimeError(f"Cannot attach {spec.name} hook to unexpected layer")
            self.handles.append(layer.register_forward_hook(self._hook(spec)))

    def _hook(self, spec: FeatureSpec):
        def capture(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            if self.input_shape is None or not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise RuntimeError(f"Invalid {spec.name} hook output")
            batch, height, width = self.input_shape
            if (height % spec.stride or width % spec.stride
                    or tuple(output.shape) != (batch, spec.channels,
                                               height // spec.stride, width // spec.stride)):
                raise RuntimeError(f"{spec.name} feature shape/stride mismatch: {tuple(output.shape)}")
            self.counts[spec.name] = self.counts.get(spec.name, 0) + 1
            self.features[spec.name] = output
        return capture

    def extract(self, backbone: nn.Sequential, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected BCHW three-channel 2.5D input")
        self.input_shape = (int(images.shape[0]), int(images.shape[2]), int(images.shape[3]))
        self.features.clear()
        self.counts.clear()
        backbone(images)
        if set(self.features) != {spec.name for spec in SPECS} or any(
            self.counts.get(spec.name) != 1 for spec in SPECS
        ):
            raise RuntimeError("P3/P4/P5 hooks did not fire exactly once")
        return dict(self.features)
