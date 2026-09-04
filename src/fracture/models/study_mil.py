"""Small dependency-free multiple-instance model for CT study classification."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class SliceEncoder(nn.Module):
    """Compact 2D encoder suitable for 15-24 GB shared GPU environments."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        channels = (3, 16, 32, 64, 128)
        blocks: list[nn.Module] = []
        for source, target in zip(channels[:-1], channels[1:], strict=True):
            blocks.extend(
                [
                    nn.Conv2d(source, target, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(target),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(target, target, kernel_size=3, padding=1, groups=target, bias=False),
                    nn.BatchNorm2d(target),
                    nn.SiLU(inplace=True),
                ]
            )
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(channels[-1], embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(images)).flatten(1)
        return self.projection(features)


def pool_slice_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    method: str = "topk",
    top_k: int = 3,
) -> torch.Tensor:
    """Pool sparse slice evidence into one logit per study."""
    if logits.ndim != 2 or mask.shape != logits.shape:
        raise ValueError("logits and mask must both have shape [batch, slices]")
    output = []
    for values, valid in zip(logits, mask, strict=True):
        selected = values[valid]
        if not selected.numel():
            raise ValueError("Every study must contain at least one valid slice")
        if method == "max":
            pooled = selected.max()
        elif method == "mean":
            pooled = selected.mean()
        elif method == "topk":
            pooled = selected.topk(min(int(top_k), selected.numel())).values.mean()
        elif method == "logsumexp":
            pooled = torch.logsumexp(selected, dim=0) - math.log(selected.numel())
        else:
            raise ValueError(f"Unsupported MIL pooling method: {method}")
        output.append(pooled)
    return torch.stack(output)


class StudyMIL(nn.Module):
    """Encode 2D slices and learn a sparse whole-study fracture score."""

    def __init__(
        self,
        embedding_dim: int = 128,
        pooling: str = "topk",
        top_k: int = 3,
        encoder_chunk_size: int = 16,
    ):
        super().__init__()
        self.encoder = SliceEncoder(embedding_dim)
        self.slice_head = nn.Linear(embedding_dim, 1)
        self.pooling = pooling
        self.top_k = int(top_k)
        self.encoder_chunk_size = int(encoder_chunk_size)

    def forward(self, images: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("images must have shape [batch, slices, channels, height, width]")
        batch, slices, channels, height, width = images.shape
        if channels != 3:
            raise ValueError("StudyMIL currently expects three-channel inputs")
        if mask is None:
            mask = torch.ones((batch, slices), dtype=torch.bool, device=images.device)
        else:
            mask = mask.to(device=images.device, dtype=torch.bool)
        flat = images.reshape(batch * slices, channels, height, width)
        chunks = [
            self.encoder(flat[start : start + self.encoder_chunk_size])
            for start in range(0, flat.shape[0], self.encoder_chunk_size)
        ]
        embeddings = torch.cat(chunks, dim=0).reshape(batch, slices, -1)
        slice_logits = self.slice_head(embeddings).squeeze(-1)
        study_logits = pool_slice_logits(
            slice_logits,
            mask,
            method=self.pooling,
            top_k=self.top_k,
        )
        return {
            "study_logits": study_logits,
            "slice_logits": slice_logits,
            "embeddings": embeddings,
        }

    def checkpoint_payload(
        self,
        preprocessing: dict[str, Any],
        training_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "state_dict": self.state_dict(),
            "model_config": {
                "embedding_dim": self.slice_head.in_features,
                "pooling": self.pooling,
                "top_k": self.top_k,
                "encoder_chunk_size": self.encoder_chunk_size,
            },
            "preprocessing": preprocessing,
            "architecture": "compact_slice_cnn_topk_mil",
            "external_pretrained_weights": False,
        }
        if training_metadata is not None:
            payload["training_metadata"] = training_metadata
        return payload
