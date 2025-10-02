"""JEPA-style encoder for EEG windows."""

from __future__ import annotations

import torch
from torch import nn

from cerebro.backbones import EEGConvBackbone, EEGConvBackboneConfig


class EEGJEPAModel(nn.Module):
    def __init__(self, backbone: EEGConvBackbone | None = None, proj_dim: int = 128):
        super().__init__()
        self.encoder = backbone or EEGConvBackbone()
        embed_dim = self.encoder.config.embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        proj = self.projector(emb)
        return nn.functional.normalize(proj, dim=-1)
