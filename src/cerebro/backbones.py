"""Custom backbone architectures for EEG representation learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .constants import N_CHANS, N_TIMES


@dataclass
class EEGConvBackboneConfig:
    input_channels: int = N_CHANS
    input_samples: int = N_TIMES
    temporal_kernel: int = 7
    temporal_dilations: tuple[int, ...] = (1, 2, 4, 8)
    hidden_channels: int = 128
    embedding_dim: int = 256
    dropout: float = 0.1


class EEGConvBackbone(nn.Module):
    """Lightweight convolutional encoder producing a fixed-length embedding."""

    def __init__(self, config: EEGConvBackboneConfig | None = None):
        super().__init__()
        cfg = config or EEGConvBackboneConfig()
        self.config = cfg

        layers = []
        in_channels = cfg.input_channels
        for dilation in cfg.temporal_dilations:
            conv = nn.Conv1d(
                in_channels,
                cfg.hidden_channels,
                kernel_size=cfg.temporal_kernel,
                padding=dilation * (cfg.temporal_kernel // 2),
                dilation=dilation,
                bias=False,
            )
            layers.extend(
                [
                    conv,
                    nn.BatchNorm1d(cfg.hidden_channels),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_channels = cfg.hidden_channels

        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Conv1d(cfg.hidden_channels, cfg.embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(cfg.embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input of shape (batch, channels, samples) to (batch, embed_dim)."""
        features = self.encoder(x)
        projected = self.projection(features)
        pooled = projected.mean(dim=-1)
        return pooled


__all__ = ["EEGConvBackbone", "EEGConvBackboneConfig"]
