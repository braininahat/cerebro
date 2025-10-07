"""Masked modeling autoencoder."""

from __future__ import annotations

import torch
from torch import nn

from cerebro.constants import N_CHANS


class EEGMaskedAutoencoder(nn.Module):
    """Lightweight convolutional autoencoder for masked reconstruction."""

    def __init__(self, hidden_channels: int = 256, depth: int = 3):
        super().__init__()
        encoder_layers = []
        in_channels = N_CHANS
        for _ in range(depth):
            encoder_layers.extend(
                [
                    nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
                    nn.BatchNorm1d(hidden_channels),
                    nn.GELU(),
                ]
            )
            in_channels = hidden_channels
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for _ in range(depth - 1):
            decoder_layers.extend(
                [
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
                    nn.BatchNorm1d(hidden_channels),
                    nn.GELU(),
                ]
            )
        decoder_layers.append(nn.Conv1d(hidden_channels, N_CHANS, kernel_size=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


__all__ = ["EEGMaskedAutoencoder"]
