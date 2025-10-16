# labram_encoder.py
"""
LaBraM-style encoder architecture for EEG processing.
Implements channel-aware feature extraction following the original LaBraM paper.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """LaBraM-style convolutional block with LayerNorm"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, C) for LayerNorm
        x = self.norm(x)
        x = self.act(x)
        return x.transpose(1, 2)  # back to (B, C, T)


class LaBraMEncoder(nn.Module):
    """
    Feature encoder matching LaBraM's architecture.
    Processes each channel independently then aggregates across channels.
    """

    def __init__(self, n_chans=129, patch_len=20, dim=256):
        super().__init__()
        self.n_chans = n_chans
        self.patch_len = patch_len
        self.dim = dim

        # Per-channel feature extraction
        self.channel_conv = nn.Sequential(
            ConvBlock(1, 64, kernel_size=3, padding=1),
            ConvBlock(64, 128, kernel_size=3, padding=1),
            nn.MaxPool1d(2),  # downsample patch
            ConvBlock(128, dim, kernel_size=3, padding=1),
        )

        # Global average pooling followed by projection
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (B, C, P, L) - batch, channels, patches, patch_length
        Returns: (B, C, P, D) - encoded features
        """
        B, C, P, L = x.shape

        # Process each channel-patch independently
        # Reshape to (B*C*P, 1, L)
        x_flat = x.reshape(B * C * P, 1, L)

        # Apply convolutional encoder
        features = self.channel_conv(x_flat)  # (B*C*P, dim, L//2)

        # Global average pooling over time dimension
        features = features.mean(dim=-1)  # (B*C*P, dim)

        # Project to final dimension
        features = self.proj(features)  # (B*C*P, dim)

        # Reshape back to (B, C, P, D)
        features = features.reshape(B, C, P, self.dim)

        return features


class LaBraMEncoderV2(nn.Module):
    """
    Alternative encoder with cross-channel interaction.
    Similar to LaBraM but with explicit channel mixing.
    """

    def __init__(self, n_chans=129, patch_len=20, dim=256):
        super().__init__()
        self.n_chans = n_chans
        self.patch_len = patch_len
        self.dim = dim

        # Per-patch temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Cross-channel aggregation
        self.channel_mix = nn.Sequential(
            nn.Linear(n_chans, n_chans),
            nn.GELU(),
            nn.LayerNorm(n_chans),
        )

        self.final_proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (B, C, P, L) - batch, channels, patches, patch_length
        Returns: (B, C, P, D) - encoded features
        """
        B, C, P, L = x.shape

        # Process each patch independently across time
        x_flat = x.reshape(B * C * P, 1, L)  # (B*C*P, 1, L)

        # Extract temporal features
        temp_features = self.temporal_conv(x_flat)  # (B*C*P, dim, L//2)
        temp_features = temp_features.mean(dim=-1)  # (B*C*P, dim)

        # Reshape to (B, P, C, dim) for channel mixing
        temp_features = temp_features.reshape(B, C, P, self.dim)
        temp_features = temp_features.permute(0, 2, 1, 3)  # (B, P, C, dim)

        # Mix across channels
        B_P_shape = temp_features.shape[:-2]
        # (B*P, C, dim)
        temp_features_flat = temp_features.reshape(-1, C, self.dim)

        # Apply channel mixing per feature dimension
        mixed = []
        for i in range(self.dim):
            feat_slice = temp_features_flat[:, :, i]  # (B*P, C)
            mixed_slice = self.channel_mix(feat_slice)  # (B*P, C)
            mixed.append(mixed_slice)

        mixed = torch.stack(mixed, dim=-1)  # (B*P, C, dim)
        mixed = mixed.reshape(B, P, C, self.dim)  # (B, P, C, dim)

        # Final projection and reshape back to (B, C, P, D)
        output = self.final_proj(mixed)  # (B, P, C, dim)
        output = output.permute(0, 2, 1, 3)  # (B, C, P, dim)

        return output
