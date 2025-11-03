"""JEPA encoder components using FNO and Mamba.

These components implement the encoder architecture from magnum_opus.md:
- MambaEncoder: Bidirectional Mamba for temporal encoding
- JEPAEncoder: Complete encoder with temporal FNO for frequency domain learning

Key design:
- Input: (B, 128, T) EEG
- Temporal FNO: Learns frequency bands (alpha/beta/gamma/delta/theta) via FFT on time dimension
- Output: (B, 96) latent split as [trait:24, state:36, event:36]
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    from neuralop.models import FNO
except ImportError:
    raise ImportError(
        "neuraloperator not found. Install with: uv add neuraloperator"
    )

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError(
        "mamba-ssm not found. Install with: uv add 'mamba-ssm[causal-conv1d]'"
    )

from .encoders import BaseEncoder


class MambaEncoder(nn.Module):
    """Mamba-based temporal encoder for long-range dependencies.

    Uses mamba-ssm for efficient temporal modeling with state-space models.
    Bidirectional processing captures both past and future context.

    Args:
        d_model: Model dimension
        d_state: State dimension for SSM
        d_conv: Convolution kernel size
        n_layers: Number of Mamba layers
        bidirectional: Whether to use bidirectional processing

    Input shape: (batch, seq_len, d_model)
    Output shape: (batch, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        n_layers: int = 4,
        bidirectional: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional

        # Stack of Mamba layers
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=2,  # Standard expansion factor
            )
            for _ in range(n_layers)
        ])

        # Layer norms for stability
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])

        if bidirectional:
            # Backward direction layers
            self.layers_backward = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=2,
                )
                for _ in range(n_layers)
            ])

            self.norms_backward = nn.ModuleList([
                nn.LayerNorm(d_model)
                for _ in range(n_layers)
            ])

            # Combine forward and backward
            self.combine = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            output: Encoded tensor (batch, seq_len, d_model)
            states: Final hidden states (optional, for analysis)
        """
        # Forward direction
        h_forward = x
        for layer, norm in zip(self.layers, self.norms):
            h_forward = layer(norm(h_forward)) + h_forward  # Residual connection

        if self.bidirectional:
            # Backward direction (flip sequence)
            h_backward = torch.flip(x, dims=[1])
            for layer, norm in zip(self.layers_backward, self.norms_backward):
                h_backward = layer(norm(h_backward)) + h_backward
            h_backward = torch.flip(h_backward, dims=[1])

            # Combine both directions
            h_combined = torch.cat([h_forward, h_backward], dim=-1)
            output = self.combine(h_combined)
        else:
            output = h_forward

        # Return output and final state for potential analysis
        return output, None


class JEPAEncoder(BaseEncoder):
    """Complete JEPA encoder: positional embed → temporal FNO → channel mixing → Mamba → attention → latent.

    Implements the encoder architecture from magnum_opus.md Phase 1.
    Learns multi-scale representations without reconstruction.

    Key architectural design:
    - Temporal FNO: Learns frequency bands (alpha/beta/gamma) via FFT on time dimension
    - Electrode positional embeddings: Distinguishes spatial locations without explicit 3D coords
    - Mamba: Captures long-range temporal dependencies

    Args:
        n_chans: Number of EEG channels (default: 129, HBN dataset includes Cz reference)
        n_times: Number of time samples (default: 400 = 4s @ 100Hz after fix)
        latent_dim: Output latent dimension (default: 96)
        sample_rate: Sampling rate in Hz (default: 100)
        fno_modes: Number of Fourier modes for temporal FNO (default: 50 Hz covers all EEG bands)
        mamba_d_state: State dimension for Mamba
        mamba_layers: Number of Mamba layers

    Input shape: (batch, n_chans, n_times)
    Output shape: (batch, latent_dim)

    Latent is split as:
    - z[:, :24]: Trait (slow, stable psychopathology)
    - z[:, 24:60]: State (medium, cognitive state)
    - z[:, 60:]: Event (fast, events/responses)
    """

    def __init__(
        self,
        n_chans: int = 129,  # HBN dataset: 129 channels including Cz reference
        n_times: int = 200,  # 2s @ 100Hz (encoder sees this after trainer splits 4s windows)
        latent_dim: int = 96,
        sample_rate: int = 100,
        fno_modes: int = 50,  # 50 Hz covers all EEG bands (delta to gamma)
        mamba_d_state: int = 16,
        mamba_layers: int = 4,
        **kwargs  # Accept extra kwargs for flexibility
    ):
        super().__init__()

        self.n_chans = n_chans
        self.n_times = n_times
        self.latent_dim = latent_dim
        self._output_dim = latent_dim
        self.sample_rate = sample_rate

        # Electrode positional embeddings (learned, no explicit 3D coordinates)
        # Each channel gets a unique learned scalar bias (shape: n_chans)
        # This distinguishes spatial locations without explicit 3D coords
        self.channel_positions = nn.Parameter(torch.zeros(n_chans))

        # Temporal FNO: Learn frequency domain patterns (alpha/beta/gamma/delta/theta)
        # Applied across TIME dimension - FFT learns which frequencies matter
        # modes=50 keeps frequencies 0-50 Hz (covers all EEG bands)
        self.temporal_fno = FNO(
            n_modes=(fno_modes,),         # Single-element tuple for 1D
            in_channels=n_chans,          # Process all 129 channels
            out_channels=n_chans,         # Keep 129 channels
            hidden_channels=64,
            n_layers=2
        )

        # Channel mixing: Learn spatial relationships (electrode interactions)
        # After FNO captures frequency info, mix across channels
        self.channel_proj = nn.Conv1d(
            in_channels=n_chans,   # 129 channels
            out_channels=64,       # Reduce to 64 features
            kernel_size=1          # Per-timepoint projection
        )

        # Temporal encoder: Mamba for long-range dependencies
        self.temporal_encoder = MambaEncoder(
            d_model=64,
            d_state=mamba_d_state,
            d_conv=4,
            n_layers=mamba_layers,
            bidirectional=True
        )

        # Global context via attention pooling
        self.context_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True
        )

        # Project to latent space
        self.to_latent = nn.Linear(64, latent_dim)

    @property
    def output_dim(self) -> int:
        """Output dimension of the encoder."""
        return self._output_dim

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor | tuple:
        """
        Args:
            x: Input EEG (batch, n_chans, n_times)
            return_intermediates: If True, return intermediate activations

        Returns:
            z: Latent representation (batch, latent_dim)
            intermediates: Dict of intermediate activations (if return_intermediates=True)
        """
        B, C, T = x.shape

        # Add electrode positional embeddings
        # channel_positions shape: (C,) - learned scalar bias per channel
        # Broadcast to (B, C, T): add same bias to all timepoints of each channel
        x_pos = x + self.channel_positions.view(1, C, 1)  # (B, C, T)

        # Temporal FNO: Learn frequency domain patterns (alpha/beta/gamma)
        # FFT operates on last dimension (time axis)
        # Input: (B, 128, 400) → FFT on 400 timepoints
        # Learns: which temporal frequencies (Hz) matter per channel
        x_freq = self.temporal_fno(x_pos)  # (B, C, T)

        # Channel mixing: Learn spatial relationships (electrode interactions)
        # After capturing frequency info, mix information across channels
        x_spatial = self.channel_proj(x_freq)  # (B, 64, T)

        # Reshape for temporal sequence processing: (B, 64, T) → (B, T, 64)
        x_seq = x_spatial.permute(0, 2, 1)

        # Temporal encoding with Mamba: Capture long-range dependencies
        x_temporal, _ = self.temporal_encoder(x_seq)  # (B, T, 64)

        # Global attention pooling
        # Use mean-pooled temporal as query, full sequence as key/value
        x_query = x_temporal.mean(dim=1, keepdim=True)  # (B, 1, 64)
        x_pooled, attn_weights = self.context_attention(
            x_query, x_temporal, x_temporal
        )  # (B, 1, 64)

        # Project to latent space
        z = self.to_latent(x_pooled.squeeze(1))  # (B, 96)

        if return_intermediates:
            intermediates = {
                'frequency': x_freq,      # After temporal FNO
                'spatial': x_spatial,     # After channel mixing
                'temporal': x_temporal,   # After Mamba
                'attention': attn_weights
            }
            return z, intermediates

        return z
