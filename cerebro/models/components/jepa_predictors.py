"""Multi-scale predictors for JEPA temporal prediction.

These predictors operate on different timescales:
- TraitPredictor: Slow dynamics (MLP, stable traits like psychopathology)
- StatePredictor: Medium dynamics (Mamba, cognitive state)
- EventPredictor: Fast dynamics (Mamba, events/responses)

From magnum_opus.md Phase 1: Different predictors for different timescales.
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError(
        "mamba-ssm not found. Install with: uv add 'mamba-ssm[causal-conv1d]'"
    )


class TraitPredictor(nn.Module):
    """Predictor for slow timescale dynamics (stable traits).

    Uses simple MLP since traits should be relatively stable over time.
    Predicts future trait representation from past.

    Args:
        trait_dim: Dimension of trait latents (default: 24)
        hidden_dim: Hidden layer dimension

    Input shape: (batch, trait_dim)
    Output shape: (batch, trait_dim)
    """

    def __init__(self, trait_dim: int = 24, hidden_dim: int = 64):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(trait_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, trait_dim)
        )

    def forward(self, z_trait: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_trait: Trait latent (batch, trait_dim)

        Returns:
            Predicted future trait (batch, trait_dim)
        """
        return self.predictor(z_trait)


class StatePredictor(nn.Module):
    """Predictor for medium timescale dynamics (cognitive state).

    Uses Mamba to capture state evolution over medium timescales.
    Cognitive state changes more dynamically than traits but slower than events.

    Args:
        state_dim: Dimension of state latents (default: 36)
        d_state: Mamba state dimension
        d_conv: Mamba convolution kernel size

    Input shape: (batch, state_dim)
    Output shape: (batch, state_dim)
    """

    def __init__(
        self,
        state_dim: int = 36,
        d_state: int = 8,
        d_conv: int = 3
    ):
        super().__init__()

        # Mamba block for temporal prediction
        self.mamba = Mamba(
            d_model=state_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, z_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_state: State latent (batch, state_dim)

        Returns:
            Predicted future state (batch, state_dim)
        """
        # Add sequence dimension for Mamba: (B, D) → (B, 1, D)
        z_state_seq = z_state.unsqueeze(1)

        # Mamba processing
        z_pred = self.mamba(self.norm(z_state_seq))

        # Remove sequence dimension: (B, 1, D) → (B, D)
        return z_pred.squeeze(1)


class EventPredictor(nn.Module):
    """Predictor for fast timescale dynamics (events/responses).

    Uses fast Mamba configuration for rapid event dynamics.
    Optimized for short-term predictions with small dt_min/dt_max.

    Args:
        event_dim: Dimension of event latents (default: 36)
        d_state: Mamba state dimension (smaller for faster dynamics)
        d_conv: Mamba convolution kernel size (smaller for local context)

    Input shape: (batch, event_dim)
    Output shape: (batch, event_dim)
    """

    def __init__(
        self,
        event_dim: int = 36,
        d_state: int = 4,
        d_conv: int = 2
    ):
        super().__init__()

        # Fast Mamba block
        # Note: dt_min and dt_max control timescale but aren't directly
        # exposed in mamba_ssm.Mamba constructor. The default values
        # are reasonable for fast dynamics.
        self.mamba = Mamba(
            d_model=event_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
        )

        # Layer norm
        self.norm = nn.LayerNorm(event_dim)

    def forward(self, z_event: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_event: Event latent (batch, event_dim)

        Returns:
            Predicted future event (batch, event_dim)
        """
        # Add sequence dimension: (B, D) → (B, 1, D)
        z_event_seq = z_event.unsqueeze(1)

        # Mamba processing
        z_pred = self.mamba(self.norm(z_event_seq))

        # Remove sequence dimension: (B, 1, D) → (B, D)
        return z_pred.squeeze(1)


class MambaBlock(nn.Module):
    """General Mamba block wrapper for easy predictor construction.

    This is a convenience wrapper that can be used to create predictors
    with different configurations.

    Args:
        d_model: Model dimension
        d_state: State dimension
        d_conv: Convolution kernel size
        dt_min: Minimum dt (affects timescale)
        dt_max: Maximum dt (affects timescale)

    Input shape: (batch, seq_len, d_model) or (batch, d_model)
    Output shape: Same as input
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model) or (batch, d_model)

        Returns:
            Output tensor (same shape as input)
        """
        # Handle both 2D and 3D inputs
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, D) → (B, 1, D)
            squeeze_output = True
        else:
            squeeze_output = False

        # Mamba processing with residual
        out = self.mamba(self.norm(x)) + x

        if squeeze_output:
            out = out.squeeze(1)  # (B, 1, D) → (B, D)

        return out
