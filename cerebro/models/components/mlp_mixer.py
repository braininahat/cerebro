"""
MLP-Mixer for EEG spatial channel mixing.

Based on DSE-Mixer (2024) and VICReg (2022) for preventing representation collapse.
Implements spatial channel mixing without graph construction - learns relationships from data.

References:
- DSE-Mixer: 95%+ accuracy on DEAP emotion recognition
- VICReg: Variance-Invariance-Covariance Regularization
- MLP-Mixer: "MLP-Mixer: An all-MLP Architecture for Vision" (Tolstikhin et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ChannelMixerBlock(nn.Module):
    """
    MLP block for mixing across channels (spatial dimension).

    Learns relationships between EEG electrodes without explicit graph structure.
    """

    def __init__(
        self,
        n_channels: int,
        hidden_dim: int,
        expansion_factor: float = 4.0,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Args:
            n_channels: Number of input channels (e.g., 129 for HBN)
            hidden_dim: Hidden dimension for features
            expansion_factor: MLP expansion ratio (default 4x)
            dropout: Dropout rate
            activation: Activation function ('gelu', 'relu', 'silu')
        """
        super().__init__()

        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        mlp_hidden = int(hidden_dim * expansion_factor)

        # Channel mixing MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_channels, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_channels)
        )

        # Layer norm for pre-normalization
        self.norm = nn.LayerNorm(n_channels)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'silu': nn.SiLU()
        }
        return activations.get(name, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mix channels using MLP.

        Args:
            x: (batch, n_channels, features) or (batch, features, n_channels)

        Returns:
            out: Same shape as input
        """
        # Determine if channels are in dim 1 or 2
        if x.size(1) == self.n_channels:
            # (batch, n_channels, features) → transpose for MLP
            x_transposed = x.transpose(1, 2)  # (batch, features, n_channels)
            x_normed = self.norm(x_transposed)
            x_mixed = self.mlp(x_normed)
            out = x_transposed + x_mixed  # Residual
            out = out.transpose(1, 2)  # Back to (batch, n_channels, features)
        else:
            # Assume (batch, features, n_channels)
            x_normed = self.norm(x)
            x_mixed = self.mlp(x_normed)
            out = x + x_mixed  # Residual

        return out


class FeatureMixerBlock(nn.Module):
    """
    MLP block for mixing across features (temporal/spectral dimension).

    Complements channel mixing for full MLP-Mixer architecture.
    """

    def __init__(
        self,
        feature_dim: int,
        expansion_factor: float = 4.0,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Args:
            feature_dim: Number of features (e.g., hidden_dim from FNO)
            expansion_factor: MLP expansion ratio
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()

        mlp_hidden = int(feature_dim * expansion_factor)

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feature_dim)
        )

        self.norm = nn.LayerNorm(feature_dim)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'silu': nn.SiLU()
        }
        return activations.get(name, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mix features using MLP.

        Args:
            x: (batch, n_channels, features)

        Returns:
            out: Same shape as input
        """
        x_normed = self.norm(x)
        x_mixed = self.mlp(x_normed)
        out = x + x_mixed  # Residual
        return out


class MLPMixerEncoder(nn.Module):
    """
    Complete MLP-Mixer encoder for EEG.

    Alternates between channel mixing (spatial) and feature mixing (temporal/spectral).
    """

    def __init__(
        self,
        n_channels: int = 129,
        feature_dim: int = 256,
        n_blocks: int = 2,
        expansion_factor: float = 4.0,
        dropout: float = 0.1
    ):
        """
        Args:
            n_channels: Number of EEG channels
            feature_dim: Feature dimension
            n_blocks: Number of mixer blocks (each has channel + feature mixing)
            expansion_factor: MLP expansion ratio
            dropout: Dropout rate
        """
        super().__init__()

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            # Channel mixing block
            channel_mixer = ChannelMixerBlock(
                n_channels=n_channels,
                hidden_dim=feature_dim,
                expansion_factor=expansion_factor,
                dropout=dropout
            )

            # Feature mixing block
            feature_mixer = FeatureMixerBlock(
                feature_dim=feature_dim,
                expansion_factor=expansion_factor,
                dropout=dropout
            )

            self.blocks.append(nn.ModuleDict({
                'channel': channel_mixer,
                'feature': feature_mixer
            }))

        # Final layer norm
        self.final_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process features through MLP-Mixer blocks.

        Args:
            x: (batch, n_channels, features)

        Returns:
            out: (batch, n_channels, features)
        """
        for block in self.blocks:
            # Channel mixing (spatial)
            x = block['channel'](x)

            # Feature mixing (temporal/spectral)
            x = block['feature'](x)

        # Final normalization
        x = self.final_norm(x)

        return x


class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization.

    Prevents representation collapse by:
    1. Variance term: Maintains std > threshold for each dimension
    2. Covariance term: Decorrelates features to prevent redundancy

    Reference: "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
               Bardes et al., ICLR 2022
    """

    def __init__(
        self,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        var_threshold: float = 1.0,
        eps: float = 1e-4
    ):
        """
        Args:
            lambda_var: Weight for variance term
            lambda_cov: Weight for covariance term
            var_threshold: Target standard deviation
            eps: Small constant for numerical stability
        """
        super().__init__()

        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.var_threshold = var_threshold
        self.eps = eps

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute VICReg loss.

        Args:
            embeddings: (batch_size, feature_dim) embeddings to regularize

        Returns:
            loss: Total VICReg loss
            metrics: Dictionary with variance and covariance losses
        """
        batch_size, feature_dim = embeddings.shape

        # Variance term: Encourage each dimension to have variance >= threshold
        std = torch.sqrt(embeddings.var(dim=0) + self.eps)
        var_loss = torch.relu(self.var_threshold - std).mean()

        # Covariance term: Minimize off-diagonal correlations
        # Center embeddings
        embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)

        # Compute covariance matrix
        cov = (embeddings_centered.T @ embeddings_centered) / (batch_size - 1)

        # Off-diagonal elements should be zero (decorrelated features)
        cov_loss = (cov ** 2).fill_diagonal_(0).sum() / feature_dim

        # Total loss
        total_loss = self.lambda_var * var_loss + self.lambda_cov * cov_loss

        metrics = {
            'vicreg_var': var_loss.item(),
            'vicreg_cov': cov_loss.item(),
            'vicreg_total': total_loss.item(),
            'embedding_std_mean': std.mean().item(),
            'embedding_std_min': std.min().item()
        }

        return total_loss, metrics


class SpatialChannelMixer(nn.Module):
    """
    Simple spatial channel mixer for post-FNO processing.

    Designed to replace weak 1x1 conv with learnable channel mixing.
    Includes VICReg regularization to prevent collapse.
    """

    def __init__(
        self,
        n_channels: int = 129,
        hidden_dim: int = 256,
        use_vicreg: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            n_channels: Number of EEG channels
            hidden_dim: Hidden dimension
            use_vicreg: Whether to apply VICReg regularization
            dropout: Dropout rate
        """
        super().__init__()

        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.use_vicreg = use_vicreg

        # Simple channel mixer (replaces 1x1 conv)
        self.mixer = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # VICReg loss (optional)
        if use_vicreg:
            self.vicreg = VICRegLoss(
                lambda_var=5.0,  # Reduced from 25.0 to prevent over-spreading
                lambda_cov=1.0,
                var_threshold=1.0
            )
        else:
            self.vicreg = None

    def forward(
        self,
        x: torch.Tensor,
        return_vicreg: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """
        Mix channels spatially.

        Args:
            x: Input features
                - If 3D: (batch, n_channels, features)
                - If 2D: (batch, n_channels) - will be unsqueezed
            return_vicreg: Whether to compute VICReg loss

        Returns:
            out: Mixed features (batch, hidden_dim)
            vicreg_loss: VICReg regularization loss (if requested)
            vicreg_metrics: Diagnostic metrics (if requested)
        """
        # Handle different input shapes
        if x.dim() == 3:
            # (batch, n_channels, features) → pool features
            x_pooled = x.mean(dim=2)  # (batch, n_channels)
        elif x.dim() == 2:
            x_pooled = x  # (batch, n_channels)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        # Mix channels
        out = self.mixer(x_pooled)  # (batch, hidden_dim)

        # Compute VICReg loss if requested
        vicreg_loss = None
        vicreg_metrics = None
        if return_vicreg and self.vicreg is not None:
            vicreg_loss, vicreg_metrics = self.vicreg(out)

        return out, vicreg_loss, vicreg_metrics


class EnhancedSpatialChannelMixer(nn.Module):
    """
    Enhanced spatial channel mixer with per-timestep processing and max pooling.

    Based on DSE-Mixer (Ma et al. 2024) - achieves 95%+ accuracy on DEAP emotion recognition.

    Key improvements over SpatialChannelMixer:
    1. Per-timestep spatial mixing (prevents attention collapse)
    2. 3-layer channel mixing (not 1-layer) - stronger spatial modeling
    3. Max pooling over time (preserves strongest activations)

    Architecture:
    - Input: (batch, n_channels, time)
    - Per-timestep 3-layer MLP: (batch, n_channels, time) → (batch, hidden_dim, time)
    - Max pooling over time: → (batch, hidden_dim)
    - Optional VICReg regularization

    Why no attention pooling? Attention-based pooling with data-dependent queries
    causes severe representation collapse (all samples get similar attention patterns).
    """

    def __init__(
        self,
        n_channels: int = 129,
        hidden_dim: int = 256,
        use_vicreg: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            n_channels: Number of EEG channels (129 for HBN)
            hidden_dim: Output dimension (latent size)
            use_vicreg: Whether to apply VICReg regularization
            dropout: Dropout rate
        """
        super().__init__()

        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.use_vicreg = use_vicreg

        # No temporal pooling inside mixer! Keep temporal dimension intact.
        # Pooling will be done AFTER spatial mixing (outside this module)
        # This prevents collapse from attention-based pooling

        # 3-layer channel mixer (inspired by DSE-Mixer)
        # Layer 1: Expand (129 → 512)
        # Layer 2: Project (512 → 256)
        # Layer 3: Refine (256 → 256)
        expansion_factor = 2
        expanded_dim = hidden_dim * expansion_factor

        self.channel_mixer = nn.Sequential(
            # Pre-normalization
            nn.LayerNorm(n_channels),

            # Layer 1: Expansion
            nn.Linear(n_channels, expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            # Layer 2: Projection
            nn.Linear(expanded_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            # Layer 3: Refinement
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Residual projection (if n_channels != hidden_dim)
        self.residual_proj = nn.Linear(n_channels, hidden_dim) if n_channels != hidden_dim else nn.Identity()

        # VICReg loss (optional)
        if use_vicreg:
            self.vicreg = VICRegLoss(
                lambda_var=5.0,  # Reduced from 25.0 to prevent over-spreading
                lambda_cov=1.0,
                var_threshold=1.0
            )
        else:
            self.vicreg = None

    def forward(
        self,
        x: torch.Tensor,
        return_vicreg: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """
        Enhanced spatial mixing with attention pooling.

        Args:
            x: Input features
                - If 3D: (batch, n_channels, time)
                - If 2D: (batch, n_channels) - will be unsqueezed
            return_vicreg: Whether to compute VICReg loss

        Returns:
            out: Mixed features (batch, hidden_dim)
            vicreg_loss: VICReg regularization loss (if requested)
            vicreg_metrics: Diagnostic metrics (if requested)
        """
        # Handle different input shapes
        if x.dim() == 3:
            # Process spatial mixing per-timestep (no temporal pooling!)
            # Input: (batch, channels, time)
            batch_size, n_chans, n_time = x.shape

            # Reshape to (batch * time, channels) to process all timesteps together
            x_flat = x.transpose(1, 2).reshape(batch_size * n_time, n_chans)

            # Apply spatial mixing (3-layer MLP): (B*T, 129) → (B*T, hidden_dim)
            # This processes each timestep independently
            mixed = self.channel_mixer(x_flat)  # (B*T, hidden_dim)

            # Reshape back to (batch, time, hidden_dim)
            mixed = mixed.view(batch_size, n_time, self.hidden_dim)

            # Global mean pooling over time: (batch, time, hidden_dim) → (batch, hidden_dim)
            # Mean pooling preserves more information for continuous EEG signals
            # (Max pooling is too aggressive and loses temporal context)
            out = mixed.mean(dim=1)  # (batch, hidden_dim)

            # Proper residual connection: pool input over time, then project to hidden_dim
            # Input: (batch, channels, time) → pool time → (batch, channels) → project → (batch, hidden_dim)
            x_pooled = x.mean(dim=2)  # (batch, channels) - pool over time
            x_res = self.residual_proj(x_pooled)  # (batch, hidden_dim)
            out = out + x_res

        elif x.dim() == 2:
            # Already pooled: (batch, n_channels) → (batch, hidden_dim)
            out = self.channel_mixer(x)  # (batch, hidden_dim)

            # Residual connection
            if self.n_channels != self.hidden_dim:
                residual = self.residual_proj(x)  # (batch, hidden_dim)
                out = out + residual

        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        # Compute VICReg loss if requested
        vicreg_loss = None
        vicreg_metrics = None
        if return_vicreg and self.vicreg is not None:
            vicreg_loss, vicreg_metrics = self.vicreg(out)

        return out, vicreg_loss, vicreg_metrics


def test_mlp_mixer():
    """Test MLP-Mixer components."""
    print("Testing MLP-Mixer components...")

    # Test channel mixer
    batch_size = 32
    n_channels = 129
    features = 256

    x = torch.randn(batch_size, n_channels, features)

    channel_mixer = ChannelMixerBlock(n_channels=n_channels, hidden_dim=features)
    out = channel_mixer(x)
    assert out.shape == x.shape
    print(f"✓ ChannelMixerBlock: {x.shape} → {out.shape}")

    # Test feature mixer
    feature_mixer = FeatureMixerBlock(feature_dim=features)
    out = feature_mixer(x)
    assert out.shape == x.shape
    print(f"✓ FeatureMixerBlock: {x.shape} → {out.shape}")

    # Test full MLP-Mixer
    mixer = MLPMixerEncoder(n_channels=n_channels, feature_dim=features, n_blocks=2)
    out = mixer(x)
    assert out.shape == x.shape
    print(f"✓ MLPMixerEncoder: {x.shape} → {out.shape}")

    # Test VICReg loss
    embeddings = torch.randn(batch_size, features)
    vicreg = VICRegLoss()
    loss, metrics = vicreg(embeddings)
    print(f"✓ VICReg loss: {loss.item():.4f}")
    print(f"  Metrics: var={metrics['vicreg_var']:.4f}, cov={metrics['vicreg_cov']:.4f}")

    # Test SpatialChannelMixer
    spatial_mixer = SpatialChannelMixer(n_channels=n_channels, hidden_dim=features)
    out, vicreg_loss, vicreg_metrics = spatial_mixer(x, return_vicreg=True)
    assert out.shape == (batch_size, features)
    print(f"✓ SpatialChannelMixer: {x.shape} → {out.shape}")
    if vicreg_loss is not None:
        print(f"  VICReg loss: {vicreg_loss.item():.4f}")

    # Test EnhancedSpatialChannelMixer
    enhanced_mixer = EnhancedSpatialChannelMixer(n_channels=n_channels, hidden_dim=features)
    out, vicreg_loss, vicreg_metrics = enhanced_mixer(x, return_vicreg=True)
    assert out.shape == (batch_size, features)
    print(f"✓ EnhancedSpatialChannelMixer: {x.shape} → {out.shape}")
    if vicreg_loss is not None:
        print(f"  VICReg loss: {vicreg_loss.item():.4f}")
    print(f"  Per-timestep spatial mixing + max pooling (no attention collapse!)")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_mlp_mixer()