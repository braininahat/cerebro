"""Complete model architectures composed from encoder and decoder components.

These are pure nn.Module classes that combine encoders and decoders
into complete models. They contain no training logic - that's handled
by the trainer modules.

Key design principles:
- Models are composable: encoder + decoder
- They can be trained with different objectives (supervised, contrastive, etc.)
- Transfer learning is supported through encoder weight loading
"""

from typing import Any, Literal, Optional

import torch
import torch.nn as nn

from .builders import build_encoder
from .components import ProjectionHead, RegressionHead, MultiTaskHead
from .components.jepa_components import JEPAEncoder
from .components.jepa_predictors import TraitPredictor, StatePredictor, EventPredictor


class RegressorModel(nn.Module):
    """Complete regression model: encoder + regression head.

    This model is designed for supervised regression tasks where we
    predict continuous values from EEG input.

    Args:
        encoder_class: Name of encoder to use ("EEGNeX", "SignalJEPA", etc.)
        n_outputs: Number of regression targets (default: 1)
        dropout: Dropout probability before regression head (default: 0)
        input_scale: Multiplier to scale input from volt to other units (default: 1.0)
            1.0 = volt scale (matches startkit baseline)
            1000.0 = millivolt scale (faster initial convergence)
            1e6 = microvolt scale
        encoder_kwargs: Additional arguments for encoder initialization (as dict)

    Example:
        >>> model = RegressorModel(encoder_class="EEGNeX", n_outputs=1)
        >>> predictions = model(eeg_input)  # Shape: (batch, 1)

        >>> # With voltage scaling and encoder params
        >>> model = RegressorModel(
        ...     encoder_class="EEGNeX",
        ...     input_scale=1000.0,
        ...     encoder_kwargs={"n_chans": 129, "n_times": 200, "sfreq": 100}
        ... )
    """

    def __init__(
        self,
        encoder_class: str = "EEGNeX",
        n_outputs: int = 1,
        dropout: float = 0.0,
        input_scale: float = 1.0,
        encoder_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        # Register input scale as buffer (part of model state, not trainable)
        self.register_buffer('scale', torch.tensor(input_scale, dtype=torch.float32))

        # Build encoder with kwargs
        if encoder_kwargs is None:
            encoder_kwargs = {}
        self.encoder = build_encoder(encoder_class, **encoder_kwargs)

        # Build regression head
        self.head = RegressionHead(
            input_dim=self.encoder.output_dim,
            output_dim=n_outputs,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and regression head.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)
               Expected in VOLT scale (e.g., -0.016 to 0.008 for typical EEG)

        Returns:
            Predictions of shape (batch_size, n_outputs)
        """
        # Apply input scaling (e.g., volt → millivolt)
        if self.scale != 1.0:
            x = x * self.scale

        features = self.encoder(x)
        predictions = self.head(features)
        return predictions

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without the regression head.

        Useful for visualization or transfer learning.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)

        Returns:
            Features of shape (batch_size, feature_dim)
        """
        return self.encoder(x)


class ContrastiveModel(nn.Module):
    """Complete contrastive model: encoder + projection head.

    This model is designed for contrastive learning where we learn
    representations by pulling similar samples together and pushing
    dissimilar samples apart in embedding space.

    The projection head is typically discarded after pretraining,
    keeping only the encoder for downstream tasks.

    Args:
        encoder_class: Name of encoder to use
        projection_dim: Output dimension of projection head (default: 128)
        hidden_dim: Hidden dimension of projection MLP (default: 256)
        dropout: Dropout in projection head (default: 0.1)
        input_scale: Multiplier to scale input from volt to other units (default: 1.0)
            1.0 = volt scale (matches startkit baseline)
            1000.0 = millivolt scale (faster initial convergence)
            1e6 = microvolt scale
        encoder_kwargs: Additional arguments for encoder initialization (as dict)

    Example:
        >>> model = ContrastiveModel(encoder_class="EEGNeX")
        >>> embeddings = model(eeg_input)  # Shape: (batch, 128)
    """

    def __init__(
        self,
        encoder_class: str = "EEGNeX",
        projection_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        input_scale: float = 1.0,
        encoder_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        # Register input scale as buffer (part of model state, not trainable)
        self.register_buffer('scale', torch.tensor(input_scale, dtype=torch.float32))

        # Build encoder with kwargs
        if encoder_kwargs is None:
            encoder_kwargs = {}
        self.encoder = build_encoder(encoder_class, **encoder_kwargs)

        # Build projection head
        self.head = ProjectionHead(
            input_dim=self.encoder.output_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and projection head.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)
               Expected in VOLT scale (e.g., -0.016 to 0.008 for typical EEG)

        Returns:
            Embeddings of shape (batch_size, projection_dim)
        """
        # Apply input scaling (e.g., volt → millivolt)
        if self.scale != 1.0:
            x = x * self.scale

        features = self.encoder(x)
        embeddings = self.head(features)
        return embeddings

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without the projection head.

        This is what you'd use after pretraining for downstream tasks.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)

        Returns:
            Features of shape (batch_size, feature_dim)
        """
        return self.encoder(x)


class MultitaskModel(nn.Module):
    """Multi-task model with shared encoder and task-specific heads.

    This model learns a shared representation that's useful for multiple
    tasks simultaneously. The encoder is shared across all tasks while
    each task gets its own head.

    Args:
        encoder_class: Name of encoder to use
        task_dims: Dictionary mapping task names to output dimensions
        dropout: Dropout probability before task heads (default: 0)
        encoder_kwargs: Additional arguments for encoder initialization (as dict)

    Example:
        >>> model = MultitaskModel(
        ...     encoder_class="EEGNeX",
        ...     task_dims={'rt': 1, 'p_factor': 1}
        ... )
        >>> outputs = model(eeg_input)
        >>> # outputs['rt'].shape = (batch, 1)
        >>> # outputs['p_factor'].shape = (batch, 1)
    """

    def __init__(
        self,
        encoder_class: str = "EEGNeX",
        task_dims: Optional[dict[str, int]] = None,
        dropout: float = 0.0,
        encoder_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        if task_dims is None:
            # Default: Challenge 1 and 2
            task_dims = {'challenge1': 1, 'challenge2': 1}

        # Build encoder with kwargs
        if encoder_kwargs is None:
            encoder_kwargs = {}
        self.encoder = build_encoder(encoder_class, **encoder_kwargs)

        # Build multi-task head
        self.head = MultiTaskHead(
            input_dim=self.encoder.output_dim,
            task_dims=task_dims,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through encoder and task-specific heads.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)

        Returns:
            Dictionary mapping task names to predictions
        """
        features = self.encoder(x)
        predictions = self.head(features)
        return predictions

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features without task heads.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)

        Returns:
            Features of shape (batch_size, feature_dim)
        """
        return self.encoder(x)


class LinearProbeModel(nn.Module):
    """Linear probe model for evaluating pretrained representations.

    Freezes a pretrained encoder and only trains a linear head.
    This is used to evaluate the quality of learned representations
    without fine-tuning the encoder.

    Args:
        encoder: Pretrained encoder (will be frozen)
        n_outputs: Number of outputs for the linear probe
        dropout: Dropout before linear layer (default: 0)

    Example:
        >>> pretrained = ContrastiveModel.load_from_checkpoint(...)
        >>> probe = LinearProbeModel(pretrained.encoder, n_outputs=1)
        >>> # Only probe.head will be trained
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_outputs: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()

        # Store encoder and freeze it
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Create trainable linear head
        self.head = RegressionHead(
            input_dim=self.encoder.output_dim,
            output_dim=n_outputs,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through frozen encoder and trainable head.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)

        Returns:
            Predictions of shape (batch_size, n_outputs)
        """
        with torch.no_grad():
            features = self.encoder(x)

        predictions = self.head(features)
        return predictions

class JEPAFoundationModel(nn.Module):
    """JEPA-based EEG foundation model with multi-scale dynamics.

    Implements the architecture from magnum_opus.md Phase 1.
    Uses Joint Embedding Predictive Architecture for self-supervised learning.

    Architecture:
    - Encoder: channel embed → spatial FNO → temporal Mamba → attention → 96-dim latent
    - Latent split: [trait:24, state:36, event:36] for multi-scale dynamics
    - Predictors: Different for each timescale (MLP for trait, Mamba for state/event)
    - Task heads: Added in later phases (behavior, psychopathology)

    Args:
        n_chans: Number of EEG channels (default: 129, HBN dataset includes Cz reference)
        n_times: Number of time samples (default: 200 = 2s @ 100Hz)
        latent_dim: Total latent dimension (default: 96)
        sample_rate: Sampling frequency (default: 100)
        fno_modes: Number of Fourier modes for spatial FNO
        mamba_d_state: State dimension for temporal Mamba
        mamba_layers: Number of Mamba layers

    Example:
        >>> model = JEPAFoundationModel(n_chans=129, latent_dim=96)
        >>> # Phase 1: Self-supervised pretraining
        >>> z = model.encode(eeg_input)
        >>> z_trait, z_state, z_event = model.split_latent(z)
        >>> # Predict future from past
        >>> z_trait_pred = model.predict_trait(z_trait)
    """

    def __init__(
        self,
        n_chans: int = 129,  # HBN dataset: 129 channels including Cz reference
        n_times: int = 200,  # 2s @ 100Hz
        latent_dim: int = 96,
        sample_rate: int = 100,
        fno_modes: int = 16,
        mamba_d_state: int = 16,
        mamba_layers: int = 4,
    ):
        super().__init__()

        self.n_chans = n_chans
        self.n_times = n_times
        self.latent_dim = latent_dim

        # Latent dimensions for each scale
        self.trait_dim = 24  # Slow: stable traits (psychopathology)
        self.state_dim = 36  # Medium: cognitive state
        self.event_dim = 36  # Fast: events/responses

        assert latent_dim == self.trait_dim + self.state_dim + self.event_dim, \
            f"latent_dim ({latent_dim}) must equal sum of trait/state/event dims (24+36+36=96)"

        # ========== ENCODER (Shared for all scales) ==========
        self.encoder = JEPAEncoder(
            n_chans=n_chans,
            n_times=n_times,
            latent_dim=latent_dim,
            sample_rate=sample_rate,
            fno_modes=fno_modes,
            mamba_d_state=mamba_d_state,
            mamba_layers=mamba_layers,
        )

        # ========== MULTI-SCALE PREDICTORS (JEPA-style) ==========
        self.trait_predictor = TraitPredictor(trait_dim=self.trait_dim)
        self.state_predictor = StatePredictor(state_dim=self.state_dim)
        self.event_predictor = EventPredictor(event_dim=self.event_dim)

        # ========== TASK HEADS (Added in Phase 3-4) ==========
        # Challenge 1: Behavioral prediction (RT + accuracy)
        self.behavior_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # RT + accuracy
        )

        # Challenge 2: Psychopathology prediction (4 factors)
        # Only uses trait dimensions
        self.psych_head = nn.Sequential(
            nn.Linear(self.trait_dim, 32),
            nn.GELU(),
            nn.Linear(32, 4)  # p_factor, internalizing, externalizing, attention
        )

    def encode(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor | tuple:
        """Encode EEG to latent representation.

        Args:
            x: Input EEG (batch, n_chans, n_times)
            return_intermediates: If True, return intermediate activations

        Returns:
            z: Latent representation (batch, latent_dim)
            intermediates: Dict of intermediate activations (if return_intermediates=True)
        """
        return self.encoder(x, return_intermediates=return_intermediates)

    def split_latent(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split latent into trait/state/event components.

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            z_trait: Trait latent (batch, 24) - slow, stable
            z_state: State latent (batch, 36) - medium dynamics
            z_event: Event latent (batch, 36) - fast dynamics
        """
        z_trait = z[..., :self.trait_dim]  # [:24]
        z_state = z[..., self.trait_dim:self.trait_dim + self.state_dim]  # [24:60]
        z_event = z[..., self.trait_dim + self.state_dim:]  # [60:]
        return z_trait, z_state, z_event

    def predict_trait(self, z_trait: torch.Tensor) -> torch.Tensor:
        """Predict future trait from past trait.

        Args:
            z_trait: Past trait latent (batch, 24)

        Returns:
            Predicted future trait (batch, 24)
        """
        return self.trait_predictor(z_trait)

    def predict_state(self, z_state: torch.Tensor) -> torch.Tensor:
        """Predict future state from past state.

        Args:
            z_state: Past state latent (batch, 36)

        Returns:
            Predicted future state (batch, 36)
        """
        return self.state_predictor(z_state)

    def predict_event(self, z_event: torch.Tensor) -> torch.Tensor:
        """Predict future event from past event.

        Args:
            z_event: Past event latent (batch, 36)

        Returns:
            Predicted future event (batch, 36)
        """
        return self.event_predictor(z_event)

    def predict_behavior(self, z: torch.Tensor) -> torch.Tensor:
        """Predict behavioral outcomes (Phase 3).

        Args:
            z: Full latent representation (batch, 96)

        Returns:
            Behavior predictions (batch, 2) - [RT, accuracy]
        """
        return self.behavior_head(z)

    def predict_psychopathology(self, z_trait: torch.Tensor) -> torch.Tensor:
        """Predict psychopathology factors (Phase 4).

        Args:
            z_trait: Trait latent (batch, 24)

        Returns:
            Psychopathology predictions (batch, 4) - [p_factor, int, ext, att]
        """
        return self.psych_head(z_trait)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for general use.

        Args:
            x: Input EEG (batch, n_chans, n_times)

        Returns:
            Latent representation (batch, latent_dim)
        """
        return self.encode(x)
