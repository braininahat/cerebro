"""Encoder components that extract features from EEG data.

Encoders are the backbone of our models, transforming raw EEG signals
into learned feature representations. They are pure nn.Module classes
with no training logic.

Key design principles:
- Each encoder exposes its output dimension via the output_dim property
- Encoders can be from braindecode or custom implementations
- They output feature vectors, not task-specific predictions
"""

from typing import Optional

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """Abstract base class for all encoder components.

    All encoders must implement:
    - forward(): Transform input to features
    - output_dim property: Report feature dimension
    """

    @property
    def output_dim(self) -> int:
        """Output dimension of the encoder features."""
        raise NotImplementedError("Subclasses must implement output_dim property")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform EEG input to feature representation.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)

        Returns:
            Feature tensor of shape (batch_size, output_dim)
        """
        raise NotImplementedError("Subclasses must implement forward method")


class EEGNeXEncoder(BaseEncoder):
    """EEGNeX encoder wrapper for feature extraction.

    Wraps braindecode's EEGNeX model, removing the final classification
    layer to expose intermediate feature representations.

    Args:
        n_chans: Number of EEG channels (default: 129 for HBN)
        n_times: Number of time samples in window (default: 200 = 2s @ 100Hz)
        sfreq: Sampling frequency in Hz (default: 100)
        **kwargs: Additional arguments passed to EEGNeX (for flexibility)
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_times: int = 200,
        sfreq: int = 100,
        **kwargs  # Accept any additional args for flexibility
    ):
        super().__init__()

        # Import here to avoid circular imports and defer braindecode dependency
        try:
            from braindecode.models import EEGNeX
        except ImportError:
            raise ImportError(
                "braindecode not found. Install with: pip install braindecode"
            )

        # Create full EEGNeX model with dummy output dimension
        # We'll extract features before the final layer
        # EEGNeX doesn't accept n_filters or n_layers, so we don't pass them
        self.full_model = EEGNeX(
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=1,  # Dummy value, we won't use the final layer
        )

        # Find the feature extraction point (before final classifier)
        # EEGNeX structure: conv blocks -> flatten -> dropout -> linear
        # We want features after flatten but before final linear
        modules = list(self.full_model.children())

        # The last layer is typically the classifier
        # Store it separately to get its input dimension
        self.final_layer = modules[-1]

        # Create feature extractor (everything except final layer)
        self.features = nn.Sequential(*modules[:-1])

        # Infer output dimension by checking final layer's input size
        if hasattr(self.final_layer, 'in_features'):
            self._output_dim = self.final_layer.in_features
        else:
            # Fallback: forward dummy input to measure
            with torch.no_grad():
                dummy = torch.zeros(1, n_chans, n_times)
                dummy_features = self.features(dummy)
                self._output_dim = dummy_features.shape[-1]

    @property
    def output_dim(self) -> int:
        """Output dimension of EEGNeX features."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from EEG input.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)

        Returns:
            Feature tensor of shape (batch_size, output_dim)
        """
        return self.features(x)


class SignalJEPAEncoder(BaseEncoder):
    """SignalJEPA encoder wrapper for feature extraction.

    Wraps braindecode's SignalJEPA_PreLocal model for self-supervised
    feature learning inspired by Joint Embedding Predictive Architectures.

    Args:
        n_chans: Number of EEG channels
        n_times: Number of time samples in window
        sfreq: Sampling frequency in Hz
        n_outputs: Feature dimension (used to infer output_dim)
        **kwargs: Additional arguments passed to SignalJEPA_PreLocal
            Common kwargs include:
            - n_spat_filters: Number of spatial filters
            - feature_encoder__conv_layers_spec: Conv layer specifications
            - feature_encoder__mode: Feature encoder mode
            - feature_encoder__conv_bias: Whether to use bias in convs
            - drop_prob: Dropout probability
            - pos_encoder__spat_dim: Spatial positional encoding dimension
            - pos_encoder__time_dim: Temporal positional encoding dimension
            - pos_encoder__sfreq_features: Sampling freq for positional encoding
            - pos_encoder__spat_kwargs: Additional spatial encoder kwargs
            - transformer__d_model: Transformer model dimension
            - transformer__num_encoder_layers: Number of encoder layers
            - transformer__num_decoder_layers: Number of decoder layers
            - transformer__nhead: Number of attention heads
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_times: int = 200,
        sfreq: int = 100,
        n_outputs: int = 128,
        **kwargs
    ):
        super().__init__()

        try:
            from braindecode.models import SignalJEPA_PreLocal
        except ImportError:
            raise ImportError(
                "braindecode not found. Install with: pip install braindecode"
            )

        # Create SignalJEPA model with all kwargs passed through
        # SignalJEPA uses double-underscore notation for nested parameters
        self.model = SignalJEPA_PreLocal(
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=n_outputs,
            **kwargs
        )

        self._output_dim = n_outputs

    @property
    def output_dim(self) -> int:
        """Output dimension of SignalJEPA features."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from EEG input using SignalJEPA.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)

        Returns:
            Feature tensor of shape (batch_size, output_dim)
        """
        return self.model(x)


class LearnableChannelPositionalEncoder(nn.Module):
    """Perceiver-style positional encoder with learnable electrode queries.

    Replaces fixed channel location encoding with learnable queries that attend
    to observed channels via cross-attention. Enables variable channel counts
    (21 for TUH, 129 for HBN) by learning a fixed set of electrode representations.

    Architecture:
        - Fixed 128 learnable electrode queries (spatial priors)
        - Cross-attention: queries=electrodes, keys/values=observed channels
        - Always outputs 128 channels regardless of input count
        - Temporal positional encoding remains sinusoidal

    Args:
        n_electrodes: Number of learnable electrode queries (default: 128)
        spat_dim: Spatial embedding dimension
        time_dim: Temporal embedding dimension
        d_model: Total embedding dimension (must be >= spat_dim + time_dim)
        nhead: Number of attention heads for cross-attention
        sfreq_features: Sampling frequency after feature encoder
        max_seconds: Maximum duration for temporal encoding
    """

    def __init__(
        self,
        n_electrodes: int = 128,
        spat_dim: int = 30,
        time_dim: int = 34,
        d_model: int = 64,
        nhead: int = 6,  # Must divide spat_dim=30 evenly (30/6=5)
        sfreq_features: float = 1.0,
        max_seconds: float = 600.0,
    ):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.spat_dim = spat_dim
        self.time_dim = time_dim
        self.max_n_times = int(max_seconds * sfreq_features)

        # Learnable electrode queries (spatial priors)
        self.electrode_queries = nn.Parameter(
            torch.randn(n_electrodes, spat_dim)
        )
        nn.init.normal_(self.electrode_queries, mean=0.0, std=0.02)

        # Cross-attention for electrode queries to attend to observed channels
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=spat_dim,
            num_heads=nhead,
            batch_first=True,
        )

        # Pre-computed sinusoidal temporal encoding
        self.encoding_time = torch.zeros(0, dtype=torch.float32, requires_grad=False)

    def _check_encoding_time(self, n_times: int):
        """Ensure temporal encoding buffer is large enough."""
        if self.encoding_time.size(0) < n_times:
            # Import from braindecode
            from braindecode.models.signal_jepa import _pos_encode_time

            self.encoding_time = self.encoding_time.new_empty((n_times, self.time_dim))
            self.encoding_time[:] = _pos_encode_time(
                n_times=n_times,
                n_dim=self.time_dim,
                max_n_times=self.max_n_times,
                device=self.encoding_time.device,
            )

    def forward(
        self,
        local_features: torch.Tensor,
        n_chans: int,
        ch_idxs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply learnable spatial encoding via cross-attention and sinusoidal temporal encoding.

        Args:
            local_features: (batch_size, n_chans * n_times_out, emb_dim)
                Features from convolutional encoder
            n_chans: Number of input channels (needed to split channel/time dimensions)
            ch_idxs: Ignored (for API compatibility with base SignalJEPA)

        Returns:
            pos_encoding: (batch_size, n_electrodes * n_times_out, emb_dim)
                Positional encoding with learned spatial component and sinusoidal temporal
        """
        batch_size, n_chans_times, emb_dim = local_features.shape
        n_times = n_chans_times // n_chans

        # Reshape to separate channels and time: (batch, n_chans, n_times, emb_dim)
        features_reshaped = local_features.view(batch_size, n_chans, n_times, emb_dim)

        # Extract spatial features for cross-attention (pool over time)
        # Shape: (batch, n_chans, emb_dim)
        channel_features = features_reshaped.mean(dim=2)  # Average over time

        # Project to spatial dimension if needed
        if emb_dim > self.spat_dim:
            channel_features = channel_features[..., :self.spat_dim]
        elif emb_dim < self.spat_dim:
            # Pad with zeros if needed
            pad = channel_features.new_zeros(batch_size, n_chans, self.spat_dim - emb_dim)
            channel_features = torch.cat([channel_features, pad], dim=-1)

        # Cross-attention: electrode queries attend to observed channels
        # queries: (batch, n_electrodes, spat_dim)
        # keys/values: (batch, n_chans, spat_dim)
        queries = self.electrode_queries.unsqueeze(0).expand(batch_size, -1, -1)
        electrode_encoding, _ = self.cross_attn(
            query=queries,
            key=channel_features,
            value=channel_features,
        )
        # electrode_encoding: (batch, n_electrodes, spat_dim)

        # Prepare output tensor: (batch, n_electrodes, n_times, emb_dim)
        pos_encoding = local_features.new_zeros(
            (batch_size, self.n_electrodes, n_times, emb_dim)
        )

        # Spatial encoding from learned electrodes
        pos_encoding[:, :, :, :self.spat_dim] = electrode_encoding[:, :, None, :]

        # Temporal encoding (sinusoidal)
        self._check_encoding_time(n_times)
        pos_encoding[:, :, :, self.spat_dim : self.spat_dim + self.time_dim].copy_(
            self.encoding_time[None, None, :n_times, :],
        )

        # Reshape to match expected output: (batch, n_electrodes * n_times, emb_dim)
        return pos_encoding.view(batch_size, self.n_electrodes * n_times, emb_dim)


class SignalJEPAWithLearnedChannels(BaseEncoder):
    """SignalJEPA encoder with learnable channel positions instead of fixed coordinates.

    This encoder adapts the base SignalJEPA architecture to work with datasets that
    lack channel location information (e.g., HBN with NaN locations) and to unify
    models across different channel counts (e.g., TUH 21 channels, HBN 129 channels).

    Key differences from base SignalJEPA:
        1. Replaces `_PosEncoder` with `LearnableChannelPositionalEncoder`
        2. Uses Perceiver-style cross-attention with 128 learnable electrode queries
        3. Always outputs 128-channel representations regardless of input
        4. No requirement for channel location metadata

    Architecture:
        Input (B, C_observed, T) →
        Feature Encoder (conv layers) →
        Learnable Position Encoder (cross-attention) →
        Transformer Encoder →
        Output (B, 128, d_model)

    Use cases:
        - Multi-phase training (Pretrain → Contrastive → Supervised)
        - Cross-dataset pretraining (TUH + HBN)
        - Any dataset with missing or unreliable channel locations

    Args:
        n_chans: Number of input channels (e.g., 21 for TUH, 129 for HBN)
        n_times: Number of time samples
        sfreq: Sampling frequency in Hz
        n_electrodes: Number of learnable electrode queries (default: 128)
        **kwargs: Additional arguments passed to SignalJEPA components
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_times: int = 200,
        sfreq: int = 100,
        n_electrodes: int = 128,
        # Feature encoder parameters
        feature_encoder__conv_layers_spec=None,
        drop_prob: float = 0.0,
        feature_encoder__mode: str = "default",
        feature_encoder__conv_bias: bool = False,
        activation: Optional[type] = None,  # Lightning CLI compatible
        # Positional encoder parameters
        pos_encoder__spat_dim: int = 30,
        pos_encoder__time_dim: int = 34,
        pos_encoder__sfreq_features: float = 1.0,
        pos_encoder__nhead: int = 6,  # Must divide spat_dim evenly
        # Transformer parameters
        transformer__d_model: int = 64,
        transformer__num_encoder_layers: int = 8,
        transformer__nhead: int = 8,
        **kwargs
    ):
        super().__init__()

        try:
            from braindecode.models.signal_jepa import (
                _ConvFeatureEncoder,
                _DEFAULT_CONV_LAYER_SPEC,
            )
        except ImportError:
            raise ImportError(
                "braindecode not found. Install with: pip install braindecode"
            )

        # Default handling for Lightning CLI compatibility
        if feature_encoder__conv_layers_spec is None:
            feature_encoder__conv_layers_spec = _DEFAULT_CONV_LAYER_SPEC
        if activation is None:
            activation = nn.GELU

        self.n_chans = n_chans
        self.n_times = n_times
        self.n_electrodes = n_electrodes

        # Feature encoder (same as base SignalJEPA)
        self.feature_encoder = _ConvFeatureEncoder(
            conv_layers_spec=feature_encoder__conv_layers_spec,
            channels=n_chans,
            drop_prob=drop_prob,
            mode=feature_encoder__mode,
            conv_bias=feature_encoder__conv_bias,
            activation=activation,
        )

        # Learnable positional encoder (replaces fixed channel locations)
        self.pos_encoder = LearnableChannelPositionalEncoder(
            n_electrodes=n_electrodes,
            spat_dim=pos_encoder__spat_dim,
            time_dim=pos_encoder__time_dim,
            d_model=transformer__d_model,
            nhead=pos_encoder__nhead,
            sfreq_features=pos_encoder__sfreq_features,
        )

        # Transformer encoder (same as base SignalJEPA)
        self.transformer = nn.Transformer(
            d_model=transformer__d_model,
            nhead=transformer__nhead,
            num_encoder_layers=transformer__num_encoder_layers,
            num_decoder_layers=0,  # Only use encoder
            batch_first=True,
        )

        self._output_dim = transformer__d_model

    @property
    def output_dim(self) -> int:
        """Output dimension of encoder features."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using SignalJEPA with learned channel positions.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)
               Can have variable n_channels (e.g., 21 or 129)

        Returns:
            Feature tensor of shape (batch_size, n_electrodes*n_times_out, d_model)
            Always outputs n_electrodes (default 128) representations
        """
        batch_size, n_chans, n_times = x.shape

        # Extract local features via conv layers
        local_features = self.feature_encoder(x)  # (B, C*T', d_model)

        # Get learnable positional encoding (transforms C → n_electrodes)
        pos_features = self.pos_encoder(local_features, n_chans)  # (B, n_electrodes*T', d_model)

        # Process through transformer encoder
        contextual_features = self.transformer.encoder(pos_features)

        # Return full sequence (B, n_electrodes*T', d_model)
        # Downstream tasks can pool/aggregate as needed
        return contextual_features


class VanillaSignalJEPAEncoder(BaseEncoder):
    """Vanilla SignalJEPA encoder from braindecode with real electrode locations.

    This encoder wraps braindecode's base SignalJEPA model for self-supervised
    pretraining. It uses actual electrode coordinates from the GSN-HydroCel-129
    montage file, providing spatial positional encoding based on real 3D positions.

    Key differences from SignalJEPAWithLearnedChannels:
        - Uses fixed electrode locations (from .sfp file)
        - Requires chs_info metadata with channel locations
        - Follows original SignalJEPA paper design

    Use this encoder when:
        - You want to use actual electrode positions
        - Comparing learned vs. fixed positional encoding
        - Following the original SignalJEPA methodology

    Args:
        n_chans: Number of input channels (default: 129 for HBN)
        n_times: Number of time samples (default: 200 = 2s @ 100Hz)
        sfreq: Sampling frequency in Hz (default: 100)
        drop_prob: Dropout probability (default: 0.0)
        transformer__d_model: Transformer model dimension (default: 64)
        transformer__num_encoder_layers: Number of encoder layers (default: 8)
        transformer__nhead: Number of attention heads (default: 8)
        **kwargs: Additional arguments passed to SignalJEPA

    Note:
        Electrode locations are automatically loaded from
        docs/GSN_HydroCel_129_AdjustedLabels.sfp using the
        cerebro.utils.electrode_locations.load_hbn_chs_info() utility.
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_times: int = 200,
        sfreq: int = 100,
        drop_prob: float = 0.0,
        transformer__d_model: int = 64,
        transformer__num_encoder_layers: int = 8,
        transformer__nhead: int = 8,
        **kwargs
    ):
        super().__init__()

        try:
            from braindecode.models import SignalJEPA
        except ImportError:
            raise ImportError(
                "braindecode not found. Install with: pip install braindecode"
            )

        # Load actual HBN electrode locations
        from cerebro.utils.electrode_locations import load_hbn_chs_info
        chs_info = load_hbn_chs_info()

        # Create SignalJEPA model with real electrode positions
        self.model = SignalJEPA(
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            chs_info=chs_info,  # CRITICAL: Provides spatial positional encoding
            drop_prob=drop_prob,
            transformer__d_model=transformer__d_model,
            transformer__num_encoder_layers=transformer__num_encoder_layers,
            transformer__nhead=transformer__nhead,
            **kwargs
        )

        self._output_dim = transformer__d_model

    @property
    def output_dim(self) -> int:
        """Output dimension of encoder features."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using vanilla SignalJEPA.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)
               Must have n_channels=129 matching the HBN electrode montage

        Returns:
            Feature tensor of shape (batch_size, n_channels*n_times_out, d_model)
            The exact shape depends on the feature encoder's downsampling
        """
        return self.model(x)


class IdentityEncoder(BaseEncoder):
    """Identity encoder for testing and baseline comparisons.

    Simply flattens the input without any learned transformations.
    Useful for testing the rest of the pipeline or as a baseline.

    Args:
        n_chans: Number of EEG channels
        n_times: Number of time samples
    """

    def __init__(self, n_chans: int = 129, n_times: int = 200, **kwargs):
        super().__init__()
        self._output_dim = n_chans * n_times
        self.flatten = nn.Flatten()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten the input tensor.

        Args:
            x: Input tensor of shape (batch_size, n_channels, n_times)

        Returns:
            Flattened tensor of shape (batch_size, n_channels * n_times)
        """
        return self.flatten(x)