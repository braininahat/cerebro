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