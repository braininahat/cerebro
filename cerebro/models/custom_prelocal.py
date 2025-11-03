"""Custom PreLocal implementation with proper pretrained weight transfer.

This module provides a custom SignalJEPA_PreLocal variant that properly handles
weight transfer from pretrained models despite the channel dimension mismatch.

The key insight: We can transfer all pretrained weights EXCEPT the first conv layer,
which needs to be retrained due to the spatial filtering changing input dimensions
from 129 channels to n_spat_filters channels.
"""

from copy import deepcopy
from typing import Optional, Any

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class CustomPreLocal(nn.Module):
    """Custom PreLocal architecture with proper pretrained weight transfer.

    Architecture:
        1. Spatial conv: 129 channels → n_spat_filters virtual channels (NEW)
        2. Feature encoder: Process virtual channels (MOSTLY PRETRAINED)
        3. Fully connected: Map to outputs (NEW)

    Weight transfer strategy:
        - Spatial conv layer: Trained from scratch (new layer)
        - Feature encoder first conv: Trained from scratch (dimension mismatch)
        - Feature encoder other layers: Transfer pretrained weights
        - Fully connected: Trained from scratch (new layer)

    This gives us ~90% of the pretrained benefits while handling the dimension issue.

    Args:
        n_chans: Number of input EEG channels (129 for HBN)
        n_times: Number of time samples (200 = 2s @ 100Hz)
        sfreq: Sampling frequency in Hz
        n_outputs: Number of outputs for downstream task
        n_spat_filters: Number of spatial filters/virtual channels (default: 4)
        pretrained_encoder: Optional pretrained SignalJEPA encoder to transfer from
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_times: int = 200,
        sfreq: int = 100,
        n_outputs: int = 1,
        n_spat_filters: int = 4,
        pretrained_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = sfreq
        self.n_outputs = n_outputs
        self.n_spat_filters = n_spat_filters

        # ===== STEP 1: Spatial Convolution (NEW) =====
        # Reduces spatial dimensions from n_chans to n_spat_filters
        self.spatial_conv = nn.Sequential(
            Rearrange("batch channels time -> batch 1 channels time"),
            nn.Conv2d(1, n_spat_filters, kernel_size=(n_chans, 1), bias=False),
            Rearrange("batch filters 1 time -> batch filters time"),
        )

        # ===== STEP 2: Feature Encoder (PARTIALLY PRETRAINED) =====
        if pretrained_encoder is not None:
            # Clone the pretrained encoder structure
            self.feature_encoder = self._create_modified_encoder(pretrained_encoder)
            self._transfer_pretrained_weights(pretrained_encoder)
        else:
            # Create from scratch if no pretrained model
            self.feature_encoder = self._create_default_encoder()

        # Calculate output dimension after feature encoding
        self.feature_dim = self._calculate_feature_dim()

        # ===== STEP 3: Classification Head (NEW) =====
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, n_outputs)
        )

    def _create_modified_encoder(self, pretrained_encoder: nn.Module) -> nn.Module:
        """Create a feature encoder with modified first layer for n_spat_filters input.

        Clones the pretrained encoder structure but replaces the first conv layer
        to accept n_spat_filters channels instead of n_chans.

        Also removes the Rearrange layer at index 0 which validates channel dimensions.
        """
        # Access the braindecode SignalJEPA's feature encoder
        if hasattr(pretrained_encoder, 'model') and hasattr(pretrained_encoder.model, 'feature_encoder'):
            original_encoder = pretrained_encoder.model.feature_encoder
        else:
            raise ValueError("Pretrained encoder doesn't have expected structure")

        # Clone the encoder structure and convert to list for modification
        new_encoder = deepcopy(original_encoder)
        new_encoder_list = list(new_encoder)

        # Note: Encoder structure has nested Sequential blocks with Rearrange layers
        # that need to be removed for n_spat_filters input compatibility

        # Braindecode's feature encoder structure (original):
        # 0: Rearrange (validates channel dimensions - must be removed!)
        # 1: Conv1d (first conv - needs modification for n_spat_filters input)
        # 2: GroupNorm
        # 3: GELU
        # 4: Dropout
        # 5: Conv1d (second conv)
        # ... etc

        # Step 1: Remove ALL Rearrange layers (they validate/reshape based on 129 channels)
        # There can be multiple: one at the start for validation, one at the end for output reshaping
        rearrange_removed = 0
        new_encoder_list_filtered = []
        for i, layer in enumerate(new_encoder_list):
            if isinstance(layer, Rearrange):
                print(f"[CustomPreLocal] Removing Rearrange layer at index {i}")
                rearrange_removed += 1
            else:
                new_encoder_list_filtered.append(layer)
        new_encoder_list = new_encoder_list_filtered
        print(f"[CustomPreLocal] Removed {rearrange_removed} Rearrange layer(s)")

        # Step 2: Replace the first Conv1d to accept n_spat_filters input
        # The Conv1d might be directly at index 0, or nested in a Sequential block
        if len(new_encoder_list) > 0:
            if isinstance(new_encoder_list[0], nn.Conv1d):
                # Direct Conv1d at index 0
                old_conv = new_encoder_list[0]
                new_encoder_list[0] = nn.Conv1d(
                    in_channels=self.n_spat_filters,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    dilation=old_conv.dilation,
                    groups=1,
                    bias=old_conv.bias is not None
                )
                print(f"[CustomPreLocal] Replaced first conv (direct): {self.n_chans}→{self.n_spat_filters} input channels")
            elif isinstance(new_encoder_list[0], nn.Sequential):
                # Conv1d nested in Sequential block at index 0
                first_block = list(new_encoder_list[0])
                if len(first_block) > 0 and isinstance(first_block[0], nn.Conv1d):
                    old_conv = first_block[0]
                    first_block[0] = nn.Conv1d(
                        in_channels=self.n_spat_filters,
                        out_channels=old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        dilation=old_conv.dilation,
                        groups=1,
                        bias=old_conv.bias is not None
                    )
                    # Replace the Sequential block with modified version
                    new_encoder_list[0] = nn.Sequential(*first_block)
                    print(f"[CustomPreLocal] Replaced first conv (nested): {old_conv.in_channels}→{self.n_spat_filters} input channels")
                else:
                    print(f"[CustomPreLocal] WARNING: First Sequential doesn't start with Conv1d")
            else:
                print(f"[CustomPreLocal] WARNING: Layer 0 is neither Conv1d nor Sequential, it's {type(new_encoder_list[0]).__name__}")

        # Convert back to Sequential
        return nn.Sequential(*new_encoder_list)

    def _transfer_pretrained_weights(self, pretrained_encoder: nn.Module):
        """Transfer pretrained weights for all layers except the first conv.

        This preserves ~90% of the pretrained representations while handling
        the dimension mismatch in the first layer.

        Note: After removing the Rearrange layer (index 0), all layer indices shift down by 1.
        Pretrained layer 2 → Current layer 1, Pretrained layer 3 → Current layer 2, etc.
        """
        if hasattr(pretrained_encoder, 'model') and hasattr(pretrained_encoder.model, 'feature_encoder'):
            pretrained_fe = pretrained_encoder.model.feature_encoder
        else:
            return  # No weights to transfer

        # Get state dicts
        pretrained_state = pretrained_fe.state_dict()
        current_state = self.feature_encoder.state_dict()

        # Transfer weights with index adjustment for nested Sequential structure
        # Structure: 0=Rearrange (removed), 1=Sequential(Conv1d, ...), 2=Sequential(...), ...
        # Skip: 0.* (Rearrange) and 1.0.* (first Conv1d with dimension mismatch)
        # Map: Pretrained N.* → Current (N-1).* for N >= 1
        transferred = 0
        skipped = 0

        for key in pretrained_state.keys():
            # Skip Rearrange layer (index 0) - it was removed
            if key.startswith("0."):
                skipped += 1
                # print(f"[CustomPreLocal] Skipping {key} (Rearrange layer removed)")
                continue

            # Skip first Conv1d (index 1.0) - dimension mismatch
            # It's nested in the first Sequential block at position 1.0
            if key.startswith("1.0."):
                skipped += 1
                print(f"[CustomPreLocal] Skipping {key} (first Conv1d dimension mismatch)")
                continue

            # Map pretrained layer N to current layer N-1 (due to Rearrange removal)
            # Examples:
            #   "1.1.weight" (Dropout in block 1) → "0.1.weight"
            #   "1.2.weight" (GroupNorm in block 1) → "0.2.weight"
            #   "2.0.weight" (Conv1d in block 2) → "1.0.weight"
            try:
                # Parse first index from key (e.g., "1.2.weight" → 1)
                parts = key.split(".", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    old_idx = int(parts[0])
                    new_idx = old_idx - 1  # Shift down by 1 due to Rearrange removal
                    new_key = f"{new_idx}.{parts[1]}"

                    if new_key in current_state and pretrained_state[key].shape == current_state[new_key].shape:
                        current_state[new_key] = pretrained_state[key]
                        transferred += 1
                    else:
                        skipped += 1
                        # Only print warnings for unexpected skips (not Rearrange or first Conv)
                        if not key.startswith("0.") and not key.startswith("1.0."):
                            print(f"[CustomPreLocal] Warning: Couldn't transfer {key} → {new_key} (shape mismatch)")
                else:
                    skipped += 1
                    print(f"[CustomPreLocal] Warning: Unexpected key format: {key}")
            except Exception as e:
                skipped += 1
                print(f"[CustomPreLocal] Error transferring {key}: {e}")

        # Load the modified state dict
        self.feature_encoder.load_state_dict(current_state)
        print(f"[CustomPreLocal] Transferred {transferred} params, skipped {skipped} params")

    def _create_default_encoder(self) -> nn.Module:
        """Create a default feature encoder from scratch."""
        # Default CNN architecture matching SignalJEPA's feature encoder
        return nn.Sequential(
            Rearrange("batch channels time -> batch channels time"),
            # Conv block 1
            nn.Conv1d(self.n_spat_filters, 8, kernel_size=25, stride=12),
            nn.GroupNorm(1, 8),
            nn.GELU(),
            nn.Dropout(0.0),
            # Conv block 2
            nn.Conv1d(8, 16, kernel_size=2, stride=2),
            nn.GroupNorm(1, 16),
            nn.GELU(),
            nn.Dropout(0.0),
            # Conv block 3
            nn.Conv1d(16, 32, kernel_size=2, stride=2),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            nn.Dropout(0.0),
            # Conv block 4
            nn.Conv1d(32, 32, kernel_size=2, stride=2),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            nn.Dropout(0.0),
            # Conv block 5
            nn.Conv1d(32, 64, kernel_size=2, stride=2),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            nn.Dropout(0.0),
        )

    def _calculate_feature_dim(self) -> int:
        """Calculate the flattened dimension after feature encoding."""
        # Create dummy input to trace dimensions
        dummy_input = torch.zeros(1, self.n_chans, self.n_times)

        # Pass through spatial conv
        spatial_out = self.spatial_conv(dummy_input)

        # Pass through feature encoder
        features = self.feature_encoder(spatial_out)

        # Flatten and get dimension
        flattened = features.flatten(start_dim=1)
        return flattened.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the custom PreLocal architecture.

        Args:
            x: Input EEG tensor of shape (batch, n_chans, n_times)

        Returns:
            Output predictions of shape (batch, n_outputs)
        """
        # Step 1: Spatial filtering (129 → n_spat_filters)
        x = self.spatial_conv(x)

        # Step 2: Feature encoding (mostly using pretrained weights)
        x = self.feature_encoder(x)

        # Step 3: Classification
        x = self.classifier(x)

        return x

    @classmethod
    def from_pretrained(
        cls,
        pretrained_encoder: nn.Module,
        n_outputs: int = 1,
        n_spat_filters: int = 4,
        **kwargs
    ) -> "CustomPreLocal":
        """Create CustomPreLocal from a pretrained SignalJEPA encoder.

        Args:
            pretrained_encoder: Pretrained VanillaSignalJEPAEncoder or SignalJEPA model
            n_outputs: Number of outputs for downstream task
            n_spat_filters: Number of spatial filters
            **kwargs: Additional arguments passed to constructor

        Returns:
            CustomPreLocal model with transferred pretrained weights
        """
        # Extract dimensions from pretrained model
        if hasattr(pretrained_encoder, 'model'):
            n_chans = pretrained_encoder.model.n_chans
            n_times = pretrained_encoder.model.n_times
            sfreq = pretrained_encoder.model.sfreq
        else:
            # Default values if not accessible
            n_chans = kwargs.get('n_chans', 129)
            n_times = kwargs.get('n_times', 200)
            sfreq = kwargs.get('sfreq', 100)

        # Create the custom PreLocal with pretrained encoder
        model = cls(
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=n_outputs,
            n_spat_filters=n_spat_filters,
            pretrained_encoder=pretrained_encoder,
        )

        print(f"[CustomPreLocal] Created model with {n_spat_filters} spatial filters")
        print(f"[CustomPreLocal] Using pretrained weights for most of feature encoder")

        return model


def create_custom_prelocal_from_checkpoint(
    checkpoint_path: str,
    n_outputs: int = 1,
    n_spat_filters: int = 4,
    device: str = 'cpu'
) -> CustomPreLocal:
    """Convenience function to create CustomPreLocal from a checkpoint file.

    Args:
        checkpoint_path: Path to pretrained checkpoint
        n_outputs: Number of outputs
        n_spat_filters: Number of spatial filters
        device: Device to load model on

    Returns:
        CustomPreLocal model with pretrained weights
    """
    import torch
    from cerebro.models.components.encoders import VanillaSignalJEPAEncoder

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create base encoder
    base_encoder = VanillaSignalJEPAEncoder()

    # Load pretrained weights into encoder
    encoder_state = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('encoder.'):
            new_key = key.replace('encoder.', '', 1)
            encoder_state[new_key] = value

    if encoder_state:
        base_encoder.model.load_state_dict(encoder_state, strict=False)
        print(f"[CustomPreLocal] Loaded {len(encoder_state)} pretrained parameters")

    # Create CustomPreLocal with pretrained encoder
    model = CustomPreLocal.from_pretrained(
        pretrained_encoder=base_encoder,
        n_outputs=n_outputs,
        n_spat_filters=n_spat_filters
    )

    return model