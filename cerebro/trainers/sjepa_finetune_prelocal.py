"""S-JEPA fine-tuning with PreLocal architecture.

PreLocal architecture (paper's best approach):
    Spatial conv (n_chans → n_spat_filters)
    → Local encoder (per-channel CNN)
    → Flatten
    → Linear head

Applies spatial filtering BEFORE the local encoder, allowing the encoder
to learn features from learned "virtual channels" rather than fixed electrodes.
This provides more flexibility than post-hoc spatial filtering.

Per paper (Figure 5), PreLocal + full fine-tuning achieves best results:
    "16s-60% × full-pre-local occupies rank 1 in two-thirds of cases"
"""

import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from braindecode.models import SignalJEPA, SignalJEPA_PreLocal

from cerebro.trainers.sjepa_finetune_base import BaseSJEPAFinetuneTrainer


class PreLocalFinetuneTrainer(BaseSJEPAFinetuneTrainer):
    """Fine-tuning trainer using SignalJEPA_PreLocal architecture.

    Loads pretrained SignalJEPA base model and transfers weights to
    SignalJEPA_PreLocal variant via from_pretrained() method.

    Args:
        pretrained_checkpoint: Path to pretrained S-JEPA checkpoint
        n_outputs: Number of outputs (1 for regression)
        n_spat_filters: Number of spatial filters (virtual channels, typically 4-16)
        freeze_encoder: If True, freeze encoder permanently. If False, unfreeze after warmup.
        warmup_epochs: Warmup period before unfreezing (paper uses 10)
        **kwargs: Additional arguments for BaseSJEPAFinetuneTrainer
    """

    def __init__(
        self,
        pretrained_checkpoint: str,
        n_outputs: int = 1,
        n_spat_filters: int = 4,
        freeze_encoder: bool = True,
        warmup_epochs: int = 10,
        **kwargs
    ):
        self.n_spat_filters = n_spat_filters
        super().__init__(
            pretrained_checkpoint=pretrained_checkpoint,
            n_outputs=n_outputs,
            freeze_encoder=freeze_encoder,
            warmup_epochs=warmup_epochs,
            **kwargs
        )

    def _load_model(self, checkpoint_path: str, n_outputs: int) -> nn.Module:
        """Instantiate PreLocal model (no pretrained weights transferred).

        IMPORTANT: PreLocal applies spatial convolution BEFORE the local encoder,
        reducing channels from 129 → n_spat_filters. This means its feature encoder
        expects n_spat_filters input channels, NOT 129.

        The pretrained feature_encoder (trained on 129 channels) cannot be transferred
        due to this architecture mismatch. PreLocal learns spatial filtering and local
        features together from scratch during fine-tuning.

        This aligns with the paper's finding that PreLocal works best with full
        fine-tuning (not frozen encoder), since it needs to learn from scratch anyway.

        Args:
            checkpoint_path: Path to pretrained S-JEPA checkpoint (currently unused)
            n_outputs: Number of outputs for fine-tuning task

        Returns:
            SignalJEPA_PreLocal model with randomly initialized weights
        """
        print(f"[PRELOCAL] Creating PreLocal model from scratch (no weight transfer)")
        print(f"[PRELOCAL] Reason: Spatial conv (129→{self.n_spat_filters}) makes feature encoder incompatible")

        # Load HBN electrode locations (required for spatial positional encoding)
        from cerebro.utils.electrode_locations import load_hbn_chs_info
        chs_info = load_hbn_chs_info()

        # Instantiate PreLocal directly (random initialization)
        model = SignalJEPA_PreLocal(
            n_chans=129,  # HBN dataset
            sfreq=100,
            n_times=200,  # 2.0s windows at 100Hz
            input_window_seconds=2.0,
            chs_info=chs_info,  # CRITICAL: Electrode positions for spatial encoding
            n_outputs=n_outputs,
            n_spat_filters=self.n_spat_filters,
            transformer__d_model=64,
            transformer__num_encoder_layers=8,
            transformer__nhead=8,
        )
        print(f"[PRELOCAL] ✓ Created PreLocal model with {self.n_spat_filters} spatial filters")

        return model

    def _extract_encoder_state(self, checkpoint: dict) -> dict:
        """Extract encoder state dict from checkpoint and normalize keys.

        Handles:
        - Stripping "encoder." prefix from keys
        - Mapping old format to new format (feature_1 → feature_encoder.1, etc.)

        Args:
            checkpoint: Loaded checkpoint dict with 'state_dict' key

        Returns:
            Normalized encoder state dict
        """
        # Extract encoder keys (remove "encoder." prefix)
        encoder_state = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '', 1)
                encoder_state[new_key] = value

        if not encoder_state:
            # Fallback: maybe keys don't have "encoder." prefix
            encoder_state = checkpoint['state_dict'].copy()

        print(f"[EXTRACT] Sample keys before normalization: {list(encoder_state.keys())[:5]}")

        # Detect checkpoint format and normalize if needed
        sample_keys = list(encoder_state.keys())[:10]
        is_old_format = any(
            'feature_1' in k or 'feature_2' in k or 'feature_3' in k or
            'pos_pos_encoder_spat' in k or
            ('transformer.layers' in k and 'transformer.encoder.layers' not in k)
            for k in sample_keys
        )

        if is_old_format:
            print("[EXTRACT] Detected old checkpoint format, normalizing keys...")
            normalized_state = {}
            for key, value in encoder_state.items():
                new_key = key
                # Map feature_N → feature_encoder.N
                new_key = re.sub(r'feature_(\d+)', r'feature_encoder.\1', new_key)
                # Map pos_pos_encoder_spat → pos_encoder.pos_encoder_spat
                if 'pos_pos_encoder_spat' in new_key:
                    new_key = new_key.replace('pos_pos_encoder_spat', 'pos_encoder.pos_encoder_spat')
                # Map pos_pos_encoder_temp → pos_encoder.pos_encoder_temp
                if 'pos_pos_encoder_temp' in new_key:
                    new_key = new_key.replace('pos_pos_encoder_temp', 'pos_encoder.pos_encoder_temp')
                # Map transformer.layers → transformer.encoder.layers
                if 'transformer.layers' in new_key and 'transformer.encoder.layers' not in new_key:
                    new_key = new_key.replace('transformer.layers', 'transformer.encoder.layers')
                # Map transformer.norm → transformer.encoder.norm
                if 'transformer.norm' in new_key and 'transformer.encoder.norm' not in new_key:
                    new_key = new_key.replace('transformer.norm', 'transformer.encoder.norm')
                normalized_state[new_key] = value

            encoder_state = normalized_state
            print(f"[EXTRACT] Normalized to new format: {list(encoder_state.keys())[:5]}")
        else:
            print("[EXTRACT] Detected new checkpoint format, no normalization needed")

        return encoder_state

    def _get_encoder_param_names(self) -> list[str]:
        """Return parameter prefixes for pretrained encoder components.

        PreLocal uses:
        - feature_encoder: Per-channel CNN (pretrained)
        - pos_encoder: Positional encoding (pretrained, but not used in forward pass)
        - transformer: Contextual encoder (pretrained, but not used in forward pass)

        New layers:
        - spatial_conv: Spatial filtering before encoder (randomly initialized)
        - final_layer: Output head (randomly initialized)

        Returns:
            List of encoder parameter prefixes
        """
        return ['feature_encoder', 'pos_encoder', 'transformer']
