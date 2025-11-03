"""S-JEPA fine-tuning with Contextual architecture.

Contextual architecture:
    Local encoder (per-channel CNN)
    → Positional encoding
    → Transformer contextual encoder
    → 3D Conv spatial filter
    → Flatten
    → Linear head

Uses the FULL pretrained encoder pipeline (local + positional + contextual)
before applying spatial filtering and the output head. This preserves the most
information from pretraining but may be less flexible for downstream tasks that
differ significantly from the pretraining objective.
"""

import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from braindecode.models import SignalJEPA, SignalJEPA_Contextual

from cerebro.trainers.sjepa_finetune_base import BaseSJEPAFinetuneTrainer


class ContextualFinetuneTrainer(BaseSJEPAFinetuneTrainer):
    """Fine-tuning trainer using SignalJEPA_Contextual architecture.

    Loads pretrained SignalJEPA base model and transfers weights to
    SignalJEPA_Contextual variant via from_pretrained() method.

    Args:
        pretrained_checkpoint: Path to pretrained S-JEPA checkpoint
        n_outputs: Number of outputs (1 for regression)
        n_spat_filters: Number of spatial filters (typically 4-16)
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
        """Load pretrained SignalJEPA and transfer to Contextual variant.

        Steps:
        1. Load pretrained checkpoint
        2. Extract encoder state dict (strip "encoder." prefix, handle format)
        3. Instantiate base SignalJEPA with same architecture
        4. Load weights into base model
        5. Call SignalJEPA_Contextual.from_pretrained()

        Args:
            checkpoint_path: Path to pretrained S-JEPA checkpoint
            n_outputs: Number of outputs for fine-tuning task

        Returns:
            SignalJEPA_Contextual model with pretrained weights + new layers
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"[CONTEXTUAL] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract encoder state dict
        encoder_state = self._extract_encoder_state(checkpoint)
        print(f"[CONTEXTUAL] Extracted {len(encoder_state)} parameters from checkpoint")

        # Load HBN electrode locations (required for spatial positional encoding)
        from cerebro.utils.electrode_locations import load_hbn_chs_info
        chs_info = load_hbn_chs_info()

        # Instantiate base SignalJEPA with pretrained architecture
        base_model = SignalJEPA(
            n_chans=129,
            sfreq=100,
            n_times=200,
            input_window_seconds=2.0,
            chs_info=chs_info,  # CRITICAL: Electrode positions for spatial encoding
            transformer__d_model=64,
            transformer__num_encoder_layers=8,
            transformer__num_decoder_layers=0,
            transformer__nhead=8,
        )

        # Load pretrained weights
        print("[CONTEXTUAL] Loading pretrained weights into base SignalJEPA...")
        base_model.load_state_dict(encoder_state, strict=False)
        print("[CONTEXTUAL] ✓ Loaded pretrained weights")

        # Transfer to Contextual variant
        print(f"[CONTEXTUAL] Creating Contextual variant with n_spat_filters={self.n_spat_filters}...")
        model = SignalJEPA_Contextual.from_pretrained(
            base_model,
            n_outputs=n_outputs,
            n_spat_filters=self.n_spat_filters
        )
        print("[CONTEXTUAL] ✓ Created Contextual model with pretrained weights")

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

        Contextual uses:
        - feature_encoder: Per-channel CNN (pretrained, USED in forward pass)
        - pos_encoder: Positional encoding (pretrained, USED in forward pass)
        - transformer: Contextual encoder (pretrained, USED in forward pass)

        New layers:
        - final_layer: Conv3d spatial filter + Linear head (randomly initialized)

        Returns:
            List of encoder parameter prefixes
        """
        return ['feature_encoder', 'pos_encoder', 'transformer']
