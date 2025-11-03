"""S-JEPA fine-tuning trainers for downstream tasks with frozen encoders.

Implements frozen encoder fine-tuning following the paper's best practices:
- Load pretrained VanillaSignalJEPAEncoder weights
- Freeze encoder permanently (no weight updates)
- Add task-specific classification head
- Train only the new head layers

Per paper (Figure 6), pre-local architecture performed best, but since we're
freezing the encoder, we use post-contextual architecture (add layers after encoder).
"""

from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SJEPAFinetuneHead(nn.Module):
    """Classification head for S-JEPA fine-tuning.

    Takes frozen encoder features and learns task-specific mapping.

    Architecture (post-contextual):
        Encoder features (B, C×t, d)
        → Temporal pooling → (B, C, d)
        → Spatial aggregation → (B, V, d) where V << C
        → Global average pooling → (B, d)
        → Fully connected → (B, 1) for regression

    Args:
        input_dim: Encoder output dimension (default: 64)
        n_virtual_channels: Number of virtual channels for spatial aggregation (default: 16)
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(
        self,
        input_dim: int = 64,
        n_virtual_channels: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Spatial aggregation: Learn weighted combination of channels
        # Conv1d over channel dimension
        self.spatial_agg = nn.Conv1d(
            in_channels=129,  # HBN has 129 channels
            out_channels=n_virtual_channels,
            kernel_size=1,  # 1x1 conv = linear combination per channel
            bias=True
        )

        # Fully connected layers for regression
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim // 2, 1)  # Single output for regression
        )

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """Apply classification head to frozen encoder features.

        Args:
            encoder_features: Output from VanillaSignalJEPAEncoder
                Shape: (B, C×t, d) where C=129 channels, t=time windows, d=64

        Returns:
            Predictions: (B, 1) regression outputs
        """
        B, L, d = encoder_features.shape

        # CRITICAL: VanillaSignalJEPAEncoder outputs (B, C×t, d)
        # We need to reshape to (B, C, t, d) then pool over time
        # Assuming C=129 channels
        C = 129
        t = L // C  # Time windows per channel

        # Reshape to separate channels and time
        x = encoder_features.reshape(B, C, t, d)  # (B, C, t, d)

        # Temporal pooling: Average over time windows
        x = x.mean(dim=2)  # (B, C, d)

        # Spatial aggregation: Conv1d over channel dimension
        # Input: (B, C, d) → Transpose to (B, d, C) for Conv1d
        x = x.transpose(1, 2)  # (B, d, C)
        x = self.spatial_agg(x.transpose(1, 2))  # (B, V, d)

        # Global average pooling over virtual channels
        x = x.mean(dim=1)  # (B, d)

        # Fully connected layers for regression
        output = self.fc(x)  # (B, 1)

        return output.squeeze(-1)  # (B,)


class SJEPAFinetuneTrainer(L.LightningModule):
    """S-JEPA fine-tuning trainer with frozen encoder.

    Loads pretrained VanillaSignalJEPAEncoder, freezes it permanently,
    and trains a task-specific classification head.

    Args:
        encoder: VanillaSignalJEPAEncoder instance (will be frozen)
        encoder_weights_path: Path to pretrained encoder weights file
        n_virtual_channels: Number of virtual channels in classification head
        dropout: Dropout probability in classification head
        lr: Learning rate
        weight_decay: AdamW weight decay
        scheduler_patience: ReduceLROnPlateau patience epochs
        scheduler_factor: ReduceLROnPlateau reduction factor
        loss_fn: Loss function ("mse" or "mae")
    """

    def __init__(
        self,
        encoder: nn.Module,  # VanillaSignalJEPAEncoder
        encoder_weights_path: Optional[str] = None,
        n_virtual_channels: int = 16,
        dropout: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        loss_fn: Literal["mse", "mae"] = "mse",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])

        # Encoder (frozen)
        self.encoder = encoder
        self._freeze_encoder()

        # Store checkpoint path for deferred loading in setup()
        self.encoder_weights_path = encoder_weights_path
        self._weights_loaded = False

        # Classification head (trainable)
        self.head = SJEPAFinetuneHead(
            input_dim=encoder.output_dim,
            n_virtual_channels=n_virtual_channels,
            dropout=dropout
        )

        # Loss function
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor

    def setup(self, stage: str):
        """Load pretrained weights after model is moved to device.

        This hook is called automatically by Lightning after the model is
        instantiated and moved to the correct device (GPU/CPU), but before
        training starts. This is the ideal time to load pretrained weights.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'
        """
        print(f"\n[SJEPA SETUP] Called with stage='{stage}'")
        print(f"[SJEPA SETUP] encoder_weights_path={self.encoder_weights_path}")
        print(f"[SJEPA SETUP] _weights_loaded={self._weights_loaded}")

        if stage == "fit" and self.encoder_weights_path and not self._weights_loaded:
            print(f"[SJEPA SETUP] Loading pretrained weights from {self.encoder_weights_path}")
            try:
                self._load_pretrained_weights(self.encoder_weights_path)
                self._weights_loaded = True
                print(f"[SJEPA SETUP] Successfully loaded weights, _weights_loaded={self._weights_loaded}")
            except Exception as e:
                print(f"[SJEPA SETUP] ERROR loading weights: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            reasons = []
            if stage != "fit":
                reasons.append(f"stage is '{stage}' not 'fit'")
            if not self.encoder_weights_path:
                reasons.append("encoder_weights_path is None")
            if self._weights_loaded:
                reasons.append("weights already loaded")
            print(f"[SJEPA SETUP] Skipping weight loading: {', '.join(reasons)}")

    def _freeze_encoder(self):
        """Freeze encoder weights permanently."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()  # Set to eval mode

    def _load_pretrained_weights(self, checkpoint_path: Path | str):
        """Load pretrained encoder weights from S-JEPA checkpoint.

        This method is called from setup() hook, so self.device is already set.
        Handles both old and new checkpoint formats via intelligent key mapping.
        """
        checkpoint_path = Path(checkpoint_path)
        print(f"[WEIGHTS] Checking if checkpoint exists: {checkpoint_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[WEIGHTS] Checkpoint exists, loading to device: {self.device}")

        # Load checkpoint directly to model's device
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"[WEIGHTS] Checkpoint loaded, extracting encoder state dict")

        # Extract encoder state dict (remove "encoder." prefix)
        encoder_state = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                encoder_state[new_key] = value

        print(f"[WEIGHTS] Extracted {len(encoder_state)} encoder parameters")
        print(f"[WEIGHTS] Sample keys (before mapping): {list(encoder_state.keys())[:5]}")

        # Detect checkpoint format and normalize keys if needed
        # Old format has: feature_1, feature_2, pos_pos_encoder_spat, transformer.layers
        # New format has: feature_encoder.1, feature_encoder.2, pos_encoder.pos_encoder_spat, transformer.encoder.layers
        sample_keys = list(encoder_state.keys())[:10] if encoder_state else []
        is_old_format = any(
            'feature_1' in k or 'feature_2' in k or 'feature_3' in k or
            'pos_pos_encoder_spat' in k or
            ('transformer.layers' in k and 'transformer.encoder.layers' not in k)
            for k in sample_keys
        )

        if is_old_format:
            print("[WEIGHTS] Detected old checkpoint format, mapping keys to new structure...")
            import re
            normalized_state = {}
            for key, value in encoder_state.items():
                new_key = key
                # Map feature_N → feature_encoder.N (for any digit N)
                new_key = re.sub(r'\.feature_(\d+)\.', r'.feature_encoder.\1.', new_key)
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
            print(f"[WEIGHTS] Normalized {len(encoder_state)} parameters to new format")
            print(f"[WEIGHTS] Sample keys (after mapping): {list(encoder_state.keys())[:5]}")
        else:
            print("[WEIGHTS] Detected new checkpoint format, no mapping needed")

        # Load into encoder
        print(f"[WEIGHTS] Loading state dict into encoder (strict=True)")
        self.encoder.load_state_dict(encoder_state, strict=True)
        print(f"✓ Loaded pretrained encoder from: {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with frozen encoder + trainable head.

        Args:
            x: Input EEG (batch_size, n_chans, n_times)

        Returns:
            Predictions (batch_size,)
        """
        # Encoder (frozen, in eval mode)
        with torch.no_grad():
            encoder_features = self.encoder(x)  # (B, C×t, d)

        # Classification head (trainable)
        predictions = self.head(encoder_features)  # (B,)

        return predictions

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: (eeg, target) where eeg is (B, C, T) and target is (B,)

        Returns:
            Loss value
        """
        eeg, target = batch

        # Forward pass
        predictions = self.forward(eeg)

        # Compute loss
        loss = self.criterion(predictions, target)

        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        eeg, target = batch

        predictions = self.forward(eeg)
        loss = self.criterion(predictions, target)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        eeg, target = batch

        predictions = self.forward(eeg)
        loss = self.criterion(predictions, target)

        self.log('test/loss', loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with ReduceLROnPlateau scheduler."""
        # Only optimize head parameters (encoder is frozen)
        optimizer = AdamW(
            self.head.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # ReduceLROnPlateau scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.scheduler_factor,
            patience=self.scheduler_patience
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
            }
        }

    def on_train_epoch_start(self):
        """Ensure encoder stays frozen at start of each epoch."""
        self.encoder.eval()
