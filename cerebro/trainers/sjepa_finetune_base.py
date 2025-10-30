"""Base trainer for S-JEPA fine-tuning with braindecode variants.

Provides shared functionality for all three architectural variants:
- SignalJEPA_PreLocal: Spatial filtering before local encoder
- SignalJEPA_PostLocal: Spatial filtering after local encoder
- SignalJEPA_Contextual: Full contextual encoder with spatial filtering

Supports both freezing strategies from the paper:
- "new-": Freeze pretrained encoder, train only new layers (fast baseline)
- "full-": Unfreeze encoder after warmup, train entire network (best performance)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseSJEPAFinetuneTrainer(L.LightningModule, ABC):
    """Abstract base class for S-JEPA fine-tuning trainers.

    Subclasses implement _load_model() to instantiate specific variant
    (PreLocal, PostLocal, or Contextual) and _get_encoder_param_names()
    to identify which parameters belong to the pretrained encoder.

    Args:
        pretrained_checkpoint: Path to pretrained S-JEPA checkpoint
        n_outputs: Number of outputs (1 for regression, N for classification)
        freeze_encoder: If True, always freeze encoder. If False, use warmup then unfreeze.
        warmup_epochs: Number of epochs to freeze encoder before unfreezing (only if freeze_encoder=False)
        lr: Learning rate for new layers (final_layer, spatial conv)
        encoder_lr_multiplier: LR multiplier for encoder when unfrozen (typically 0.1)
        weight_decay: AdamW weight decay
        scheduler_patience: ReduceLROnPlateau patience
        scheduler_factor: ReduceLROnPlateau factor
        loss_fn: Loss function ("mse" or "mae")
    """

    def __init__(
        self,
        pretrained_checkpoint: str,
        n_outputs: int = 1,
        freeze_encoder: bool = True,
        warmup_epochs: int = 10,
        lr: float = 1e-3,
        encoder_lr_multiplier: float = 0.1,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        loss_fn: Literal["mse", "mae"] = "mse",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pretrained_checkpoint = pretrained_checkpoint
        self.n_outputs = n_outputs
        self.freeze_encoder = freeze_encoder
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.encoder_lr_multiplier = encoder_lr_multiplier
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor

        # Model will be loaded in setup() after device is set
        self.model: Optional[nn.Module] = None
        self._model_loaded = False

        # Loss function
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

    @abstractmethod
    def _load_model(self, checkpoint_path: str, n_outputs: int) -> nn.Module:
        """Load pretrained model and instantiate variant-specific architecture.

        Subclasses must implement this to:
        1. Load pretrained SignalJEPA base model from checkpoint
        2. Call variant.from_pretrained(base_model, n_outputs, ...)
        3. Return the fine-tuning model

        Args:
            checkpoint_path: Path to pretrained checkpoint
            n_outputs: Number of outputs for fine-tuning task

        Returns:
            Fine-tuning model (SignalJEPA_PreLocal, PostLocal, or Contextual)
        """
        pass

    @abstractmethod
    def _get_encoder_param_names(self) -> list[str]:
        """Return list of parameter name prefixes that belong to pretrained encoder.

        Used to separate encoder params from new layers for optimizer param groups.

        Returns:
            List of parameter name prefixes (e.g., ['feature_encoder', 'transformer'])
        """
        pass

    def setup(self, stage: str):
        """Load model after moving to device.

        This hook is called after model is moved to GPU/CPU but before training.
        Ideal time to load pretrained weights.
        """
        if stage == "fit" and not self._model_loaded:
            print(f"\n[SETUP] Loading model from {self.pretrained_checkpoint}")
            self.model = self._load_model(self.pretrained_checkpoint, self.n_outputs)
            self._model_loaded = True
            print(f"[SETUP] Model loaded: {self.model.__class__.__name__}")

            # Apply initial freezing if needed
            if self.freeze_encoder:
                self._freeze_encoder_components()
                print("[SETUP] Encoder frozen (freeze_encoder=True)")
            else:
                print(f"[SETUP] Encoder will be frozen for {self.warmup_epochs} warmup epochs, then unfrozen")

    def _freeze_encoder_components(self):
        """Freeze pretrained encoder components."""
        encoder_prefixes = self._get_encoder_param_names()
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if any(prefix in name for prefix in encoder_prefixes):
                param.requires_grad = False
                frozen_count += 1
        print(f"[FREEZE] Froze {frozen_count} encoder parameters")

    def _unfreeze_encoder_components(self):
        """Unfreeze pretrained encoder components."""
        encoder_prefixes = self._get_encoder_param_names()
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if any(prefix in name for prefix in encoder_prefixes):
                param.requires_grad = True
                unfrozen_count += 1
        print(f"[UNFREEZE] Unfroze {unfrozen_count} encoder parameters")

    def on_train_epoch_start(self):
        """Handle encoder freezing/unfreezing based on warmup schedule."""
        if not self.freeze_encoder and self.current_epoch == self.warmup_epochs:
            # Transition from warmup to full fine-tuning
            self._unfreeze_encoder_components()
            print(f"\n[EPOCH {self.current_epoch}] Warmup complete, encoder unfrozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fine-tuning model.

        Args:
            x: Input EEG (batch_size, n_chans, n_times)

        Returns:
            Predictions (batch_size,) for regression or (batch_size, n_outputs) for classification
        """
        return self.model(x)

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

        # Handle shape: predictions might be (B, 1) or (B,)
        if predictions.ndim > 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)

        # Compute loss
        loss = self.criterion(predictions, target)

        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        eeg, target = batch

        predictions = self.forward(eeg)
        if predictions.ndim > 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)

        loss = self.criterion(predictions, target)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        eeg, target = batch

        predictions = self.forward(eeg)
        if predictions.ndim > 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)

        loss = self.criterion(predictions, target)

        self.log('test/loss', loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with param groups and ReduceLROnPlateau scheduler.

        If encoder is frozen or in warmup phase:
            - Only optimize new layers (final_layer, spatial_conv)

        If encoder is unfrozen after warmup:
            - Encoder params: lr * encoder_lr_multiplier (typically 0.1x)
            - New params: lr (1.0x)
        """
        encoder_prefixes = self._get_encoder_param_names()

        # Separate encoder and new layer parameters
        encoder_params = []
        new_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen params

            if any(prefix in name for prefix in encoder_prefixes):
                encoder_params.append(param)
            else:
                new_params.append(param)

        # Build param groups
        if self.freeze_encoder or self.current_epoch < self.warmup_epochs:
            # Only new layers are trainable
            param_groups = [{'params': new_params, 'lr': self.lr}]
            print(f"[OPTIMIZER] Optimizing {len(new_params)} new layer parameters")
        else:
            # Both encoder and new layers are trainable (after warmup)
            param_groups = [
                {'params': encoder_params, 'lr': self.lr * self.encoder_lr_multiplier},
                {'params': new_params, 'lr': self.lr}
            ]
            print(f"[OPTIMIZER] Optimizing {len(encoder_params)} encoder + {len(new_params)} new layer parameters")

        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)

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
