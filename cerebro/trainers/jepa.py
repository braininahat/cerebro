"""JEPA trainers for multi-phase EEG foundation model training.

Phase 1: Self-supervised pretraining with temporal and masked prediction
Phase 2-5: Planned for future implementation

From magnum_opus.md:
- Phase 1: SSL on all data (2 weeks)
- Phase 2: Multi-task alignment (1 week)
- Phase 3: Behavioral grounding (1 week)
- Phase 4: Psychopathology regression (1 week)
- Phase 5: End-to-end fine-tuning (3 days)
"""

from typing import Any, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPhase1Trainer(L.LightningModule):
    """Phase 1: Self-supervised pretraining trainer.

    Implements JEPA-style temporal prediction with multi-scale losses.
    No labels needed - learns from data structure alone.

    Training strategies:
    1. Temporal prediction (every batch): Split window → predict future from past
    2. Masked prediction (every 5th batch): Mask 25% of timepoints → predict full

    Multi-scale losses (from magnum_opus.md):
    - L_state: MSE(state_pred, state_true) × 1.0
    - L_event: MSE(event_pred, event_true) × 0.5
    - L_trait: MSE(trait_pred, trait_true) × 0.1
    - L_stability: MSE(trait_past, trait_future) × 0.5  # Traits should be stable
    - L_mask: MSE(z_masked, z_full) × 0.5 (every 5th batch)

    Args:
        model: JEPAFoundationModel instance
        lr: Learning rate for AdamW optimizer
        weight_decay: Weight decay for regularization
        warmup_epochs: Number of warmup epochs
        loss_weights: Dict with keys [state, event, trait, stability, mask]
        mask_prob: Probability of masking each timepoint (default: 0.25)
        mask_every_n_batches: Apply masked prediction every N batches (default: 5)

    Note:
        Total epochs for cosine annealing scheduler is automatically read from
        trainer.max_epochs (no need for separate epochs parameter).

    Example:
        >>> from cerebro.models.architectures import JEPAFoundationModel
        >>> model = JEPAFoundationModel(n_chans=129, latent_dim=96)
        >>> trainer = JEPAPhase1Trainer(model, lr=1e-4, weight_decay=1e-4)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 0,
        loss_weights: Optional[dict[str, float]] = None,
        mask_prob: float = 0.25,
        mask_every_n_batches: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Store model (JEPAFoundationModel)
        self.model = model

        # Loss weights (default from magnum_opus.md)
        if loss_weights is None:
            loss_weights = {
                "state": 1.0,
                "event": 0.5,
                "trait": 0.1,
                "stability": 0.5,
                "mask": 0.5,
            }
        self.loss_weights = loss_weights

        # Masking config
        self.mask_prob = mask_prob
        self.mask_every_n_batches = mask_every_n_batches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input EEG (batch, n_chans, n_times)

        Returns:
            Latent representation (batch, latent_dim)
        """
        return self.model(x)

    def _temporal_prediction_loss(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute temporal prediction loss (strategy 1).

        Split window in half, predict future from past.

        Args:
            eeg: Input EEG (batch, n_chans, n_times)

        Returns:
            Dictionary of losses and metrics
        """
        B, C, T = eeg.shape
        t_split = T // 2

        # Encode past and future
        z_past = self.model.encode(eeg[..., :t_split])
        z_future_true = self.model.encode(eeg[..., t_split:])

        # Split latents into trait/state/event
        z_trait_p, z_state_p, z_event_p = self.model.split_latent(z_past)
        z_trait_f, z_state_f, z_event_f = self.model.split_latent(z_future_true)

        # Predict future from past
        z_trait_pred = self.model.predict_trait(z_trait_p)
        z_state_pred = self.model.predict_state(z_state_p)
        z_event_pred = self.model.predict_event(z_event_p)

        # Multi-scale losses
        loss_trait = F.mse_loss(z_trait_pred, z_trait_f.detach())
        loss_state = F.mse_loss(z_state_pred, z_state_f.detach())
        loss_event = F.mse_loss(z_event_pred, z_event_f.detach())

        # Trait stability: traits should not change much over time
        loss_stability = F.mse_loss(z_trait_p, z_trait_f.detach())

        # Weighted combination
        loss_temporal = (
            self.loss_weights["state"] * loss_state
            + self.loss_weights["event"] * loss_event
            + self.loss_weights["trait"] * loss_trait
            + self.loss_weights["stability"] * loss_stability
        )

        return {
            "loss_temporal": loss_temporal,
            "loss_trait": loss_trait,
            "loss_state": loss_state,
            "loss_event": loss_event,
            "loss_stability": loss_stability,
        }

    def _masked_prediction_loss(self, eeg: torch.Tensor) -> torch.Tensor:
        """Compute masked prediction loss (strategy 2).

        Mask random timepoints, predict full latent from masked input.

        Args:
            eeg: Input EEG (batch, n_chans, n_times)

        Returns:
            Masked prediction loss
        """
        B, C, T = eeg.shape

        # Create random mask: (T,) with self.mask_prob fraction True
        mask = torch.rand(T, device=eeg.device) > (1 - self.mask_prob)

        # Apply mask: zero out masked timepoints
        # mask shape: (T,) → (1, 1, T) for broadcasting
        eeg_masked = eeg * (~mask).unsqueeze(0).unsqueeze(0).float()

        # Encode both masked and full
        z_visible = self.model.encode(eeg_masked)
        with torch.no_grad():
            z_full = self.model.encode(eeg)

        # Loss: visible should predict full
        loss_mask = F.mse_loss(z_visible, z_full)

        return loss_mask

    def training_step(self, batch: torch.Tensor | dict, batch_idx: int) -> torch.Tensor:
        """Training step with temporal and masked prediction.

        Args:
            batch: EEG tensor (batch, n_chans, n_times) or dict with 'eeg' key
            batch_idx: Batch index

        Returns:
            Total loss for backpropagation
        """
        # Handle both tensor and dict inputs
        if isinstance(batch, dict):
            eeg = batch["eeg"]
        else:
            eeg = batch

        eeg = eeg.float()

        # Strategy 1: Temporal prediction (every batch)
        losses_temporal = self._temporal_prediction_loss(eeg)
        loss = losses_temporal["loss_temporal"]

        # Strategy 2: Masked prediction (every Nth batch)
        if batch_idx % self.mask_every_n_batches == 0:
            loss_mask = self._masked_prediction_loss(eeg)
            loss += self.loss_weights["mask"] * loss_mask

            # Log mask loss
            self.log("train_loss_mask", loss_mask, on_step=False, on_epoch=True)

        # Log all losses
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_trait", losses_temporal["loss_trait"], on_step=False, on_epoch=True)
        self.log("train_loss_state", losses_temporal["loss_state"], on_step=False, on_epoch=True)
        self.log("train_loss_event", losses_temporal["loss_event"], on_step=False, on_epoch=True)
        self.log("train_loss_stability", losses_temporal["loss_stability"], on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor | dict, batch_idx: int) -> torch.Tensor:
        """Validation step (temporal prediction only, no masking).

        Args:
            batch: EEG tensor or dict with 'eeg' key
            batch_idx: Batch index

        Returns:
            Validation loss
        """
        # Handle both tensor and dict inputs
        if isinstance(batch, dict):
            eeg = batch["eeg"]
        else:
            eeg = batch

        eeg = eeg.float()

        # Only temporal prediction for validation
        losses_temporal = self._temporal_prediction_loss(eeg)
        loss = losses_temporal["loss_temporal"]

        # Log validation losses
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_trait", losses_temporal["loss_trait"], on_step=False, on_epoch=True)
        self.log("val_loss_state", losses_temporal["loss_state"], on_step=False, on_epoch=True)
        self.log("val_loss_event", losses_temporal["loss_event"], on_step=False, on_epoch=True)
        self.log("val_loss_stability", losses_temporal["loss_stability"], on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW optimizer with cosine annealing scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # AdamW optimizer (from magnum_opus.md Phase 1)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing scheduler (optional warmup)
        # Use trainer.max_epochs directly instead of separate config parameter
        max_epochs = self.trainer.max_epochs

        if self.hparams.warmup_epochs > 0:
            # Linear warmup followed by cosine annealing
            def lr_lambda(epoch):
                if epoch < self.hparams.warmup_epochs:
                    # Linear warmup
                    return epoch / self.hparams.warmup_epochs
                else:
                    # Cosine annealing after warmup
                    progress = (epoch - self.hparams.warmup_epochs) / (
                        max_epochs - self.hparams.warmup_epochs
                    )
                    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # Standard cosine annealing without warmup
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def load_state_dict(self, state_dict, strict=True):
        """Override to handle neuraloperator's _metadata key.

        NeuralOperator v1.0+ adds _metadata to state_dict for version tracking,
        but Lightning's batch size finder restoration fails with strict=True.
        Filter it out to maintain compatibility.

        Args:
            state_dict: State dictionary to load
            strict: Whether to strictly enforce key matching (default: True)

        Returns:
            Missing and unexpected keys after loading
        """
        # Filter out _metadata key added by neuraloperator v1.0+
        # This is a version tracking field, not actual model weights
        if "_metadata" in state_dict:
            state_dict = {k: v for k, v in state_dict.items() if k != "_metadata"}

        # Call parent's load_state_dict
        return super().load_state_dict(state_dict, strict=strict)
