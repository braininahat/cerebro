"""Contrastive learning trainer with support for multiple pairing strategies.

This trainer handles the training loop for contrastive learning tasks
with flexible pairing strategies:
- Triplet: (anchor, positive, negative) with single sampled negative
- All-pairs: (anchor, positive) using all batch samples as negatives (SimCLR style)

The trainer is responsible for:
- Training step logic with InfoNCE loss
- Similarity monitoring
- Temperature scheduling (optional)
- Optimizer configuration

The model is responsible for:
- Forward pass (input -> embeddings)
"""

from typing import Any, Literal, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from cerebro.losses import info_nce_all_pairs, info_nce_triplet


class ContrastiveTrainer(L.LightningModule):
    """Lightning trainer for contrastive learning tasks with flexible pairing strategies.

    This trainer works with any model that transforms inputs to embeddings.
    Supports two pairing strategies:
    - "triplet": Expects (anchor, positive, negative), uses single negative
    - "all_pairs": Expects (anchor, positive), uses all batch samples as negatives

    Args:
        model: Any nn.Module that outputs embeddings
        temperature: Temperature scaling for InfoNCE loss
        temperature_decay: Decay factor for temperature per epoch (1.0 = no decay)
        min_temperature: Minimum temperature value
        pairing_strategy: "triplet" (single negative) or "all_pairs" (SimCLR style)
        lr: Learning rate for AdamW optimizer
        weight_decay: Weight decay for regularization
        epochs: Total epochs for cosine annealing scheduler
        warmup_epochs: Number of warmup epochs

    Example:
        >>> from cerebro.models.architectures import ContrastiveModel
        >>> model = ContrastiveModel(encoder_class="EEGNeX", projection_dim=128)
        >>> # Triplet strategy (simpler, works with any batch size)
        >>> trainer = ContrastiveTrainer(model, pairing_strategy="triplet")
        >>> # All-pairs strategy (stronger signal, needs batch_size >= 128)
        >>> trainer = ContrastiveTrainer(model, pairing_strategy="all_pairs")
    """

    def __init__(
        self,
        model: nn.Module,
        temperature: float = 0.07,
        temperature_decay: float = 1.0,
        min_temperature: float = 0.01,
        pairing_strategy: Literal["triplet", "all_pairs"] = "triplet",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Store model (contains no training logic)
        self.model = model

        # Temperature for InfoNCE (can be scheduled)
        self.temperature = temperature

        # Pairing strategy
        self.pairing_strategy = pairing_strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Embedding vectors
        """
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """Training step with InfoNCE loss.

        Args:
            batch: Triplet (anchor, positive, negative) or pair (anchor, positive)
            batch_idx: Batch index

        Returns:
            Loss value for backpropagation
        """
        # Unpack batch based on pairing strategy
        if self.pairing_strategy == "all_pairs":
            # All-pairs: only anchor and positive, use all batch as negatives
            anchor, positive = batch[0], batch[1]
            anchor = anchor.float()
            positive = positive.float()

            # Forward pass
            z_anchor = self(anchor)
            z_positive = self(positive)

            # Compute all-pairs InfoNCE loss
            loss = info_nce_all_pairs(z_anchor, z_positive, temperature=self.temperature)

            # Compute similarities for monitoring
            with torch.no_grad():
                z_anchor_norm = F.normalize(z_anchor, p=2, dim=1)
                z_positive_norm = F.normalize(z_positive, p=2, dim=1)
                pos_sim = F.cosine_similarity(z_anchor_norm, z_positive_norm).mean()

            # Log metrics (no neg_sim or margin for all-pairs)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("pos_sim", pos_sim, on_step=False, on_epoch=True, prog_bar=True)
            self.log("temperature", self.temperature, on_step=False, on_epoch=True)

        else:  # triplet strategy
            # Unpack triplet
            anchor, positive, negative = batch[0], batch[1], batch[2]
            anchor = anchor.float()
            positive = positive.float()
            negative = negative.float()

            # Forward pass through model
            z_anchor = self(anchor)
            z_positive = self(positive)
            z_negative = self(negative)

            # Compute InfoNCE loss
            loss = info_nce_triplet(
                z_anchor, z_positive, z_negative, temperature=self.temperature
            )

            # Compute similarities for monitoring (with gradient detached)
            with torch.no_grad():
                # Normalize embeddings for cosine similarity
                z_anchor_norm = F.normalize(z_anchor, p=2, dim=1)
                z_positive_norm = F.normalize(z_positive, p=2, dim=1)
                z_negative_norm = F.normalize(z_negative, p=2, dim=1)

                # Compute mean similarities
                pos_sim = F.cosine_similarity(z_anchor_norm, z_positive_norm).mean()
                neg_sim = F.cosine_similarity(z_anchor_norm, z_negative_norm).mean()
                margin = pos_sim - neg_sim

            # Log metrics
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("pos_sim", pos_sim, on_step=False, on_epoch=True)
            self.log("neg_sim", neg_sim, on_step=False, on_epoch=True)
            self.log("margin", margin, on_step=False, on_epoch=True, prog_bar=True)
            self.log("temperature", self.temperature, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Triplet (anchor, positive, negative) or pair (anchor, positive)
            batch_idx: Batch index

        Returns:
            Loss value for logging
        """
        # Unpack batch based on pairing strategy
        if self.pairing_strategy == "all_pairs":
            # All-pairs: only anchor and positive
            anchor, positive = batch[0], batch[1]
            anchor = anchor.float()
            positive = positive.float()

            # Forward pass
            z_anchor = self(anchor)
            z_positive = self(positive)

            # Compute all-pairs InfoNCE loss
            loss = info_nce_all_pairs(z_anchor, z_positive, temperature=self.temperature)

            # Compute similarities for monitoring
            with torch.no_grad():
                z_anchor_norm = F.normalize(z_anchor, p=2, dim=1)
                z_positive_norm = F.normalize(z_positive, p=2, dim=1)
                pos_sim = F.cosine_similarity(z_anchor_norm, z_positive_norm).mean()

            # Log metrics (no neg_sim or margin for all-pairs)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_pos_sim", pos_sim, on_step=False, on_epoch=True)

        else:  # triplet strategy
            # Unpack triplet
            anchor, positive, negative = batch[0], batch[1], batch[2]
            anchor = anchor.float()
            positive = positive.float()
            negative = negative.float()

            # Forward pass
            z_anchor = self(anchor)
            z_positive = self(positive)
            z_negative = self(negative)

            # Compute loss
            loss = info_nce_triplet(
                z_anchor, z_positive, z_negative, temperature=self.temperature
            )

            # Compute similarities
            with torch.no_grad():
                z_anchor_norm = F.normalize(z_anchor, p=2, dim=1)
                z_positive_norm = F.normalize(z_positive, p=2, dim=1)
                z_negative_norm = F.normalize(z_negative, p=2, dim=1)

                pos_sim = F.cosine_similarity(z_anchor_norm, z_positive_norm).mean()
                neg_sim = F.cosine_similarity(z_anchor_norm, z_negative_norm).mean()
                margin = pos_sim - neg_sim

            # Log metrics
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_pos_sim", pos_sim, on_step=False, on_epoch=True)
            self.log("val_neg_sim", neg_sim, on_step=False, on_epoch=True)
            self.log("val_margin", margin, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Update temperature at end of epoch if decay is enabled."""
        if self.hparams.temperature_decay < 1.0:
            # Decay temperature
            old_temp = self.temperature
            self.temperature *= self.hparams.temperature_decay
            self.temperature = max(self.temperature, self.hparams.min_temperature)

            if old_temp != self.temperature:
                print(f"Temperature decayed: {old_temp:.4f} -> {self.temperature:.4f}")

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW optimizer with cosine annealing scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # AdamW optimizer (typically uses higher weight decay than supervised)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Warmup + Cosine annealing scheduler
        if self.hparams.warmup_epochs > 0:
            # Linear warmup followed by cosine annealing
            def lr_lambda(epoch):
                if epoch < self.hparams.warmup_epochs:
                    # Linear warmup
                    return epoch / self.hparams.warmup_epochs
                else:
                    # Cosine annealing after warmup
                    progress = (epoch - self.hparams.warmup_epochs) / (
                        self.hparams.epochs - self.hparams.warmup_epochs
                    )
                    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # Standard cosine annealing without warmup
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.epochs,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


class ContrastiveFineTuner(ContrastiveTrainer):
    """Contrastive trainer with encoder freezing support.

    This variant allows freezing the encoder for the first N epochs,
    then unfreezing for fine-tuning. Useful for stabilizing projection
    head training before updating the encoder.

    Args:
        model: Model with encoder and projection head
        freeze_encoder_epochs: Number of epochs to freeze encoder (0 = never freeze)
        **kwargs: Additional arguments for ContrastiveTrainer

    Example:
        >>> trainer = ContrastiveFineTuner(
        ...     model,
        ...     freeze_encoder_epochs=5,  # Freeze for first 5 epochs
        ...     temperature=0.07
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        freeze_encoder_epochs: int = 0,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.freeze_encoder_epochs = freeze_encoder_epochs

        # Initially freeze encoder if requested
        if freeze_encoder_epochs > 0:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        if hasattr(self.model, 'encoder'):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen")

    def _unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters."""
        if hasattr(self.model, 'encoder'):
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            print("Encoder unfrozen")

    def on_train_epoch_start(self) -> None:
        """Check if encoder should be unfrozen."""
        if self.freeze_encoder_epochs > 0:
            if self.current_epoch == self.freeze_encoder_epochs:
                self._unfreeze_encoder()
                print(f"Unfreezing encoder at epoch {self.current_epoch}")