"""Supervised learning trainer for models with (X, y) data.

This trainer handles the training loop for supervised learning tasks
where we have input-target pairs. It's model-agnostic - any model
that outputs predictions matching the target shape will work.

The trainer is responsible for:
- Training step logic
- Validation/test step logic
- Loss computation
- Metric logging
- Optimizer configuration

The model is responsible for:
- Forward pass (input -> predictions)
"""

from typing import Any, Literal, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedTrainer(L.LightningModule):
    """Lightning trainer for supervised learning tasks.

    This trainer works with any model that transforms inputs to predictions
    matching the target shape. It handles regression and classification tasks
    through configurable loss functions.

    Args:
        model: Any nn.Module that outputs predictions
        loss_fn: Loss function name ("mse", "mae", "huber", "cross_entropy")
        lr: Learning rate for AdamW optimizer
        weight_decay: Weight decay for regularization
        epochs: Total epochs for cosine annealing scheduler
        warmup_epochs: Number of warmup epochs (0 = no warmup)

    Example:
        >>> from cerebro.models.architectures import RegressorModel
        >>> model = RegressorModel(encoder_class="EEGNeX", n_outputs=1)
        >>> trainer = SupervisedTrainer(model, loss_fn="mse")
        >>> lightning_trainer.fit(trainer, datamodule)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Literal["mse", "mae", "huber", "cross_entropy"] = "mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        warmup_epochs: int = 0,
        pretrained_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Store model (contains no training logic)
        self.model = model

        # Load pretrained encoder weights if checkpoint provided
        if pretrained_checkpoint is not None:
            self._load_pretrained_encoder(pretrained_checkpoint)

        # Initialize loss function
        self.loss_fn = self._build_loss_fn(loss_fn)

        # For NRMSE calculation in validation
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []

    def _build_loss_fn(self, loss_name: str) -> nn.Module:
        """Build loss function from name.

        Args:
            loss_name: Name of loss function

        Returns:
            Loss function module

        Raises:
            ValueError: If loss_name is not recognized
        """
        loss_functions = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
            "huber": nn.HuberLoss(),
            "cross_entropy": nn.CrossEntropyLoss(),
        }

        if loss_name not in loss_functions:
            available = ", ".join(loss_functions.keys())
            raise ValueError(
                f"Unknown loss function: {loss_name}. "
                f"Available: {available}"
            )

        return loss_functions[loss_name]

    def _load_pretrained_encoder(self, checkpoint_path: str) -> None:
        """Load pretrained encoder weights from checkpoint.

        This method extracts encoder weights from a Lightning checkpoint
        and loads them into the model's encoder. It handles key mismatches
        by stripping common prefixes like 'model.', 'encoder.', etc.

        Args:
            checkpoint_path: Path to pretrained checkpoint (.ckpt file)

        Notes:
            - Only encoder weights are loaded; head is initialized randomly
            - Handles Lightning checkpoints (extracts from state_dict)
            - Strips 'model.', 'module.', 'encoder.' prefixes automatically
            - Logs missing/unexpected keys for debugging
        """
        import pathlib

        print(f"[SupervisedTrainer] Loading pretrained encoder from: {checkpoint_path}")

        # Load checkpoint
        ckpt_path = pathlib.Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract state dict (handle Lightning checkpoint format)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Filter and fix keys for encoder
        encoder_state = {}
        for key, value in state_dict.items():
            # Strip common prefixes
            clean_key = key
            for prefix in ["model.", "module.", "_orig_mod."]:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]

            # Only keep encoder weights
            if clean_key.startswith("encoder."):
                # Strip 'encoder.' prefix to match model.encoder state dict
                encoder_key = clean_key[len("encoder."):]
                encoder_state[encoder_key] = value

        if not encoder_state:
            print("[SupervisedTrainer] WARNING: No encoder weights found in checkpoint!")
            print(f"  Available keys: {list(state_dict.keys())[:5]}...")
            return

        # Load into encoder (strict=False to allow head mismatch)
        missing, unexpected = self.model.encoder.load_state_dict(encoder_state, strict=False)

        print(f"[SupervisedTrainer] Loaded {len(encoder_state)} encoder parameters")
        if missing:
            print(f"  Missing keys: {len(missing)} (e.g., {missing[:3]})")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Model predictions
        """
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch from DataLoader (X, y, ...)
            batch_idx: Batch index

        Returns:
            Loss value for backpropagation
        """
        # Unpack batch (handles variable length tuples)
        X, y = batch[0], batch[1]
        X = X.float()
        y = y.float()

        # Forward pass
        y_pred = self(X)

        # Reshape if needed (for single output regression)
        if y_pred.dim() == 2 and y_pred.shape[1] == 1 and y.dim() == 1:
            y_pred = y_pred.squeeze(1)
        elif y.dim() == 1 and y_pred.dim() == 2:
            y = y.unsqueeze(1)

        # Compute loss
        loss = self.loss_fn(y_pred, y)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Batch from DataLoader (X, y, ...)
            batch_idx: Batch index

        Returns:
            Loss value for logging
        """
        X, y = batch[0], batch[1]
        X = X.float()
        y = y.float()

        # Forward pass
        y_pred = self(X)

        # Reshape if needed
        if y_pred.dim() == 2 and y_pred.shape[1] == 1 and y.dim() == 1:
            y_pred = y_pred.squeeze(1)
        elif y.dim() == 1 and y_pred.dim() == 2:
            y = y.unsqueeze(1)

        # Compute loss
        loss = self.loss_fn(y_pred, y)

        # Store predictions for epoch-level metrics
        self.val_preds.append(y_pred.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        # Log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Calculate epoch-level validation metrics."""
        if not self.val_preds:
            return

        # Concatenate all predictions and targets
        all_preds = torch.cat(self.val_preds, dim=0)
        all_targets = torch.cat(self.val_targets, dim=0)

        # Calculate RMSE
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2))

        # Calculate NRMSE (normalized by target standard deviation)
        target_std = torch.std(all_targets)
        nrmse = rmse / target_std if target_std > 0 else rmse

        # Calculate MAE
        mae = torch.mean(torch.abs(all_preds - all_targets))

        # Calculate R-squared
        ss_res = torch.sum((all_targets - all_preds) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(0.0)

        # Log metrics
        self.log("val_rmse", rmse, prog_bar=False)
        self.log("val_nrmse", nrmse, prog_bar=True)
        self.log("val_mae", mae, prog_bar=False)
        self.log("val_r2", r2, prog_bar=False)

        # Clear stored predictions
        self.val_preds.clear()
        self.val_targets.clear()

    def test_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """Test step.

        Args:
            batch: Batch from DataLoader (X, y, ...)
            batch_idx: Batch index

        Returns:
            Loss value for logging
        """
        X, y = batch[0], batch[1]
        X = X.float()
        y = y.float()

        # Forward pass
        y_pred = self(X)

        # Reshape if needed
        if y_pred.dim() == 2 and y_pred.shape[1] == 1 and y.dim() == 1:
            y_pred = y_pred.squeeze(1)
        elif y.dim() == 1 and y_pred.dim() == 2:
            y = y.unsqueeze(1)

        # Compute loss
        loss = self.loss_fn(y_pred, y)

        # Store predictions for epoch-level metrics
        self.test_preds.append(y_pred.detach().cpu())
        self.test_targets.append(y.detach().cpu())

        # Log loss
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self) -> None:
        """Calculate epoch-level test metrics."""
        if not self.test_preds:
            return

        # Concatenate all predictions and targets
        all_preds = torch.cat(self.test_preds, dim=0)
        all_targets = torch.cat(self.test_targets, dim=0)

        # Calculate metrics
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2))
        target_std = torch.std(all_targets)
        nrmse = rmse / target_std if target_std > 0 else rmse
        mae = torch.mean(torch.abs(all_preds - all_targets))

        # Log metrics
        self.log("test_rmse", rmse)
        self.log("test_nrmse", nrmse)
        self.log("test_mae", mae)

        # Print results
        print(f"\nTest Results:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  NRMSE: {nrmse:.6f}")
        print(f"  MAE: {mae:.6f}")

        # Clear stored predictions
        self.test_preds.clear()
        self.test_targets.clear()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW optimizer with cosine annealing scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing scheduler with optional warmup
        if self.hparams.warmup_epochs > 0:
            # Linear warmup followed by cosine annealing
            def lr_lambda(epoch):
                if epoch < self.hparams.warmup_epochs:
                    # Linear warmup
                    return epoch / self.hparams.warmup_epochs
                else:
                    # Cosine annealing
                    progress = (epoch - self.hparams.warmup_epochs) / (
                        self.hparams.epochs - self.hparams.warmup_epochs
                    )
                    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # Standard cosine annealing
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