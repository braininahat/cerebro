"""Challenge 1 Module: Response Time Prediction from CCD Task.

Extracted from notebook 04_train_challenge1.py. Implements EEGNeX model
with MSE loss and NRMSE metric for response time prediction.

Key features:
- MSE loss for training
- NRMSE metric for validation (normalized by target std)
- AdamW optimizer with CosineAnnealingLR scheduler
- Prediction storage for epoch-level metric calculation
"""

import logging
from typing import List

import lightning as L
import torch
from braindecode.models import EEGNeX

logger = logging.getLogger(__name__)


class Challenge1Module(L.LightningModule):
    """LightningModule for Challenge 1 (RT prediction from CCD task).

    Args:
        n_chans: Number of EEG channels (default: 129 for HBN)
        n_outputs: Number of output neurons (default: 1 for regression)
        n_times: Window length in samples (default: 200 = 2s @ 100Hz)
        sfreq: Sampling frequency in Hz (default: 100)
        lr: Learning rate for AdamW (default: 0.001)
        weight_decay: Weight decay for AdamW (default: 0.00001)
        epochs: Total training epochs for scheduler (default: 100)

    Example:
        >>> model = Challenge1Module(lr=0.001, weight_decay=0.00001)
        >>> trainer = Trainer(max_epochs=100)
        >>> trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_outputs: int = 1,
        n_times: int = 200,
        sfreq: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = EEGNeX(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
        )

        # Loss
        self.loss_fn = torch.nn.MSELoss()

        # Metrics storage for epoch-level NRMSE calculation
        self.val_preds: List[torch.Tensor] = []
        self.val_targets: List[torch.Tensor] = []
        self.test_preds: List[torch.Tensor] = []
        self.test_targets: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EEGNeX.

        Args:
            x: Input tensor of shape (batch_size, n_chans, n_times)

        Returns:
            Predictions of shape (batch_size, n_outputs)
        """
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Tuple of (X, y, ...) from DataLoader
            batch_idx: Batch index

        Returns:
            Loss value
        """
        # Batch: (X, y, ...)
        X, y = batch[0], batch[1]
        X = X.float()
        y = y.float().view(-1, 1)

        # Forward
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)

        # Log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step.

        Stores predictions for epoch-level NRMSE calculation.

        Args:
            batch: Tuple of (X, y, ...) from DataLoader
            batch_idx: Batch index

        Returns:
            Loss value
        """
        X, y = batch[0], batch[1]
        X = X.float()
        y = y.float().view(-1, 1)

        # Forward
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)

        # Store for NRMSE computation
        self.val_preds.append(y_pred.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        # Log
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Calculate and log NRMSE at end of validation epoch."""
        # Compute NRMSE (normalized RMSE)
        all_preds = torch.cat(self.val_preds, dim=0).squeeze()
        all_targets = torch.cat(self.val_targets, dim=0).squeeze()

        # RMSE
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2))

        # Normalize by target standard deviation
        target_std = torch.std(all_targets)
        nrmse = rmse / target_std if target_std > 0 else rmse

        # Log
        self.log("val_rmse", rmse, prog_bar=False)
        self.log("val_nrmse", nrmse, prog_bar=True)

        # Clear for next epoch
        self.val_preds.clear()
        self.val_targets.clear()

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Test step.

        Stores predictions for epoch-level NRMSE calculation.

        Args:
            batch: Tuple of (X, y, ...) from DataLoader
            batch_idx: Batch index

        Returns:
            Loss value
        """
        X, y = batch[0], batch[1]
        X = X.float()
        y = y.float().view(-1, 1)

        # Forward
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)

        # Store for NRMSE computation
        self.test_preds.append(y_pred.detach().cpu())
        self.test_targets.append(y.detach().cpu())

        # Log
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self) -> None:
        """Calculate and log NRMSE at end of test epoch."""
        # Compute NRMSE
        all_preds = torch.cat(self.test_preds, dim=0).squeeze()
        all_targets = torch.cat(self.test_targets, dim=0).squeeze()

        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2))
        target_std = torch.std(all_targets)
        nrmse = rmse / target_std if target_std > 0 else rmse

        self.log("test_rmse", rmse)
        self.log("test_nrmse", nrmse)

        print(f"\nTest Results:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  NRMSE: {nrmse:.6f}")

        # Clear
        self.test_preds.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        """Configure AdamW optimizer with CosineAnnealingLR scheduler.

        Returns:
            Tuple of ([optimizer], [scheduler])
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
        )
        return [optimizer], [scheduler]
