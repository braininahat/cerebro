# finetune_challenge.py
"""
Fine-tuning modules for EEG challenges using pretrained LaBraM tokenizer.
Supports progressive unfreezing and task-specific heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional

# Import your tokenizer - adjust based on which version you use
try:
    from tokenizer_vq_snp_v2 import TokenizerVQSNP
except ImportError:
    from tokenizer_vq_snp import TokenizerVQSNP


class FineTuneChallenge1(pl.LightningModule):
    """
    Fine-tune VQ-SNP tokenizer for Challenge 1 (RT prediction).

    Features:
    - Load pretrained tokenizer
    - Optional encoder freezing for progressive training
    - Task-specific regression head
    - Proper regularization
    """

    def __init__(
        self,
        tokenizer_ckpt: str,
        freeze_encoder: bool = True,
        freeze_vq: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        dropout: float = 0.3,
        hidden_sizes: list = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained tokenizer
        print(f"Loading tokenizer from {tokenizer_ckpt}")
        try:
            tokenizer = TokenizerVQSNP.load_from_checkpoint(tokenizer_ckpt)
        except Exception as e:
            print(f"Warning: Could not load from checkpoint: {e}")
            print("Creating new tokenizer instead")
            tokenizer = TokenizerVQSNP()

        # Extract components
        self.encoder = tokenizer.enc
        self.vq = tokenizer.vq
        self.patchify = tokenizer.patchify

        # Store dimensions
        self.n_chans = tokenizer.hparams.n_chans if hasattr(
            tokenizer.hparams, 'n_chans') else 129
        self.patch_len = tokenizer.patch_len
        self.P = tokenizer.P
        self.dim = tokenizer.hparams.dim if hasattr(
            tokenizer.hparams, 'dim') else 256

        # Freeze components if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("✓ Encoder frozen")

        if freeze_vq:
            for param in self.vq.parameters():
                param.requires_grad = False
            print("✓ VQ frozen")

        # Task-specific head for RT prediction
        if hidden_sizes is None:
            hidden_sizes = [512, 128]

        # Calculate input size: C * P * D
        input_size = self.n_chans * self.P * self.dim

        # Build regression head
        layers = []
        layers.append(nn.Flatten())
        layers.append(nn.LayerNorm(input_size))

        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))  # RT prediction

        self.head = nn.Sequential(*layers)

        # Loss
        self.loss_fn = nn.MSELoss()

        # For validation metrics
        self.val_preds = []
        self.val_targets = []

    def forward(self, x):
        """
        x: (B, C, T)
        Returns: (B, 1) - predicted RT
        """
        B, C, T = x.shape
        xp = self.patchify(x)  # (B, C, P, L)

        # Encode
        z = self.encoder(xp)  # (B, C, P, D) or depends on encoder version

        # Handle different encoder outputs
        if not isinstance(z, torch.Tensor):
            # If encoder returns tuple, take first element
            z = z[0]

        # Ensure correct shape
        if z.dim() == 4:  # (B, C, P, D)
            pass
        elif z.dim() == 3:  # (B, C*P, D)
            z = z.reshape(B, C, self.P, self.dim)
        else:
            raise ValueError(f"Unexpected encoder output shape: {z.shape}")

        # Quantize (optional, can skip if frozen)
        if not self.hparams.freeze_vq:
            zq, _, _ = self.vq(z)
        else:
            with torch.no_grad():
                zq, _, _ = self.vq(z)

        # Predict RT
        rt_pred = self.head(zq)

        return rt_pred

    def training_step(self, batch, batch_idx):
        X, y = batch[0], batch[1].view(-1, 1)  # Ensure y is (B, 1)
        y_pred = self(X)

        loss = self.loss_fn(y_pred, y)

        # Calculate metrics
        with torch.no_grad():
            mae = F.l1_loss(y_pred, y)

        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch[0], batch[1].view(-1, 1)
        y_pred = self(X)

        loss = self.loss_fn(y_pred, y)

        # Store for epoch-end metrics
        self.val_preds.append(y_pred.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return

        preds = torch.cat(self.val_preds).squeeze()
        targets = torch.cat(self.val_targets).squeeze()

        # Calculate metrics
        mse = torch.mean((preds - targets) ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(preds - targets))

        # Normalized RMSE
        target_std = torch.std(targets)
        nrmse = rmse / target_std if target_std > 0 else rmse

        # R-squared
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0)

        self.log("val_rmse", rmse, prog_bar=True)
        self.log("val_nrmse", nrmse, prog_bar=True)
        self.log("val_mae", mae)
        self.log("val_r2", r2)

        # Clear cache
        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        # Different learning rates for frozen/unfrozen parts
        if self.hparams.freeze_encoder and self.hparams.freeze_vq:
            # Only training head
            params = self.head.parameters()
        else:
            # Train everything with potentially different LRs
            params = [
                {'params': self.head.parameters(), 'lr': self.hparams.lr},
                {'params': self.encoder.parameters(), 'lr': self.hparams.lr * 0.1},
                {'params': self.vq.parameters(), 'lr': self.hparams.lr * 0.1},
            ]

        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if hasattr(self, 'trainer') else 50,
            eta_min=self.hparams.lr * 0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }


class FineTuneChallenge2(pl.LightningModule):
    """
    Fine-tune VQ-SNP tokenizer for Challenge 2 (binary classification).
    Similar structure to Challenge 1 but with classification head.
    """

    def __init__(
        self,
        tokenizer_ckpt: str,
        freeze_encoder: bool = True,
        freeze_vq: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        dropout: float = 0.3,
        hidden_sizes: list = None,
        pos_weight: Optional[float] = None,  # For class imbalance
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained tokenizer
        print(f"Loading tokenizer from {tokenizer_ckpt}")
        tokenizer = TokenizerVQSNP.load_from_checkpoint(tokenizer_ckpt)

        self.encoder = tokenizer.enc
        self.vq = tokenizer.vq
        self.patchify = tokenizer.patchify

        self.n_chans = tokenizer.hparams.n_chans if hasattr(
            tokenizer.hparams, 'n_chans') else 129
        self.patch_len = tokenizer.patch_len
        self.P = tokenizer.P
        self.dim = tokenizer.hparams.dim if hasattr(
            tokenizer.hparams, 'dim') else 256

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if freeze_vq:
            for param in self.vq.parameters():
                param.requires_grad = False

        # Build classification head
        if hidden_sizes is None:
            hidden_sizes = [512, 128]

        input_size = self.n_chans * self.P * self.dim

        layers = []
        layers.append(nn.Flatten())
        layers.append(nn.LayerNorm(input_size))

        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))  # Binary classification

        self.head = nn.Sequential(*layers)

        # Loss with optional class weighting
        if pos_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        self.val_preds = []
        self.val_targets = []

    def forward(self, x):
        """x: (B, C, T) -> (B, 1) logits"""
        B, C, T = x.shape
        xp = self.patchify(x)

        z = self.encoder(xp)

        if not isinstance(z, torch.Tensor):
            z = z[0]

        if z.dim() == 3:
            z = z.reshape(B, C, self.P, self.dim)

        if not self.hparams.freeze_vq:
            zq, _, _ = self.vq(z)
        else:
            with torch.no_grad():
                zq, _, _ = self.vq(z)

        logits = self.head(zq)
        return logits

    def training_step(self, batch, batch_idx):
        X, y = batch[0], batch[1].view(-1, 1).float()
        logits = self(X)

        loss = self.loss_fn(logits, y)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            acc = ((probs > 0.5) == y).float().mean()

        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch[0], batch[1].view(-1, 1).float()
        logits = self(X)

        loss = self.loss_fn(logits, y)

        self.val_preds.append(torch.sigmoid(logits).detach().cpu())
        self.val_targets.append(y.detach().cpu())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return

        preds = torch.cat(self.val_preds).squeeze()
        targets = torch.cat(self.val_targets).squeeze()

        # Accuracy
        acc = ((preds > 0.5) == targets).float().mean()

        # AUC-ROC (if sklearn available)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(targets.numpy(), preds.numpy())
            self.log("val_auc", auc, prog_bar=True)
        except ImportError:
            pass

        self.log("val_acc", acc, prog_bar=True)

        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        if self.hparams.freeze_encoder and self.hparams.freeze_vq:
            params = self.head.parameters()
        else:
            params = [
                {'params': self.head.parameters(), 'lr': self.hparams.lr},
                {'params': self.encoder.parameters(), 'lr': self.hparams.lr * 0.1},
                {'params': self.vq.parameters(), 'lr': self.hparams.lr * 0.1},
            ]

        optimizer = torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if hasattr(
                self, 'trainer') else 50
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
