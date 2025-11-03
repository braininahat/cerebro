"""Masked Autoencoder (MAE) pretraining for EEG encoders.

Implements self-supervised pretraining via masking and reconstruction:
- Random masking in time dimension (e.g., 50% of timesteps)
- Random masking in channel dimension (e.g., 30% of channels)
- Reconstruction loss on masked regions only

Works with any encoder-decoder architecture.
"""

from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAutoencoderTrainer(L.LightningModule):
    """Masked Autoencoder pretraining for EEG encoders.

    Masks random timesteps and channels, then reconstructs the original signal.
    Loss is computed only on masked regions (standard MAE approach).

    Architecture:
        Input (B, C, T) →
        Apply mask →
        Encoder →
        Decoder →
        Reconstruct (B, C, T) →
        Compute loss on masked regions only

    Args:
        encoder: Encoder module (e.g., SignalJEPA_LearnedChannels)
        decoder: Decoder module (reconstructs from encoder features)
        time_mask_ratio: Fraction of timesteps to mask (default: 0.5)
        channel_mask_ratio: Fraction of channels to mask (default: 0.3)
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        warmup_epochs: Number of warmup epochs

    Example:
        >>> from cerebro.models.components import SignalJEPAWithLearnedChannels
        >>> encoder = SignalJEPAWithLearnedChannels(n_chans=129, n_times=200)
        >>> decoder = SimpleDecoder(input_dim=encoder.output_dim, n_chans=129, n_times=200)
        >>> trainer = MaskedAutoencoderTrainer(encoder, decoder)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        time_mask_ratio: float = 0.5,
        channel_mask_ratio: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])

        self.encoder = encoder
        self.decoder = decoder

        self.time_mask_ratio = time_mask_ratio
        self.channel_mask_ratio = channel_mask_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

    def apply_mask(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking to input EEG.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Tuple of:
                - masked_x: Input with masked regions set to 0
                - mask: Boolean mask (True = keep, False = masked)
        """
        batch_size, n_chans, n_times = x.shape
        device = x.device

        # Create mask (True = keep, False = mask)
        mask = torch.ones(batch_size, n_chans, n_times, dtype=torch.bool, device=device)

        # Time masking: randomly mask time_mask_ratio of timesteps
        n_time_masked = int(n_times * self.time_mask_ratio)
        for b in range(batch_size):
            time_indices = torch.randperm(n_times, device=device)[:n_time_masked]
            mask[b, :, time_indices] = False

        # Channel masking: randomly mask channel_mask_ratio of channels
        n_chan_masked = int(n_chans * self.channel_mask_ratio)
        for b in range(batch_size):
            chan_indices = torch.randperm(n_chans, device=device)[:n_chan_masked]
            mask[b, chan_indices, :] = False

        # Apply mask (set masked regions to 0)
        masked_x = x.clone()
        masked_x[~mask] = 0.0

        return masked_x, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input EEG.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Encoded features
        """
        return self.encoder(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training step with masked reconstruction.

        Args:
            batch: Either EEG tensor or tuple of (eeg, target)
            batch_idx: Batch index

        Returns:
            Loss value
        """
        # Handle both formats: tensor or (eeg, target) tuple
        if isinstance(batch, tuple):
            eeg, _ = batch  # Ignore labels for unsupervised pretraining
        else:
            eeg = batch

        # Apply random masking
        masked_eeg, mask = self.apply_mask(eeg)

        # Encode masked input
        features = self.encoder(masked_eeg)

        # Decode to reconstruct original signal
        reconstructed = self.decoder(features)

        # Compute loss only on masked regions
        loss = F.mse_loss(reconstructed[~mask], eeg[~mask])

        # Additional metrics
        mask_ratio = (~mask).float().mean()
        reconstruction_quality = F.mse_loss(reconstructed[mask], eeg[mask])  # Quality on unmasked regions

        # Logging
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("mask_ratio", mask_ratio, prog_bar=False, on_step=False, on_epoch=True)
        self.log("reconstruction_quality", reconstruction_quality, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Either EEG tensor or tuple of (eeg, target)
            batch_idx: Batch index

        Returns:
            Loss value
        """
        # Handle both formats: tensor or (eeg, target) tuple
        if isinstance(batch, tuple):
            eeg, _ = batch
        else:
            eeg = batch

        # Apply masking (deterministic for validation)
        masked_eeg, mask = self.apply_mask(eeg)

        # Encode and decode
        features = self.encoder(masked_eeg)
        reconstructed = self.decoder(features)

        # Loss on masked regions
        loss = F.mse_loss(reconstructed[~mask], eeg[~mask])

        # Logging
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with warmup + cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Warmup + cosine annealing scheduler
        # Note: trainer.max_epochs is available during training
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.warmup_epochs
            else:
                # Cosine annealing after warmup
                progress = (epoch - self.warmup_epochs) / (
                    self.trainer.max_epochs - self.warmup_epochs
                )
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class SimpleDecoder(nn.Module):
    """Simple decoder for MAE reconstruction.

    Upsamples encoder features back to original EEG dimensions.

    Args:
        input_dim: Encoder output dimension (e.g., 64 from SignalJEPA)
        n_chans: Number of output channels
        n_times: Number of output timesteps
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        input_dim: int,
        n_chans: int,
        n_times: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_chans = n_chans
        self.n_times = n_times

        # Project encoder features to hidden dimension
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_chans * n_times)

        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to reconstruct EEG.

        Args:
            features: Encoder output (batch, seq_len, input_dim) or (batch, input_dim)

        Returns:
            Reconstructed EEG (batch, n_chans, n_times)
        """
        # Handle sequence outputs (e.g., from transformer)
        if features.dim() == 3:
            # Global average pooling over sequence dimension
            features = features.mean(dim=1)  # (batch, input_dim)

        # Decode
        x = self.activation(self.fc1(features))
        x = self.fc2(x)  # (batch, n_chans * n_times)

        # Reshape to EEG dimensions
        x = x.view(-1, self.n_chans, self.n_times)

        return x
