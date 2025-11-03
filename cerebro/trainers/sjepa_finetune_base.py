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
        warmup_reruns_lr_finder: If True, re-run LR finder at warmup_epochs when encoder unfreezes
        lr_finder_min: Minimum LR for LR finder search range
        lr_finder_max: Maximum LR for LR finder search range
        lr_finder_num_training: Number of training steps for LR finder
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
        warmup_reruns_lr_finder: bool = True,
        lr_finder_min: float = 1e-8,
        lr_finder_max: float = 1e-2,
        lr_finder_num_training: int = 100,
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
        self.warmup_reruns_lr_finder = warmup_reruns_lr_finder
        self.lr_finder_min = lr_finder_min
        self.lr_finder_max = lr_finder_max
        self.lr_finder_num_training = lr_finder_num_training
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor

        # Model will be loaded in setup() after device is set
        self.model: Optional[nn.Module] = None
        self._model_loaded = False
        self._warmup_lr_finder_executed = False  # Guard flag to prevent infinite loop

        # Loss function
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        # For competition metrics calculation (RMSE, NRMSE, R²)
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []

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

            # Log total parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"[SETUP] Total parameters: {total_params:,}")

            # Apply initial freezing if needed
            if self.freeze_encoder:
                self._freeze_encoder_components()
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"[SETUP] Encoder frozen (freeze_encoder=True)")
                print(f"[SETUP] Trainable parameters: {trainable_params:,} / {total_params:,} "
                      f"({100 * trainable_params / total_params:.1f}%)")
            else:
                # Initially freeze encoder for warmup
                self._freeze_encoder_components()
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"[SETUP] Encoder will be frozen for {self.warmup_epochs} warmup epochs, then unfrozen")
                print(f"[SETUP] Initial trainable parameters (warmup): {trainable_params:,} / {total_params:,} "
                      f"({100 * trainable_params / total_params:.1f}%)")

    def _freeze_encoder_components(self):
        """Freeze pretrained encoder components."""
        encoder_prefixes = self._get_encoder_param_names()
        frozen_count = 0
        frozen_numel = 0
        for name, param in self.model.named_parameters():
            if any(prefix in name for prefix in encoder_prefixes):
                param.requires_grad = False
                frozen_count += 1
                frozen_numel += param.numel()
        print(f"[FREEZE] Froze {frozen_count} encoder parameters ({frozen_numel:,} elements)")

    def _unfreeze_encoder_components(self):
        """Unfreeze pretrained encoder components."""
        encoder_prefixes = self._get_encoder_param_names()
        unfrozen_count = 0
        unfrozen_numel = 0
        for name, param in self.model.named_parameters():
            if any(prefix in name for prefix in encoder_prefixes):
                param.requires_grad = True
                unfrozen_count += 1
                unfrozen_numel += param.numel()

        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"[UNFREEZE] Unfroze {unfrozen_count} encoder parameters ({unfrozen_numel:,} elements)")
        print(f"[UNFREEZE] Total trainable: {total_trainable:,} / {total_params:,} "
              f"({100 * total_trainable / total_params:.1f}%)")

    def on_train_epoch_start(self):
        """Handle encoder freezing/unfreezing based on warmup schedule."""
        if not self.freeze_encoder and self.current_epoch == self.warmup_epochs:
            # RE-ENTRY GUARD: Prevent infinite loop if LR finder re-triggers this hook
            # tuner.lr_find() internally calls trainer.fit() which re-executes this hook
            if self._warmup_lr_finder_executed:
                print(f"[EPOCH {self.current_epoch}] Warmup transition already executed, skipping")
                return

            self._warmup_lr_finder_executed = True

            # Transition from warmup to full fine-tuning
            self._unfreeze_encoder_components()
            print(f"\n[EPOCH {self.current_epoch}] Warmup complete, encoder unfrozen")

            # Optionally re-run LR finder with full model (encoder + new layers)
            # IMPORTANT: Run BEFORE reconfiguring optimizer so we can apply suggested LR
            if self.warmup_reruns_lr_finder:
                self._run_warmup_lr_finder()

            # CRITICAL: Reconfigure optimizer to include encoder parameters
            # Lightning only calls configure_optimizers() once at training start,
            # so we must manually update param_groups when encoder unfreezes
            self._reconfigure_optimizer_with_encoder()

    def _reconfigure_optimizer_with_encoder(self):
        """Reconfigure optimizer to include unfrozen encoder parameters.

        Called at epoch=warmup_epochs when encoder is unfrozen.
        Adds encoder parameters as a new param_group using PyTorch's add_param_group() API.
        This properly initializes optimizer state (momentum buffers) for encoder parameters.

        CRITICAL: Using add_param_group() instead of direct param_groups assignment to:
        1. Preserve existing optimizer state for new-layer parameters
        2. Initialize momentum buffers (exp_avg, exp_avg_sq) for encoder parameters
        3. Ensure optimizer can actually update the encoder weights
        """
        encoder_prefixes = self._get_encoder_param_names()

        # Collect encoder parameters (now unfrozen)
        encoder_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(prefix in name for prefix in encoder_prefixes):
                encoder_params.append(param)

        if not encoder_params:
            print(f"[OPTIMIZER] WARNING: No encoder parameters found to add!")
            return

        # Add encoder parameters as a new group using PyTorch's official API
        # This properly initializes optimizer state (momentum, variance) for new params
        optimizer = self.optimizers()

        optimizer.add_param_group({
            'params': encoder_params,
            'lr': self.lr * self.encoder_lr_multiplier,
            'weight_decay': self.weight_decay,
        })

        # Log reconfiguration
        encoder_param_count = sum(p.numel() for p in encoder_params)
        print(f"[OPTIMIZER] ✓ Added encoder param group: {len(encoder_params)} params "
              f"({encoder_param_count:,} elements) with lr={self.lr * self.encoder_lr_multiplier:.2e}")
        print(f"[OPTIMIZER] Total param groups: {len(optimizer.param_groups)} "
              f"(group 0: new layers @ {self.lr:.2e}, group 1: encoder @ {self.lr * self.encoder_lr_multiplier:.2e})")

        # Verify encoder params are actually in optimizer
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_in_optimizer = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        print(f"[OPTIMIZER] Trainable params: {total_trainable:,} | In optimizer: {total_in_optimizer:,}")

        if total_trainable != total_in_optimizer:
            print(f"[WARNING] Mismatch between trainable and optimized parameters!")

    def _run_warmup_lr_finder(self):
        """Re-run LR finder at epoch=warmup_epochs when encoder unfreezes.

        This finds an optimal LR for the full model (encoder + new layers) rather
        than using a hardcoded encoder_lr_multiplier. The suggested LR is used for
        both encoder and new layers (full fine-tuning with same LR).

        IMPORTANT: Called BEFORE _reconfigure_optimizer_with_encoder() so we can
        apply the suggested LR to both parameter groups.

        Behavior:
        - Runs LR finder with full model (encoder + new layers both trainable)
        - Updates self.lr with suggested LR
        - Sets encoder_lr_multiplier = 1.0 (same LR for both)
        - Falls back to hardcoded encoder_lr_multiplier if LR finder fails
        """
        print(f"\n[LR FINDER] Re-running at epoch {self.current_epoch} (encoder just unfrozen)")
        print(f"[LR FINDER] Searching range: [{self.lr_finder_min:.2e}, {self.lr_finder_max:.2e}]")
        print(f"[LR FINDER] Strategy: Full model optimization (encoder + new layers)")

        try:
            from lightning.pytorch.tuner import Tuner

            # Create tuner
            tuner = Tuner(self.trainer)

            # Run LR finder
            # NOTE: lr_find temporarily modifies model state, then restores it
            lr_finder = tuner.lr_find(
                self,
                self.trainer.datamodule,
                min_lr=self.lr_finder_min,
                max_lr=self.lr_finder_max,
                num_training=self.lr_finder_num_training,
                mode="exponential",
                update_attr=False,  # Don't auto-update self.lr (we'll do it manually)
            )

            suggested_lr = lr_finder.suggestion()
            print(f"[LR FINDER] ✓ Suggested LR for full model: {suggested_lr:.6e}")

            # Update learning rate for full model
            # Since we're doing full fine-tuning, use same LR for both encoder and new layers
            self.lr = suggested_lr
            self.encoder_lr_multiplier = 1.0

            print(f"[LR FINDER] ✓ Updated self.lr = {self.lr:.6e}")
            print(f"[LR FINDER] ✓ Updated encoder_lr_multiplier = {self.encoder_lr_multiplier:.2f}")
            print(f"[LR FINDER]   (Both encoder and new layers will use lr={self.lr:.2e})")

            # Log to wandb if available
            if hasattr(self.logger, 'experiment'):
                try:
                    import wandb
                    import tempfile
                    from pathlib import Path

                    # Log suggested LR as hyperparameter
                    if hasattr(self.logger.experiment, 'config'):
                        self.logger.experiment.config.update({
                            "encoder_lr_suggested": suggested_lr,
                            "encoder_lr_multiplier_final": self.encoder_lr_multiplier,
                        })

                    # Save and upload LR finder plot
                    with tempfile.TemporaryDirectory() as tmpdir:
                        plot_path = Path(tmpdir) / "lr_finder_warmup.png"
                        fig = lr_finder.plot(suggest=True)
                        fig.savefig(plot_path)
                        self.logger.experiment.log({
                            f"lr_finder_warmup_epoch{self.current_epoch}": wandb.Image(str(plot_path))
                        })
                        print("[LR FINDER] ✓ Plot uploaded to wandb")
                except Exception as e:
                    print(f"[LR FINDER] Warning: Could not upload to wandb: {e}")

        except Exception as e:
            print(f"[LR FINDER] ⚠ LR finder failed: {e}")
            print(f"[LR FINDER] Falling back to hardcoded encoder_lr_multiplier={self.encoder_lr_multiplier}")
            print(f"[LR FINDER] Training will continue with encoder_lr={self.lr * self.encoder_lr_multiplier:.2e}")

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

    def on_after_backward(self):
        """Hook called after backward pass (gradients computed).

        Periodically logs encoder gradient norms to verify optimizer is updating encoder parameters.
        This helps debug issues where encoder parameters aren't being trained despite being unfrozen.
        """
        # Only log every 100 steps to avoid spam
        if self.global_step % 100 == 0 and self.current_epoch >= self.warmup_epochs:
            encoder_prefixes = self._get_encoder_param_names()

            # Find first encoder parameter with gradients and log its gradient norm
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if any(prefix in name for prefix in encoder_prefixes):
                        grad_norm = param.grad.norm().item()
                        self.log('debug/encoder_grad_norm', grad_norm, on_step=True, on_epoch=False)

                        # Also log to console periodically for verification
                        if self.global_step % 500 == 0:
                            print(f"[DEBUG] Step {self.global_step}: Encoder grad norm = {grad_norm:.6f} "
                                  f"(param: {name})")
                        break  # Just check one encoder param

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        eeg, target = batch

        predictions = self.forward(eeg)
        if predictions.ndim > 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)

        loss = self.criterion(predictions, target)

        # Store predictions for epoch-level metrics (RMSE, NRMSE, R²)
        self.val_preds.append(predictions.detach().cpu())
        self.val_targets.append(target.detach().cpu())

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Calculate epoch-level validation metrics matching competition scoring."""
        if not self.val_preds:
            return

        # Concatenate all predictions and targets from the epoch
        all_preds = torch.cat(self.val_preds, dim=0)
        all_targets = torch.cat(self.val_targets, dim=0)

        # Calculate RMSE (Root Mean Squared Error)
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2))

        # Calculate NRMSE (Normalized RMSE using target standard deviation)
        # This matches the competition scoring in startkit/local_scoring.py
        target_std = torch.std(all_targets)
        nrmse = rmse / target_std if target_std > 0 else rmse

        # Calculate MAE (Mean Absolute Error) for additional insight
        mae = torch.mean(torch.abs(all_preds - all_targets))

        # Calculate R-squared (coefficient of determination)
        # Note: Competition uses negative R² but we log the positive value for clarity
        ss_res = torch.sum((all_targets - all_preds) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(0.0)

        # Log all metrics
        self.log("val/rmse", rmse, prog_bar=False)
        self.log("val/nrmse", nrmse, prog_bar=True)  # Primary competition metric
        self.log("val/mae", mae, prog_bar=False)
        self.log("val/r2", r2, prog_bar=False)

        # Clear stored predictions for next epoch
        self.val_preds.clear()
        self.val_targets.clear()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        eeg, target = batch

        predictions = self.forward(eeg)
        if predictions.ndim > 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)

        loss = self.criterion(predictions, target)

        # Store predictions for epoch-level metrics (RMSE, NRMSE, R²)
        self.test_preds.append(predictions.detach().cpu())
        self.test_targets.append(target.detach().cpu())

        self.log('test/loss', loss, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self) -> None:
        """Calculate epoch-level test metrics matching competition scoring."""
        if not self.test_preds:
            return

        # Concatenate all predictions and targets
        all_preds = torch.cat(self.test_preds, dim=0)
        all_targets = torch.cat(self.test_targets, dim=0)

        # Calculate metrics (same as validation)
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2))
        target_std = torch.std(all_targets)
        nrmse = rmse / target_std if target_std > 0 else rmse
        mae = torch.mean(torch.abs(all_preds - all_targets))

        # R-squared
        ss_res = torch.sum((all_targets - all_preds) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(0.0)

        # Log metrics
        self.log("test/rmse", rmse)
        self.log("test/nrmse", nrmse)
        self.log("test/mae", mae)
        self.log("test/r2", r2)

        # Print test results summary (matches competition output format)
        print(f"\nTest Results (Competition Metrics):")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  NRMSE: {nrmse:.6f} ← Primary competition metric")
        print(f"  MAE:   {mae:.6f}")
        print(f"  R²:    {r2:.6f}")

        # Clear stored predictions
        self.test_preds.clear()
        self.test_targets.clear()

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
            new_param_count = sum(p.numel() for p in new_params)
            print(f"[OPTIMIZER] Initial setup: Optimizing {len(new_params)} new layer params "
                  f"({new_param_count:,} elements, lr={self.lr:.2e})")
        else:
            # Both encoder and new layers are trainable (after warmup)
            # NOTE: This branch should rarely execute since configure_optimizers is called once at epoch 0
            param_groups = [
                {'params': encoder_params, 'lr': self.lr * self.encoder_lr_multiplier},
                {'params': new_params, 'lr': self.lr}
            ]
            encoder_param_count = sum(p.numel() for p in encoder_params)
            new_param_count = sum(p.numel() for p in new_params)
            print(f"[OPTIMIZER] Initial setup with unfrozen encoder: {len(encoder_params)} encoder params "
                  f"({encoder_param_count:,} elements, lr={self.lr * self.encoder_lr_multiplier:.2e}) + "
                  f"{len(new_params)} new layer params ({new_param_count:,} elements, lr={self.lr:.2e})")

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
