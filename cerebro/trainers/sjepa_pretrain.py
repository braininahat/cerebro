"""S-JEPA (Signal Joint-Embedding Predictive Architecture) pretraining trainer.

Implements the exact methodology from /home/varun/repos/cerebro/docs/signaljepa.pdf:
- Spatial block masking strategy (masks channels within radius of random center)
- L1 loss on latent token predictions (not pixel reconstruction)
- EMA-updated target encoder for stable targets
- 4-layer Transformer predictor

Key differences from MAE:
- MAE: Reconstructs raw pixels with MSE loss
- S-JEPA: Predicts latent embeddings with L1 loss + EMA targets

Architecture (Figure 1 from paper):
    Input (B, C, T)
    → Local encoder (per-channel CNN)
    → Tokens (B, C×t, d)
    → + Positional encoding (temporal cosine + spatial trainable)
    → Spatial block masking
    → Contextual encoder (only unmasked tokens)
    → Predictor → Predict masked tokens
    → L1 loss vs. EMA target encoder (all tokens)
"""

from copy import deepcopy
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class SpatialBlockMasker:
    """Spatial block masking for EEG electrodes based on 3D coordinates.

    Masks all channels within a spherical region around a randomly chosen center channel.
    This is the novel contribution from the SignalJEPA paper vs. random/temporal masking.

    Args:
        chs_info: List of channel info dicts (MNE format) with 3D coordinates
        mask_diameter_pct: Diameter of mask as percentage of head size (40, 60, or 80)

    Implementation follows Figure 2 from paper:
    - Pick random center electrode
    - Calculate distances to all other electrodes in 3D space
    - Mask all electrodes within radius threshold
    """

    def __init__(self, chs_info: list[dict[str, Any]], mask_diameter_pct: int = 60):
        self.chs_info = chs_info
        self.mask_diameter_pct = mask_diameter_pct
        self.n_chans = len(chs_info)

        # Extract 3D coordinates (loc[3:6] for braindecode compatibility)
        # loc[0:3] would be MNE format, but braindecode uses loc[3:6]
        self.coords = np.array([ch['loc'][3:6] for ch in chs_info])  # (n_chans, 3)

        # Compute pairwise distances between all electrodes
        # Shape: (n_chans, n_chans)
        self.distances = self._compute_distances()

        # Set radius threshold as percentage of maximum head distance
        max_distance = self.distances.max()
        self.radius_threshold = (mask_diameter_pct / 100.0) * max_distance

    def _compute_distances(self) -> np.ndarray:
        """Compute Euclidean distance matrix between all electrode pairs."""
        # coords: (n_chans, 3)
        # Expand to (n_chans, 1, 3) and (1, n_chans, 3) for broadcasting
        diff = self.coords[:, None, :] - self.coords[None, :, :]  # (n_chans, n_chans, 3)
        distances = np.sqrt((diff ** 2).sum(axis=2))  # (n_chans, n_chans)
        return distances

    def get_mask(self, center_ch_idx: Optional[int] = None) -> torch.Tensor:
        """Generate spatial block mask centered on a random (or specified) channel.

        Args:
            center_ch_idx: Center channel index (None = random)

        Returns:
            Binary mask tensor of shape (n_chans,) where 1 = masked, 0 = visible
        """
        if center_ch_idx is None:
            center_ch_idx = np.random.randint(0, self.n_chans)

        # Mask all channels within radius of center
        mask = self.distances[center_ch_idx] <= self.radius_threshold
        return torch.from_numpy(mask).float()

    def get_batch_masks(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate independent masks for each sample in batch.

        Args:
            batch_size: Number of masks to generate
            device: Device to place masks on

        Returns:
            Batch of masks (batch_size, n_chans) where 1 = masked, 0 = visible
        """
        masks = []
        for _ in range(batch_size):
            masks.append(self.get_mask())
        return torch.stack(masks).to(device)  # (batch_size, n_chans)


class SJEPATrainer(L.LightningModule):
    """S-JEPA pretraining trainer following the exact paper methodology.

    Architecture components:
    1. Local encoder: VanillaSignalJEPAEncoder (already includes local + contextual)
    2. Contextual target encoder: EMA-updated copy (non-trainable)
    3. Predictor: 4-layer Transformer decoder
    4. Spatial block masker: Masks spherical electrode regions

    Training procedure:
    1. Local encoder processes all channels → tokens (B, C×t, d)
    2. Spatial block masking removes ~40-80% of channel tokens
    3. Contextual encoder processes only unmasked tokens
    4. EMA target encoder processes ALL tokens (no masking) → targets
    5. Predictor reconstructs masked tokens from context
    6. L1 loss between predictions and EMA targets

    Args:
        encoder: VanillaSignalJEPAEncoder instance
        mask_diameter_pct: Spatial mask diameter as % of head size (40, 60, or 80)
        ema_momentum: EMA decay rate for target encoder (default: 0.996)
        predictor_depth: Number of Transformer decoder layers (default: 4)
        predictor_heads: Number of attention heads in predictor (default: 8)
        lr: Learning rate (default: 1e-4)
        weight_decay: AdamW weight decay (default: 0.05)
        warmup_epochs: Cosine schedule warmup epochs (default: 10)
        max_epochs: Total epochs for cosine schedule (default: 100)
    """

    def __init__(
        self,
        encoder: nn.Module,  # VanillaSignalJEPAEncoder
        mask_diameter_pct: int = 60,
        ema_momentum: float = 0.996,
        predictor_depth: int = 4,
        predictor_heads: int = 8,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])

        # Encoder (trainable)
        self.encoder = encoder
        d_model = encoder.output_dim

        # Target encoder (EMA-updated, non-trainable)
        self.target_encoder = deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.target_encoder.eval()  # Set to eval mode for consistency

        # Predictor: 4-layer Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=predictor_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerDecoder(decoder_layer, num_layers=predictor_depth)

        # Learnable mask token for predictor queries
        self.mask_token = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Spatial block masker (initialized in setup() when we have chs_info)
        self.masker: Optional[SpatialBlockMasker] = None

        self.ema_momentum = ema_momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

    def setup(self, stage: str):
        """Initialize spatial masker with electrode coordinates."""
        if self.masker is None:
            # Load electrode coordinates
            from cerebro.utils.electrode_locations import load_hbn_chs_info
            chs_info = load_hbn_chs_info()
            self.masker = SpatialBlockMasker(
                chs_info=chs_info,
                mask_diameter_pct=self.hparams.mask_diameter_pct
            )

    @torch.no_grad()
    def _update_target_encoder(self):
        """Update target encoder parameters via exponential moving average."""
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(self.ema_momentum).add_(param_q.data, alpha=1 - self.ema_momentum)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with spatial block masking following JEPA paper architecture.

        Architecture (per SignalJEPA paper):
        1. Local encoding (per-channel CNN)
        2. Positional encoding (spatial + temporal)
        3. Masking and token separation
        4. Contextual encoding (transformer, ONLY on visible tokens)
        5. Target encoding (EMA contextual encoder on ALL tokens)
        6. Predictor (reconstruct masked from visible context)

        Args:
            x: Input EEG (batch_size, n_chans, n_times)
            mask: Optional precomputed channel mask (batch_size, n_chans)

        Returns:
            predicted: Predicted masked tokens (batch_size, n_masked_tokens, d_model)
            target: Target embeddings from EMA encoder (batch_size, n_masked_tokens, d_model)
            mask: Channel mask used (batch_size, n_chans)
        """
        B, C, T = x.shape

        # Generate spatial block masks if not provided
        if mask is None:
            mask = self.masker.get_batch_masks(B, x.device)  # (B, C)

        # ===== STEP 1: Local encoding (per-channel features) =====
        # Use braindecode's feature_encoder component directly (public API)
        local_features = self.encoder.model.feature_encoder(x)  # (B, C×t, d)
        n_tokens_per_chan = local_features.shape[1] // C  # t time windows per channel

        # ===== STEP 2: Positional encoding =====
        # Add spatial (channel location) and temporal positional encodings
        pos_encoding = self.encoder.model.pos_encoder(local_features)  # (B, C×t, d)
        local_features = local_features + pos_encoding

        # ===== STEP 3: Masking and token separation =====
        # Expand channel mask to cover all time windows per channel
        # mask: (B, C) → (B, C, t) → (B, C×t)
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, n_tokens_per_chan).reshape(B, -1)
        visible_mask = mask_expanded == 0  # (B, C×t)
        masked_mask = mask_expanded == 1   # (B, C×t)

        # Extract visible local features (BEFORE contextual encoding)
        visible_local = []
        for i in range(B):
            visible_local.append(local_features[i][visible_mask[i]])

        # Pad visible tokens for batching
        max_visible = max(t.shape[0] for t in visible_local)
        visible_padded = torch.zeros(B, max_visible, local_features.shape[-1], device=x.device)
        for i, tokens in enumerate(visible_local):
            visible_padded[i, :len(tokens)] = tokens

        # ===== STEP 4: Contextual encoding (ONLY on visible tokens) =====
        # This is the key fix: contextual encoder sees only partial information
        visible_context = self.encoder.model.transformer.encoder(visible_padded)  # (B, max_visible, d)

        # ===== STEP 5: Predictor (reconstruct masked from visible context) =====
        # Create learnable queries for masked positions
        masked_tokens_list = []
        for i in range(B):
            n_masked = masked_mask[i].sum().item()
            masked_tokens_list.append(self.mask_token.expand(n_masked, -1))

        max_masked = max(t.shape[0] for t in masked_tokens_list)
        masked_queries = torch.zeros(B, max_masked, local_features.shape[-1], device=x.device)
        for i, tokens in enumerate(masked_tokens_list):
            masked_queries[i, :len(tokens)] = tokens

        # Predictor: cross-attend masked queries to visible context
        predicted = self.predictor(
            tgt=masked_queries,        # Queries for masked positions
            memory=visible_context     # Context from visible tokens (now correctly contextual!)
        )  # (B, max_masked, d)

        # ===== STEP 6: Target encoding (EMA encoder on ALL tokens) =====
        with torch.no_grad():
            # Get local features from target encoder
            target_local = self.target_encoder.model.feature_encoder(x)  # (B, C×t, d)
            target_pos = self.target_encoder.model.pos_encoder(target_local)
            target_local = target_local + target_pos

            # Apply contextual encoding to ALL tokens (target sees full context)
            target_context = self.target_encoder.model.transformer.encoder(target_local)  # (B, C×t, d)

        # Extract target tokens at masked positions
        target_masked_list = []
        for i in range(B):
            target_masked_list.append(target_context[i][masked_mask[i]])

        # Pad targets
        target_masked = torch.zeros(B, max_masked, local_features.shape[-1], device=x.device)
        for i, tokens in enumerate(target_masked_list):
            target_masked[i, :len(tokens)] = tokens

        return predicted, target_masked, mask

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step with L1 loss on masked token predictions.

        Args:
            batch: EEG tensor (batch_size, n_chans, n_times)

        Returns:
            L1 loss between predicted and target masked tokens
        """
        x = batch  # RawEEGDataset returns just EEG tensors, no labels

        # Forward pass with masking
        predicted, target, mask = self.forward(x)

        # L1 loss (per paper, not MSE)
        loss = F.l1_loss(predicted, target, reduction='mean')

        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/mask_ratio', mask.mean(), on_step=False, on_epoch=True)

        # Update target encoder via EMA
        self._update_target_encoder()

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x = batch

        predicted, target, mask = self.forward(x)
        loss = F.l1_loss(predicted, target, reduction='mean')

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mask_ratio', mask.mean(), on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine annealing schedule."""
        # Separate weight decay for different parameter groups
        params_with_decay = []
        params_without_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # No weight decay for biases and layer norms
            if 'bias' in name or 'norm' in name:
                params_without_decay.append(param)
            else:
                params_with_decay.append(param)

        param_groups = [
            {'params': params_with_decay, 'weight_decay': self.weight_decay},
            {'params': params_without_decay, 'weight_decay': 0.0},
        ]

        optimizer = AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))

        # Cosine annealing schedule
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - self.warmup_epochs,
            eta_min=self.lr * 0.01,  # End at 1% of initial LR
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
