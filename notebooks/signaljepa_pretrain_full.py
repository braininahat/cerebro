# %% [markdown]
# # SignalJEPA Pretraining: Combined MAE + Contrastive + Auxiliary Demographics
#
# **Multi-objective pretraining strategy:**
# 1. **MAE reconstruction**: Spatial + temporal masking with L1 loss on latent tokens
# 2. **Movie contrastive**: InfoNCE on temporally aligned inter-subject movie responses
# 3. **Auxiliary demographics**: Age, sex, p_factor, attention prediction heads
#
# **Data**: 6 passive tasks (excluding Challenge 1):
# - restingState: Spontaneous neural activity
# - 4 movies: DespicableMe, ThePresent, DiaryOfAWimpyKid, FunwithFractals
# - surroundSupp: Perceptual visual suppression task
#
# **Releases**: R1-R4, R6-R11 (all subjects, no validation split during pretraining)
#
# **Output**: Pretrained encoder checkpoint for transfer learning to Challenge 1/2

# %% imports
from pathlib import Path
import math
import os
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim

# Muon optimizer (optional, falls back to AdamW if not available)
try:
    from muon import SingleDeviceMuonWithAuxAdam  # For single-GPU training
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("⚠️  Muon optimizer not available. Install with: uv pip install git+https://github.com/KellerJordan/Muon")

# SignalJEPA imports
from braindecode.models import SignalJEPA
from cerebro.utils.electrode_locations import load_hbn_chs_info

# ISC contrastive learning utilities (no Lightning dependencies)
from cerebro.utils.movie_windows import load_and_window_movies
from cerebro.utils.contrastive_dataset import ContrastivePairDataset

# Braindecode and startkit imports for data loading
from eegdash import EEGChallengeDataset
from eegdash.dataset import EEGChallengeDataset
from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_fixed_length_windows
from eegdash.hbn.windows import add_extras_columns

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Load environment variables
load_dotenv()

# Suppress verbose warnings
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='eegdash')
warnings.filterwarnings('ignore', category=UserWarning, module='eegdash')

# %% config
class Config:
    # Data paths from environment
    HBN_ROOT = Path(os.getenv("HBN_ROOT", "/media/varun/OS/Users/varun/DATASETS/HBN"))
    CACHE_PATH = Path(os.getenv("CACHE_PATH", "/home/varun/repos/cerebro/cache"))

    # Model architecture - SignalJEPA base encoder
    n_channels = 129
    n_times = 200  # 2.0 seconds at 100 Hz
    sfreq = 100
    input_window_seconds = 2.0

    # SignalJEPA hyperparameters
    transformer_d_model = 64  # Must match feature encoder output (last conv layer)
    transformer_num_encoder_layers = 12  # More layers for pretraining
    transformer_nhead = 8
    dropout = 0.1  # Add dropout for pretraining

    # EMA target encoder
    ema_momentum = 0.996  # Momentum for EMA updates

    # Predictor (reconstructs masked tokens)
    predictor_depth = 4
    predictor_nhead = 8

    # Masking strategy
    spatial_mask_diameter = 60  # 40%, 60%, or 80% of head diameter
    temporal_mask_ratio = 0.4  # Mask 40% of time steps
    temporal_mask_span_ms = 200  # Mask spans of 200ms (20 samples @ 100Hz)

    # Feature toggles
    use_mae = False  # Enable/disable MAE reconstruction (spatial+temporal masking)
    use_contrastive = False  # Enable/disable contrastive learning (ISC movie pairing)
    use_auxiliary = True    # Enable/disable all auxiliary demographic heads

    # Auxiliary head dimensions
    aux_hidden_dim = 128
    use_separable_aux_heads = True  # Use SignalJEPA-style Conv3d heads (vs simple MLPs)
    aux_n_spat_filters = 4  # Spatial filters per head (4-16 typical range)
    aux_heads = {
        'age': {'type': 'regression', 'n_outputs': 1, 'weight': 0.15, 'enabled': True},
        'sex': {'type': 'classification', 'n_outputs': 2, 'weight': 0.05, 'enabled': True},
        'p_factor': {'type': 'regression', 'n_outputs': 1, 'weight': 0.1, 'enabled': True},  # Safe - C2 is externalizing
        'attention': {'type': 'regression', 'n_outputs': 1, 'weight': 0.05, 'enabled': True},
        'internalizing': {'type': 'regression', 'n_outputs': 1, 'weight': 0.05, 'enabled': True},
        'ehq_total': {'type': 'regression', 'n_outputs': 1, 'weight': 0.05, 'enabled': True},
        # EXCLUDED: externalizing (Challenge 2 target), rt_from_stimulus (Challenge 1 target)
    }

    # Loss weights (initial values for GradNorm)
    mae_loss_weight = 0.45  # MAE reconstruction
    contrastive_loss_weight = 0.5  # Movie contrastive
    aux_total_weight = 0.05  # Total weight for all auxiliary heads
    auxiliary_loss_weight = 0.05  # Alias for aux_total_weight

    # GradNorm - Dynamic loss weight balancing
    use_gradnorm = True  # Enable GradNorm for automatic weight balancing
    gradnorm_alpha = 1.5  # Restoring force strength (1.5 = moderate balancing)
    gradnorm_warmup_epochs = 0  # Wait 5 epochs before enabling GradNorm
    gradnorm_update_freq = 10  # Update weights every 10 batches

    # Contrastive learning (ISC-based pairing)
    temperature = 0.07  # Temperature for InfoNCE
    contrastive_batch_size = 128  # Larger for contrastive
    use_triplet_negatives = False  # Use explicit hard negatives from ISC triplets (toggleable)
    time_bin_size_s = 1.0  # Time bin size for ISC pairing (1s = good balance)
    pos_strategy = "same_movie_time"  # ISC positive pairs: same (movie, time_bin)
    neg_strategy = "diff_movie_mixed"  # Hard negatives: different movies
    val_frac = 0.1  # Fraction of subjects for validation (subject-level split)

    # Training
    batch_size = 512  # MAE batch size
    learning_rate = 0.0003  # AdamW learning rate
    weight_decay = 0.05  # Higher for pretraining
    n_epochs = 1  # Long pretraining
    warmup_epochs = 10
    grad_clip = 1.0
    early_stopping_patience = 15  # Stop if no improvement for 15 epochs

    # Optimizer selection
    use_muon = True  # Disabled for stability (NaN issues with high LR)

    # Muon hyperparameters (for 2D weight matrices)
    muon_lr = 0.02              # 10-50x higher than AdamW (paper recommendation)
    muon_momentum = 0.95        # Nesterov momentum
    muon_weight_decay = 0.01    # Moderate regularization
    muon_ns_steps = 5           # Newton-Schulz iterations (informational - uses default)

    # AdamW hyperparameters (for 1D params when using Muon)
    adamw_aux_lr = 3e-4         # AdamW LR for biases/layer norms
    adamw_aux_betas = (0.9, 0.95)  # AdamW betas (informational - uses default)
    adamw_aux_weight_decay = 0.01  # Match Muon weight decay

    # Differential learning rates & encoder freezing
    encoder_lr_multiplier: float = 0.1  # Encoder LR = muon_lr * multiplier (e.g., 0.1 = 10x slower)
    freeze_encoder: bool = False  # Freeze encoder weights completely (requires_grad=False)
    freeze_encoder_epochs: int = 0  # Auto-unfreeze after N epochs (0 = never auto-unfreeze)

    # Data loading
    num_workers = 16
    window_len = 2.0  # seconds
    sfreq = 100

    # Masked pretraining tasks (MAE only - smaller dataset for faster caching)
    masked_tasks = [
        "RestingState",      # Spontaneous neural activity
        "surroundSupp"       # Perceptual SSVEP processing
    ]

    # Movie tasks (contrastive only - temporal alignment for ISC)
    movie_tasks = [
        "DespicableMe",
        "ThePresent",
        "DiaryOfAWimpyKid",
        "FunwithFractals"
    ]
    # Note: Movies NOT used for MAE, only for contrastive learning

    # 90/10 train/val split (excluding R5 test)
    # Split optimized from signaljepa_c1_full.py to balance recording counts
    train_releases = ["R11", "R2", "R3", "R4", "R7", "R8", "R9", "R10"]  # ~90%
    val_releases = ["R1", "R6"]  # ~10%
    test_release = "R5"  # Competition held-out set

    # Full dataset (not mini)
    use_mini = False

    # Tracking
    use_wandb = True
    experiment_name = f"signaljepa_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Checkpoint loading (for progressive training: MAE → +Contrastive → +Aux)
    pretrained_checkpoint_path: Optional[str] = "/home/varun/repos/cerebro/cache/signaljepa_pretrain/base_mae_1/best_pretrain.pt"  # Path to pretrained checkpoint
    # "/home/varun/repos/cerebro/cache/signaljepa_pretrain/base_mae_1/best_pretrain.pt"
    load_encoder_only: bool = True  # Only load encoder weights (not optimizer/scheduler)

    # Output
    checkpoint_dir = CACHE_PATH / "signaljepa_pretrain" / experiment_name

cfg = Config()

# Validation: Ensure at least one loss is enabled
if not cfg.use_mae and not cfg.use_contrastive and not cfg.use_auxiliary:
    raise ValueError(
        "At least one of use_mae, use_contrastive, or use_auxiliary must be enabled! "
        "All training objectives are currently disabled."
    )

# Validation: Ensure frozen encoder has trainable components
if cfg.freeze_encoder:
    if not cfg.use_mae and not cfg.use_contrastive and not cfg.use_auxiliary:
        raise ValueError(
            "Encoder is frozen but no training objectives are enabled! "
            "Enable at least one of: use_mae, use_contrastive, or use_auxiliary."
        )

cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Set seeds
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)

print(f"Configuration:")
print(f"  HBN_ROOT: {cfg.HBN_ROOT}")
print(f"  CACHE_PATH: {cfg.CACHE_PATH}")
print(f"  Masked (MAE) tasks: {cfg.masked_tasks}")
print(f"  Contrastive tasks: {cfg.movie_tasks}")
print(f"  Train releases (90%): {cfg.train_releases}")
print(f"  Val releases (10%): {cfg.val_releases}")
print(f"  Test release: {cfg.test_release}")
print(f"  Using mini: {cfg.use_mini}")
print(f"  MAE batch size: {cfg.batch_size}")
print(f"  Contrastive batch size: {cfg.contrastive_batch_size}")
print(f"  Model: SignalJEPA with {cfg.transformer_num_encoder_layers} encoder layers")

# %% Helper Functions

def set_encoder_frozen(model, frozen: bool):
    """Freeze or unfreeze encoder parameters.

    Args:
        model: SignalJEPA model with context_encoder and target_encoder
        frozen: If True, set requires_grad=False (freeze). If False, set requires_grad=True (unfreeze).
    """
    for param in model.context_encoder.parameters():
        param.requires_grad = not frozen
    for param in model.target_encoder.parameters():
        param.requires_grad = not frozen

# %% Temporal Masking Implementation
class TemporalBlockMasker:
    """Temporal block masking for EEG time series.

    Masks contiguous spans of time to encourage learning temporal dependencies.

    Args:
        n_times: Number of time steps in signal
        mask_ratio: Fraction of time steps to mask (e.g., 0.4 = 40%)
        span_length: Length of each masked span in samples
    """

    def __init__(self, n_times: int, mask_ratio: float = 0.4, span_length: int = 20):
        self.n_times = n_times
        self.mask_ratio = mask_ratio
        self.span_length = span_length

        # Calculate number of spans to mask
        n_masked_samples = int(n_times * mask_ratio)
        self.n_spans = max(1, n_masked_samples // span_length)

    def get_mask(self) -> torch.Tensor:
        """Generate temporal block mask.

        Returns:
            Binary mask of shape (n_times,) where 1 = masked, 0 = visible
        """
        mask = torch.zeros(self.n_times)

        for _ in range(self.n_spans):
            # Random start position
            start = np.random.randint(0, max(1, self.n_times - self.span_length))
            end = min(start + self.span_length, self.n_times)
            mask[start:end] = 1

        return mask

    def get_batch_masks(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate independent masks for each sample in batch.

        Args:
            batch_size: Number of masks to generate
            device: Device to place masks on

        Returns:
            Batch of masks (batch_size, n_times)
        """
        masks = torch.stack([self.get_mask() for _ in range(batch_size)])
        return masks.to(device)


# %% Spatial Masking (from sjepa_pretrain.py)
class SpatialBlockMasker:
    """Spatial block masking for EEG electrodes based on 3D coordinates.

    Masks spherical electrode regions around randomly chosen centers.
    """

    def __init__(self, chs_info: list[dict], mask_diameter_pct: int = 60):
        self.chs_info = chs_info
        self.mask_diameter_pct = mask_diameter_pct
        self.n_chans = len(chs_info)

        # Extract 3D coordinates
        self.coords = np.array([ch['loc'][3:6] for ch in chs_info])

        # Compute pairwise distances
        self.distances = self._compute_distances()

        # Set radius threshold
        max_distance = self.distances.max()
        self.radius_threshold = (mask_diameter_pct / 100.0) * max_distance

    def _compute_distances(self) -> np.ndarray:
        """Compute Euclidean distance matrix between all electrode pairs."""
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=2))
        return distances

    def get_mask(self, center_ch_idx: Optional[int] = None) -> torch.Tensor:
        """Generate spatial block mask.

        Returns:
            Binary mask of shape (n_chans,) where 1 = masked, 0 = visible
        """
        if center_ch_idx is None:
            center_ch_idx = np.random.randint(0, self.n_chans)

        mask = self.distances[center_ch_idx] <= self.radius_threshold
        return torch.from_numpy(mask).float()

    def get_batch_masks(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate independent masks for each sample in batch.

        Returns:
            Batch of masks (batch_size, n_chans)
        """
        masks = torch.stack([self.get_mask() for _ in range(batch_size)])
        return masks.to(device)


# %% Auxiliary Prediction Heads
def filter_enabled_aux_heads(aux_heads_config: Optional[dict]) -> Optional[dict]:
    """Filter auxiliary heads to only include enabled ones.

    Args:
        aux_heads_config: Dictionary of auxiliary head configurations

    Returns:
        Filtered dict with only enabled heads, or None if no heads enabled
    """
    if not aux_heads_config:
        return None

    enabled_heads = {k: v for k, v in aux_heads_config.items() if v.get('enabled', True)}

    return enabled_heads if enabled_heads else None


# %% Separable Auxiliary Head (SignalJEPA-style)
class SeparableAuxiliaryHead(nn.Module):
    """SignalJEPA-style separable head with Conv3d spatial filtering.

    Architecture:
        1. Rearrange: (batch, n_chans, d_model) → (batch, 1, n_chans, 1, d_model)
        2. Conv3d: Learn spatial aggregation across channels
        3. Flatten: Prepare for final linear layer
        4. Linear: Output prediction

    This matches the head architecture used in SignalJEPA_Contextual and
    SignalJEPA_PostLocal from braindecode, allowing task-specific spatial
    learning while maintaining a shared encoder.
    """

    def __init__(
        self,
        n_chans: int,
        d_model: int,
        n_outputs: int,
        n_spat_filters: int = 4,
    ):
        """Initialize separable auxiliary head.

        Args:
            n_chans: Number of EEG channels (e.g., 129 for HBN)
            d_model: Feature dimension per channel (from encoder)
            n_outputs: Output dimension (1 for regression, n_classes for classification)
            n_spat_filters: Number of spatial filters (default: 4, range: 4-16)
        """
        super().__init__()
        from einops.layers.torch import Rearrange

        self.n_chans = n_chans
        self.d_model = d_model
        self.n_outputs = n_outputs
        self.n_spat_filters = n_spat_filters

        # Output embedding dimension after spatial filtering
        out_emb_dim = n_spat_filters * d_model

        self.head = nn.Sequential(
            # Reshape: (batch, n_chans, d_model) → (batch, 1, n_chans, 1, d_model)
            Rearrange("b (n_chans tokens) d -> b 1 n_chans tokens d",
                     n_chans=n_chans, tokens=1),

            # Spatial filtering: Learn channel aggregation patterns
            # (batch, 1, n_chans, 1, d_model) → (batch, n_spat_filters, 1, 1, d_model)
            nn.Conv3d(1, n_spat_filters, kernel_size=(n_chans, 1, 1)),

            # Flatten: (batch, n_spat_filters, 1, 1, d_model) → (batch, n_spat_filters * d_model)
            nn.Flatten(start_dim=1),

            # Output layer
            nn.Linear(out_emb_dim, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through separable head.

        Args:
            x: Encoder output (batch, n_chans, d_model)

        Returns:
            Predictions (batch, n_outputs)
        """
        return self.head(x)


class AuxiliaryHeads(nn.Module):
    """Auxiliary prediction heads for demographic/phenotypic variables.

    Trained jointly with MAE to organize latent space by demographics.

    Supports two architectures:
    - Simple MLP: 2-layer feedforward (for flattened embeddings)
    - Separable: SignalJEPA-style Conv3d spatial filtering (for channel-wise features)
    """

    def __init__(
        self,
        input_dim: int,
        heads_config: dict,
        use_separable: bool = False,
        n_chans: Optional[int] = None,
        n_spat_filters: int = 4,
    ):
        """Initialize auxiliary heads.

        Args:
            input_dim: Input feature dimension (d_model for separable, flattened dim for MLP)
            heads_config: Dictionary of head configurations {name: {type, n_outputs, ...}}
            use_separable: Use SignalJEPA-style Conv3d heads (vs simple MLPs)
            n_chans: Number of EEG channels (required if use_separable=True)
            n_spat_filters: Spatial filters per head (default: 4, only used if separable)
        """
        super().__init__()
        self.heads_config = heads_config
        self.use_separable = use_separable
        self.heads = nn.ModuleDict()

        if use_separable:
            if n_chans is None:
                raise ValueError("n_chans must be provided when use_separable=True")

            # Create separable heads with Conv3d spatial filtering
            for name, config in heads_config.items():
                self.heads[name] = SeparableAuxiliaryHead(
                    n_chans=n_chans,
                    d_model=input_dim,
                    n_outputs=config['n_outputs'],
                    n_spat_filters=n_spat_filters,
                )
        else:
            # Create simple 2-layer MLP heads
            for name, config in heads_config.items():
                self.heads[name] = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, config['n_outputs'])
                )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through all heads.

        Args:
            x: Encoder output
               - If use_separable=True: (batch, n_chans, d_model) - channel-wise features
               - If use_separable=False: (batch, d_model) - flattened embedding

        Returns:
            Dictionary of predictions for each head {name: predictions}
        """
        return {name: head(x) for name, head in self.heads.items()}


# %% Combined Pretraining Model
class SignalJEPAPretrainer(nn.Module):
    """Combined SignalJEPA pretraining with MAE + Contrastive + Auxiliary.

    Components:
    1. Context encoder (trainable) - processes visible tokens
    2. Target encoder (EMA) - generates stable targets for masked tokens
    3. Predictor - reconstructs masked tokens from context
    4. Auxiliary heads - predict demographics from encoder output
    5. Projection head - for contrastive learning
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: int,
        input_window_seconds: float,
        chs_info: list[dict],
        d_model: int = 96,
        num_encoder_layers: int = 12,
        nhead: int = 8,
        dropout: float = 0.1,
        ema_momentum: float = 0.996,
        predictor_depth: int = 4,
        aux_heads_config: Optional[dict] = None,
        use_separable_aux_heads: bool = False,
        aux_n_spat_filters: int = 4,
    ):
        super().__init__()

        self.n_chans = n_chans
        self.d_model = d_model
        self.ema_momentum = ema_momentum
        self.use_separable_aux_heads = use_separable_aux_heads

        # Context encoder (trainable)
        self.context_encoder = SignalJEPA(
            n_outputs=None,  # No final layer for pretraining
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            input_window_seconds=input_window_seconds,
            chs_info=chs_info,
            transformer__d_model=d_model,
            transformer__num_encoder_layers=num_encoder_layers,
            transformer__nhead=nhead,
            drop_prob=dropout,
        )

        # Target encoder (EMA, non-trainable)
        self.target_encoder = SignalJEPA(
            n_outputs=None,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            input_window_seconds=input_window_seconds,
            chs_info=chs_info,
            transformer__d_model=d_model,
            transformer__num_encoder_layers=num_encoder_layers,
            transformer__nhead=nhead,
            drop_prob=dropout,
        )

        # Initialize target encoder with same weights
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        # Freeze target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor (Transformer decoder)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.predictor = nn.TransformerDecoder(decoder_layer, num_layers=predictor_depth)

        # Learnable mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # Auxiliary heads (optional)
        if aux_heads_config is not None:
            self.aux_heads = AuxiliaryHeads(
                input_dim=d_model,
                heads_config=aux_heads_config,
                use_separable=use_separable_aux_heads,
                n_chans=n_chans if use_separable_aux_heads else None,
                n_spat_filters=aux_n_spat_filters,
            )
        else:
            self.aux_heads = None

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128)  # Project to 128-dim space
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None):
        """EMA update of target encoder parameters.

        Args:
            momentum: Optional EMA momentum. If None, uses self.ema_momentum.
                     Paper uses cosine schedule: 0.996 → 1.0 over training.
        """
        if momentum is None:
            momentum = self.ema_momentum

        for param_c, param_t in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_t.data = (
                momentum * param_t.data +
                (1 - momentum) * param_c.data
            )

    def forward_mae(
        self,
        x: torch.Tensor,
        spatial_mask: torch.Tensor,
        temporal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for MAE reconstruction.

        Args:
            x: Input EEG (batch, n_chans, n_times)
            spatial_mask: Spatial mask (batch, n_chans) - 1 = masked, 0 = visible
            temporal_mask: Temporal mask (batch, n_times) - 1 = masked, 0 = visible

        Returns:
            predictions: Predicted tokens (batch, seq_len, d_model)
            targets: Target encoder outputs (batch, seq_len, d_model)
            combined_mask: Mask for loss computation (batch, seq_len) - 1 = masked
        """
        batch_size = x.shape[0]

        # Apply masking to input: zero out masked positions
        # spatial_mask: (batch, n_chans) → (batch, n_chans, 1)
        # temporal_mask: (batch, n_times) → (batch, 1, n_times)
        # Broadcasting creates (batch, n_chans, n_times) mask
        spatial_mask_3d = spatial_mask.unsqueeze(2)  # (batch, n_chans, 1)
        temporal_mask_3d = temporal_mask.unsqueeze(1)  # (batch, 1, n_times)

        # Combined mask: 1 if EITHER spatial OR temporal is masked (union)
        input_mask = torch.maximum(spatial_mask_3d, temporal_mask_3d)  # (batch, n_chans, n_times)

        # Apply mask: zero out masked positions for context encoder
        x_masked = x * (1.0 - input_mask)  # Masked positions become 0

        # Get context encoder output on visible tokens
        context_output = self.context_encoder(x_masked)  # (batch, seq_len, d_model)

        # Get target encoder output (no masking, all tokens)
        with torch.no_grad():
            target_output = self.target_encoder(x)  # (batch, seq_len, d_model)

        # Create sequence-level mask for loss computation
        # SignalJEPA produces one token per channel, so map spatial mask to sequence
        seq_len = context_output.shape[1]

        # Spatial mask directly maps to sequence positions (one token per channel)
        # Note: If seq_len != n_chans, this needs adjustment based on encoder architecture
        if seq_len == spatial_mask.shape[1]:
            # Direct mapping: each sequence position corresponds to a channel
            combined_mask = spatial_mask  # (batch, seq_len)
        else:
            # Fallback: use spatial mask as proxy (may need refinement)
            # Interpolate or repeat to match seq_len
            combined_mask = F.interpolate(
                spatial_mask.unsqueeze(1).float(),  # (batch, 1, n_chans)
                size=seq_len,
                mode='nearest'
            ).squeeze(1)  # (batch, seq_len)

        # Create mask tokens for ALL sequence positions
        mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)  # (batch, seq_len, d_model)

        # Predict tokens using transformer decoder (cross-attention with context)
        predictions = self.predictor(mask_tokens, context_output)  # (batch, seq_len, d_model)

        return predictions, target_output, combined_mask

    def forward_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for contrastive learning.

        Args:
            x: Input EEG (batch, n_chans, n_times)

        Returns:
            Projected embeddings (batch, 128)
        """
        # Get encoder output
        encoder_output = self.context_encoder(x)  # (batch, seq_len, d_model)

        # Global average pooling
        pooled = encoder_output.mean(dim=1)  # (batch, d_model)

        # Project to contrastive space
        projected = self.projection_head(pooled)  # (batch, 128)

        # L2 normalize
        projected = F.normalize(projected, p=2, dim=1)

        return projected

    def forward_auxiliary(self, x: torch.Tensor) -> Optional[dict]:
        """Forward pass for auxiliary demographic prediction.

        Args:
            x: Input EEG (batch, n_chans, n_times)

        Returns:
            Dictionary of predictions for each auxiliary head
        """
        if self.aux_heads is None:
            return None

        # Get encoder output
        encoder_output = self.context_encoder(x)  # (batch, n_chans, d_model)

        if self.use_separable_aux_heads:
            # Use channel-wise features (no pooling) for separable heads
            # Shape: (batch, n_chans, d_model)
            features = encoder_output
        else:
            # Global average pooling for simple MLP heads
            # Shape: (batch, d_model)
            features = encoder_output.mean(dim=1)

        # Auxiliary predictions
        return self.aux_heads(features)


# %% GradNorm - Dynamic Loss Weight Balancing
class GradNorm:
    """GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.

    Dynamically adjusts loss weights to balance gradient magnitudes across tasks.
    Paper: https://arxiv.org/abs/1711.02257

    Args:
        n_tasks: Number of tasks
        alpha: Restoring force strength (higher = more aggressive balancing, default=1.5)
        warmup_epochs: Number of epochs before enabling GradNorm (default=5)
        update_freq: Update weights every N batches (default=10)
    """

    def __init__(self, n_tasks: int = 3, alpha: float = 1.5, warmup_epochs: int = 0, update_freq: int = 10):
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs
        self.update_freq = update_freq

        # Initial task weights (learnable)
        self.task_weights = torch.nn.Parameter(torch.ones(n_tasks))

        # Track initial task losses for relative weighting
        self.initial_losses = None

        # Batch counter
        self.batch_count = 0

    def compute_grad_norm(self, loss: torch.Tensor, parameters) -> float:
        """Compute L2 norm of gradients for a single task loss.

        Args:
            loss: Scalar loss for one task
            parameters: Model parameters to compute gradients for

        Returns:
            L2 norm of gradients
        """
        # Compute gradients for this task
        grads = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )

        # Compute L2 norm (filter out None gradients)
        grad_norm = 0.0
        for grad in grads:
            if grad is not None:
                grad_norm += (grad ** 2).sum().item()

        return np.sqrt(grad_norm)

    def update_weights(
        self,
        task_losses: list[torch.Tensor],
        shared_parameters,
        current_epoch: int
    ) -> torch.Tensor:
        """Update task weights based on gradient norm balancing.

        Args:
            task_losses: List of scalar losses [mae_loss, contrastive_loss, aux_loss]
            shared_parameters: Shared model parameters (e.g., encoder)
            current_epoch: Current training epoch

        Returns:
            Normalized task weights
        """
        self.batch_count += 1

        # Skip update if in warmup or not at update frequency
        if current_epoch < self.warmup_epochs or self.batch_count % self.update_freq != 0:
            # Return current weights (softmax normalized)
            return torch.softmax(self.task_weights, dim=0)

        # Initialize initial losses on first update
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([l.item() for l in task_losses])

        # Compute gradient norms for each task
        grad_norms = []
        for task_loss in task_losses:
            grad_norm = self.compute_grad_norm(task_loss, shared_parameters)
            grad_norms.append(grad_norm)

        grad_norms = np.array(grad_norms)

        # Compute average gradient norm
        avg_grad_norm = grad_norms.mean()

        # Compute relative inverse training rates
        current_losses = torch.tensor([l.item() for l in task_losses])
        loss_ratios = current_losses / self.initial_losses

        # Compute target gradient norms (GradNorm formula)
        # G_W(t) = avg_grad * [r_i(t)]^alpha where r_i is relative inverse training rate
        target_grad_norms = avg_grad_norm * (loss_ratios.numpy() ** self.alpha)

        # Update task weights to push gradient norms toward targets
        # Simple gradient step on L1 distance
        grad_diff = grad_norms - target_grad_norms

        # Update weights (gradient descent)
        with torch.no_grad():
            self.task_weights -= 0.01 * torch.from_numpy(grad_diff).float()

            # Clamp weights to prevent collapse
            self.task_weights.clamp_(min=0.1, max=10.0)

        # Return softmax-normalized weights
        return torch.softmax(self.task_weights, dim=0)

    def get_weights(self) -> torch.Tensor:
        """Get current normalized task weights."""
        return torch.softmax(self.task_weights, dim=0)


# %% InfoNCE Loss
def info_nce_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.07,
    z_neg: Optional[torch.Tensor] = None,
    use_hard_negative: bool = False
) -> torch.Tensor:
    """InfoNCE loss for contrastive learning with optional explicit hard negatives.

    Two modes:
    1. Pair-based (default): Uses batch negatives only
    2. Triplet-based (use_hard_negative=True): Adds explicit hard negatives from ISC

    Args:
        z_i: Embeddings for anchors (batch, dim)
        z_j: Embeddings for positives (batch, dim)
        temperature: Temperature scaling parameter
        z_neg: Embeddings for hard negatives (batch, dim), optional
        use_hard_negative: If True and z_neg provided, add explicit hard negatives

    Returns:
        Scalar loss
    """
    batch_size = z_i.shape[0]

    if use_hard_negative and z_neg is not None:
        # Triplet mode: Include explicit hard negatives
        # Concatenate to create 3N samples
        z = torch.cat([z_i, z_j, z_neg], dim=0)  # (3*batch, dim)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / temperature  # (3*batch, 3*batch)

        # CRITICAL: Clamp logits to prevent exp() overflow (NaN prevention)
        # Standard practice in SimCLR/MoCo: [-20, 20] allows exp() without overflow
        sim_matrix = torch.clamp(sim_matrix, min=-20.0, max=20.0)

        # Remove self-similarities
        mask = torch.eye(3 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, float('-inf'))

        # Positive similarities: (anchor, positive) pairs
        pos_sim = torch.cat([
            sim_matrix[:batch_size, batch_size:2*batch_size].diag(),  # (i, i+N)
            sim_matrix[batch_size:2*batch_size, :batch_size].diag()   # (i+N, i)
        ])  # (2*batch,)

        # Negative similarities: All other pairs (includes hard negatives)
        neg_sim = sim_matrix[:2*batch_size]  # Only need rows for anchors and positives

    else:
        # Pair mode: Original implementation with batch negatives only
        # Concatenate to create 2N samples
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch, dim)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / temperature  # (2*batch, 2*batch)

        # CRITICAL: Clamp logits to prevent exp() overflow (NaN prevention)
        # Standard practice in SimCLR/MoCo: [-20, 20] allows exp() without overflow
        sim_matrix = torch.clamp(sim_matrix, min=-20.0, max=20.0)

        # Create positive pair mask
        # Positive pairs: (i, i+N) and (i+N, i)
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, float('-inf'))  # Remove self-similarities

        # Positive similarities
        pos_sim = torch.cat([
            sim_matrix[:batch_size, batch_size:].diag(),  # (i, i+N)
            sim_matrix[batch_size:, :batch_size].diag()   # (i+N, i)
        ])  # (2*batch,)

        # Negative similarities (all other pairs)
        neg_sim = sim_matrix

    # LogSumExp for numerical stability
    loss = -pos_sim + torch.logsumexp(neg_sim, dim=1)

    return loss.mean()


# %% Pickle Cache Functions for Windowed Datasets
import pickle

def get_dataset_cache_key(tasks, releases, window_len, stride, mini):
    """Generate unique cache key for dataset configuration."""
    tasks_str = "_".join(sorted(tasks))
    releases_str = "_".join(sorted(releases))
    mini_str = "mini" if mini else "full"
    return f"{tasks_str}_{releases_str}_win{window_len}_stride{stride}_{mini_str}"

def save_windowed_dataset(dataset, cache_dir, split_name, cache_key):
    """Save windowed dataset to pickle file."""
    cache_path = cache_dir / f"{split_name}_{cache_key}.pkl"
    print(f"  💾 Saving {split_name} dataset to cache...")
    with open(cache_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"     → Saved to {cache_path}")
    print(f"     → Size: {cache_path.stat().st_size / 1024 / 1024:.1f} MB")
    return cache_path

def load_windowed_dataset(cache_dir, split_name, cache_key):
    """Load windowed dataset from pickle file if it exists."""
    cache_path = cache_dir / f"{split_name}_{cache_key}.pkl"
    if cache_path.exists():
        print(f"  📦 Loading {split_name} dataset from cache...")
        with open(cache_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"     → Loaded {len(dataset)} windows from {cache_path}")
        return dataset
    return None


# %% Custom Collate Function for Metadata
def collate_with_metadata(batch):
    """Custom collate function that preserves metadata as list of dicts.

    PyTorch's default collate tries to batch dict values, but we need
    to preserve the list structure for per-sample metadata.

    Handles both numpy arrays and tensors from Braindecode datasets.
    """
    import numpy as np

    # Separate components
    X_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]
    window_inds_list = [item[2] for item in batch]
    infos_list = [item[3] for item in batch]  # Keep as list of dicts

    # Defensive check: ensure all arrays have the same shape
    if len(X_list) > 0:
        shapes = [x.shape for x in X_list]
        if len(set(shapes)) > 1:
            shape_counts = {}
            for s in shapes:
                shape_counts[s] = shape_counts.get(s, 0) + 1
            raise ValueError(
                f"Shape mismatch in batch! Cannot stack arrays with different shapes.\n"
                f"  Found {len(set(shapes))} different shapes in batch of {len(X_list)} items:\n"
                f"  Shape distribution: {shape_counts}\n"
                f"  This indicates that validation failed to filter inconsistent windows.\n"
                f"  Check validate_windowed_dataset_shapes() is being called correctly."
            )

    # Convert to tensors and stack
    # Braindecode datasets return numpy arrays, so convert first
    if isinstance(X_list[0], np.ndarray):
        X_batch = torch.from_numpy(np.stack(X_list))
    else:
        X_batch = torch.stack(X_list)

    if isinstance(y_list[0], np.ndarray):
        y_batch = torch.from_numpy(np.stack(y_list))
    elif isinstance(y_list[0], torch.Tensor):
        y_batch = torch.stack(y_list)
    else:
        y_batch = torch.tensor(y_list)

    if isinstance(window_inds_list[0], np.ndarray):
        window_inds_batch = torch.from_numpy(np.stack(window_inds_list))
    elif isinstance(window_inds_list[0], torch.Tensor):
        window_inds_batch = torch.stack(window_inds_list)
    else:
        window_inds_batch = torch.tensor(window_inds_list)

    return X_batch, y_batch, window_inds_batch, infos_list


def collate_triplets(batch):
    """Custom collate for (anchor, positive, negative) triplets.

    Handles tensors from ContrastivePairDataset that may have non-resizable storage.
    Creates new tensors with proper storage for batching.
    """
    import numpy as np

    anchors = []
    positives = []
    negatives = []

    for anchor, pos, neg in batch:
        # Ensure contiguous tensors with resizable storage
        if isinstance(anchor, np.ndarray):
            anchors.append(torch.from_numpy(anchor.copy()))
            positives.append(torch.from_numpy(pos.copy()))
            negatives.append(torch.from_numpy(neg.copy()))
        else:
            # Handle tensors - clone if not contiguous to ensure resizable storage
            anchors.append(anchor.clone() if not anchor.is_contiguous() else anchor.contiguous())
            positives.append(pos.clone() if not pos.is_contiguous() else pos.contiguous())
            negatives.append(neg.clone() if not neg.is_contiguous() else neg.contiguous())

    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives)
    )


def filter_short_trials(dataset, min_samples):
    """Filter out trials/recordings shorter than min_samples from a BaseConcatDataset.

    Args:
        dataset: BaseConcatDataset containing individual trial/recording datasets
        min_samples: Minimum number of samples required (e.g., 200 for 2s windows at 100Hz)

    Returns:
        Filtered BaseConcatDataset with only trials >= min_samples
    """
    from braindecode.datasets import BaseConcatDataset

    valid_trials = []
    filtered_count = 0

    for trial_ds in dataset.datasets:
        try:
            # Check if trial has .raw attribute (MNE Raw object with n_times)
            if hasattr(trial_ds, 'raw') and hasattr(trial_ds.raw, 'n_times'):
                trial_length = trial_ds.raw.n_times
            # Otherwise, check total duration across all epochs
            elif len(trial_ds) > 0:
                # Sum up all epoch durations (each epoch is (data, metadata))
                # Calculate total samples across all epochs
                total_samples = sum(trial_ds[i][0].shape[1] for i in range(len(trial_ds)))
                trial_length = total_samples
            else:
                # Empty trial, skip
                filtered_count += 1
                continue

            if trial_length >= min_samples:
                valid_trials.append(trial_ds)
            else:
                filtered_count += 1

        except Exception:
            # Skip trials that can't be accessed
            filtered_count += 1
            pass

    if len(valid_trials) == 0:
        # No valid trials found - return empty dataset
        return BaseConcatDataset([])

    return BaseConcatDataset(valid_trials)


def validate_windowed_dataset_shapes(windows_dataset, expected_n_channels=129, expected_n_times=200):
    """Validate that all windows have consistent shapes (channels and timepoints).

    Filters out recordings with incorrect channel counts or time lengths.

    Args:
        windows_dataset: BaseConcatDataset of windowed data
        expected_n_channels: Expected number of channels (default 129 for HBN)
        expected_n_times: Expected number of timepoints (default 200 for 2s windows at 100Hz)

    Returns:
        Filtered BaseConcatDataset with only valid windows
    """
    from braindecode.datasets import BaseConcatDataset

    valid_datasets = []
    filtered_count = 0
    channels_mismatch = 0
    timepoints_mismatch = 0
    shape_details = {}  # Track what shapes we're seeing

    for ds in windows_dataset.datasets:
        if len(ds) == 0:
            filtered_count += 1
            continue

        try:
            # Check first window's shape
            first_window = ds[0][0]  # (data, metadata)
            n_channels = first_window.shape[0]
            timepoints = first_window.shape[1]

            # Track shape occurrences for debugging
            shape_key = f"({n_channels}, {timepoints})"
            shape_details[shape_key] = shape_details.get(shape_key, 0) + 1

            # Strict validation: both dimensions must match exactly
            if n_channels != expected_n_channels:
                channels_mismatch += 1
                filtered_count += 1
            elif timepoints != expected_n_times:
                timepoints_mismatch += 1
                filtered_count += 1
            else:
                valid_datasets.append(ds)
        except Exception:
            filtered_count += 1

    if filtered_count > 0:
        print(f"    ⚠️  Filtered {filtered_count} recording(s) with incorrect shape")
        if channels_mismatch > 0:
            print(f"       ├─ {channels_mismatch} with wrong channel count (expected {expected_n_channels})")
        if timepoints_mismatch > 0:
            print(f"       └─ {timepoints_mismatch} with wrong timepoint count (expected {expected_n_times})")

        # Show what shapes we encountered (for debugging)
        wrong_shapes = {k: v for k, v in shape_details.items() if k != f"({expected_n_channels}, {expected_n_times})"}
        if wrong_shapes and len(wrong_shapes) <= 5:  # Only show if not too many
            print(f"       Incorrect shapes found: {wrong_shapes}")

    if len(valid_datasets) == 0:
        return BaseConcatDataset([])

    return BaseConcatDataset(valid_datasets)


def _count_task_release_recordings(task, release, hbn_root, use_mini):
    """Count recordings for a single (task, release) pair (for parallel execution).

    Args:
        task: Task name (e.g., "RestingState")
        release: Release name (e.g., "R11")
        hbn_root: Path to HBN data root
        use_mini: Whether to use mini dataset

    Returns:
        Tuple of (task, release, count)
    """
    try:
        ds = EEGChallengeDataset(
            task=task,
            release=release,
            cache_dir=Path(str(hbn_root)),
            mini=use_mini,
            description_fields=["subject", "age", "sex", "p_factor", "attention", "internalizing", "ehq_total"]
        )
        count = len(ds.datasets)
        return (task, release, count)
    except Exception:
        return (task, release, 0)


def _process_task_windows(task, task_idx, total_tasks, task_release_counts, train_releases, hbn_root, use_mini, window_len, sfreq):
    """Process all releases for a single task (for parallel execution).

    Args:
        task: Task name (e.g., "RestingState")
        task_idx: Task index (for progress reporting)
        total_tasks: Total number of tasks
        task_release_counts: Dict mapping task -> release -> count
        train_releases: List of release names to process
        hbn_root: Path to HBN data root
        use_mini: Whether to use mini dataset
        window_len: Window length in seconds
        sfreq: Sampling frequency in Hz

    Returns:
        List of windowed datasets for this task
    """
    print(f"\n{'='*60}")
    print(f"Processing Task {task_idx+1}/{total_tasks}: {task}")
    print(f"{'='*60}")

    task_windows = []

    for release_idx, release in enumerate(train_releases):
        n_recordings = task_release_counts[task].get(release, 0)

        if n_recordings == 0:
            print(f"  [{release_idx+1}/{len(train_releases)}] {release}: Skipped (no recordings)")
            continue

        print(f"  [{release_idx+1}/{len(train_releases)}] {release}: Windowing {n_recordings} recordings...", end="", flush=True)

        try:
            # Load this specific (task, release) pair
            ds = EEGChallengeDataset(
                task=task,
                release=release,
                cache_dir=Path(str(hbn_root)),
                mini=use_mini,
                description_fields=["subject", "age", "sex", "p_factor", "attention", "internalizing", "ehq_total"]
            )

            # Filter out trials shorter than window size
            min_samples = int(window_len * sfreq)  # 200 samples for 2s
            original_count = len(ds.datasets)
            ds = filter_short_trials(ds, min_samples)
            filtered_count = original_count - len(ds.datasets)

            if len(ds.datasets) == 0:
                print(f" ⚠️  All {original_count} trials too short, skipping")
                continue

            # Window the filtered dataset
            release_windows = create_fixed_length_windows(
                ds,
                window_size_samples=min_samples,
                window_stride_samples=min_samples,  # No overlap
                drop_last_window=True,
                preload=True
            )

            # Validate channel counts and shape consistency (129 channels x 200 timepoints)
            release_windows = validate_windowed_dataset_shapes(release_windows, expected_n_channels=129, expected_n_times=200)

            if len(release_windows.datasets) == 0:
                print(f" ⚠️  All recordings filtered due to shape issues, skipping")
                continue

            task_windows.append(release_windows)

            # Report results
            if filtered_count > 0:
                print(f" ✓ {len(release_windows)} windows ({filtered_count}/{original_count} trials filtered)")
            else:
                print(f" ✓ {len(release_windows)} windows")

        except Exception as e:
            print(f" ✗ Error: {e}")

    return task_windows


# %% DatasetWrapper for Subject Metadata
class DatasetWrapper(BaseDataset):
    """Wraps windowed dataset to add subject metadata for auxiliary loss."""

    def __init__(self, windows_dataset):
        self.dataset = windows_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get original data
        X, y, window_inds = self.dataset[index]

        # Extract subject ID from description
        subject = self.dataset.description.get('subject', 'unknown')
        infos = {'subject': subject}

        return X, y, window_inds, infos


# %% Load Data - MAE Dataset (masked pretraining only)
print("\n" + "="*60)
print("Loading MAE pretraining data (2 tasks: RestingState + surroundSupp)")
print("="*60)

# Load masked pretraining tasks only (count recordings first)
print("\n📊 Counting recordings per task...")

# Count all (task, release) pairs in parallel (2-4x speedup)
count_results = Parallel(n_jobs=16, backend='loky', verbose=0)(
    delayed(_count_task_release_recordings)(
        task=task,
        release=release,
        hbn_root=cfg.HBN_ROOT,
        use_mini=cfg.use_mini
    )
    for task in cfg.masked_tasks
    for release in cfg.train_releases
)

# Build task_release_counts dict from results
task_release_counts = {}
total_recordings = 0
for task, release, count in count_results:
    if task not in task_release_counts:
        task_release_counts[task] = {}
    task_release_counts[task][release] = count
    total_recordings += count

print(f"\n✅ Found {total_recordings} total recordings across {len(cfg.masked_tasks)} tasks")

# Setup windowing cache
print("\n" + "="*60)
print("Checking for cached windowed datasets...")
print("="*60)
PRETRAIN_CACHE_DIR = cfg.CACHE_PATH / "signaljepa_pretrain_windows"
PRETRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

mae_cache_key = get_dataset_cache_key(
    cfg.masked_tasks,  # Only RestingState + surroundSupp
    cfg.train_releases,
    cfg.window_len,
    cfg.window_len,  # stride = window_len (no overlap)
    cfg.use_mini
)
print(f"  MAE cache key: {mae_cache_key}")

# Try loading from cache
mae_windows = load_windowed_dataset(PRETRAIN_CACHE_DIR, "mae", mae_cache_key)

if mae_windows is None:
    # Cache miss - create windows with per-release progress
    print("\n⚠️  Cache miss - creating MAE windows from scratch...")
    print("  ⏱️  Estimated time:")
    print(f"     - {len(cfg.masked_tasks)} tasks × {len(cfg.train_releases)} releases")
    print(f"     - ~{total_recordings} recordings")
    print(f"     - Expected: 10-20 minutes total (mini: ~2-5 min)")
    print("  💡 Future runs will use cache and load in ~10 seconds")
    print("")

    import time
    start = time.time()

    # Process each (task, release) pair separately with progress
    all_windowed_datasets = []

    # Process tasks in parallel (2x speedup: ~8min → ~4min)
    task_results = Parallel(n_jobs=2, backend='loky', verbose=0)(
        delayed(_process_task_windows)(
            task=task,
            task_idx=task_idx,
            total_tasks=len(cfg.masked_tasks),
            task_release_counts=task_release_counts,
            train_releases=cfg.train_releases,
            hbn_root=cfg.HBN_ROOT,
            use_mini=cfg.use_mini,
            window_len=cfg.window_len,
            sfreq=cfg.sfreq
        )
        for task_idx, task in enumerate(cfg.masked_tasks)
    )

    # Flatten results: [task1_windows, task2_windows] → all_windowed_datasets
    all_windowed_datasets = [w for task_ws in task_results for w in task_ws]

    # Concatenate all windowed datasets
    print(f"\n{'='*60}")
    print("Concatenating all windowed datasets...")
    mae_windows = BaseConcatDataset([w for ws in all_windowed_datasets for w in ws.datasets])

    elapsed = time.time() - start
    print(f"✅ Windowing completed in {elapsed/60:.1f} minutes")
    print(f"   Total windows: {len(mae_windows):,}")

    # Save to cache for future runs
    print(f"\n💾 Saving to cache...")
    save_windowed_dataset(mae_windows, PRETRAIN_CACHE_DIR, "mae", mae_cache_key)
    print(f"✅ MAE dataset cached - next run will be much faster!")
else:
    print(f"✅ Using cached MAE dataset")
    print(f"   Total windows: {len(mae_windows):,}")

# Wrap with subject metadata for auxiliary loss
print("\n🔄 Wrapping MAE dataset with subject metadata...")
mae_windows_wrapped = BaseConcatDataset([DatasetWrapper(w) for w in mae_windows.datasets])
print(f"   ✅ Wrapped {len(mae_windows_wrapped)} windows")

# Create DataLoader with custom collate function
mae_loader = DataLoader(
    mae_windows_wrapped,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=1,
    collate_fn=collate_with_metadata  # Preserve metadata as list of dicts
)

# %% Load Data - Contrastive Dataset (ISC-based movie pairing)
if cfg.use_contrastive:
    print("\n" + "="*60)
    print("Loading ISC contrastive data (4 movie tasks)")
    print("="*60)

    print(f"\n📽️  Movie tasks: {cfg.movie_tasks}")
    print(f"  Train releases: {cfg.train_releases}")
    print(f"  Window length: {cfg.window_len}s, Stride: 1.0s (50% overlap)")
    print(f"  Time bin size: {cfg.time_bin_size_s}s (for ISC pairing)")
    print(f"  ISC strategy: pos={cfg.pos_strategy}, neg={cfg.neg_strategy}")

    # Load and window movies with ISC metadata
    print("\n⏳ Loading movies with ISC metadata...")
    movie_windows = load_and_window_movies(
        movie_names=cfg.movie_tasks,
        dataset_class=EEGChallengeDataset,
        cache_dir=cfg.HBN_ROOT,
        releases=cfg.train_releases,
        mini=cfg.use_mini,
        window_len_s=cfg.window_len,
        stride_s=1.0,  # 1s stride for more contrastive pairs
        sfreq=cfg.sfreq,
        time_bin_size_s=cfg.time_bin_size_s,
        preload=True
    )

    print(f"✅ Loaded {len(movie_windows):,} movie windows with ISC metadata")

    # Wrap with ContrastivePairDataset for ISC-based triplet sampling
    print("\n🔗 Creating ISC triplet dataset...")
    contrastive_dataset = ContrastivePairDataset(
        movie_windows,
        pos_strategy=cfg.pos_strategy,  # Same (movie, time_bin), different subjects
        neg_strategy=cfg.neg_strategy,  # Different movies
        return_triplets=True,  # Return (anchor, positive, negative)
        random_state=2025
    )

    # Get stats
    stats = contrastive_dataset.get_stats()
    print(f"  ✅ ISC Dataset Statistics:")
    print(f"     Valid anchors: {stats['valid_anchors']:,} (windows with ISC pairs)")
    print(f"     Positive groups: {stats['pos_groups']:,} ((movie, time_bin) combinations)")
    print(f"     Subjects: {stats['subjects']} unique subjects")
    print(f"     Movies: {stats['movies']}")

    # Create DataLoader for ISC triplets
    contrastive_loader = DataLoader(
        contrastive_dataset,
        batch_size=cfg.contrastive_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
        collate_fn=collate_triplets  # Handle non-resizable tensor storage
    )

    print(f"  📦 Contrastive batches: {len(contrastive_loader)} (batch_size={cfg.contrastive_batch_size})")
else:
    print("\n⚠️  Contrastive learning DISABLED (use_contrastive=False)")
    print("   Skipping movie data loading - using MAE-only training")
    movie_windows = None
    contrastive_loader = None

# %% Load Demographics
print("\n" + "="*60)
print("Loading demographic data from datasets")
print("="*60)

# Extract demographics from windowed datasets (already in dataset.description)
demographics = {}

# Collect from both MAE and movie datasets (if movie datasets exist)
all_datasets = list(mae_windows.datasets)
if movie_windows is not None:
    all_datasets.extend(list(movie_windows.datasets))

for ds in all_datasets:
    subject = ds.description.get('subject')
    if subject and subject not in demographics:
        # Extract available demographics (matching aux_heads config)
        demo_dict = {
            'age': ds.description.get('age'),
            'sex': ds.description.get('sex'),
            'p_factor': ds.description.get('p_factor'),
            'attention': ds.description.get('attention'),
            'internalizing': ds.description.get('internalizing'),
            'ehq_total': ds.description.get('ehq_total'),
        }

        # Filter out NaN values and None
        import math
        demo_dict = {k: v for k, v in demo_dict.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}

        if demo_dict:  # Only add if we have at least some demographics
            demographics[subject] = demo_dict

print(f"✅ Loaded demographics for {len(demographics)} subjects")
if demographics:
    sample_subject = next(iter(demographics.keys()))
    print(f"  Available fields: {list(demographics[sample_subject].keys())}")
    print(f"  Example (subject {sample_subject}): {demographics[sample_subject]}")
else:
    print("⚠️  No demographics found - auxiliary heads will be disabled")
    demographics = None

# %% Load Validation Data (R1, R6)
print("\n" + "="*60)
print(f"Loading validation data from {cfg.val_releases}")
print("="*60)

# Generate validation cache keys
mae_val_cache_key = get_dataset_cache_key(
    cfg.masked_tasks,
    cfg.val_releases,
    cfg.window_len,
    cfg.window_len,
    cfg.use_mini
)
movie_val_cache_key = get_dataset_cache_key(
    cfg.movie_tasks,
    cfg.val_releases,
    cfg.window_len,
    1.0,  # stride = 1s
    cfg.use_mini
)

# Load validation MAE windows
print(f"\n📦 Loading validation MAE windows (cache key: {mae_val_cache_key[:50]}...)")
mae_val_windows = load_windowed_dataset(PRETRAIN_CACHE_DIR, "mae_val", mae_val_cache_key)

if mae_val_windows is None:
    print("  Creating validation MAE windows from scratch...")
    # Load and window validation data (simplified - mirror training logic)
    all_mae_val_windows = []
    for task in cfg.masked_tasks:
        for release in cfg.val_releases:
            try:
                ds = EEGChallengeDataset(
                    task=task,
                    release=release,
                    cache_dir=Path(str(cfg.HBN_ROOT)),
                    mini=cfg.use_mini,
                    description_fields=["subject", "age", "sex", "p_factor", "attention", "internalizing", "ehq_total"]
                )

                # Filter out trials shorter than window size
                min_samples = int(cfg.window_len * cfg.sfreq)
                original_count = len(ds.datasets)
                ds = filter_short_trials(ds, min_samples)
                filtered_count = original_count - len(ds.datasets)

                if len(ds.datasets) == 0:
                    continue  # Skip if all trials too short

                # Window the filtered dataset
                windows = create_fixed_length_windows(
                    ds,
                    window_size_samples=min_samples,
                    window_stride_samples=min_samples,
                    drop_last_window=False,
                    preload=True
                )

                # Validate channel counts and shape consistency (129 channels x 200 timepoints)
                windows = validate_windowed_dataset_shapes(windows, expected_n_channels=129, expected_n_times=200)

                if len(windows.datasets) == 0:
                    continue  # Skip if all recordings filtered

                all_mae_val_windows.extend(windows.datasets)

                # Report results
                if filtered_count > 0:
                    print(f"    {task}/{release}: {len(windows)} windows ({filtered_count}/{original_count} trials filtered)")
                else:
                    print(f"    {task}/{release}: {len(windows)} windows")
            except:
                pass

    mae_val_windows = BaseConcatDataset(all_mae_val_windows)

    save_windowed_dataset(mae_val_windows, PRETRAIN_CACHE_DIR, "mae_val", mae_val_cache_key)
else:
    print(f"  ✅ Loaded {len(mae_val_windows)} windows from cache")

# Load validation movie windows with ISC metadata (if contrastive enabled)
if cfg.use_contrastive:
    print(f"\n📦 Loading validation movie windows with ISC metadata...")
    movie_val_windows = load_and_window_movies(
        movie_names=cfg.movie_tasks,
        dataset_class=EEGChallengeDataset,
        cache_dir=cfg.HBN_ROOT,
        releases=cfg.val_releases,
        mini=cfg.use_mini,
        window_len_s=cfg.window_len,
        stride_s=1.0,
        sfreq=cfg.sfreq,
        time_bin_size_s=cfg.time_bin_size_s,
        preload=True
    )
    print(f"  ✅ Loaded {len(movie_val_windows)} validation movie windows")

    # Wrap with ContrastivePairDataset for ISC validation
    print("\n🔗 Creating validation ISC triplet dataset...")
    contrastive_val_dataset = ContrastivePairDataset(
        movie_val_windows,
        pos_strategy=cfg.pos_strategy,
        neg_strategy=cfg.neg_strategy,
        return_triplets=True,
        random_state=2025
    )

    # Get validation ISC stats
    val_stats = contrastive_val_dataset.get_stats()
    print(f"  ✅ Validation ISC Statistics:")
    print(f"     Valid anchors: {val_stats['valid_anchors']:,}")
    print(f"     Positive groups: {val_stats['pos_groups']:,}")
    print(f"     Subjects: {val_stats['subjects']}")

    print(f"\n✅ Validation data ready: {len(mae_val_windows)} MAE windows, {len(movie_val_windows)} movie windows")
else:
    movie_val_windows = None
    contrastive_val_dataset = None
    print(f"\n✅ Validation data ready: {len(mae_val_windows)} MAE windows (contrastive disabled)")

# Wrap validation MAE dataset with subject metadata for auxiliary loss
print("\n🔄 Wrapping validation MAE dataset with subject metadata...")
mae_val_windows_wrapped = BaseConcatDataset([DatasetWrapper(w) for w in mae_val_windows.datasets])
print(f"   ✅ Wrapped {len(mae_val_windows_wrapped)} MAE windows")

# Create validation dataloaders
mae_val_loader = DataLoader(
    mae_val_windows_wrapped,  # Use wrapped version for auxiliary loss
    batch_size=cfg.batch_size,
    shuffle=False,  # No shuffle for validation
    num_workers=cfg.num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=1,
    collate_fn=collate_with_metadata  # Preserve metadata as list of dicts
)

if cfg.use_contrastive and contrastive_val_dataset is not None:
    contrastive_val_loader = DataLoader(
        contrastive_val_dataset,  # Use ISC triplet dataset
        batch_size=cfg.contrastive_batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
        collate_fn=collate_triplets  # Handle non-resizable tensor storage
    )
    print(f"  Val MAE batches: {len(mae_val_loader)}")
    print(f"  Val contrastive batches: {len(contrastive_val_loader)}")
else:
    contrastive_val_loader = None
    print(f"  Val MAE batches: {len(mae_val_loader)}")

# %% Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️ Using device: {device}")

# Load electrode locations
print("📍 Loading electrode locations...")
chs_info = load_hbn_chs_info()

# Initialize maskers
spatial_masker = SpatialBlockMasker(
    chs_info,
    mask_diameter_pct=cfg.spatial_mask_diameter
)
temporal_masker = TemporalBlockMasker(
    n_times=cfg.n_times,
    mask_ratio=cfg.temporal_mask_ratio,
    span_length=int(cfg.temporal_mask_span_ms / 1000 * cfg.sfreq)
)

# Initialize model
# Filter auxiliary heads: respect use_auxiliary flag and individual enabled flags
if cfg.use_auxiliary and demographics is not None:
    filtered_aux_config = filter_enabled_aux_heads(cfg.aux_heads)
    if filtered_aux_config:
        print(f"📋 Auxiliary heads enabled: {list(filtered_aux_config.keys())}")

        # Show which head architecture is being used
        if cfg.use_separable_aux_heads:
            print(f"   Architecture: SignalJEPA-style separable heads (Conv3d spatial filtering)")
            print(f"   Spatial filters per head: {cfg.aux_n_spat_filters}")
            print(f"   Input: channel-wise features (batch, {cfg.n_channels}, {cfg.transformer_d_model})")
        else:
            print(f"   Architecture: Simple 2-layer MLPs")
            print(f"   Input: globally-pooled features (batch, {cfg.transformer_d_model})")
    else:
        print("⚠️  All auxiliary heads disabled via 'enabled' flags")
else:
    filtered_aux_config = None
    if not cfg.use_auxiliary:
        print("⚠️  Auxiliary heads DISABLED (use_auxiliary=False)")

model = SignalJEPAPretrainer(
    n_chans=cfg.n_channels,
    n_times=cfg.n_times,
    sfreq=cfg.sfreq,
    input_window_seconds=cfg.input_window_seconds,
    chs_info=chs_info,
    d_model=cfg.transformer_d_model,
    num_encoder_layers=cfg.transformer_num_encoder_layers,
    nhead=cfg.transformer_nhead,
    dropout=cfg.dropout,
    ema_momentum=cfg.ema_momentum,
    predictor_depth=cfg.predictor_depth,
    aux_heads_config=filtered_aux_config,
    use_separable_aux_heads=cfg.use_separable_aux_heads,
    aux_n_spat_filters=cfg.aux_n_spat_filters,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Model parameters: {total_params:,} (trainable: {trainable_params:,})")

# Load pretrained checkpoint if provided (for progressive training)
if cfg.pretrained_checkpoint_path is not None:
    print(f"\n📦 Loading pretrained checkpoint...")
    print(f"   Path: {cfg.pretrained_checkpoint_path}")

    checkpoint = torch.load(cfg.pretrained_checkpoint_path, map_location=device, weights_only=False)

    if cfg.load_encoder_only:
        # Load only context encoder (for progressive training: MAE → +Contrastive → +Aux)
        # Try multiple checkpoint formats in order of preference
        encoder_loaded = False

        # Try 1: Direct context_encoder key (periodic checkpoint format)
        if 'context_encoder' in checkpoint:
            model.context_encoder.load_state_dict(checkpoint['context_encoder'])
            model.target_encoder.load_state_dict(checkpoint['context_encoder'])
            print("  ✅ Loaded from context_encoder (periodic checkpoint format)")
            encoder_loaded = True

        # Try 2: Direct encoder_state_dict key (best checkpoint format)
        elif 'encoder_state_dict' in checkpoint:
            model.context_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            model.target_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("  ✅ Loaded from encoder_state_dict (best checkpoint format)")
            encoder_loaded = True

        # Try 3: Extract from full_model_state_dict (best checkpoint fallback)
        elif 'full_model_state_dict' in checkpoint:
            full_state = checkpoint['full_model_state_dict']
            context_encoder_state = {
                k.replace('context_encoder.', ''): v
                for k, v in full_state.items()
                if k.startswith('context_encoder.')
            }
            model.context_encoder.load_state_dict(context_encoder_state)
            model.target_encoder.load_state_dict(context_encoder_state)
            print("  ✅ Extracted encoder from full_model_state_dict")
            encoder_loaded = True

        # Try 4: Extract from model_state_dict (periodic checkpoint fallback)
        elif 'model_state_dict' in checkpoint:
            full_state = checkpoint['model_state_dict']
            context_encoder_state = {
                k.replace('context_encoder.', ''): v
                for k, v in full_state.items()
                if k.startswith('context_encoder.')
            }
            model.context_encoder.load_state_dict(context_encoder_state)
            model.target_encoder.load_state_dict(context_encoder_state)
            print("  ✅ Extracted encoder from model_state_dict")
            encoder_loaded = True

        if not encoder_loaded:
            raise KeyError(
                f"Could not find encoder weights in checkpoint. "
                f"Available keys: {list(checkpoint.keys())}"
            )
    else:
        # Load full model state (including projection head, aux heads if present)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  ✅ Loaded full model state")

    print(f"   Checkpoint was trained for {checkpoint.get('epoch', 'unknown')} epochs")

# Initialize GradNorm for dynamic loss weight balancing
# Auto-disable GradNorm when encoder is frozen (no shared encoder gradients to balance)
if cfg.use_gradnorm and cfg.freeze_encoder:
    print("\n⚠️  Encoder frozen: Auto-disabling GradNorm")
    print("   (GradNorm balances w.r.t. shared encoder parameters, but encoder is frozen)")
    gradnorm = None
    shared_params = None
elif cfg.use_gradnorm:
    # Dynamically compute number of tasks based on enabled features
    n_tasks = 0
    task_names = []

    if cfg.use_mae:
        n_tasks += 1
        task_names.append("MAE")

    if cfg.use_contrastive:
        n_tasks += 1
        task_names.append("Contrastive")

    # Per-head GradNorm: each auxiliary head is a separate task
    if filtered_aux_config is not None:
        n_tasks += len(filtered_aux_config)
        task_names.extend([f"aux_{name}" for name in filtered_aux_config.keys()])

    print(f"\n🎯 Initializing GradNorm for dynamic loss balancing")
    print(f"   Tasks: {', '.join(task_names)} (n={n_tasks})")
    print(f"   Alpha: {cfg.gradnorm_alpha} (restoring force)")
    print(f"   Warmup: {cfg.gradnorm_warmup_epochs} epochs")
    print(f"   Update frequency: every {cfg.gradnorm_update_freq} batches")

    gradnorm = GradNorm(
        n_tasks=n_tasks,
        alpha=cfg.gradnorm_alpha,
        warmup_epochs=cfg.gradnorm_warmup_epochs,
        update_freq=cfg.gradnorm_update_freq
    )
    # Get shared parameters (context encoder) for gradient norm computation
    shared_params = list(model.context_encoder.parameters())
else:
    gradnorm = None
    shared_params = None  # Not used when GradNorm disabled
    print("\n📊 Using fixed loss weights (GradNorm disabled)")

# %% Optimizer Factory Functions
def create_muon_optimizer(model, cfg):
    """Create hybrid Muon + AdamW optimizer for single-GPU training.

    Muon optimizes 2D weight matrices (Linear/Conv layers) with orthogonalization.
    AdamW optimizes 1D parameters (biases, layer norms, mask token).

    Supports differential learning rates: encoder can have different LR than other components.

    Args:
        model: SignalJEPAPretrainer model
        cfg: Config object with Muon hyperparameters

    Returns:
        SingleDeviceMuonWithAuxAdam optimizer with hybrid parameter groups
    """
    # Separate parameters by component and dimensionality
    encoder_muon_params = []  # Encoder 2D params (weight matrices)
    other_muon_params = []    # Non-encoder 2D params (predictor, projection head)
    adamw_params = []         # All 1D params (biases, layer norms)

    print("\n🔍 Parameter grouping for Muon optimizer:")
    print(f"   Encoder LR multiplier: {cfg.encoder_lr_multiplier}x")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Skip target encoder (EMA-updated, not optimized)
        if 'target_encoder' in name:
            continue

        # Check if parameter belongs to encoder
        is_encoder = 'context_encoder' in name

        if param.ndim >= 2:
            # 2D+ parameters → Muon (with orthogonalization)
            if is_encoder:
                encoder_muon_params.append(param)
                lr_info = f"(lr={cfg.muon_lr * cfg.encoder_lr_multiplier:.6f})"
                print(f"  ✓ Encoder Muon: {name:50s} {str(param.shape):20s} {lr_info}")
            else:
                other_muon_params.append(param)
                print(f"  ✓ Other Muon:   {name:50s} {str(param.shape):20s} (lr={cfg.muon_lr:.6f})")
        else:
            # 1D parameters → AdamW (standard adaptive LR)
            adamw_params.append(param)
            component = "Encoder" if is_encoder else "Other"
            print(f"  ○ {component} AdamW: {name:50s} {str(param.shape):20s}")

    print(f"\n📊 Encoder Muon parameters: {len(encoder_muon_params)} (LR: {cfg.muon_lr * cfg.encoder_lr_multiplier:.6f})")
    print(f"📊 Other Muon parameters: {len(other_muon_params)} (LR: {cfg.muon_lr:.6f})")
    print(f"📊 AdamW parameters: {len(adamw_params)} (LR: {cfg.adamw_aux_lr:.6f})")

    # Create parameter groups
    # NOTE: SingleDeviceMuonWithAuxAdam requires different keys based on use_muon flag:
    #   - Muon groups: params, lr, momentum, weight_decay, use_muon
    #   - Adam groups: params, lr, betas, eps, weight_decay, use_muon
    param_groups = []

    # Encoder Muon group (with differential LR)
    if encoder_muon_params:
        param_groups.append(dict(
            params=encoder_muon_params,
            lr=cfg.muon_lr * cfg.encoder_lr_multiplier,
            momentum=cfg.muon_momentum,
            weight_decay=cfg.muon_weight_decay,
            use_muon=True
        ))

    # Other Muon group (predictor, projection head)
    if other_muon_params:
        param_groups.append(dict(
            params=other_muon_params,
            lr=cfg.muon_lr,
            momentum=cfg.muon_momentum,
            weight_decay=cfg.muon_weight_decay,
            use_muon=True
        ))

    # AdamW group (all 1D params: biases, layer norms)
    if adamw_params:
        param_groups.append(dict(
            params=adamw_params,
            lr=cfg.adamw_aux_lr,
            betas=cfg.adamw_aux_betas,
            eps=1e-10,
            weight_decay=cfg.adamw_aux_weight_decay,
            use_muon=False
        ))

    return SingleDeviceMuonWithAuxAdam(param_groups)

# %% Create Optimizer
# Optimizer and scheduler
if cfg.use_muon:
    print("📊 Using hybrid Muon+AdamW optimizer (Muon for 2D weights, AdamW for 1D params)")
    optimizer = create_muon_optimizer(model, cfg)
else:
    print("📊 Using standard AdamW optimizer (stable for multi-task learning)")
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

# Create warmup + cosine annealing scheduler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# Warmup scheduler: linear ramp from 0 to target LR over warmup_epochs
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,  # Start at 10% of target LR
    end_factor=1.0,    # End at 100% of target LR
    total_iters=cfg.warmup_epochs
)

# Main scheduler: cosine annealing after warmup
main_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=cfg.n_epochs - cfg.warmup_epochs,
    eta_min=1e-6
)

# Combined scheduler: warmup → cosine annealing
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[cfg.warmup_epochs]
)

print(f"📈 LR schedule: {cfg.warmup_epochs} epoch warmup → cosine annealing")

# Apply initial encoder freeze if configured
if cfg.freeze_encoder:
    print(f"\n🔒 Freezing encoder (requires_grad=False)")
    if cfg.freeze_encoder_epochs > 0:
        print(f"   Will auto-unfreeze at epoch {cfg.freeze_encoder_epochs}")
    else:
        print(f"   Encoder will remain frozen throughout training")
    set_encoder_frozen(model, frozen=True)

    # Verify frozen status
    n_frozen = sum(1 for p in model.context_encoder.parameters() if not p.requires_grad)
    n_total = sum(1 for _ in model.context_encoder.parameters())
    print(f"   ✓ Frozen {n_frozen}/{n_total} encoder parameters")

# Initialize wandb
if cfg.use_wandb:
    tags = ["signaljepa", "pretraining", "mae", "contrastive", "auxiliary", "adamw", "stable"]

    wandb.init(
        project="cerebro-signaljepa-pretrain",
        name=cfg.experiment_name,
        config=vars(cfg),
        tags=tags
    )

# %% Auxiliary Loss Function
def compute_auxiliary_loss(model, X, batch_infos, demographics, aux_config, device):
    """
    Compute auxiliary loss for demographics prediction.

    Args:
        model: SignalJEPAPretrainer model with auxiliary heads
        X: Input EEG tensor [batch, channels, time]
        batch_infos: List of dicts with 'subject' keys
        demographics: Dict mapping subject IDs to demographic values
        aux_config: Dict of auxiliary head configurations
        device: torch device

    Returns:
        Dict of per-head losses {head_name: loss_tensor}
        Returns empty dict if no valid targets
    """
    if demographics is None or not aux_config:
        return {}

    # Get batch subjects
    batch_subjects = [info['subject'] for info in batch_infos]

    # DEBUG: Print first batch to diagnose issue
    if not hasattr(compute_auxiliary_loss, '_debug_printed'):
        print(f"\n🔍 DEBUG: Auxiliary Loss Matching")
        print(f"  First 3 batch subjects: {batch_subjects[:3]}")
        print(f"  First 3 demographics keys: {list(demographics.keys())[:3]}")
        print(f"  Batch subject in demographics? {batch_subjects[0] in demographics}")
        if batch_subjects[0] in demographics:
            print(f"  Available fields for {batch_subjects[0]}: {list(demographics[batch_subjects[0]].keys())}")
        print(f"  Aux heads requested: {list(aux_config.keys())}")
        compute_auxiliary_loss._debug_printed = True

    # Forward through auxiliary heads
    aux_preds = model.forward_auxiliary(X)

    # Store per-head losses (unweighted - GradNorm will apply dynamic weights)
    head_losses = {}

    for head_name, head_config in aux_config.items():
        # Collect targets for this batch
        targets = []
        valid_indices = []

        for i, subj in enumerate(batch_subjects):
            if subj in demographics and head_name in demographics[subj]:
                target_val = demographics[subj][head_name]
                targets.append(target_val)
                valid_indices.append(i)

        # Skip if no valid targets in this batch
        if not targets:
            continue

        # Get predictions for valid samples
        preds = aux_preds[head_name][valid_indices]

        # Compute loss based on head type
        try:
            if head_config['type'] == 'regression':
                # Convert numeric targets to float tensor
                targets_tensor = torch.tensor(targets, device=device, dtype=torch.float32)
                loss = F.mse_loss(preds.squeeze(), targets_tensor)
            else:  # classification
                # Convert sex strings to indices: 'M'=0, 'F'=1
                if head_name == 'sex':
                    targets_indices = [0 if t == 'M' else 1 for t in targets]
                    targets_tensor = torch.tensor(targets_indices, device=device, dtype=torch.long)
                else:
                    # Generic classification: assume integer labels
                    targets_tensor = torch.tensor(targets, device=device, dtype=torch.long)
                loss = F.cross_entropy(preds, targets_tensor)

            # Check for NaN/Inf in individual head loss
            if torch.isnan(loss) or torch.isinf(loss):
                if not hasattr(compute_auxiliary_loss, f'_warned_{head_name}_nan'):
                    print(f"⚠️  NaN/Inf in {head_name} loss - skipping this head")
                    setattr(compute_auxiliary_loss, f'_warned_{head_name}_nan', True)
                continue  # Skip this head

            # Store unweighted loss (GradNorm will apply dynamic weights)
            head_losses[head_name] = loss

        except Exception as e:
            # Catch any unexpected errors (tensor shape mismatches, etc.)
            if not hasattr(compute_auxiliary_loss, f'_warned_{head_name}_error'):
                print(f"⚠️  Error computing {head_name} loss: {e}")
                setattr(compute_auxiliary_loss, f'_warned_{head_name}_error', True)
            continue

    # Warn once if no valid targets in batch
    if not head_losses and not hasattr(compute_auxiliary_loss, '_warned_empty_batch'):
        print("⚠️  WARNING: Auxiliary loss batch has no valid demographic data")
        print("   This is expected if subjects lack p_factor/attention/etc.")
        compute_auxiliary_loss._warned_empty_batch = True

    return head_losses

# %% Validation Function
def validate_epoch(model, mae_loader, contrastive_loader, device, cfg, demographics, aux_config):
    """Validate for one epoch."""
    model.eval()

    epoch_losses = {
        'mae': [],
        'contrastive': [],
        'auxiliary': [],
        'total': []
    }

    with torch.no_grad():
        # Create iterator for contrastive loader (if enabled)
        if contrastive_loader is not None:
            contrastive_iter = iter(contrastive_loader)
            n_batches = min(len(mae_loader), len(contrastive_loader))
        else:
            contrastive_iter = None
            n_batches = len(mae_loader)

        for batch_idx, mae_batch in enumerate(mae_loader):
            if batch_idx >= n_batches:
                break

            # === MAE Step ===
            X_mae, _, _, batch_infos = mae_batch  # Unpack 4-tuple with infos
            X_mae = X_mae.to(device, dtype=torch.float32)
            batch_size = X_mae.shape[0]

            if cfg.use_mae:
                # Generate masks
                spatial_mask = spatial_masker.get_batch_masks(batch_size, device)
                temporal_mask = temporal_masker.get_batch_masks(batch_size, device)

                # Forward MAE
                predictions, targets, mask = model.forward_mae(X_mae, spatial_mask, temporal_mask)

                # L1 loss on masked tokens only
                mask_expanded = mask.unsqueeze(-1).expand_as(predictions)
                if mask.sum() > 0:
                    mae_loss = (F.l1_loss(predictions, targets, reduction='none') * mask_expanded).sum() / mask_expanded.sum()
                else:
                    mae_loss = F.l1_loss(predictions, targets)
            else:
                # MAE disabled
                mae_loss = torch.tensor(0.0, device=device)

            # === Contrastive Step (ISC Triplets) ===
            if contrastive_loader is not None:
                try:
                    contrast_batch = next(contrastive_iter)
                except StopIteration:
                    contrastive_iter = iter(contrastive_loader)
                    contrast_batch = next(contrastive_iter)

                # Unpack ISC triplets: (anchor, positive, negative)
                anchor, positive, negative = contrast_batch
                anchor = anchor.to(device, dtype=torch.float32)
                positive = positive.to(device, dtype=torch.float32)
                negative = negative.to(device, dtype=torch.float32)

                # Forward through encoder
                z_anchor = model.forward_contrastive(anchor)
                z_positive = model.forward_contrastive(positive)
                z_negative = model.forward_contrastive(negative)

                # InfoNCE with optional hard negatives
                contrastive_loss = info_nce_loss(
                    z_anchor, z_positive, cfg.temperature,
                    z_neg=z_negative,
                    use_hard_negative=cfg.use_triplet_negatives
                )
            else:
                contrastive_loss = torch.tensor(0.0, device=device)

            # === Auxiliary Step (Per-Head) ===
            if aux_config is not None:
                aux_loss_dict = compute_auxiliary_loss(model, X_mae, batch_infos, demographics, aux_config, device)
            else:
                aux_loss_dict = {}

            # === Combined Loss ===
            mae_weight = cfg.mae_loss_weight if cfg.use_mae else 0.0
            contrast_weight = cfg.contrastive_loss_weight if contrastive_loader is not None else 0.0

            # Compute weighted auxiliary loss using fixed config weights
            aux_loss_combined = sum(
                aux_config[head_name]['weight'] * loss
                for head_name, loss in aux_loss_dict.items()
            ) if aux_loss_dict else torch.tensor(0.0, device=device)

            total_loss = (
                mae_weight * mae_loss +
                contrast_weight * contrastive_loss +
                aux_loss_combined
            )

            # Track
            epoch_losses['mae'].append(mae_loss.item())
            epoch_losses['contrastive'].append(contrastive_loss.item())
            epoch_losses['auxiliary'].append(aux_loss_combined.item())
            epoch_losses['total'].append(total_loss.item())

    # Return average losses
    return {
        'mae': np.mean(epoch_losses['mae']),
        'contrastive': np.mean(epoch_losses['contrastive']),
        'auxiliary': np.mean(epoch_losses['auxiliary']),
        'total': np.mean(epoch_losses['total'])
    }

# %% EMA Momentum Schedule
def get_ema_momentum(current_epoch: int, max_epochs: int, start: float = 0.996, end: float = 1.0) -> float:
    """Compute EMA momentum using cosine schedule.

    Paper (S-JEPA): "exponential moving average (EMA) with cosine schedule"
    Schedules momentum from start to end over training.

    Args:
        current_epoch: Current epoch (0-indexed)
        max_epochs: Total number of epochs
        start: Starting momentum (default: 0.996)
        end: Ending momentum (default: 1.0)

    Returns:
        Current EMA momentum value
    """
    # Cosine annealing: smoothly transitions from start to end
    progress = current_epoch / max(1, max_epochs - 1)  # 0 to 1
    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))  # 1 to 0
    momentum = end + (start - end) * cosine_factor
    return momentum

# %% Training Loop
print("\n" + "="*60)
print("🚀 Starting combined pretraining...")
print("="*60)

print(f"\n🎯 Loss Toggles:")
print(f"  MAE: {'✅ Enabled' if cfg.use_mae else '❌ Disabled'}")
print(f"  Contrastive: {'✅ Enabled' if cfg.use_contrastive else '❌ Disabled'}")
print(f"  Auxiliary: {'✅ Enabled' if cfg.use_auxiliary else '❌ Disabled'}")

print(f"\n📋 Training Configuration:")
print(f"  Model: SignalJEPA with {total_params:,} parameters")
if cfg.use_mae:
    print(f"  MAE loss weight: {cfg.mae_loss_weight}")
if cfg.use_contrastive:
    print(f"  Contrastive loss weight: {cfg.contrastive_loss_weight}")
if cfg.use_auxiliary:
    print(f"  Auxiliary loss weight: {cfg.aux_total_weight}")
print(f"  Epochs: {cfg.n_epochs}")
print(f"  MAE batch size: {cfg.batch_size}")
print(f"  Contrastive batch size: {cfg.contrastive_batch_size}")

best_loss = float('inf')
patience_counter = 0
training_history = []

# EMA tracking for loss stability
loss_ema = None
loss_ema_alpha = 0.1  # EMA smoothing factor (lower = smoother)

# Checkpoint tracking
checkpoint_counter = 0
last_checkpoint_epoch = -1

for epoch in range(cfg.n_epochs):
    print(f"\n📅 Epoch {epoch+1}/{cfg.n_epochs}")
    print("-" * 40)

    # Auto-unfreeze encoder if configured
    if cfg.freeze_encoder and cfg.freeze_encoder_epochs > 0 and epoch == cfg.freeze_encoder_epochs:
        print(f"\n🔓 Unfreezing encoder at epoch {epoch+1} (frozen for {cfg.freeze_encoder_epochs} epochs)")
        print(f"   Encoder will train with LR multiplier: {cfg.encoder_lr_multiplier}x")
        set_encoder_frozen(model, frozen=False)

        # Verify unfrozen status
        n_trainable = sum(1 for p in model.context_encoder.parameters() if p.requires_grad)
        n_total = sum(1 for _ in model.context_encoder.parameters())
        print(f"   ✓ Unfrozen {n_trainable}/{n_total} encoder parameters")

    model.train()
    epoch_losses = {
        'mae': [],
        'contrastive': [],
        'auxiliary': []
    }

    # Alternate between MAE and contrastive batches
    mae_iter = iter(mae_loader)

    if contrastive_loader is not None:
        contrastive_iter = iter(contrastive_loader)
        n_batches = min(len(mae_loader), len(contrastive_loader))
    else:
        contrastive_iter = None
        n_batches = len(mae_loader)

    pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}")

    for batch_idx in pbar:
        # === MAE Step ===
        try:
            mae_batch = next(mae_iter)
        except StopIteration:
            mae_iter = iter(mae_loader)
            mae_batch = next(mae_iter)

        X_mae, _, _, batch_infos = mae_batch  # Unpack 4-tuple with infos
        X_mae = X_mae.to(device, dtype=torch.float32)
        batch_size = X_mae.shape[0]

        if cfg.use_mae:
            # Generate masks
            spatial_mask = spatial_masker.get_batch_masks(batch_size, device)
            temporal_mask = temporal_masker.get_batch_masks(batch_size, device)

            # Forward MAE
            predictions, targets, mask = model.forward_mae(X_mae, spatial_mask, temporal_mask)

            # === VERIFICATION: Print masking statistics for first batch ===
            if epoch == 0 and batch_idx == 0:
                print(f"\n{'='*60}")
                print("🔍 MASKING VERIFICATION (First Batch)")
                print(f"{'='*60}")
                print(f"  Input shape: {X_mae.shape}")
                print(f"  Spatial mask shape: {spatial_mask.shape}, masked channels: {spatial_mask[0].sum().item():.0f}/{spatial_mask.shape[1]}")
                print(f"  Temporal mask shape: {temporal_mask.shape}, masked timepoints: {temporal_mask[0].sum().item():.0f}/{temporal_mask.shape[1]}")
                print(f"  Sequence mask shape: {mask.shape}, masked tokens: {mask[0].sum().item():.0f}/{mask.shape[1]}")
                print(f"  Predictions shape: {predictions.shape}")
                print(f"  Targets shape: {targets.shape}")
                print(f"  Masked token ratio: {mask.float().mean().item():.2%}")
                print(f"{'='*60}\n")

            # L1 loss on masked tokens only
            # mask: (batch, seq_len) where 1 = masked, 0 = visible
            # Expand mask to match prediction dimensions: (batch, seq_len, d_model)
            mask_expanded = mask.unsqueeze(-1).expand_as(predictions)

            # Compute loss only on masked positions
            if mask.sum() > 0:  # Ensure we have masked tokens
                mae_loss = (F.l1_loss(predictions, targets, reduction='none') * mask_expanded).sum() / mask_expanded.sum()
            else:
                # No masked tokens (shouldn't happen, but handle gracefully)
                mae_loss = F.l1_loss(predictions, targets)
        else:
            # MAE disabled - set to zero, but keep batch data for auxiliary heads
            mae_loss = torch.tensor(0.0, device=device)

        # === Contrastive Step (ISC Triplets) ===
        if cfg.use_contrastive and contrastive_loader is not None:
            try:
                contrast_batch = next(contrastive_iter)
            except StopIteration:
                contrastive_iter = iter(contrastive_loader)
                contrast_batch = next(contrastive_iter)

            # Unpack ISC triplets: (anchor, positive, negative)
            anchor, positive, negative = contrast_batch
            anchor = anchor.to(device, dtype=torch.float32)
            positive = positive.to(device, dtype=torch.float32)
            negative = negative.to(device, dtype=torch.float32)

            # Forward through encoder
            z_anchor = model.forward_contrastive(anchor)
            z_positive = model.forward_contrastive(positive)
            z_negative = model.forward_contrastive(negative)

            # InfoNCE with optional hard negatives
            contrastive_loss = info_nce_loss(
                z_anchor, z_positive, cfg.temperature,
                z_neg=z_negative,
                use_hard_negative=cfg.use_triplet_negatives
            )
        else:
            # Contrastive disabled - set to zero
            contrastive_loss = torch.tensor(0.0, device=device)
            z_anchor = z_positive = z_negative = None

        # === Auxiliary Step (Per-Head) ===
        if cfg.use_auxiliary and filtered_aux_config is not None:
            aux_loss_dict = compute_auxiliary_loss(model, X_mae, batch_infos, demographics, filtered_aux_config, device)
        else:
            # Auxiliary disabled - empty dict
            aux_loss_dict = {}

        # === GradNorm Weight Update (if enabled) ===
        if cfg.use_gradnorm and gradnorm is not None:
            # Build task_losses list based on enabled tasks (per-head for aux)
            task_losses = []

            if cfg.use_mae:
                task_losses.append(mae_loss)

            if cfg.use_contrastive:
                task_losses.append(contrastive_loss)

            # Per-head auxiliary losses (each head is a separate task)
            if aux_loss_dict:
                task_losses.extend(aux_loss_dict.values())

            # Update task weights based on gradient norms
            dynamic_weights = gradnorm.update_weights(task_losses, shared_params, epoch)

            # Extract weights (order: [MAE], [Contrastive], [aux_head1, aux_head2, ...])
            weight_idx = 0

            if cfg.use_mae:
                mae_weight = dynamic_weights[weight_idx].item()
                weight_idx += 1
            else:
                mae_weight = 0.0

            if cfg.use_contrastive:
                contrast_weight = dynamic_weights[weight_idx].item()
                weight_idx += 1
            else:
                contrast_weight = 0.0

            # Extract per-head auxiliary weights
            if aux_loss_dict:
                aux_head_weights = {
                    head_name: dynamic_weights[weight_idx + i].item()
                    for i, head_name in enumerate(aux_loss_dict.keys())
                }
            else:
                aux_head_weights = {}
        else:
            # Use fixed config weights (respect enabled flags)
            mae_weight = cfg.mae_loss_weight if cfg.use_mae else 0.0
            contrast_weight = cfg.contrastive_loss_weight if cfg.use_contrastive else 0.0

            # Fixed per-head weights from config
            if aux_loss_dict:
                aux_head_weights = {
                    head_name: filtered_aux_config[head_name]['weight']
                    for head_name in aux_loss_dict.keys()
                }
            else:
                aux_head_weights = {}

        # === Combined Loss with Dynamic/Fixed Weights ===
        # Compute weighted auxiliary loss
        aux_loss_combined = sum(
            aux_head_weights[head_name] * loss
            for head_name, loss in aux_loss_dict.items()
        ) if aux_loss_dict else torch.tensor(0.0, device=device)

        total_loss = (
            mae_weight * mae_loss +
            contrast_weight * contrastive_loss +
            aux_loss_combined
        )

        # === NaN Detection (Pre-Backward) ===
        # Check for NaN/Inf in losses BEFORE backprop
        if torch.isnan(mae_loss) or torch.isinf(mae_loss):
            print(f"🛑 NaN/Inf in MAE loss at epoch {epoch+1}, batch {batch_idx}")
            print(f"   MAE: {mae_loss.item()}, Contrast: {contrastive_loss.item()}, Aux: {aux_loss_combined.item()}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
            print(f"🛑 NaN/Inf in contrastive loss at epoch {epoch+1}, batch {batch_idx}")
            print(f"   MAE: {mae_loss.item()}, Contrast: {contrastive_loss.item()}, Aux: {aux_loss_combined.item()}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if torch.isnan(aux_loss_combined) or torch.isinf(aux_loss_combined):
            print(f"🛑 NaN/Inf in auxiliary loss at epoch {epoch+1}, batch {batch_idx}")
            print(f"   MAE: {mae_loss.item()}, Contrast: {contrastive_loss.item()}, Aux: {aux_loss_combined.item()}")
            if aux_loss_dict:
                print(f"   Per-head losses: {', '.join(f'{k}={v.item():.4f}' for k, v in aux_loss_dict.items())}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"🛑 NaN/Inf in total loss at epoch {epoch+1}, batch {batch_idx}")
            print(f"   MAE: {mae_loss.item()}, Contrast: {contrastive_loss.item()}, Aux: {aux_loss_combined.item()}")
            optimizer.zero_grad(set_to_none=True)
            continue

        # Backward
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        # === NaN Detection (Post-Backward) ===
        # Check for NaN/Inf in gradients AFTER backprop
        has_nan_grad = False
        max_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"⚠️  NaN/Inf gradient in {name}")
                    has_nan_grad = True

        if has_nan_grad:
            print(f"🛑 NaN gradients detected at epoch {epoch+1}, batch {batch_idx}")
            print(f"   MAE: {mae_loss.item():.4f}, Contrast: {contrastive_loss.item():.4f}, Aux: {aux_loss_combined.item():.4f}")
            print(f"   Skipping batch to prevent optimizer corruption")
            optimizer.zero_grad(set_to_none=True)
            continue

        # Gradient clipping (only if no NaN)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Update target encoder with scheduled EMA momentum
        # Paper uses cosine schedule: 0.996 → 1.0 over training
        current_momentum = get_ema_momentum(epoch, cfg.n_epochs)
        model.update_target_encoder(momentum=current_momentum)

        # Track losses
        epoch_losses['mae'].append(mae_loss.item())
        epoch_losses['contrastive'].append(contrastive_loss.item())
        epoch_losses['auxiliary'].append(aux_loss_combined.item())

        # Update progress
        pbar.set_postfix({
            'mae': f"{mae_loss.item():.4f}",
            'contrast': f"{contrastive_loss.item():.4f}",
            'aux': f"{aux_loss_combined.item():.4f}",
            'total': f"{total_loss.item():.4f}",
            'grad': f"{max_grad_norm:.2f}",
            'ema_mom': f"{current_momentum:.4f}"
        })

        # Periodic detailed monitoring (every 100 batches)
        if batch_idx % 100 == 0 and cfg.use_wandb:
            log_dict = {
                'batch_mae_loss': mae_loss.item(),
                'batch_contrastive_loss': contrastive_loss.item(),
                'batch_auxiliary_loss': aux_loss_combined.item(),
                'batch_total_loss': total_loss.item(),
                'max_gradient_norm': max_grad_norm,
                'ema_momentum': current_momentum,
                'batch_idx': batch_idx + epoch * len(mae_loader)
            }

            # Log per-head auxiliary losses
            if aux_loss_dict:
                for head_name, head_loss in aux_loss_dict.items():
                    log_dict[f'batch_aux_{head_name}_loss'] = head_loss.item()

            # Log per-head auxiliary weights
            if aux_head_weights:
                for head_name, head_weight in aux_head_weights.items():
                    log_dict[f'weight_aux_{head_name}'] = head_weight

            # Compute similarity statistics (only if contrastive enabled)
            if cfg.use_contrastive and z_anchor is not None:
                with torch.no_grad():
                    z_anchor_monitor = model.forward_contrastive(anchor)
                    z_positive_monitor = model.forward_contrastive(positive)
                    sim_monitor = torch.mm(z_anchor_monitor, z_positive_monitor.t())

                log_dict.update({
                    'similarity_mean': sim_monitor.mean().item(),
                    'similarity_std': sim_monitor.std().item(),
                    'similarity_max': sim_monitor.max().item(),
                    'similarity_min': sim_monitor.min().item(),
                })

            # Add GradNorm weights if enabled
            if cfg.use_gradnorm and gradnorm is not None:
                log_dict.update({
                    'gradnorm_mae_weight': mae_weight,
                    'gradnorm_contrastive_weight': contrast_weight,
                    'gradnorm_active': epoch >= cfg.gradnorm_warmup_epochs
                })
                # Per-head aux weights already logged above

            wandb.log(log_dict)

    # Epoch summary
    avg_mae = np.mean(epoch_losses['mae'])
    avg_contrast = np.mean(epoch_losses['contrastive'])
    avg_aux = np.mean(epoch_losses['auxiliary'])
    avg_total = (
        (avg_mae * cfg.mae_loss_weight if cfg.use_mae else 0.0) +
        (avg_contrast * cfg.contrastive_loss_weight if cfg.use_contrastive else 0.0) +
        (avg_aux * cfg.auxiliary_loss_weight if cfg.use_auxiliary else 0.0)
    )

    # Update EMA for loss stability tracking
    if loss_ema is None:
        loss_ema = avg_total
    else:
        loss_ema = loss_ema_alpha * avg_total + (1 - loss_ema_alpha) * loss_ema

    print(f"\n📈 Train Epoch {epoch+1} Summary:")
    print(f"  MAE Loss: {avg_mae:.4f}")
    print(f"  Contrastive Loss: {avg_contrast:.4f}")
    print(f"  Auxiliary Loss: {avg_aux:.4f}")
    print(f"  Total Loss: {avg_total:.4f}")
    print(f"  Loss EMA: {loss_ema:.4f}")

    # Validate
    print(f"\n📊 Running validation...")
    val_losses = validate_epoch(model, mae_val_loader, contrastive_val_loader, device, cfg, demographics, filtered_aux_config)
    print(f"  Val MAE Loss: {val_losses['mae']:.4f}")
    print(f"  Val Contrastive Loss: {val_losses['contrastive']:.4f}")
    print(f"  Val Auxiliary Loss: {val_losses['auxiliary']:.4f}")
    print(f"  Val Total Loss: {val_losses['total']:.4f}")

    training_history.append({
        'epoch': epoch + 1,
        'train_mae_loss': avg_mae,
        'train_contrastive_loss': avg_contrast,
        'train_auxiliary_loss': avg_aux,
        'train_total_loss': avg_total,
        'val_mae_loss': val_losses['mae'],
        'val_contrastive_loss': val_losses['contrastive'],
        'val_auxiliary_loss': val_losses['auxiliary'],
        'val_total_loss': val_losses['total']
    })

    # Log to wandb
    if cfg.use_wandb:
        wandb.log({
            'epoch_train_mae_loss': avg_mae,
            'epoch_train_contrastive_loss': avg_contrast,
            'epoch_train_auxiliary_loss': avg_aux,
            'epoch_train_total_loss': avg_total,
            'epoch_val_mae_loss': val_losses['mae'],
            'epoch_val_contrastive_loss': val_losses['contrastive'],
            'epoch_val_auxiliary_loss': val_losses['auxiliary'],
            'epoch_val_total_loss': val_losses['total'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch + 1,
            # Loss component ratios (for debugging balance)
            'mae_contribution': (avg_mae * cfg.mae_loss_weight) / avg_total,
            'contrastive_contribution': (avg_contrast * cfg.contrastive_loss_weight) / avg_total,
            'auxiliary_contribution': (avg_aux * cfg.auxiliary_loss_weight) / avg_total,
        })

    # Detect loss explosion (safeguard against instability)
    if loss_ema > 100.0 or (epoch > 10 and val_losses['total'] > 2 * best_loss):
        print(f"\n🛑 LOSS EXPLOSION DETECTED!")
        print(f"   Current val loss: {val_losses['total']:.4f}")
        print(f"   Best val loss: {best_loss:.4f}")
        print(f"   Loss EMA: {loss_ema:.4f}")
        print(f"   Training has become unstable. Stopping to prevent divergence.")
        break

    # Save checkpoint based on VALIDATION loss
    if val_losses['total'] < best_loss:
        improvement = best_loss - val_losses['total']
        best_loss = val_losses['total']
        patience_counter = 0  # Reset patience
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': model.context_encoder.state_dict(),
            'full_model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_losses['total'],
            'loss_ema': loss_ema,
            'config': vars(cfg),
            'history': training_history
        }
        torch.save(checkpoint, cfg.checkpoint_dir / "best_pretrain.pt")
        print(f"✅ NEW BEST! Val loss: {val_losses['total']:.4f} (improved by {improvement:.4f})")
    else:
        patience_counter += 1
        print(f"⏳ Patience: {patience_counter}/{cfg.early_stopping_patience} (best: {best_loss:.4f})")

        # Early stopping
        if patience_counter >= cfg.early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            print(f"   No improvement for {cfg.early_stopping_patience} epochs")
            print(f"   Best Val Loss: {best_loss:.4f}")
            break

    # Periodic checkpoint saving (every 10 epochs) for rollback capability
    if (epoch + 1) % 10 == 0 and epoch != last_checkpoint_epoch:
        checkpoint_counter += 1
        periodic_checkpoint = {
            'epoch': epoch,
            'context_encoder': model.context_encoder.state_dict(),  # For encoder-only loading
            'model_state_dict': model.state_dict(),  # For full model loading
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_losses['total'],
            'loss_ema': loss_ema,
            'config': vars(cfg),
            'history': training_history
        }
        checkpoint_path = cfg.checkpoint_dir / f"checkpoint_epoch{epoch+1:03d}.pt"
        torch.save(periodic_checkpoint, checkpoint_path)
        last_checkpoint_epoch = epoch
        print(f"💾 Periodic checkpoint saved: {checkpoint_path.name}")

    # Step scheduler
    scheduler.step()

# %% Final Summary
print("\n" + "="*60)
print("🏁 Pretraining Complete!")
print("="*60)
print(f"Best Total Loss: {best_loss:.4f}")
print(f"Total epochs: {len(training_history)}")
print(f"\n📁 Pretrained encoder saved to:")
print(f"  {cfg.checkpoint_dir / 'best_pretrain.pt'}")
print(f"\n✨ Ready for fine-tuning on Challenge 1/2!")

# %% Plot Training Curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

epochs = [h['epoch'] for h in training_history]
mae_losses = [h['mae_loss'] for h in training_history]
contrast_losses = [h['contrastive_loss'] for h in training_history]
total_losses = [h['total_loss'] for h in training_history]

axes[0].plot(epochs, mae_losses, linewidth=2, color='blue')
axes[0].set_title('MAE Reconstruction Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('L1 Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, contrast_losses, linewidth=2, color='green')
axes[1].set_title('Contrastive Loss (InfoNCE)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs, total_losses, linewidth=2, color='red')
axes[2].set_title('Total Combined Loss')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Weighted Loss')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(cfg.checkpoint_dir / "pretraining_curves.png", dpi=150)
plt.show()

print(f"\n📊 Training curves saved to: {cfg.checkpoint_dir / 'pretraining_curves.png'}")
