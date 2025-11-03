# %% [markdown]
# # Challenge 1: Full HBN Dataset Training with Mamba2 (Release-Level Split)
#
# **CRITICAL FIXES APPLIED (2025-11-02):**
# 1. NRMSE calculation: Fixed to use std() instead of (max-min) to match competition metric
# 2. Window length: Fixed from 2.5s to 2.0s to match competition preprocessing
# 3. Model input shape: Fixed from 250 to 200 samples to accept competition data
#
# **Release-level train/val split to improve generalization to R5**
#
# Previous approach: Subject-level split within all releases
# - Validation NRMSE: 0.1661
# - R5 test NRMSE: 1.0310 (6.2× worse!)
#
# New approach: Release-level split (entire releases to train OR val)
# - ALL releases R1-R11 have Challenge 1 data (total: 4,645 recordings)
# - Train: R11, R2, R3, R4, R7, R8 (3,253 recordings, 70.03%)
# - Val: R1, R10, R6, R9 (1,392 recordings, 29.97%)
# - Test: R5 (held out for competition)
#
# Hypothesis: Better simulates generalization to unseen data batches

# %% imports
from pathlib import Path
import math
import os
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm.auto import tqdm
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
# Removed mixed precision for numerical stability with Mamba2
# from torch.amp import GradScaler, autocast

# Mamba imports
from mamba_ssm import Mamba2

# Braindecode and startkit imports for data loading
from eegdash import EEGChallengeDataset
from eegdash.dataset import EEGChallengeDataset
from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Load environment variables
load_dotenv()

# Suppress verbose warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='eegdash')
warnings.filterwarnings('ignore', category=UserWarning, module='eegdash')

# Global constants
ANCHOR = "stimulus_anchor"  # Event anchor for stimulus-locked windowing

# %% config
class Config:
    # Data paths from environment
    HBN_ROOT = Path(os.getenv("HBN_ROOT", "/media/varun/OS/Users/varun/DATASETS/HBN"))
    CACHE_PATH = Path(os.getenv("CACHE_PATH", "/home/varun/repos/cerebro/cache"))

    # Model architecture
    n_channels = 129
    n_times = 200  # 2.0 seconds at 100 Hz (FIXED: was 250, must match competition)
    d_model = 256
    d_state = 32  # Increased for full dataset
    n_layers = 6   # Deeper model for full dataset
    expand_factor = 2
    conv_size = 4
    dropout = 0.1

    # Spatial processing
    spatial_kernel_size = 3
    n_spatial_filters = 128  # More filters for full dataset

    # Training
    batch_size = 128  # Increased back to 128 for more stable gradients
    learning_rate = 0.0001  # Reduced 5x for Mamba2 SSM stability (was 0.0005)
    weight_decay = 0.00001
    n_epochs = 50  # Fewer epochs needed with more data
    early_stopping_patience = 10
    warmup_epochs = 2
    grad_clip = 5.0  # Relaxed from 1.0 to allow larger gradient updates

    # Data loading
    num_workers = 8  # TODO: Set back to 8 after debugging
    window_len = 2.0  # seconds (FIXED: was 2.5, must be 2.0 to match competition)
    shift_after_stim = 0.5  # seconds after stimulus
    sfreq = 100

    # Validation
    seed = 2025  # Competition year

    # Release-level split (70:30) - optimized split from scripts/optimize_release_splits.py
    # All releases R1-R11 have Challenge 1 data (verified 2025-11-02)
    # Total available: 4,645 recordings (excluding R5 test set)
    train_releases = ["R11", "R2", "R3", "R4", "R7", "R8"]  # 3,253 recordings (70.03%)
    val_releases = ["R1", "R10", "R6", "R9"]  # 1,392 recordings (29.97%)
    test_release = "R5"  # Competition validation set

    # Full dataset (not mini)
    use_mini = False

    # Tracking
    use_wandb = True
    experiment_name = f"mamba_c1_release60_40_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Output
    checkpoint_dir = CACHE_PATH / "mamba_checkpoints" / experiment_name

cfg = Config()
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Set seeds for reproducibility
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

print(f"Configuration:")
print(f"  HBN_ROOT: {cfg.HBN_ROOT}")
print(f"  CACHE_PATH: {cfg.CACHE_PATH}")
print(f"  Train releases (70%): {cfg.train_releases}")
print(f"  Val releases (30%): {cfg.val_releases}")
print(f"  Test release: {cfg.test_release}")
print(f"  Using mini: {cfg.use_mini}")
print(f"  Batch size: {cfg.batch_size}")
print(f"  Model: {cfg.n_layers} layers, d_model={cfg.d_model}")

# %% Pickle Cache Functions for Windowed Datasets
import pickle

def get_dataset_cache_key(releases, window_len, shift, mini):
    """Generate unique cache key for dataset configuration."""
    # Sort releases for consistent key
    releases_str = "_".join(sorted(releases))
    mini_str = "mini" if mini else "full"
    return f"{releases_str}_win{window_len}_shift{shift}_{mini_str}"

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

# %% Load Data with Startkit Approach (Braindecode + Pickle Cache)

# Setup cache directory
WINDOW_CACHE_DIR = cfg.CACHE_PATH / "mamba_c1_windows"
WINDOW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Generate cache keys
train_cache_key = get_dataset_cache_key(cfg.train_releases, cfg.window_len, cfg.shift_after_stim, cfg.use_mini)
val_cache_key = get_dataset_cache_key(cfg.val_releases, cfg.window_len, cfg.shift_after_stim, cfg.use_mini)

print("\n" + "="*60)
print("Checking for cached windowed datasets...")
print("="*60)
print(f"  Train cache key: {train_cache_key}")
print(f"  Val cache key: {val_cache_key}")

# Try to load from cache
train_set_cached = load_windowed_dataset(WINDOW_CACHE_DIR, "train", train_cache_key)
val_set_cached = load_windowed_dataset(WINDOW_CACHE_DIR, "val", val_cache_key)

if train_set_cached is not None and val_set_cached is not None:
    # Cache hit - use cached datasets
    print("\n✅ Using cached datasets (skipping preprocessing + windowing)")
    train_set = train_set_cached
    val_set = val_set_cached

    print(f"\n📊 Dataset statistics (from cache):")
    print(f"  Train windows: {len(train_set)}")
    print(f"  Val windows: {len(val_set)}")
else:
    # Cache miss - need to create datasets
    print("\n⚠️  Cache miss - creating datasets from scratch...")
    print("   (This will be cached for future runs)")
    print("\n" + "="*60)
    print("Loading data with STARTKIT approach (braindecode + pickle)")
    print("="*60)

    # Load dataset - must load each release separately and concatenate
    print("\n⏳ Loading EEGChallengeDataset...")
    all_releases_to_load = cfg.train_releases + cfg.val_releases
    print(f"  Releases: {all_releases_to_load}")

    # Load each release separately (EEGChallengeDataset only accepts single release string)
    release_datasets = []
    for release in all_releases_to_load:
        print(f"  Loading {release}...")
        ds = EEGChallengeDataset(
            task="contrastChangeDetection",
            release=release,  # Single release string
            cache_dir=Path(str(cfg.HBN_ROOT)),
            mini=cfg.use_mini
        )
        release_datasets.append(ds)
        print(f"    → {len(ds.datasets)} recordings")

    # Concatenate all releases into one dataset
    dataset_ccd = BaseConcatDataset(
        [recording for ds in release_datasets for recording in ds.datasets]
    )

    print(f"  Total loaded: {len(dataset_ccd.datasets)} recordings across {len(all_releases_to_load)} releases")

    # Preprocess: annotate trials with targets
    print("\n⏳ Preprocessing: annotating trials with RT targets...")
    transformation_offline = [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus",
            epoch_length=cfg.window_len,
            require_stimulus=True,
            require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    preprocess(dataset_ccd, transformation_offline, n_jobs=8)

    # Create stimulus-locked windows
    print("\n⏳ Creating stimulus-locked windows...")
    # ANCHOR defined globally at top of file

    # Keep only recordings with stimulus anchors
    dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)
    print(f"  Recordings with {ANCHOR}: {len(dataset.datasets)}")

    # Create windows (2.0s windows, 0.5s after stimulus)
    single_windows = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(cfg.shift_after_stim * cfg.sfreq),  # +0.5s
        trial_stop_offset_samples=int((cfg.shift_after_stim + cfg.window_len) * cfg.sfreq),  # +2.5s
        window_size_samples=int(cfg.window_len * cfg.sfreq),  # 2.0s window
        window_stride_samples=cfg.sfreq,
        preload=True,  # Load into memory (pickle cache)
    )

    print(f"  Total windows created: {len(single_windows.datasets)}")

    # Add metadata columns
    single_windows = add_extras_columns(
        single_windows,
        dataset,
        desc=ANCHOR,
        keys=("target", "rt_from_stimulus", "rt_from_trialstart",
              "stimulus_onset", "response_onset", "correct", "response_type")
    )

    # Split by release
    print("\n📊 Splitting by release...")
    meta_information = single_windows.get_metadata()

    train_indices = []
    val_indices = []

    # Get release for each window
    for i in range(len(single_windows.datasets)):
        window_desc = single_windows.datasets[i].description
        # Extract release from description
        # The release is stored in the window's raw metadata
        raw = single_windows.datasets[i].raw
        if hasattr(raw, 'filenames') and raw.filenames:
            # Extract release from filename path (e.g., /path/to/ds005504-bdf/... -> R1)
            # Release mapping: ds005504=R1, ds005505=R2, ..., ds005516=R11
            # Note: ds005513 doesn't exist (skipped in HBN numbering)
            filename = str(raw.filenames[0])  # Convert PosixPath to string
            if 'ds005504' in filename: release = 'R1'
            elif 'ds005505' in filename: release = 'R2'
            elif 'ds005506' in filename: release = 'R3'
            elif 'ds005507' in filename: release = 'R4'
            elif 'ds005508' in filename: release = 'R5'
            elif 'ds005509' in filename: release = 'R6'
            elif 'ds005510' in filename: release = 'R7'
            elif 'ds005511' in filename: release = 'R8'
            elif 'ds005512' in filename: release = 'R9'
            elif 'ds005514' in filename: release = 'R9'  # R9 also uses ds005514
            elif 'ds005515' in filename: release = 'R10'  # FIXED: Added missing R10 mapping
            elif 'ds005516' in filename: release = 'R11'
            else:
                print(f"  ⚠️  Unknown dataset in filename: {filename}")
                continue

            if release in cfg.train_releases:
                train_indices.append(i)
            elif release in cfg.val_releases:
                val_indices.append(i)

    # Create subsets
    train_set = BaseConcatDataset([single_windows.datasets[i] for i in train_indices])
    val_set = BaseConcatDataset([single_windows.datasets[i] for i in val_indices])

    print(f"\n📊 Dataset statistics:")
    print(f"  Train windows: {len(train_set)} from {len(cfg.train_releases)} releases {cfg.train_releases}")
    print(f"  Val windows: {len(val_set)} from {len(cfg.val_releases)} releases {cfg.val_releases}")
    print(f"  Release-level split ensures no subject overlap between train/val")
    print(f"  Pickle cache: preloaded into memory ⚡")

    # Save datasets to cache for future runs
    print("\n" + "="*60)
    print("Saving datasets to cache for future runs...")
    print("="*60)

    save_windowed_dataset(train_set, WINDOW_CACHE_DIR, "train", train_cache_key)
    save_windowed_dataset(val_set, WINDOW_CACHE_DIR, "val", val_cache_key)

    print("\n✅ Datasets cached - next run will be much faster!")

# Create DataLoaders (works with both cached and freshly created datasets)
train_loader = DataLoader(
    train_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True
)
val_loader = DataLoader(
    val_set,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True
)

print(f"  Total steps per epoch: {len(train_loader)}")

# %% [markdown]
# ## Model Architecture
#
# ```mermaid
# graph TB
#     subgraph Input["Input Layer"]
#         A["EEG Signal<br/>(batch, 129 channels, 200 timesteps)<br/>2.0s @ 100Hz"]
#     end
#
#     subgraph SpatialEncoder["Spatial Channel Encoder"]
#         B["Channel Embeddings<br/>Embedding(129, 256)"]
#         C["Spatial Conv1D<br/>129 → 128 filters<br/>kernel=3, padding=1"]
#         D["Channel Mixer<br/>Linear(128 → 256)"]
#         E["LayerNorm + Dropout<br/>(0.1)"]
#     end
#
#     subgraph MambaBlocks["Mamba2 Sequence Processing<br/>6 Layers"]
#         F1["Mamba2 Block 1<br/>d_model=256, d_state=32<br/>expand=2, d_conv=4"]
#         G1["LayerNorm + Residual"]
#         F2["Mamba2 Block 2<br/>d_model=256, d_state=32"]
#         G2["LayerNorm + Residual"]
#         F3["..."]
#         F6["Mamba2 Block 6<br/>d_model=256, d_state=32"]
#         G6["LayerNorm + Residual"]
#     end
#
#     subgraph OutputHead["Regression Head"]
#         H["Adaptive AvgPool1D<br/>(batch, 256, 200) → (batch, 256, 1)"]
#         I["Flatten<br/>(batch, 256)"]
#         J["Linear(256 → 128)<br/>+ GELU + Dropout(0.1)"]
#         K["Linear(128 → 1)<br/>Response Time"]
#     end
#
#     subgraph Output["Output"]
#         L["Predicted RT<br/>(batch, 1)"]
#     end
#
#     A --> C
#     B -.-> D
#     C --> D
#     D --> E
#     E --> F1
#
#     F1 --> G1
#     G1 --> F2
#     F2 --> G2
#     G2 --> F3
#     F3 --> F6
#     F6 --> G6
#
#     G6 --> H
#     H --> I
#     I --> J
#     J --> K
#     K --> L
#
#     classDef inputStyle fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
#     classDef spatialStyle fill:#fff9c4,stroke:#f57f17,stroke-width:2px
#     classDef mambaStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
#     classDef headStyle fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
#     classDef outputStyle fill:#ffccbc,stroke:#e64a19,stroke-width:2px
#
#     class A inputStyle
#     class B,C,D,E spatialStyle
#     class F1,F2,F3,F6,G1,G2,G6 mambaStyle
#     class H,I,J,K headStyle
#     class L outputStyle
# ```
#
# ### Architecture Summary
#
# | Component | Input Shape | Output Shape | Parameters |
# |-----------|-------------|--------------|------------|
# | **Spatial Encoder** | (B, 129, 200) | (B, 200, 256) | ~50K |
# | **6× Mamba2 Blocks** | (B, 200, 256) | (B, 200, 256) | ~2.5M |
# | **Pooler + Head** | (B, 200, 256) | (B, 1) | ~33K |
# | **Total** | | | **~2.6M params** |
#
# **Key Design Choices:**
# - **Spatial Encoder**: Learns channel relationships before temporal processing
# - **Mamba2 SSM**: Efficient long-range dependencies (O(N) complexity vs O(N²) for attention)
# - **Residual Connections**: Stable training for deep networks (6 layers)
# - **Adaptive Pooling**: Aggregates temporal information for regression

# %% Model Definition
class ImprovedSpatialEncoder(nn.Module):
    """Spatial encoder with attention to prevent representation collapse."""

    def __init__(self, n_channels, d_model, n_heads=8):
        super().__init__()

        # Simple projection from channels to model dimension
        self.input_proj = nn.Conv1d(n_channels, d_model, kernel_size=1)

        # Spatial attention - CRITICAL for creating diverse representations
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1
        )

        # Layer norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        # Optional: Feedforward for additional processing
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_times)
        Returns:
            (batch, n_times, d_model) with diverse spatial representations
        """
        # Project channels to model dimension
        x = self.input_proj(x)  # (batch, d_model, n_times)
        x = x.transpose(1, 2)  # (batch, n_times, d_model)

        # Self-attention across time steps (captures spatial patterns)
        # This forces different representations for different inputs
        x_att, attn_weights = self.spatial_attention(x, x, x)
        x = self.norm1(x + self.dropout(x_att))  # Residual connection

        # Feedforward for additional non-linearity
        x_ff = self.ff(x)
        x = self.norm2(x + self.dropout(x_ff))

        return x


class MambaEEGModel(nn.Module):
    """Mamba2-based model for EEG response time prediction."""

    def __init__(self, cfg):
        super().__init__()

        # Improved spatial encoder with attention
        self.spatial_encoder = ImprovedSpatialEncoder(
            n_channels=cfg.n_channels,
            d_model=cfg.d_model,
            n_heads=8  # 8 heads for 256 dim (32 dim per head)
        )

        # Mamba2 blocks with residual connections
        self.mamba_blocks = nn.ModuleList([
            Mamba2(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                expand=cfg.expand_factor,
                d_conv=cfg.conv_size,
                headdim=32,
            )
            for _ in range(cfg.n_layers)
        ])

        # Layer norms - Pre-norm residual pattern (official Mamba pattern)
        self.norm_in = nn.LayerNorm(cfg.d_model)  # Initial norm before first Mamba block
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)
        ])

        # Additional spatial attention after Mamba (helps prevent collapse)
        self.post_mamba_attention = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        self.post_attention_norm = nn.LayerNorm(cfg.d_model)

        # Output head
        self.pooler = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1)
        )

        # For diagnostics
        self.debug_stats = {}

    def forward(self, x, return_debug=False):
        """
        Args:
            x: (batch, n_channels, n_times)
            return_debug: If True, store intermediate representations for debugging
        Returns:
            (batch, 1) response time predictions
        """
        # Encode spatial relationships with attention
        x = self.spatial_encoder(x)  # (batch, n_times, d_model)

        if return_debug:
            self.debug_stats['after_spatial'] = x.std().item()
            self.debug_stats['after_spatial_mean'] = x.mean().item()

        # Initial normalization before first Mamba block
        x = self.norm_in(x)

        # Pre-norm residual: Add → LN → Mamba (official Mamba pattern for stability)
        # Prevents SSM recurrent dynamics from becoming unstable
        for i, (mamba, norm) in enumerate(zip(self.mamba_blocks, self.layer_norms)):
            residual = x
            x = norm(x)       # Normalize BEFORE Mamba (pre-norm)
            x = mamba(x)      # Apply Mamba block
            x = x + residual  # Add residual AFTER Mamba

            if return_debug and i == len(self.mamba_blocks) - 1:
                self.debug_stats['after_mamba'] = x.std().item()
                self.debug_stats['after_mamba_mean'] = x.mean().item()

        # Additional spatial attention for global context (prevents collapse)
        x_att, attn_weights = self.post_mamba_attention(x, x, x)
        x = self.post_attention_norm(x + x_att)  # Residual connection

        if return_debug:
            self.debug_stats['after_post_attention'] = x.std().item()
            self.debug_stats['attention_std'] = attn_weights.std().item() if attn_weights is not None else 0

        # Pool and predict
        x = x.transpose(1, 2)  # (batch, d_model, n_times)
        x_pooled = self.pooler(x)

        if return_debug:
            self.debug_stats['after_pooling'] = x_pooled.std().item()
            # Check diversity across batch
            batch_diversity = x_pooled.std(dim=0).mean().item()
            self.debug_stats['batch_diversity'] = batch_diversity

        output = self.regression_head(x_pooled)

        return output

# %% Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️ Using device: {device}")

model = MambaEEGModel(cfg).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Model parameters: {total_params:,} (trainable: {trainable_params:,})")

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
# Switch to ReduceLROnPlateau for adaptive learning rate reduction
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',  # Minimize validation loss
    factor=0.5,  # Halve LR on plateau
    patience=5,  # Wait 5 epochs before reducing
    min_lr=1e-6  # Don't go below this
)
# Removed GradScaler - using full FP32 precision for numerical stability
# scaler = GradScaler('cuda')
print(f"🔧 Using full FP32 precision (no mixed precision) for Mamba2 stability")

# Initialize wandb
if cfg.use_wandb:
    wandb.init(
        project="cerebro-mamba",
        name=cfg.experiment_name,
        config=vars(cfg),
        tags=["mamba2", "challenge1", "full_hbn"]
    )

# %% Training utilities
def calculate_nrmse(y_true, y_pred):
    """Calculate Normalized RMSE as in competition."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    # CRITICAL FIX: Use std() not (max-min) to match competition metric
    y_std = y_true.std()
    nrmse = rmse / y_std if y_std > 0 else rmse
    return nrmse

def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    losses = []
    predictions = []
    targets = []
    grad_clips = []  # Track when gradients are clipped

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} Training",
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')

    for batch_idx, batch in enumerate(pbar):
        X, y = batch[0], batch[1]  # Braindecode returns (X, y, i) tuple
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        # Squeeze targets to 1D if needed (braindecode returns [batch, 1])
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)  # (batch, 1) -> (batch)

        # Forward pass (full FP32 precision)
        y_pred = model(X).squeeze(-1)  # (batch, 1) -> (batch)
        mae_loss = F.l1_loss(y_pred, y)  # L1 loss (MAE) instead of MSE

        # Variance penalty to discourage constant predictions
        pred_std = y_pred.std()
        target_std = 0.4  # Target standard deviation (based on actual target std ~0.412)
        variance_penalty = 0.0

        # Apply penalty only if predictions have low variance
        if pred_std < 0.05:  # If std is less than 5% of typical target range
            variance_penalty = 0.1 * (target_std - pred_std) ** 2
            if batch_idx % 50 == 0:
                print(f"    Variance penalty active: pred_std={pred_std:.4f}, penalty={variance_penalty:.4f}")

        # Combined loss
        loss = mae_loss + variance_penalty

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping with tracking
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        grad_clipped = total_norm > cfg.grad_clip
        grad_clips.append(grad_clipped)

        # Log gradient norm and predictions for first epoch
        if epoch == 0 and batch_idx < 10:
            with torch.no_grad():
                pred_vals = y_pred.detach().cpu().numpy()[:5]  # First 5 predictions
                target_vals = y.detach().cpu().numpy()[:5]  # First 5 targets
            print(f"    Batch {batch_idx}: grad_norm={total_norm:.4f} (clipped={grad_clipped})")
            print(f"      First 5 preds: {pred_vals}")
            print(f"      First 5 targets: {target_vals}")
            print(f"      Pred std: {y_pred.std().item():.4f}, MAE: {mae_loss.item():.4f}")
        elif grad_clipped and batch_idx % 50 == 0:
            print(f"    Batch {batch_idx}: Large gradient! norm={total_norm:.4f} > {cfg.grad_clip}")

        optimizer.step()

        # ReduceLROnPlateau steps on validation, not per batch
        # scheduler.step()  # REMOVED - will step after validation

        # Track
        current_loss = loss.item()
        losses.append(current_loss)
        with torch.no_grad():
            predictions.extend(y_pred.detach().cpu().numpy())
            targets.extend(y.detach().cpu().numpy())

        # Update progress with TRANSPARENT metrics
        current_lr = optimizer.param_groups[0]['lr']  # Get LR from optimizer for ReduceLROnPlateau
        avg_loss = np.mean(losses)
        recent_avg = np.mean(losses[-100:]) if len(losses) > 100 else avg_loss

        # Detailed progress every 10 batches
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "avg": f"{recent_avg:.4f}",
                "lr": f"{current_lr:.2e}",
                "clip": f"{sum(grad_clips[-10:])}/10" if len(grad_clips) >= 10 else f"{sum(grad_clips)}/{len(grad_clips)}"
            })

        # Detailed batch reporting every 50 batches
        if batch_idx % 200 == 0 and batch_idx > 0:
            clip_rate = sum(grad_clips) / len(grad_clips) * 100
            loss_std = np.std(losses[-50:]) if len(losses) >= 50 else np.std(losses)
            print(f"  Batch {batch_idx:4d}/{len(loader)} | "
                  f"Loss: {current_loss:.4f} | "
                  f"Avg50: {np.mean(losses[-50:]):.4f} (±{loss_std:.4f}) | "
                  f"LR: {current_lr:.2e} | "
                  f"GradClip: {clip_rate:.1f}%")

    # Calculate metrics
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    nrmse = calculate_nrmse(targets, predictions)

    # Analyze training predictions
    pred_std = predictions.std()
    target_std = targets.std()
    print(f"\n  📊 Training predictions analysis:")
    print(f"     Predictions: mean={predictions.mean():.4f}, std={pred_std:.4f}")
    print(f"     Targets:     mean={targets.mean():.4f}, std={target_std:.4f}")
    print(f"     Std ratio:   {pred_std/target_std:.2%} (100% = perfect variance match)")

    # Print gradient clipping statistics
    total_clips = sum(grad_clips)
    clip_percentage = (total_clips / len(grad_clips)) * 100 if grad_clips else 0
    print(f"  📊 Epoch {epoch+1} training stats:")
    print(f"     Batches with clipped gradients: {total_clips}/{len(grad_clips)} ({clip_percentage:.1f}%)")
    print(f"     Final loss: {losses[-1]:.4f} | Epoch avg: {np.mean(losses):.4f} (±{np.std(losses):.4f})")

    return np.mean(losses), nrmse

def validate(model, loader, device):
    """Validate the model."""
    model.eval()
    losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validation")):
            X, y = batch[0], batch[1]  # Braindecode returns (X, y, i) tuple
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            # Squeeze targets to 1D if needed (braindecode returns [batch, 1])
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.squeeze(1)  # (batch, 1) -> (batch)

            # DIAGNOSTIC: Check intermediate embeddings for first batch
            if batch_idx == 0:
                print(f"\n  🔍 Diagnostic: Checking representation diversity...")

                # Get debug stats from model
                _ = model(X, return_debug=True)
                debug_stats = model.debug_stats

                print(f"\n     📊 Representation std at each stage:")
                print(f"        After spatial encoder: {debug_stats.get('after_spatial', 0):.6f}")
                print(f"        After Mamba blocks: {debug_stats.get('after_mamba', 0):.6f}")
                print(f"        After post-attention: {debug_stats.get('after_post_attention', 0):.6f}")
                print(f"        After pooling: {debug_stats.get('after_pooling', 0):.6f}")
                print(f"        Batch diversity: {debug_stats.get('batch_diversity', 0):.6f}")
                print(f"        Attention std: {debug_stats.get('attention_std', 0):.6f}")

                # Check for collapse
                if debug_stats.get('after_spatial', 0) < 0.01:
                    print(f"        ⚠️  WARNING: Spatial encoder producing near-constant outputs!")
                if debug_stats.get('batch_diversity', 0) < 0.01:
                    print(f"        ⚠️  WARNING: All samples have similar representations!")
                if debug_stats.get('after_spatial', 0) > 0.1:
                    print(f"        ✅ Good diversity in spatial representations!")
                if debug_stats.get('batch_diversity', 0) > 0.05:
                    print(f"        ✅ Good diversity across batch samples!")

            # Full FP32 precision (no autocast)
            y_pred = model(X).squeeze(-1)  # (batch, 1) -> (batch)
            loss = F.l1_loss(y_pred, y)  # L1 loss (MAE) instead of MSE

            losses.append(loss.item())
            predictions.extend(y_pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    # Calculate metrics
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    nrmse = calculate_nrmse(targets, predictions)

    # Prediction analysis to detect constant outputs
    print(f"\n  Prediction Analysis:")
    print(f"    Predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}")
    print(f"    Targets:     mean={targets.mean():.4f}, std={targets.std():.4f}")
    print(f"    Pred range:  [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"    Target range:[{targets.min():.4f}, {targets.max():.4f}]")
    print(f"    Pred variance: {predictions.var():.6f}")

    # Warning if predictions have very low variance (likely predicting constants)
    if predictions.std() < 0.01:
        print(f"    ⚠️  WARNING: Predictions have very low variance! Model may be stuck predicting constants.")

    return np.mean(losses), nrmse

# %% Training loop
print("\n" + "="*50)
print("🚀 Starting training on full HBN dataset...")
print("="*50)

# Announce training configuration transparently
print("\n📋 Training Configuration:")
print(f"  Model: MambaEEG with {total_params:,} parameters")
print(f"  Optimizer: AdamW (lr={cfg.learning_rate:.2e}, weight_decay={cfg.weight_decay:.2e})")
print(f"  Scheduler: ReduceLROnPlateau")
print(f"    - Factor: 0.5 (halves LR on plateau)")
print(f"    - Patience: 5 epochs before reduction")
print(f"    - Min LR: 1e-6")
print(f"  Precision: Full FP32 (no mixed precision)")
print(f"  Gradient Clipping: max_norm={cfg.grad_clip}")
print(f"  Early Stopping: patience={cfg.early_stopping_patience} epochs")
print(f"  Batch Size: {cfg.batch_size} (reduced for better exploration)")
print(f"  Total steps per epoch: {len(train_loader)}")

# Verify batch structure and targets once before training
print("\n🔍 Verifying data loading and targets...")
sample_batch = next(iter(train_loader))
if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
    X_sample, y_sample = sample_batch[0], sample_batch[1]  # Braindecode returns (X, y, i)
    print(f"✅ Batch format: (X: {X_sample.shape}, y: {y_sample.shape})")
    print(f"  Target stats: mean={y_sample.mean():.4f}, std={y_sample.std():.4f}")
    print(f"  Target range: [{y_sample.min():.4f}, {y_sample.max():.4f}]")
    print(f"  Non-zero targets: {(y_sample != 0).sum().item()}/{len(y_sample)}")

    # Check if targets look reasonable
    if y_sample.std() < 0.01:
        print("⚠️  WARNING: Target values have very low variance!")
    if y_sample.min() == y_sample.max():
        print("⚠️  WARNING: All targets in batch are identical!")
else:
    raise ValueError(f"❌ Unexpected batch format: {type(sample_batch)} with length {len(sample_batch)}")

best_val_nrmse = float('inf')
patience_counter = 0
training_history = {
    'train_loss': [], 'val_loss': [],
    'train_nrmse': [], 'val_nrmse': []
}

for epoch in range(cfg.n_epochs):
    print(f"\n📅 Epoch {epoch+1}/{cfg.n_epochs}")
    print("-" * 40)

    # Show current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"📊 Current Learning Rate: {current_lr:.2e}")

    # Train
    train_loss, train_nrmse = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)

    # Validate
    val_loss, val_nrmse = validate(model, val_loader, device)

    # Track history
    training_history['train_loss'].append(train_loss)
    training_history['val_loss'].append(val_loss)
    training_history['train_nrmse'].append(train_nrmse)
    training_history['val_nrmse'].append(val_nrmse)

    print(f"\n📈 Train - Loss: {train_loss:.4f}, NRMSE: {train_nrmse:.4f}")
    print(f"📊 Val   - Loss: {val_loss:.4f}, NRMSE: {val_nrmse:.4f}")

    # Step the scheduler based on validation NRMSE (ReduceLROnPlateau)
    scheduler.step(val_nrmse)

    # Log to wandb
    if cfg.use_wandb:
        wandb.log({
            "train_loss": train_loss,
            "train_nrmse": train_nrmse,
            "val_loss": val_loss,
            "val_nrmse": val_nrmse,
            "lr": optimizer.param_groups[0]['lr'],  # Get LR from optimizer
            "epoch": epoch + 1
        })

    # Save best model and track patience
    if val_nrmse < best_val_nrmse:
        improvement = best_val_nrmse - val_nrmse
        best_val_nrmse = val_nrmse
        patience_counter = 0

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_nrmse': val_nrmse,
            'config': vars(cfg),
            'history': training_history
        }
        torch.save(checkpoint, cfg.checkpoint_dir / "best_model.pt")
        print(f"✅ NEW BEST! Val NRMSE: {val_nrmse:.4f} (improved by {improvement:.4f})")
    else:
        patience_counter += 1
        degradation = val_nrmse - best_val_nrmse
        print(f"⏳ Patience: {patience_counter}/{cfg.early_stopping_patience} "
              f"(best: {best_val_nrmse:.4f}, current worse by: {degradation:.4f})")

    # Report progress to targets
    target_1_0_gap = val_nrmse - 1.0
    target_0_95_gap = val_nrmse - 0.95

    print(f"🎯 Progress to targets:")
    print(f"   Target 1.00: {'✓ ACHIEVED' if target_1_0_gap < 0 else f'{abs(target_1_0_gap):.4f} away'}")
    print(f"   Target 0.95: {'✓ ACHIEVED' if target_0_95_gap < 0 else f'{abs(target_0_95_gap):.4f} away'}")

    # Early stopping check
    if patience_counter >= cfg.early_stopping_patience:
        print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
        print(f"   No improvement for {cfg.early_stopping_patience} epochs")
        print(f"   Best Val NRMSE: {best_val_nrmse:.4f}")
        break

# %% Final results
print("\n" + "="*50)
print("🏁 Training Complete!")
print("="*50)
print(f"Best Val NRMSE: {best_val_nrmse:.4f}")
print(f"Total epochs trained: {len(training_history['train_loss'])}")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
axes[0].plot(training_history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(training_history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training Progress - Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# NRMSE curves
axes[1].plot(training_history['train_nrmse'], label='Train NRMSE', linewidth=2)
axes[1].plot(training_history['val_nrmse'], label='Val NRMSE', linewidth=2)
axes[1].axhline(y=1.0, color='r', linestyle='--', label='Target (1.0)', alpha=0.5)
axes[1].axhline(y=0.95, color='g', linestyle='--', label='Excellent (0.95)', alpha=0.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('NRMSE')
axes[1].set_title('Training Progress - NRMSE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(cfg.checkpoint_dir / "training_curves.png", dpi=150)
plt.show()

# Save final metrics
final_metrics = {
    'best_val_nrmse': best_val_nrmse,
    'final_train_nrmse': training_history['train_nrmse'][-1],
    'final_val_nrmse': training_history['val_nrmse'][-1],
    'total_epochs': len(training_history['train_loss']),
    'model_params': total_params,
    'config': vars(cfg)
}

import json
with open(cfg.checkpoint_dir / "metrics.json", 'w') as f:
    json.dump(final_metrics, f, indent=2, default=str)

print(f"\n📁 All results saved to: {cfg.checkpoint_dir}")
print("\n✨ Challenge 1 Full Dataset Training Complete!")

# %% ============================================================
# Export Model to TorchScript (for submission)
# ============================================================
print("\n" + "="*60)
print("Exporting best model to TorchScript for submission")
print("="*60)

# Load best checkpoint
checkpoint_path = cfg.checkpoint_dir / "best_model.pt"
# PyTorch 2.6 requires weights_only=False for numpy arrays in checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
best_model = MambaEEGModel(cfg).to(device)
best_model.load_state_dict(checkpoint["model_state_dict"])
best_model.eval()
print(f"✓ Loaded best checkpoint from epoch {checkpoint['epoch']}")

# Trace with example input (B, C, T) = (1, 129, 200)
# Shape: (batch, channels, timesteps) for 2.0s at 100Hz (FIXED: was 250)
example_input = torch.randn(1, cfg.n_channels, cfg.n_times).to(device)

print(f"Tracing model with input shape: {example_input.shape}")
scripted_model = torch.jit.trace(best_model, example_input)
scripted_model = torch.jit.optimize_for_inference(scripted_model)

# Save for submission
output_path = cfg.checkpoint_dir / "mamba_c1_model.pt"
scripted_model.save(str(output_path))
print(f"✓ Saved TorchScript model to {output_path}")
print(f"  Model size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# %% ============================================================
# Evaluate on R5 Test Set (Competition Validation)
# ============================================================
print("\n" + "="*60)
print("Evaluating on R5 Test Set (Competition Metric)")
print("="*60)

# Load TorchScript model (exactly like submission.py does)
test_model = torch.jit.load(str(output_path), map_location=device)
test_model.eval()
print(f"✓ Loaded TorchScript model from {output_path}")

# Load R5 test data using startkit approach (braindecode + preload)
print("\nLoading R5 test data with braindecode...")

# Load R5 dataset
r5_dataset = EEGChallengeDataset(
    task="contrastChangeDetection",
    release=cfg.test_release,  # "R5"
    cache_dir=Path(str(cfg.HBN_ROOT)),
    mini=cfg.use_mini
)

# Preprocess: annotate trials
transformation_offline = [
    Preprocessor(
        annotate_trials_with_target,
        target_field="rt_from_stimulus",
        epoch_length=cfg.window_len,
        require_stimulus=True,
        require_response=True,
        apply_on_array=False,
    ),
    Preprocessor(add_aux_anchors, apply_on_array=False),
]
preprocess(r5_dataset, transformation_offline, n_jobs=8)

# Create windows
r5_dataset_filtered = keep_only_recordings_with(ANCHOR, r5_dataset)
r5_windows = create_windows_from_events(
    r5_dataset_filtered,
    mapping={ANCHOR: 0},
    trial_start_offset_samples=int(cfg.shift_after_stim * cfg.sfreq),
    trial_stop_offset_samples=int((cfg.shift_after_stim + cfg.window_len) * cfg.sfreq),
    window_size_samples=int(cfg.window_len * cfg.sfreq),
    window_stride_samples=cfg.sfreq,
    preload=True,  # Pickle cache in memory
)

# Add metadata
r5_windows = add_extras_columns(
    r5_windows,
    r5_dataset_filtered,
    desc=ANCHOR,
    keys=("target", "rt_from_stimulus", "rt_from_trialstart",
          "stimulus_onset", "response_onset", "correct", "response_type")
)

print(f"  R5 recordings: {len(r5_dataset_filtered.datasets)}")
print(f"  R5 windows: {len(r5_windows):,}")

# Create test dataloader
r5_loader = DataLoader(
    r5_windows,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True
)

# Run inference (exactly like local_scoring.py line 207-218)
print("\nRunning inference on R5...")
y_preds = []
y_trues = []

with torch.inference_mode():
    for batch in tqdm(r5_loader, desc="R5 Inference"):
        X, y = batch
        X = X.to(dtype=torch.float32, device=device)
        y = y.to(dtype=torch.float32, device=device)

        # Forward pass
        y_pred = test_model(X)

        # Collect predictions
        y_preds.extend(y_pred.cpu().numpy().flatten())
        y_trues.extend(y.cpu().numpy().flatten())

y_preds = np.array(y_preds)
y_trues = np.array(y_trues)

print(f"\nCollected {len(y_preds)} predictions")

# Calculate metrics (exactly like local_scoring.py line 76-91)
def nrmse_metric(y_true, y_pred):
    """Normalized RMSE (competition metric)"""
    from sklearn.metrics import root_mean_squared_error as rmse
    return rmse(y_true, y_pred) / y_true.std()

def r2_score_neg(y_true, y_pred):
    """Negative R² (for info only)"""
    from sklearn.metrics import r2_score
    return -r2_score(y_true, y_pred)

r5_rmse = np.sqrt(((y_trues - y_preds) ** 2).mean())
r5_nrmse = nrmse_metric(y_trues, y_preds)
r5_r2 = r2_score_neg(y_trues, y_preds)

# Print results (matching local_scoring.py format)
print("\n" + "="*60)
print("Challenge 1 R5 Test Results:")
print("="*60)
print(f"RMSE: {r5_rmse:.4f}")
print(f"NRMSE: {r5_nrmse:.4f}  (competition metric - minimize this)")
print(f"R^2: {r5_r2:.4f} (for information only, not used in scoring)")
print("="*60)

# Log to wandb
if wandb.run is not None:
    wandb.log({
        "r5_rmse": r5_rmse,
        "r5_nrmse": r5_nrmse,
        "r5_r2": r5_r2,
    })

# Save predictions for submission validation
predictions_path = cfg.checkpoint_dir / "r5_predictions.npz"
np.savez(
    predictions_path,
    y_true=y_trues,
    y_pred=y_preds,
    # Extract subject IDs from windows metadata (sample only, for size limit)
    recording_ids=[w.description.get("subject", "unknown") for w in r5_windows.datasets[:100]]
)
print(f"\n✓ Saved predictions to {predictions_path}")
print(f"\n🏆 Ready for submission! TorchScript model: {output_path}")