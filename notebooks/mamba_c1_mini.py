# %% [markdown]
# # Challenge 1: Mini Dataset Training with Mamba2 (Debug Version)
#
# **Starting with R1 only for debugging**
#
# This is a minimal version to debug the data loading issues before running the full dataset.

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
from torch.cuda.amp import GradScaler, autocast

# Mamba imports
from mamba_ssm import Mamba2

# Cerebro infrastructure
from cerebro.data.tasks.challenge1 import Challenge1Task
from cerebro.data.unified_cache import UniversalCacheManager
from eegdash import EEGChallengeDataset
from braindecode.datasets import BaseConcatDataset, BaseDataset

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Load environment variables
load_dotenv()

# %% config
class Config:
    # Data paths from environment
    HBN_ROOT = Path(os.getenv("HBN_ROOT", "/media/varun/OS/Users/varun/DATASETS/HBN"))
    CACHE_PATH = Path(os.getenv("CACHE_PATH", "/home/varun/repos/cerebro/cache"))

    # Model architecture (smaller for debugging)
    n_channels = 129
    n_times = 250  # 2.5 seconds at 100 Hz
    d_model = 128  # Smaller for debugging
    d_state = 16
    n_layers = 2   # Fewer layers for debugging
    expand_factor = 2
    conv_size = 4
    dropout = 0.1

    # Spatial processing
    spatial_kernel_size = 3
    n_spatial_filters = 64

    # Training (smaller for debugging)
    batch_size = 32  # Smaller batch for debugging
    learning_rate = 0.001
    weight_decay = 0.00001
    n_epochs = 2  # Just 2 epochs for debugging
    early_stopping_patience = 10
    warmup_epochs = 0
    grad_clip = 1.0

    # Data loading
    num_workers = 0  # Set to 0 for debugging
    window_len = 2.5  # seconds
    shift_after_stim = 0.5  # seconds after stimulus
    sfreq = 100

    # Validation
    seed = 2025

    # MINI DEBUG: Just use R1
    train_releases = ["R1"]  # Just R1 for debugging
    val_releases = ["R1"]     # Same release for debugging (will split by recordings)
    test_release = "R5"       # Not used in mini debug

    # Mini dataset for faster debugging
    use_mini = True

    # Tracking
    use_wandb = False  # Disable for debugging
    experiment_name = f"mamba_c1_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
print(f"  Train releases: {cfg.train_releases}")
print(f"  Val releases: {cfg.val_releases}")
print(f"  Using mini: {cfg.use_mini}")
print(f"  Batch size: {cfg.batch_size}")
print(f"  Model: {cfg.n_layers} layers, d_model={cfg.d_model}")

# %% Load Data with Debug Info
print("\n" + "="*60)
print("Loading MINI data for debugging")
print("="*60)

# Initialize unified cache manager
cache = UniversalCacheManager(
    cache_root=str(cfg.CACHE_PATH / "unified_challenge1"),
    data_dir=str(cfg.HBN_ROOT),
    preprocessing_params={
        "sfreq": cfg.sfreq,
        "bandpass": None,  # Already filtered
        "n_channels": cfg.n_channels,
        "standardize": False,
    }
)

# Build raw cache if needed
print(f"\n⏳ Building/loading raw cache...")
cache.build_raw(
    dataset="hbn",
    releases=cfg.train_releases,  # Just R1
    tasks=["contrastChangeDetection"],
    mini=cfg.use_mini
)

# Query raw cache
print(f"\n📊 Querying raw cache...")
all_recordings = cache.query_raw(
    dataset="hbn",
    releases=cfg.train_releases,  # Just R1
    tasks=["contrastChangeDetection"],
    mini=cfg.use_mini
)

print(f"  Total recordings: {len(all_recordings)}")

# Get event-locked windowed datasets
print(f"\n⏳ Loading stimulus-locked windows...")

# For Challenge 1, we use EVENT-LOCKED windowing
event_config = {
    'event_type': 'stimulus_anchor',
    'shift_after_event': cfg.shift_after_stim,  # 0.5s after stimulus
    'target_field': 'rt_from_stimulus'  # Extract RT as target
}

# Build window cache
all_windows = cache.get_windowed_dataset(
    recordings=all_recordings,
    window_len_s=cfg.window_len,
    stride_s=None,  # Event-locked mode
    event_config=event_config,
    mode='train'
)

print(f"  Total windows created: {len(all_windows):,}")

# Debug: Check metadata structure
print(f"\n🔍 DEBUG - Metadata info:")
print(f"  Metadata shape: {all_windows.metadata.shape}")
print(f"  Metadata columns: {list(all_windows.metadata.columns)}")
print(f"  First 5 rows:")
print(all_windows.metadata.head())

# Check if target column exists and has values
if 'target' in all_windows.metadata.columns:
    print(f"\n  Target column found!")
    print(f"  Target stats:")
    print(f"    Non-null: {all_windows.metadata['target'].notna().sum()}")
    print(f"    Null: {all_windows.metadata['target'].isna().sum()}")
    print(f"    Min: {all_windows.metadata['target'].min():.3f}")
    print(f"    Max: {all_windows.metadata['target'].max():.3f}")
    print(f"    Mean: {all_windows.metadata['target'].mean():.3f}")
else:
    print(f"\n  ⚠️ WARNING: No 'target' column found in metadata!")

# Split data for train/val (by recording to avoid leakage)
unique_recordings = all_windows.metadata['recording_id'].unique()
n_recordings = len(unique_recordings)
n_train = int(0.8 * n_recordings)

train_recordings = unique_recordings[:n_train]
val_recordings = unique_recordings[n_train:]

print(f"\n📊 Splitting data by recording:")
print(f"  Total recordings: {n_recordings}")
print(f"  Train recordings: {len(train_recordings)}")
print(f"  Val recordings: {len(val_recordings)}")

# Create train/val metadata
train_metadata = all_windows.metadata[
    all_windows.metadata['recording_id'].isin(train_recordings)
].reset_index(drop=True)

val_metadata = all_windows.metadata[
    all_windows.metadata['recording_id'].isin(val_recordings)
].reset_index(drop=True)

# Create datasets with split metadata
from cerebro.data.unified_cache.lazy_dataset import MemmapWindowDataset

train_windows = MemmapWindowDataset(
    memmap_path=all_windows.memmap_path,
    metadata=train_metadata,
    crop_len_s=None,  # No cropping for debugging
    sfreq=cfg.sfreq,
    mode='train'
)

val_windows = MemmapWindowDataset(
    memmap_path=all_windows.memmap_path,
    metadata=val_metadata,
    crop_len_s=None,
    sfreq=cfg.sfreq,
    mode='val'
)

print(f"\n📊 Dataset statistics:")
print(f"  Train windows: {len(train_windows):,}")
print(f"  Val windows: {len(val_windows):,}")

# Test loading a single item
print(f"\n🔍 Testing single item from train dataset...")
sample = train_windows[0]
if isinstance(sample, tuple):
    x, y = sample
    print(f"  ✅ Returns tuple: (X, y)")
    print(f"  X shape: {x.shape}")
    print(f"  y value: {y.item():.3f}")
else:
    print(f"  ⚠️ Returns single tensor (no target)!")
    print(f"  Shape: {sample.shape}")

# Create DataLoaders
train_loader = DataLoader(
    train_windows,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=False  # Disable for debugging
)
val_loader = DataLoader(
    val_windows,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=False
)

# Test DataLoader
print(f"\n🔍 Testing DataLoader...")
for batch_idx, batch in enumerate(train_loader):
    print(f"  Batch {batch_idx} structure:")

    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        X, y = batch
        print(f"    ✅ Returns (X, y) tuple")
        print(f"    X shape: {X.shape}")
        print(f"    y shape: {y.shape}")
        print(f"    X dtype: {X.dtype}")
        print(f"    y dtype: {y.dtype}")
    elif isinstance(batch, torch.Tensor):
        print(f"    ⚠️ Returns single tensor!")
        print(f"    Shape: {batch.shape}")
        print(f"    Dtype: {batch.dtype}")
    else:
        print(f"    ❌ Unexpected structure: {type(batch)}")

    if batch_idx >= 2:  # Test first 3 batches
        break

# %% Model Definition
class SpatialChannelEncoder(nn.Module):
    """Encode spatial relationships between EEG channels."""

    def __init__(self, n_channels, d_model, kernel_size=3, n_filters=64):
        super().__init__()

        # Learnable channel embeddings
        self.channel_embed = nn.Embedding(n_channels, d_model)

        # Depthwise separable convolution for spatial relationships
        self.spatial_conv = nn.Conv1d(
            n_channels, n_filters,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=1  # Not depthwise, allows channel mixing
        )

        # Project to model dimension
        self.channel_mixer = nn.Linear(n_filters, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_times)
        Returns:
            (batch, n_times, d_model)
        """
        batch_size, n_channels, n_times = x.shape

        # Apply spatial convolution
        x_spatial = self.spatial_conv(x)  # (batch, n_filters, n_times)

        # Transpose for mixing
        x_spatial = x_spatial.transpose(1, 2)  # (batch, n_times, n_filters)

        # Mix into model dimension
        x_mixed = self.channel_mixer(x_spatial)  # (batch, n_times, d_model)

        # Add channel embeddings
        channel_ids = torch.arange(n_channels, device=x.device)
        channel_embeds = self.channel_embed(channel_ids).mean(dim=0)
        x_mixed = x_mixed + channel_embeds

        return self.dropout(self.norm(x_mixed))


class MambaEEGModel(nn.Module):
    """Mamba2-based model for EEG response time prediction."""

    def __init__(self, cfg):
        super().__init__()

        # Spatial encoder
        self.spatial_encoder = SpatialChannelEncoder(
            n_channels=cfg.n_channels,
            d_model=cfg.d_model,
            kernel_size=cfg.spatial_kernel_size,
            n_filters=cfg.n_spatial_filters
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

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)
        ])

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

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_times)
        Returns:
            (batch, 1) response time predictions
        """
        # Encode spatial relationships
        x = self.spatial_encoder(x)  # (batch, n_times, d_model)

        # Pass through Mamba blocks
        for i, (mamba, norm) in enumerate(zip(self.mamba_blocks, self.layer_norms)):
            residual = x
            x = mamba(x)
            x = norm(x + residual)

        # Pool and predict
        x = x.transpose(1, 2)  # (batch, d_model, n_times)
        x = self.pooler(x)
        output = self.regression_head(x)

        return output

# %% Quick Training Test
if len(train_windows) > 0:
    print("\n" + "="*60)
    print("Testing training pipeline...")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model
    model = MambaEEGModel(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Try one training step
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        try:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                X, y = batch
                X = X.to(device)
                y = y.to(device).reshape(-1, 1)

                # Forward pass
                y_pred = model(X)
                loss = F.mse_loss(y_pred, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"✅ Training step successful!")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Predictions shape: {y_pred.shape}")
                break
            else:
                print(f"❌ Batch structure issue - not a (X, y) tuple")
                break

        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            break

print("\n✅ Debug script complete!")