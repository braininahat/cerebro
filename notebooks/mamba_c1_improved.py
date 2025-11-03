# %% [markdown]
# # Challenge 1: Mamba2 with Diversity-Preserving Improvements
#
# **IMPROVEMENTS APPLIED (2025-11-02):**
# 1. Enhanced loss function with diversity preservation
# 2. Adjusted hyperparameters for better exploration
# 3. Attention-weighted pooling instead of simple averaging
# 4. Comprehensive collapse diagnostics
# 5. Better initialization and warmup strategy
#
# **Previous issues fixed:**
# - Severe representation collapse (pred std: 0.0001 vs target std: 0.4081)
# - All samples getting identical embeddings (batch diversity: 0.0002)
# - Model outputting constant predictions
#
# **Release-level train/val split:**
# - Train: R11, R2, R3, R4, R7, R8 (3,253 recordings, 70.03%)
# - Val: R1, R10, R6, R9 (1,392 recordings, 29.97%)
# - Test: R5 (held out for competition)

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

# Mamba imports
from mamba_ssm import Mamba2

# Import our custom diversity-preserving loss
import sys
sys.path.append('/home/varun/repos/cerebro')
from cerebro.losses.diversity_preserving import (
    DiversityPreservingLoss,
    AdaptiveDiversityLoss,
    compute_attention_entropy,
    diagnose_collapse
)

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
    n_times = 200  # 2.0 seconds at 100 Hz
    d_model = 320  # INCREASED from 256 for more capacity
    d_state = 32
    n_layers = 4   # REDUCED from 6 to prevent cumulative collapse
    expand_factor = 2
    conv_size = 4
    dropout = 0.15  # INCREASED from 0.1 for regularization
    n_attention_heads = 16  # INCREASED from 8 for more diverse attention

    # Spatial processing
    spatial_kernel_size = 3
    n_spatial_filters = 128

    # Training (KEY CHANGES FOR DIVERSITY)
    batch_size = 64  # REDUCED from 128 for better gradient diversity
    learning_rate = 0.0001  # Keep same
    weight_decay = 0.00001
    n_epochs = 50
    early_stopping_patience = 10
    warmup_epochs = 5  # INCREASED from 2 for SSM stabilization
    grad_clip = 1.0  # TIGHTENED from 5.0 to prevent collapse-inducing updates

    # Loss function parameters
    loss_lambda_variance = 0.1  # Weight for variance preservation
    loss_lambda_diversity = 0.05  # Weight for batch diversity
    loss_lambda_entropy = 0.01  # Weight for attention entropy bonus
    loss_min_std_ratio = 0.5  # Minimum acceptable pred_std/target_std ratio

    # Data loading
    num_workers = 8
    window_len = 2.0  # seconds
    shift_after_stim = 0.5  # seconds after stimulus
    sfreq = 100

    # Validation
    seed = 2025  # Competition year

    # Release-level split (70:30)
    train_releases = ["R11", "R2", "R3", "R4", "R7", "R8"]  # 3,253 recordings (70.03%)
    val_releases = ["R1", "R10", "R6", "R9"]  # 1,392 recordings (29.97%)
    test_release = "R5"  # Competition validation set

    # Full dataset (not mini)
    use_mini = False

    # Tracking
    use_wandb = True
    experiment_name = f"mamba_c1_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
print(f"  Batch size: {cfg.batch_size} (reduced for diversity)")
print(f"  Model: {cfg.n_layers} layers, d_model={cfg.d_model}")
print(f"  Gradient clipping: {cfg.grad_clip} (tightened)")
print(f"  Warmup epochs: {cfg.warmup_epochs}")

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
            # Extract release from filename path
            filename = str(raw.filenames[0])
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
            elif 'ds005515' in filename: release = 'R10'
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

# Create DataLoaders
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

# %% Model Definition

class AttentionWeightedPooling(nn.Module):
    """Attention-weighted pooling to preserve diversity."""

    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, d_model, n_times)
        Returns:
            (batch, d_model)
        """
        x = x.transpose(1, 2)  # (batch, n_times, d_model)

        # Compute attention weights
        attn_weights = self.attention(x)  # (batch, n_times, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted average
        pooled = (x * attn_weights).sum(dim=1)  # (batch, d_model)

        return pooled, attn_weights.squeeze(-1)  # Also return weights for entropy calculation


class ImprovedSpatialEncoder(nn.Module):
    """Spatial encoder with attention to prevent representation collapse."""

    def __init__(self, n_channels, d_model, n_heads=8, dropout=0.15):
        super().__init__()

        # Simple projection from channels to model dimension
        self.input_proj = nn.Conv1d(n_channels, d_model, kernel_size=1)

        # Spatial attention - CRITICAL for creating diverse representations
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout  # INCREASED dropout
        )

        # Layer norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Feedforward for additional processing
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
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
        x_att, attn_weights = self.spatial_attention(x, x, x)
        x = self.norm1(x + self.dropout(x_att))  # Residual connection

        # Feedforward for additional non-linearity
        x_ff = self.ff(x)
        x = self.norm2(x + self.dropout(x_ff))

        return x, attn_weights  # Return attention weights for monitoring


class MambaEEGModel(nn.Module):
    """Mamba2-based model with diversity-preserving improvements."""

    def __init__(self, cfg):
        super().__init__()

        # Improved spatial encoder with attention
        self.spatial_encoder = ImprovedSpatialEncoder(
            n_channels=cfg.n_channels,
            d_model=cfg.d_model,
            n_heads=cfg.n_attention_heads,  # More heads for diversity
            dropout=cfg.dropout
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
        self.norm_in = nn.LayerNorm(cfg.d_model)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)
        ])

        # Additional spatial attention after Mamba (helps prevent collapse)
        self.post_mamba_attention = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_attention_heads,
            batch_first=True,
            dropout=cfg.dropout
        )
        self.post_attention_norm = nn.LayerNorm(cfg.d_model)

        # NEW: Attention-weighted pooling instead of simple averaging
        self.pooler = AttentionWeightedPooling(cfg.d_model)

        self.regression_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1)
        )

        # For diagnostics
        self.debug_stats = {}
        self.attention_weights = {}

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Custom initialization for better diversity."""
        for name, param in self.named_parameters():
            if 'mamba' in name:
                if 'dt_proj' in name:  # Delta parameter - most critical
                    nn.init.uniform_(param, -0.1, 0.1)
                elif 'norm' in name and 'weight' in name:
                    nn.init.ones_(param)
                elif 'norm' in name and 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x, return_debug=False):
        """
        Args:
            x: (batch, n_channels, n_times)
            return_debug: If True, store intermediate representations for debugging
        Returns:
            (batch, 1) response time predictions, attention_weights
        """
        # Encode spatial relationships with attention
        x, spatial_attn = self.spatial_encoder(x)  # (batch, n_times, d_model)
        self.attention_weights['spatial'] = spatial_attn

        if return_debug:
            self.debug_stats['after_spatial'] = x.std().item()
            self.debug_stats['after_spatial_mean'] = x.mean().item()

        # Initial normalization before first Mamba block
        x = self.norm_in(x)

        # Pre-norm residual: Add → LN → Mamba
        for i, (mamba, norm) in enumerate(zip(self.mamba_blocks, self.layer_norms)):
            residual = x
            x = norm(x)       # Normalize BEFORE Mamba (pre-norm)
            x = mamba(x)      # Apply Mamba block
            x = x + residual  # Add residual AFTER Mamba

            if return_debug and i == len(self.mamba_blocks) - 1:
                self.debug_stats['after_mamba'] = x.std().item()
                self.debug_stats['after_mamba_mean'] = x.mean().item()

        # Additional spatial attention for global context (prevents collapse)
        x_att, post_attn = self.post_mamba_attention(x, x, x)
        x = self.post_attention_norm(x + x_att)  # Residual connection
        self.attention_weights['post_mamba'] = post_attn

        if return_debug:
            self.debug_stats['after_post_attention'] = x.std().item()
            self.debug_stats['attention_std'] = post_attn.std().item() if post_attn is not None else 0

        # NEW: Attention-weighted pooling
        x = x.transpose(1, 2)  # (batch, d_model, n_times)
        x_pooled, pool_attn = self.pooler(x)
        self.attention_weights['pooling'] = pool_attn

        if return_debug:
            self.debug_stats['after_pooling'] = x_pooled.std().item()
            # Check diversity across batch
            batch_diversity = x_pooled.std(dim=0).mean().item()
            self.debug_stats['batch_diversity'] = batch_diversity

        output = self.regression_head(x_pooled)

        return output, self.attention_weights

# %% Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️ Using device: {device}")

model = MambaEEGModel(cfg).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Model parameters: {total_params:,} (trainable: {trainable_params:,})")

# Initialize diversity-preserving loss
criterion = AdaptiveDiversityLoss(
    primary_loss="mae",
    lambda_variance=cfg.loss_lambda_variance,
    lambda_diversity=cfg.loss_lambda_diversity,
    lambda_entropy=cfg.loss_lambda_entropy,
    min_std_ratio=cfg.loss_min_std_ratio,
    warmup_epochs=cfg.warmup_epochs
)
print(f"🎯 Using AdaptiveDiversityLoss with:")
print(f"   λ_variance={cfg.loss_lambda_variance}")
print(f"   λ_diversity={cfg.loss_lambda_diversity}")
print(f"   λ_entropy={cfg.loss_lambda_entropy}")

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.7,  # Gentler reduction than 0.5
    patience=3,  # React faster to plateaus
    min_lr=1e-6
)
print(f"🔧 Using full FP32 precision (no mixed precision) for Mamba2 stability")

# Initialize wandb
if cfg.use_wandb:
    wandb.init(
        project="cerebro-mamba",
        name=cfg.experiment_name,
        config=vars(cfg),
        tags=["mamba2", "challenge1", "diversity_preserving", "improved"]
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

def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    """Train for one epoch with diversity preservation."""
    model.train()
    criterion.set_epoch(epoch)  # Update adaptive weights

    losses = []
    loss_components = []
    predictions = []
    targets = []
    grad_clips = []
    attention_entropies = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} Training",
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')

    for batch_idx, batch in enumerate(pbar):
        X, y = batch[0], batch[1]  # Braindecode returns (X, y, i) tuple
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        # Squeeze targets to 1D if needed (braindecode returns [batch, 1])
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)  # (batch, 1) -> (batch)

        # Forward pass with attention weights
        y_pred, attn_weights = model(X)
        y_pred = y_pred.squeeze(-1)  # (batch, 1) -> (batch)

        # Compute attention entropy for monitoring
        if 'post_mamba' in attn_weights:
            attn_entropy = compute_attention_entropy(attn_weights['post_mamba'])
            attention_entropies.append(attn_entropy.item())

        # Enhanced loss with diversity preservation
        loss, components = criterion(
            y_pred, y,
            attention_weights=attn_weights.get('post_mamba'),
            return_components=True
        )
        loss_components.append(components)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping with tracking
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        grad_clipped = total_norm > cfg.grad_clip
        grad_clips.append(grad_clipped)

        # Log detailed info for first epoch
        if epoch == 0 and batch_idx < 10:
            with torch.no_grad():
                pred_vals = y_pred.detach().cpu().numpy()[:5]
                target_vals = y.detach().cpu().numpy()[:5]
            print(f"    Batch {batch_idx}: grad_norm={total_norm:.4f} (clipped={grad_clipped})")
            print(f"      Preds: {pred_vals}")
            print(f"      Targets: {target_vals}")
            print(f"      Pred std: {y_pred.std().item():.4f}")
            print(f"      Components: primary={components['primary_loss']:.4f}, "
                  f"var_penalty={components['variance_penalty']:.4f}, "
                  f"div_penalty={components['diversity_penalty']:.4f}")
        elif grad_clipped and batch_idx % 50 == 0:
            print(f"    Batch {batch_idx}: Large gradient! norm={total_norm:.4f} > {cfg.grad_clip}")

        optimizer.step()

        # Track
        current_loss = loss.item()
        losses.append(current_loss)
        with torch.no_grad():
            predictions.extend(y_pred.detach().cpu().numpy())
            targets.extend(y.detach().cpu().numpy())

        # Update progress
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = np.mean(losses)
        recent_avg = np.mean(losses[-100:]) if len(losses) > 100 else avg_loss

        # Progress every 10 batches
        if batch_idx % 10 == 0:
            recent_components = {
                k: np.mean([c[k] for c in loss_components[-10:]])
                for k in loss_components[0].keys()
                if len(loss_components) >= 10
            }

            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "avg": f"{recent_avg:.4f}",
                "lr": f"{current_lr:.2e}",
                "std_ratio": f"{recent_components.get('std_ratio', 0):.2f}" if recent_components else "0",
                "clip": f"{sum(grad_clips[-10:])}/10" if len(grad_clips) >= 10 else f"{sum(grad_clips)}/{len(grad_clips)}"
            })

        # Detailed batch reporting every 50 batches
        if batch_idx % 50 == 0 and batch_idx > 0:
            clip_rate = sum(grad_clips) / len(grad_clips) * 100
            loss_std = np.std(losses[-50:]) if len(losses) >= 50 else np.std(losses)

            # Average components over last 50 batches
            avg_components = {
                k: np.mean([c[k] for c in loss_components[-50:]])
                for k in loss_components[0].keys()
            }

            print(f"  Batch {batch_idx:4d}/{len(loader)} | "
                  f"Loss: {current_loss:.4f} | "
                  f"Avg50: {np.mean(losses[-50:]):.4f} (±{loss_std:.4f}) | "
                  f"LR: {current_lr:.2e} | "
                  f"GradClip: {clip_rate:.1f}%")
            print(f"      Loss components - Primary: {avg_components['primary_loss']:.4f}, "
                  f"Var: {avg_components['variance_penalty']:.4f}, "
                  f"Div: {avg_components['diversity_penalty']:.4f}, "
                  f"StdRatio: {avg_components['std_ratio']:.2f}")

    # Calculate metrics
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    nrmse = calculate_nrmse(targets, predictions)

    # Comprehensive collapse diagnostics
    collapse_diagnostics = diagnose_collapse(
        torch.tensor(predictions),
        torch.tensor(targets)
    )

    # Analyze training predictions
    pred_std = predictions.std()
    target_std = targets.std()
    print(f"\n  📊 Training predictions analysis:")
    print(f"     Predictions: mean={predictions.mean():.4f}, std={pred_std:.4f}")
    print(f"     Targets:     mean={targets.mean():.4f}, std={target_std:.4f}")
    print(f"     Std ratio:   {pred_std/target_std:.2%} (100% = perfect variance match)")
    print(f"     Collapse severity: {collapse_diagnostics['collapse_severity']:.2f} (0=healthy, 1=collapsed)")
    print(f"     Batch correlation: {collapse_diagnostics.get('batch_correlation', 0):.3f}")

    if attention_entropies:
        print(f"     Attention entropy: {np.mean(attention_entropies):.3f}")

    # Gradient clipping statistics
    total_clips = sum(grad_clips)
    clip_percentage = (total_clips / len(grad_clips)) * 100 if grad_clips else 0
    print(f"  📊 Epoch {epoch+1} training stats:")
    print(f"     Batches with clipped gradients: {total_clips}/{len(grad_clips)} ({clip_percentage:.1f}%)")
    print(f"     Final loss: {losses[-1]:.4f} | Epoch avg: {np.mean(losses):.4f} (±{np.std(losses):.4f})")

    return np.mean(losses), nrmse, collapse_diagnostics

def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    losses = []
    predictions = []
    targets = []
    embeddings = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validation")):
            X, y = batch[0], batch[1]
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            # Squeeze targets to 1D if needed
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.squeeze(1)

            # Diagnostic check for first batch
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

                # Check for collapse warnings
                if debug_stats.get('after_spatial', 0) < 0.01:
                    print(f"        ⚠️  WARNING: Spatial encoder producing near-constant outputs!")
                if debug_stats.get('batch_diversity', 0) < 0.01:
                    print(f"        ⚠️  WARNING: All samples have similar representations!")
                if debug_stats.get('after_spatial', 0) > 0.1:
                    print(f"        ✅ Good diversity in spatial representations!")
                if debug_stats.get('batch_diversity', 0) > 0.05:
                    print(f"        ✅ Good diversity across batch samples!")

            # Forward pass
            y_pred, attn_weights = model(X)
            y_pred = y_pred.squeeze(-1)

            # Compute loss
            loss = criterion(y_pred, y, attention_weights=attn_weights.get('post_mamba'))

            losses.append(loss.item())
            predictions.extend(y_pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    # Calculate metrics
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    nrmse = calculate_nrmse(targets, predictions)

    # Comprehensive diagnostics
    collapse_diagnostics = diagnose_collapse(
        torch.tensor(predictions),
        torch.tensor(targets)
    )

    # Prediction analysis
    print(f"\n  Prediction Analysis:")
    print(f"    Predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}")
    print(f"    Targets:     mean={targets.mean():.4f}, std={targets.std():.4f}")
    print(f"    Pred range:  [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"    Target range:[{targets.min():.4f}, {targets.max():.4f}]")
    print(f"    Pred variance: {predictions.var():.6f}")
    print(f"    Collapse severity: {collapse_diagnostics['collapse_severity']:.2f}")

    # Warning if predictions have very low variance
    if predictions.std() < 0.01:
        print(f"    ⚠️  WARNING: Predictions have very low variance! Model may be stuck predicting constants.")
    elif predictions.std() < 0.1:
        print(f"    ⚠️  Warning: Predictions have low variance.")
    else:
        print(f"    ✅ Good prediction diversity!")

    return np.mean(losses), nrmse, collapse_diagnostics

# %% Training loop
print("\n" + "="*50)
print("🚀 Starting training with diversity preservation...")
print("="*50)

# Training configuration summary
print("\n📋 Training Configuration:")
print(f"  Model: MambaEEG with {total_params:,} parameters")
print(f"  Architecture: {cfg.n_layers} layers, d_model={cfg.d_model}, heads={cfg.n_attention_heads}")
print(f"  Optimizer: AdamW (lr={cfg.learning_rate:.2e}, weight_decay={cfg.weight_decay:.2e})")
print(f"  Scheduler: ReduceLROnPlateau (factor=0.7, patience=3)")
print(f"  Loss: AdaptiveDiversityLoss")
print(f"  Gradient Clipping: max_norm={cfg.grad_clip}")
print(f"  Warmup: {cfg.warmup_epochs} epochs")
print(f"  Early Stopping: patience={cfg.early_stopping_patience} epochs")
print(f"  Batch Size: {cfg.batch_size}")
print(f"  Total steps per epoch: {len(train_loader)}")

# Verify data
print("\n🔍 Verifying data loading and targets...")
sample_batch = next(iter(train_loader))
if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
    X_sample, y_sample = sample_batch[0], sample_batch[1]
    print(f"✅ Batch format: (X: {X_sample.shape}, y: {y_sample.shape})")
    print(f"  Target stats: mean={y_sample.mean():.4f}, std={y_sample.std():.4f}")
    print(f"  Target range: [{y_sample.min():.4f}, {y_sample.max():.4f}]")
    print(f"  Non-zero targets: {(y_sample != 0).sum().item()}/{len(y_sample)}")

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
    'train_nrmse': [], 'val_nrmse': [],
    'collapse_severity': []
}

for epoch in range(cfg.n_epochs):
    print(f"\n📅 Epoch {epoch+1}/{cfg.n_epochs}")
    print("-" * 40)

    # Show current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"📊 Current Learning Rate: {current_lr:.2e}")

    # Show adaptive loss weights
    print(f"📊 Loss weights - λ_var: {criterion.lambda_variance:.3f}, λ_div: {criterion.lambda_diversity:.3f}")

    # Train
    train_loss, train_nrmse, train_collapse = train_epoch(
        model, train_loader, criterion, optimizer, scheduler, device, epoch
    )

    # Validate
    val_loss, val_nrmse, val_collapse = validate(model, val_loader, criterion, device)

    # Track history
    training_history['train_loss'].append(train_loss)
    training_history['val_loss'].append(val_loss)
    training_history['train_nrmse'].append(train_nrmse)
    training_history['val_nrmse'].append(val_nrmse)
    training_history['collapse_severity'].append(val_collapse['collapse_severity'])

    print(f"\n📈 Train - Loss: {train_loss:.4f}, NRMSE: {train_nrmse:.4f}")
    print(f"📊 Val   - Loss: {val_loss:.4f}, NRMSE: {val_nrmse:.4f}")
    print(f"🎯 Collapse Severity: {val_collapse['collapse_severity']:.2f} (0=healthy, 1=collapsed)")

    # Step the scheduler based on validation NRMSE
    scheduler.step(val_nrmse)

    # Log to wandb
    if cfg.use_wandb:
        wandb.log({
            "train_loss": train_loss,
            "train_nrmse": train_nrmse,
            "val_loss": val_loss,
            "val_nrmse": val_nrmse,
            "lr": optimizer.param_groups[0]['lr'],
            "collapse_severity": val_collapse['collapse_severity'],
            "std_ratio": val_collapse.get('std_ratio', 0),
            "batch_correlation": val_collapse.get('batch_correlation', 0),
            "epoch": epoch + 1
        })

    # Save best model
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
            'collapse_severity': val_collapse['collapse_severity'],
            'config': vars(cfg),
            'history': training_history
        }
        torch.save(checkpoint, cfg.checkpoint_dir / "best_model.pt")
        print(f"✅ NEW BEST! Val NRMSE: {val_nrmse:.4f} (improved by {improvement:.4f})")

        # Check if we've achieved good diversity
        if val_collapse['collapse_severity'] < 0.3:
            print(f"🎉 Model has good diversity! Collapse severity: {val_collapse['collapse_severity']:.2f}")
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
print(f"Final collapse severity: {training_history['collapse_severity'][-1]:.2f}")

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Loss curves
axes[0,0].plot(training_history['train_loss'], label='Train Loss', linewidth=2)
axes[0,0].plot(training_history['val_loss'], label='Val Loss', linewidth=2)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('Training Progress - Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# NRMSE curves
axes[0,1].plot(training_history['train_nrmse'], label='Train NRMSE', linewidth=2)
axes[0,1].plot(training_history['val_nrmse'], label='Val NRMSE', linewidth=2)
axes[0,1].axhline(y=1.0, color='r', linestyle='--', label='Target (1.0)', alpha=0.5)
axes[0,1].axhline(y=0.95, color='g', linestyle='--', label='Excellent (0.95)', alpha=0.5)
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('NRMSE')
axes[0,1].set_title('Training Progress - NRMSE')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Collapse severity
axes[1,0].plot(training_history['collapse_severity'], label='Collapse Severity', linewidth=2, color='orange')
axes[1,0].axhline(y=0.3, color='g', linestyle='--', label='Good (<0.3)', alpha=0.5)
axes[1,0].axhline(y=0.7, color='r', linestyle='--', label='Severe (>0.7)', alpha=0.5)
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Severity')
axes[1,0].set_title('Representation Collapse Monitoring')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Learning rate
lrs = [optimizer.param_groups[0]['lr']] * len(training_history['train_loss'])
axes[1,1].plot(lrs, label='Learning Rate', linewidth=2, color='purple')
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('LR')
axes[1,1].set_title('Learning Rate Schedule')
axes[1,1].set_yscale('log')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(cfg.checkpoint_dir / "training_curves.png", dpi=150)
plt.show()

# Save final metrics
final_metrics = {
    'best_val_nrmse': best_val_nrmse,
    'final_train_nrmse': training_history['train_nrmse'][-1] if training_history['train_nrmse'] else None,
    'final_val_nrmse': training_history['val_nrmse'][-1] if training_history['val_nrmse'] else None,
    'final_collapse_severity': training_history['collapse_severity'][-1] if training_history['collapse_severity'] else None,
    'total_epochs': len(training_history['train_loss']),
    'model_params': total_params,
    'config': vars(cfg)
}

import json
with open(cfg.checkpoint_dir / "metrics.json", 'w') as f:
    json.dump(final_metrics, f, indent=2, default=str)

print(f"\n📁 All results saved to: {cfg.checkpoint_dir}")
print("\n✨ Challenge 1 Training with Diversity Preservation Complete!")