# %% [markdown]
# # Challenge 1: FNO + MLP-Mixer for EEG Response Time Prediction
#
# **Testing FNO + MLP-Mixer architecture (2024 state-of-the-art):**
# - FNO operates in frequency domain for temporal patterns (0-50Hz)
# - MLP-Mixer for spatial channel mixing (no graph construction needed)
# - VICReg regularization to prevent representation collapse
#
# **Why this approach:**
# - Top 2024 EEG models (EEGPT, EEGMamba, DSE-Mixer) avoid message passing GNNs
# - MLP channel mixing is simpler, faster, and learns from data
# - VICReg directly addresses our collapse problem (pred std was only 16% of target std)
#
# **Architecture:**
# - FNO temporal blocks → MLP spatial mixer → VICReg regularization → Regression
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

# FNO imports from neuraloperator library (installed as neuralop)
try:
    from neuralop.models import FNO
    print("✓ Using neuraloperator library FNO")
except ImportError:
    print("⚠️ neuraloperator not installed. Install with: pip install neuraloperator")
    # Fallback to custom implementation if needed
    raise ImportError("Please install neuraloperator: pip install neuraloperator")

# MLP-Mixer and VICReg imports
from cerebro.models.components.mlp_mixer import EnhancedSpatialChannelMixer, VICRegLoss
print("✓ Using Enhanced MLP-Mixer spatial encoder (DSE-Mixer inspired) with VICReg regularization")

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

    # Model architecture - FNO specific
    n_channels = 129
    n_times = 200  # 2.0 seconds at 100 Hz

    # FNO parameters
    n_fno_layers = 8  # Number of FNO blocks (was n_layers=6 for Mamba)
    fno_modes = 5  # Fourier modes to keep (0-50Hz covers all EEG bands)
    fno_width = 256  # Hidden dimension in FNO (was d_model=256)
    d_model = 256  # Output dimension after FNO
    dropout = 0.1

    # Training (same as Mamba for fair comparison)
    batch_size = 64
    learning_rate = 0.0001
    weight_decay = 0.00001
    n_epochs = 50
    early_stopping_patience = 10
    warmup_epochs = 2
    grad_clip = 5.0

    # Data loading
    num_workers = 8
    window_len = 2.0  # seconds
    shift_after_stim = 0.5  # seconds after stimulus
    sfreq = 100

    # Validation
    seed = 2025

    # Release-level split (70:30)
    train_releases = ["R11", "R2", "R3", "R4", "R7", "R8"]  # 3,253 recordings (70.03%)
    val_releases = ["R1", "R10", "R6", "R9"]  # 1,392 recordings (29.97%)
    test_release = "R5"  # Competition validation set

    # Full dataset (not mini)
    use_mini = False

    # Tracking
    use_wandb = True
    experiment_name = f"fno_mlpmixer_vicreg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Output
    checkpoint_dir = CACHE_PATH / "fno_mlpmixer_checkpoints" / experiment_name

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
print(f"  Model: FNO + MLP-Mixer with {cfg.n_fno_layers} layers, {cfg.fno_modes} modes, VICReg regularization")

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

# Setup cache directory (FNO-specific to prevent cache pollution)
WINDOW_CACHE_DIR = cfg.CACHE_PATH / "fno_c1_windows"
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

# Create DataLoaders with performance optimizations
train_loader = DataLoader(
    train_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
    persistent_workers=True,  # Keep workers alive across epochs
    prefetch_factor=4  # Prefetch 4 batches per worker
)
val_loader = DataLoader(
    val_set,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
    persistent_workers=True,  # Keep workers alive across epochs
    prefetch_factor=4  # Prefetch 4 batches per worker
)

print(f"  Total steps per epoch: {len(train_loader)}")

# %% [markdown]
# ## FNO Model Architecture
#
# Key differences from Mamba:
# - **FNO layers**: Learn in frequency domain (perfect for EEG bands)
# - **Global receptive field**: Each layer sees entire 2s window
# - **Direct frequency learning**: No need for many layers to capture long-range dependencies
# - **Interpretable**: Can visualize which frequencies are important

# %% Model Definition
class FNOEEGModel(nn.Module):
    """FNO + MLP-Mixer model for EEG response time prediction."""

    def __init__(self, cfg):
        super().__init__()

        # 1. Channel-wise normalization (per-channel statistics)
        self.channel_norm = nn.BatchNorm1d(cfg.n_channels)

        # 2. Feature expansion: 129 channels → d_model (256) for more capacity
        # This gives FNO much more feature space to work with
        self.channel_expansion = nn.Sequential(
            nn.Conv1d(cfg.n_channels, cfg.d_model, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(cfg.d_model)
        )

        # 3. FNO blocks for temporal frequency learning (now operating on expanded features)
        self.fno_blocks = nn.ModuleList([
            FNO(
                n_modes=(cfg.fno_modes,),  # 50 modes = 0-50Hz
                in_channels=cfg.d_model,  # Work with expanded features
                out_channels=cfg.d_model,  # Maintain feature dimension
                hidden_channels=cfg.fno_width,  # Internal hidden dim
                n_layers=1,
                factorization='tucker',
                rank=0.42
            )
            for i in range(cfg.n_fno_layers)
        ])

        # 4. Enhanced spatial channel mixing (DSE-Mixer inspired)
        # Key improvements: per-timestep spatial mixing + max pooling (no attention collapse!)
        # Now operates on expanded features (d_model channels)
        self.spatial_mixer = EnhancedSpatialChannelMixer(
            n_channels=cfg.d_model,  # 256 channels (after expansion)
            hidden_dim=cfg.d_model,  # 256 output dim
            use_vicreg=False,  # Disabled: VICReg inflates embeddings (std=1.0 target) causing over-wide predictions
            dropout=cfg.dropout
        )

        # 4. Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1)
        )

        # For diagnostics
        self.debug_stats = {}

    def forward(self, x, return_debug=False, return_vicreg=False):
        """
        Args:
            x: (batch, n_channels, n_times)
            return_debug: If True, store intermediate representations for debugging
            return_vicreg: If True, compute VICReg regularization loss
        Returns:
            output: (batch, 1) response time predictions
            vicreg_loss: VICReg loss (if return_vicreg=True)
            vicreg_metrics: Diagnostic metrics (if return_vicreg=True)
        """
        # Normalize per channel
        x = self.channel_norm(x)

        # Expand features: 129 → 256 channels for more capacity
        x = self.channel_expansion(x)

        if return_debug:
            self.debug_stats['after_norm'] = x.std().item()
            self.debug_stats['input_mean'] = x.mean().item()

        # Apply FNO blocks (learns frequency patterns in expanded feature space)
        for i, fno in enumerate(self.fno_blocks):
            x_res = x  # Save input for residual connection
            x = fno(x)

            # Add residual connection for ALL layers (improves gradient flow)
            x = x + x_res

            if return_debug and i == len(self.fno_blocks) - 1:
                self.debug_stats['after_fno'] = x.std().item()
                self.debug_stats['after_fno_mean'] = x.mean().item()

        # Spatial channel mixing via MLP-Mixer
        # Input: (batch, 256, time) → pools time → Output: (batch, hidden_dim)
        x_mixed, vicreg_loss, vicreg_metrics = self.spatial_mixer(x, return_vicreg=return_vicreg)

        if return_debug:
            self.debug_stats['after_spatial'] = x_mixed.std().item()
            # Check diversity across batch
            batch_diversity = x_mixed.std(dim=0).mean().item()
            self.debug_stats['batch_diversity'] = batch_diversity
            if vicreg_metrics:
                self.debug_stats.update(vicreg_metrics)

        # Predict response time
        output = self.regression_head(x_mixed)

        if return_vicreg:
            return output, vicreg_loss, vicreg_metrics
        else:
            return output

class AttentionPooling(nn.Module):
    """Attention-weighted pooling to aggregate temporal information."""

    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(d_model, d_model // 4, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(d_model // 4, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, d_model, n_times)
        Returns:
            (batch, d_model)
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # (batch, 1, n_times)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted average
        pooled = (x * attn_weights).sum(dim=-1)  # (batch, d_model)

        return pooled

# %% Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️ Using device: {device}")

model = FNOEEGModel(cfg).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Model parameters: {total_params:,} (trainable: {trainable_params:,})")
print(f"   FNO blocks: {cfg.n_fno_layers}")
print(f"   Fourier modes: {cfg.fno_modes} (0-{cfg.fno_modes}Hz)")
print(f"   Width: {cfg.fno_width}")

# Optimizer and scheduler (same as Mamba for fair comparison)
optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',  # Minimize validation loss
    factor=0.5,  # Halve LR on plateau
    patience=5,  # Wait 5 epochs before reducing
    min_lr=1e-6  # Don't go below this
)
print(f"🔧 Using full FP32 precision for FNO stability")

# Initialize wandb
if cfg.use_wandb:
    wandb.init(
        project="cerebro-fno-mlpmixer",
        name=cfg.experiment_name,
        config=vars(cfg),
        tags=["fno", "mlp-mixer", "vicreg", "challenge1"]
    )

# %% Training utilities
def calculate_nrmse(y_true, y_pred):
    """Calculate Normalized RMSE as in competition."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    # Use std() to match competition metric
    y_std = y_true.std()
    nrmse = rmse / y_std if y_std > 0 else rmse
    return nrmse

def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    losses = []
    predictions = []
    targets = []
    grad_clips = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} Training",
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')

    for batch_idx, batch in enumerate(pbar):
        X, y = batch[0], batch[1]  # Braindecode returns (X, y, i) tuple
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        # Squeeze targets to 1D if needed
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)

        # Forward pass (VICReg disabled - architectural fixes prevent collapse)
        y_pred = model(X)
        y_pred = y_pred.squeeze(-1)

        # Loss: L1 (MAE) - more robust to outliers than MSE
        loss = F.l1_loss(y_pred, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        grad_clipped = total_norm > cfg.grad_clip
        grad_clips.append(grad_clipped)

        # Log for first epoch
        if epoch == 0 and batch_idx < 10:
            with torch.no_grad():
                pred_vals = y_pred.detach().cpu().numpy()[:5]
                target_vals = y.detach().cpu().numpy()[:5]
            print(f"    Batch {batch_idx}: grad_norm={total_norm:.4f} (clipped={grad_clipped})")
            print(f"      Preds: {pred_vals}")
            print(f"      Targets: {target_vals}")
            print(f"      Pred std: {y_pred.std().item():.4f}, L1: {loss.item():.4f}")

        optimizer.step()

        # Track
        losses.append(loss.item())
        with torch.no_grad():
            predictions.extend(y_pred.detach().cpu().numpy())
            targets.extend(y.detach().cpu().numpy())

        # Update progress
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{np.mean(losses):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

    # Calculate metrics
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    nrmse = calculate_nrmse(targets, predictions)

    # Analysis
    print(f"\n  📊 Training predictions analysis:")
    print(f"     Predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}")
    print(f"     Targets:     mean={targets.mean():.4f}, std={targets.std():.4f}")
    print(f"     Std ratio:   {predictions.std()/targets.std():.2%}")

    return np.mean(losses), nrmse

def validate(model, loader, device):
    """Validate the model."""
    model.eval()
    losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validation")):
            X, y = batch[0], batch[1]
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            if y.ndim == 2 and y.shape[1] == 1:
                y = y.squeeze(1)

            # Diagnostic for first batch (general stats, not VICReg-specific)
            if batch_idx == 0:
                print(f"\n  🔍 Diagnostic: Checking representation quality...")
                _ = model(X, return_debug=True)
                debug_stats = model.debug_stats

                print(f"\n     📊 Representation std at each stage:")
                print(f"        After norm: {debug_stats.get('after_norm', 0):.6f}")
                print(f"        After FNO: {debug_stats.get('after_fno', 0):.6f}")
                print(f"        After spatial: {debug_stats.get('after_spatial', 0):.6f}")
                print(f"        Batch diversity: {debug_stats.get('batch_diversity', 0):.6f}")

                if debug_stats.get('batch_diversity', 0) < 0.01:
                    print(f"        ⚠️  WARNING: Low batch diversity!")
                elif debug_stats.get('batch_diversity', 0) > 0.05:
                    print(f"        ✅ Good batch diversity!")

            y_pred = model(X).squeeze(-1)
            loss = F.l1_loss(y_pred, y)

            losses.append(loss.item())
            predictions.extend(y_pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    # Calculate metrics
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    nrmse = calculate_nrmse(targets, predictions)

    print(f"\n  Prediction Analysis:")
    print(f"    Predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}")
    print(f"    Targets:     mean={targets.mean():.4f}, std={targets.std():.4f}")
    print(f"    Pred range:  [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"    Target range:[{targets.min():.4f}, {targets.max():.4f}]")

    if predictions.std() < 0.01:
        print(f"    ⚠️  WARNING: Very low prediction variance!")

    return np.mean(losses), nrmse

# %% Frequency Analysis Functions
def visualize_fno_spectrum(model, loader, device, save_path=None):
    """Visualize what frequencies FNO is learning."""
    model.eval()

    with torch.no_grad():
        # Get a batch of data
        batch = next(iter(loader))
        X, y = batch[0], batch[1]
        X = X.to(device, dtype=torch.float32)

        # Get FFT of input
        X_fft = torch.fft.rfft(X, dim=-1)
        input_power = X_fft.abs().mean(dim=[0, 1]).cpu().numpy()

        # Pass through normalization, expansion, and first FNO block
        x = model.channel_norm(X)
        x = model.channel_expansion(x)  # Apply expansion (129 → 256)
        x = model.fno_blocks[0](x)

        # Get FFT of FNO output
        x_fft = torch.fft.rfft(x, dim=-1)
        output_power = x_fft.abs().mean(dim=[0, 1]).cpu().numpy()

        # Frequency axis
        freqs = np.fft.rfftfreq(cfg.n_times, 1/cfg.sfreq)

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Input spectrum
        axes[0].plot(freqs[:len(input_power)], input_power, 'b-', alpha=0.7)
        axes[0].axvspan(0.5, 4, alpha=0.2, color='gray', label='Delta')
        axes[0].axvspan(4, 8, alpha=0.2, color='purple', label='Theta')
        axes[0].axvspan(8, 12, alpha=0.2, color='blue', label='Alpha')
        axes[0].axvspan(12, 30, alpha=0.2, color='green', label='Beta')
        axes[0].axvspan(30, 50, alpha=0.2, color='red', label='Gamma')
        axes[0].set_ylabel('Power')
        axes[0].set_title('Input EEG Spectrum')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # FNO output spectrum
        axes[1].plot(freqs[:len(output_power)], output_power, 'r-', alpha=0.7)
        axes[1].axvspan(0.5, 4, alpha=0.2, color='gray')
        axes[1].axvspan(4, 8, alpha=0.2, color='purple')
        axes[1].axvspan(8, 12, alpha=0.2, color='blue')
        axes[1].axvspan(12, 30, alpha=0.2, color='green')
        axes[1].axvspan(30, 50, alpha=0.2, color='red')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power')
        axes[1].set_title('After FNO Block 1')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

        # Compute frequency band importance
        bands = {
            'Delta (0.5-4Hz)': (0.5, 4),
            'Theta (4-8Hz)': (4, 8),
            'Alpha (8-12Hz)': (8, 12),
            'Beta (12-30Hz)': (12, 30),
            'Gamma (30-50Hz)': (30, 50)
        }

        print("\n📊 Frequency Band Analysis (FNO amplification):")
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if mask.any():
                input_band = input_power[mask].mean()
                output_band = output_power[mask].mean()
                ratio = output_band / (input_band + 1e-8)
                print(f"  {band_name}: {ratio:.2f}x")

# %% Training loop
print("\n" + "="*50)
print("🚀 Starting FNO + MLP-Mixer training...")
print("="*50)

print("\n📋 Training Configuration:")
print(f"  Model: FNO + MLP-Mixer + VICReg with {total_params:,} parameters")
print(f"  Architecture: {cfg.n_fno_layers} FNO blocks, {cfg.fno_modes} Fourier modes, MLP channel mixing")
print(f"  Optimizer: AdamW (lr={cfg.learning_rate:.2e}, weight_decay={cfg.weight_decay:.2e})")
print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
print(f"  Batch Size: {cfg.batch_size}")
print(f"  Total steps per epoch: {len(train_loader)}")

# Verify data
print("\n🔍 Verifying data loading...")
sample_batch = next(iter(train_loader))
if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
    X_sample, y_sample = sample_batch[0], sample_batch[1]
    print(f"✅ Batch format: (X: {X_sample.shape}, y: {y_sample.shape})")
    print(f"  Target stats: mean={y_sample.mean():.4f}, std={y_sample.std():.4f}")
    print(f"  Target range: [{y_sample.min():.4f}, {y_sample.max():.4f}]")

best_val_nrmse = float('inf')
patience_counter = 0
training_history = {
    'train_loss': [], 'val_loss': [],
    'train_nrmse': [], 'val_nrmse': []
}

# Visualize initial frequency response
print("\n📊 Initial FNO frequency response:")
visualize_fno_spectrum(model, val_loader, device, cfg.checkpoint_dir / "initial_spectrum.png")

for epoch in range(cfg.n_epochs):
    print(f"\n📅 Epoch {epoch+1}/{cfg.n_epochs}")
    print("-" * 40)

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

    # Step scheduler
    scheduler.step(val_nrmse)

    # Log to wandb
    if cfg.use_wandb:
        wandb.log({
            "train_loss": train_loss,
            "train_nrmse": train_nrmse,
            "val_loss": val_loss,
            "val_nrmse": val_nrmse,
            "lr": optimizer.param_groups[0]['lr'],
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
            'config': vars(cfg),
            'history': training_history
        }
        torch.save(checkpoint, cfg.checkpoint_dir / "best_model.pt")
        print(f"✅ NEW BEST! Val NRMSE: {val_nrmse:.4f} (improved by {improvement:.4f})")

        # Visualize learned frequencies for best model
        if epoch > 0:  # Skip first epoch
            visualize_fno_spectrum(model, val_loader, device, cfg.checkpoint_dir / f"best_spectrum_epoch{epoch}.png")
    else:
        patience_counter += 1
        print(f"⏳ Patience: {patience_counter}/{cfg.early_stopping_patience}")

    # Progress to targets
    print(f"🎯 Progress to targets:")
    print(f"   Target 1.00: {'✓ ACHIEVED' if val_nrmse < 1.0 else f'{val_nrmse - 1.0:.4f} away'}")
    print(f"   Target 0.95: {'✓ ACHIEVED' if val_nrmse < 0.95 else f'{val_nrmse - 0.95:.4f} away'}")

    # Early stopping
    if patience_counter >= cfg.early_stopping_patience:
        print(f"\n⚠️ Early stopping after {epoch+1} epochs")
        break

# %% Final results
print("\n" + "="*50)
print("🏁 Training Complete!")
print("="*50)
print(f"Best Val NRMSE: {best_val_nrmse:.4f}")
print(f"Total epochs: {len(training_history['train_loss'])}")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
axes[0].plot(training_history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(training_history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('L1 Loss (MAE)')
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
import json
final_metrics = {
    'best_val_nrmse': best_val_nrmse,
    'final_train_nrmse': training_history['train_nrmse'][-1] if training_history['train_nrmse'] else None,
    'final_val_nrmse': training_history['val_nrmse'][-1] if training_history['val_nrmse'] else None,
    'total_epochs': len(training_history['train_loss']),
    'model_params': total_params,
    'config': vars(cfg)
}

with open(cfg.checkpoint_dir / "metrics.json", 'w') as f:
    json.dump(final_metrics, f, indent=2, default=str)

print(f"\n📁 All results saved to: {cfg.checkpoint_dir}")
print("\n✨ FNO Challenge 1 Training Complete!")

# Final frequency analysis
print("\n📊 Final learned frequency response:")
checkpoint = torch.load(cfg.checkpoint_dir / "best_model.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
visualize_fno_spectrum(model, val_loader, device, cfg.checkpoint_dir / "final_spectrum.png")

# %% ============================================================
# Export Model to TorchScript (for submission)
# ============================================================
print("\n" + "="*60)
print("Exporting best model to TorchScript for submission")
print("="*60)

# Load best checkpoint
checkpoint_path = cfg.checkpoint_dir / "best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Recreate model with same configuration
best_model = FNOEEGModel(cfg).to(device)
best_model.load_state_dict(checkpoint["model_state_dict"])
best_model.eval()
print(f"✓ Loaded best checkpoint from epoch {checkpoint['epoch']}")

# Trace with example input (B, C, T) = (1, 129, 200)
# Shape: (batch, channels, timesteps) for 2.0s at 100Hz
example_input = torch.randn(1, cfg.n_channels, cfg.n_times).to(device)

print(f"Tracing model with input shape: {example_input.shape}")
scripted_model = torch.jit.trace(best_model, example_input)
scripted_model = torch.jit.optimize_for_inference(scripted_model)

# Save for submission
output_path = cfg.checkpoint_dir / "fno_mlpmixer_c1_model.pt"
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
test_model = test_model.to(device)
test_model.eval()
print(f"✓ Loaded TorchScript model from {output_path}")

# Load R5 test data using startkit approach (braindecode + preload)
print("\nLoading R5 test data with braindecode...")

# Load R5 dataset
r5_dataset = EEGChallengeDataset(
    task="contrastChangeDetection",
    release="R5",  # Competition validation set
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
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

# Run inference (exactly like local_scoring.py)
print("\nRunning inference on R5...")
y_preds = []
y_trues = []

with torch.inference_mode():
    for batch in tqdm(r5_loader, desc="R5 Inference"):
        X, y = batch[0], batch[1]  # Braindecode returns (X, y, i) tuple
        X = X.to(device, non_blocking=True)

        # Run model
        outputs = test_model(X)
        y_pred = outputs.squeeze(-1)

        # Collect predictions and targets
        y_preds.append(y_pred.cpu())
        y_trues.append(y)

# Concatenate all batches
y_pred_all = torch.cat(y_preds).numpy()
y_true_all = torch.cat(y_trues).numpy()

# Compute NRMSE (competition metric)
nrmse = np.sqrt(np.mean((y_pred_all - y_true_all) ** 2)) / np.std(y_true_all)

# Stats
pred_std = np.std(y_pred_all)
target_std = np.std(y_true_all)
pred_mean = np.mean(y_pred_all)
target_mean = np.mean(y_true_all)

print("\n" + "="*60)
print("Challenge 1 R5 Test Results:")
print("="*60)
print(f"  NRMSE: {nrmse:.4f}")
print(f"  Prediction std: {pred_std:.4f} ({100*pred_std/target_std:.1f}% of target)")
print(f"  Target std:     {target_std:.4f}")
print(f"  Prediction mean: {pred_mean:.4f}")
print(f"  Target mean:     {target_mean:.4f}")
print(f"  N samples: {len(y_pred_all):,}")

# Save R5 results
r5_results = {
    "r5_nrmse": float(nrmse),
    "r5_pred_std": float(pred_std),
    "r5_target_std": float(target_std),
    "r5_pred_mean": float(pred_mean),
    "r5_target_mean": float(target_mean),
    "r5_n_samples": int(len(y_pred_all))
}

with open(cfg.checkpoint_dir / "r5_results.json", "w") as f:
    json.dump(r5_results, f, indent=2)

print(f"\n🏆 Ready for submission! TorchScript model: {output_path}")