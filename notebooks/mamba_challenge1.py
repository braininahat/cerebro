# %% [markdown]
# # Challenge 1: Response Time Prediction with Mamba2
#
# Direct supervision using Mamba2 SSM for predicting response times in contrast change detection task.
# This notebook implements a pure sequence modeling approach without attention mechanisms.

# %% imports
from pathlib import Path
import math
import os
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.cuda.amp import GradScaler, autocast

# Mamba imports
from mamba_ssm import Mamba2
from mamba_ssm.models.config_mamba import Mamba2Config

# EEG processing
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset
from eegdash import EEGChallengeDataset

# Cerebro caching infrastructure
from cerebro.data.unified_cache import UniversalCacheManager

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Load environment variables for cache path
load_dotenv()

# %% config
class Config:
    # Data paths from environment
    DATA_DIR = Path(os.getenv("HBN_ROOT", "/media/varun/OS/Users/varun/DATASETS/HBN"))
    CACHE_PATH = Path(os.getenv("CACHE_PATH", "/home/varun/repos/cerebro/cache"))
    CACHE_DIR = CACHE_PATH / "mamba_runs"

    # Model architecture
    n_channels = 129
    n_times = 200  # 2 seconds at 100 Hz
    d_model = 256
    d_state = 16
    n_layers = 4
    expand_factor = 2
    conv_size = 4
    dropout = 0.1

    # Spatial processing
    spatial_kernel_size = 3  # For depthwise conv
    n_spatial_filters = 64

    # Training
    batch_size = 128
    learning_rate = 0.001
    weight_decay = 0.00001
    n_epochs = 100
    early_stopping_patience = 10
    warmup_steps = 500
    grad_clip = 1.0

    # Data loading
    num_workers = 4
    window_size_seconds = 4
    window_stride_seconds = 2
    crop_size_seconds = 2
    sfreq = 100

    # Validation
    val_split = 0.2
    seed = 42

    # Tracking
    use_wandb = False  # Set to True if you want wandb logging
    experiment_name = "mamba_challenge1"

cfg = Config()
cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set seeds
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

# %% Initialize Cache Manager
print("Initializing cache manager...")

# Create cache manager for efficient data loading
cache_mgr = UniversalCacheManager(
    cache_root=str(cfg.CACHE_PATH / "challenge1_mamba"),
    preprocessing_params={
        "sfreq": cfg.sfreq,
        "bandpass": None,  # Already done by EEGChallengeDataset
        "n_channels": cfg.n_channels,
        "standardize": False,
        "window_size": cfg.window_size_seconds,
        "window_stride": cfg.window_stride_seconds,
    },
    data_dir=str(cfg.DATA_DIR)
)

# %% data loading
print("Loading HBN data...")

# Load releases (excluding R5 for validation)
release_list = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"]
# For quick testing, use only R1 mini
# release_list = ["R1"]  # Uncomment for quick testing

# Check if we have cached data
mini = True  # Using mini for faster iteration
cache_key = f"challenge1_{'mini' if mini else 'full'}_{'_'.join(release_list)}"

all_datasets_list = [
    EEGChallengeDataset(
        release=release,
        task="contrastChangeDetection",
        mini=mini,
        description_fields=[
            "subject", "session", "run", "task",
            "age", "gender", "sex", "p_factor"
        ],
        cache_dir=cfg.DATA_DIR,
    )
    for release in release_list
]

all_datasets = BaseConcatDataset(all_datasets_list)
print(f"Loaded {len(all_datasets.datasets)} recordings")

# Filter out excluded subjects from startkit
excluded_subjects = [
    "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
    "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"
]

filtered_datasets = BaseConcatDataset([
    ds for ds in all_datasets.datasets
    if ds.description.subject not in excluded_subjects
    and ds.raw.n_times >= 4 * cfg.sfreq
    and len(ds.raw.ch_names) == 129
])

print(f"After filtering: {len(filtered_datasets.datasets)} recordings")

# %% dataset wrapper
class MambaChallenge1Dataset(BaseDataset):
    """Dataset wrapper for Challenge 1 with proper windowing and response time targets."""

    def __init__(
        self,
        dataset: EEGWindowsDataset,
        crop_size_samples: int,
        target_name: str = "response_time",
        seed=None,
    ):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name = target_name
        self.rng = random.Random(seed)

        # Extract response times from annotations
        self.response_times = self._extract_response_times()

    def _extract_response_times(self):
        """Extract response times from raw annotations."""
        raw = self.dataset.raw
        events = raw.annotations

        response_times = []
        for i in range(len(events)):
            if 'trial' in events[i]['description']:
                # Parse response time from trial annotation
                # This matches the startkit approach
                desc = events[i]['description']
                if 'RT' in desc:
                    rt = float(desc.split('RT:')[1].split(',')[0])
                    response_times.append(rt)

        return response_times

    def __len__(self):
        return min(len(self.dataset), len(self.response_times))

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]

        # Get target response time
        if index < len(self.response_times):
            target = self.response_times[index]
        else:
            target = np.nan  # Will be filtered out

        # Skip if invalid target
        if np.isnan(target):
            # Return a neighboring valid sample
            return self.__getitem__((index + 1) % len(self))

        # Normalize response time (log transform as in some experiments)
        target = np.log(target + 1e-6)
        target = float(target)

        # Random crop to desired length
        i_window_in_trial, i_start, i_stop = crop_inds
        if i_stop - i_start >= self.crop_size_samples:
            start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
            i_start = i_start + start_offset
            i_stop = i_start + self.crop_size_samples
            X = X[:, start_offset : start_offset + self.crop_size_samples]
        else:
            # Pad if necessary
            X = X[:, :self.crop_size_samples]

        # Additional metadata
        infos = {
            "subject": self.dataset.description["subject"],
            "sex": self.dataset.description.get("sex", "unknown"),
            "age": float(self.dataset.description.get("age", 0)),
        }

        return X, target, infos

# %% model definition
class SpatialChannelEncoder(nn.Module):
    """Encode spatial relationships between EEG channels without explicit coordinates."""

    def __init__(self, n_channels, d_model, kernel_size=3):
        super().__init__()

        # Learnable channel embeddings (position-like)
        self.channel_embed = nn.Embedding(n_channels, d_model)

        # Depthwise conv for local spatial relationships
        # Treats channels as spatial dimension
        self.spatial_conv = nn.Conv1d(
            n_channels, n_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=n_channels  # Depthwise
        )

        # Project to model dimension
        self.channel_mixer = nn.Linear(n_channels, d_model)

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

        # Apply spatial convolution for local relationships
        x_spatial = self.spatial_conv(x)  # (batch, n_channels, n_times)

        # Transpose for mixing
        x_spatial = x_spatial.transpose(1, 2)  # (batch, n_times, n_channels)

        # Mix channels into model dimension
        x_mixed = self.channel_mixer(x_spatial)  # (batch, n_times, d_model)

        # Add channel position embeddings
        channel_ids = torch.arange(n_channels, device=x.device)
        channel_embeds = self.channel_embed(channel_ids)  # (n_channels, d_model)
        # Average pool channel embeddings and add
        x_mixed = x_mixed + channel_embeds.mean(dim=0)

        return self.dropout(self.norm(x_mixed))


class MambaEEGModel(nn.Module):
    """Mamba2-based model for EEG response time prediction."""

    def __init__(self, cfg):
        super().__init__()

        # Spatial encoder
        self.spatial_encoder = SpatialChannelEncoder(
            n_channels=cfg.n_channels,
            d_model=cfg.d_model,
            kernel_size=cfg.spatial_kernel_size
        )

        # Mamba2 blocks
        self.mamba_blocks = nn.ModuleList([
            Mamba2(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                expand=cfg.expand_factor,
                d_conv=cfg.conv_size,
                conv_init=None,  # Use default
                headdim=32,  # Structured state space dimension
            )
            for _ in range(cfg.n_layers)
        ])

        # Layer norms between blocks
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)
        ])

        # Global pooling and prediction head
        self.pooler = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
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

        # Pass through Mamba blocks with residual connections
        for i, (mamba, norm) in enumerate(zip(self.mamba_blocks, self.layer_norms)):
            residual = x
            x = mamba(x)
            x = norm(x + residual)

        # Global pooling across time
        x = x.transpose(1, 2)  # (batch, d_model, n_times)
        x = self.pooler(x)  # (batch, d_model)

        # Predict response time
        output = self.regression_head(x)  # (batch, 1)

        return output

# %% training utilities
def calculate_nrmse(y_true, y_pred):
    """Calculate Normalized RMSE as in competition."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    y_range = y_true.max() - y_true.min()
    nrmse = rmse / y_range if y_range > 0 else rmse
    return nrmse


def split_subjects(datasets, val_split=0.2, seed=42):
    """Split data at subject level to prevent leakage."""
    # Get unique subjects
    subjects = list(set([ds.description["subject"] for ds in datasets.datasets]))

    # Shuffle and split
    random.Random(seed).shuffle(subjects)
    n_val = int(len(subjects) * val_split)
    val_subjects = set(subjects[:n_val])
    train_subjects = set(subjects[n_val:])

    # Split datasets
    train_ds = BaseConcatDataset([
        ds for ds in datasets.datasets
        if ds.description["subject"] in train_subjects
    ])

    val_ds = BaseConcatDataset([
        ds for ds in datasets.datasets
        if ds.description["subject"] in val_subjects
    ])

    print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")
    print(f"Train recordings: {len(train_ds.datasets)}, Val recordings: {len(val_ds.datasets)}")

    return train_ds, val_ds

# %% prepare data
print("\nPreparing train/val split...")
train_datasets, val_datasets = split_subjects(filtered_datasets, cfg.val_split, cfg.seed)

# Create windows
print("Creating fixed-length windows...")
train_windows = create_fixed_length_windows(
    train_datasets,
    window_size_samples=cfg.window_size_seconds * cfg.sfreq,
    window_stride_samples=cfg.window_stride_seconds * cfg.sfreq,
    drop_last_window=True,
)

val_windows = create_fixed_length_windows(
    val_datasets,
    window_size_samples=cfg.window_size_seconds * cfg.sfreq,
    window_stride_samples=cfg.window_stride_seconds * cfg.sfreq,
    drop_last_window=True,
)

# Wrap in our dataset
train_dataset = BaseConcatDataset([
    MambaChallenge1Dataset(ds, crop_size_samples=cfg.crop_size_seconds * cfg.sfreq)
    for ds in train_windows.datasets
])

val_dataset = BaseConcatDataset([
    MambaChallenge1Dataset(ds, crop_size_samples=cfg.crop_size_seconds * cfg.sfreq)
    for ds in val_windows.datasets
])

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# %% data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

# %% initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MambaEEGModel(cfg).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
scaler = GradScaler()

# Initialize wandb if requested
if cfg.use_wandb:
    wandb.init(
        project="cerebro-mamba",
        name=cfg.experiment_name,
        config=cfg.__dict__
    )

# %% training loop
def train_epoch(model, loader, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    losses = []
    predictions = []
    targets = []

    pbar = tqdm(loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        X, y, infos = batch
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32).unsqueeze(1)

        # Mixed precision training
        with autocast():
            y_pred = model(X)
            loss = F.mse_loss(y_pred, y)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Track
        losses.append(loss.item())
        predictions.extend(y_pred.detach().cpu().numpy())
        targets.extend(y.detach().cpu().numpy())

        # Update progress bar
        pbar.set_postfix({"loss": np.mean(losses[-100:])})

    # Calculate NRMSE
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    nrmse = calculate_nrmse(targets, predictions)

    return np.mean(losses), nrmse


def validate(model, loader, device):
    """Validate the model."""
    model.eval()
    losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            X, y, infos = batch
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).unsqueeze(1)

            with autocast():
                y_pred = model(X)
                loss = F.mse_loss(y_pred, y)

            losses.append(loss.item())
            predictions.extend(y_pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    # Calculate NRMSE
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    nrmse = calculate_nrmse(targets, predictions)

    return np.mean(losses), nrmse

# %% main training
print("\nStarting training...")
best_val_nrmse = float('inf')
patience_counter = 0

train_losses = []
val_losses = []
train_nrmses = []
val_nrmses = []

for epoch in range(cfg.n_epochs):
    print(f"\nEpoch {epoch+1}/{cfg.n_epochs}")

    # Train
    train_loss, train_nrmse = train_epoch(model, train_loader, optimizer, scaler, device)

    # Validate
    val_loss, val_nrmse = validate(model, val_loader, device)

    # Scheduler step
    scheduler.step()

    # Track
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_nrmses.append(train_nrmse)
    val_nrmses.append(val_nrmse)

    print(f"Train Loss: {train_loss:.4f}, Train NRMSE: {train_nrmse:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val NRMSE: {val_nrmse:.4f}")
    print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Log to wandb
    if cfg.use_wandb:
        wandb.log({
            "train_loss": train_loss,
            "train_nrmse": train_nrmse,
            "val_loss": val_loss,
            "val_nrmse": val_nrmse,
            "lr": scheduler.get_last_lr()[0],
            "epoch": epoch + 1
        })

    # Save best model
    if val_nrmse < best_val_nrmse:
        best_val_nrmse = val_nrmse
        patience_counter = 0

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_nrmse': val_nrmse,
            'config': cfg.__dict__,
        }, cfg.CACHE_DIR / "best_mamba_challenge1.pt")

        print(f"✓ Saved best model with Val NRMSE: {val_nrmse:.4f}")
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= cfg.early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

    # Check if we reached target
    if val_nrmse < 1.0:
        print(f"🎉 Reached target Val NRMSE < 1.0: {val_nrmse:.4f}")
        # Continue training to see if we can get even better

print(f"\n✅ Training complete! Best Val NRMSE: {best_val_nrmse:.4f}")

# %% visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses, label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training Progress')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# NRMSE curves
axes[1].plot(train_nrmses, label='Train NRMSE')
axes[1].plot(val_nrmses, label='Val NRMSE')
axes[1].axhline(y=1.0, color='r', linestyle='--', label='Target (1.0)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('NRMSE')
axes[1].set_title('NRMSE Progress')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(cfg.CACHE_DIR / "training_curves_challenge1.png", dpi=150)
plt.show()

# %% final evaluation
# Load best model
checkpoint = torch.load(cfg.CACHE_DIR / "best_mamba_challenge1.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Final validation
final_val_loss, final_val_nrmse = validate(model, val_loader, device)
print(f"\n📊 Final Results:")
print(f"Best Val NRMSE: {checkpoint['val_nrmse']:.4f}")
print(f"Final Val NRMSE: {final_val_nrmse:.4f}")

# %% save predictions for analysis
model.eval()
val_predictions = []
val_targets = []
val_subjects = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Collecting predictions"):
        X, y, infos = batch
        X = X.to(device, dtype=torch.float32)

        with autocast():
            y_pred = model(X)

        val_predictions.extend(y_pred.cpu().numpy())
        val_targets.extend(y.numpy())
        val_subjects.extend(infos["subject"])

# Convert to arrays
val_predictions = np.array(val_predictions).flatten()
val_targets = np.array(val_targets).flatten()

# Create scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(val_targets, val_predictions, alpha=0.5)
plt.plot([val_targets.min(), val_targets.max()],
         [val_targets.min(), val_targets.max()], 'r--', lw=2)
plt.xlabel('True Response Time (log)')
plt.ylabel('Predicted Response Time (log)')
plt.title(f'Challenge 1: Response Time Predictions\nNRMSE = {final_val_nrmse:.4f}')
plt.grid(True, alpha=0.3)
plt.savefig(cfg.CACHE_DIR / "predictions_scatter_challenge1.png", dpi=150)
plt.show()

print("\n✨ Challenge 1 notebook complete!")
print(f"Model and results saved in {cfg.CACHE_DIR}")

if final_val_nrmse < 0.95:
    print("🎯 EXCELLENT! Val NRMSE < 0.95 achieved. Ready for contrastive pretraining!")
elif final_val_nrmse < 1.0:
    print("✓ Good! Val NRMSE < 1.0 achieved. Consider hyperparameter tuning.")
else:
    print("⚠️ Val NRMSE > 1.0. Consider architecture modifications or longer training.")