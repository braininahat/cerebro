# %% [markdown]
# # Challenge 2: Full HBN Dataset P-Factor Prediction with Mamba2
#
# Training on complete HBN dataset with subject-level aggregation across all tasks
# Predicting externalizing psychopathology factor (p_factor)

# %% imports
from pathlib import Path
import math
import os
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.cuda.amp import GradScaler, autocast

# Mamba imports
from mamba_ssm import Mamba2

# EEG processing
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import BaseConcatDataset
from eegdash import EEGChallengeDataset

# Cerebro infrastructure
from cerebro.data.unified_cache import UniversalCacheManager

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

    # Model architecture
    n_channels = 129
    n_times = 200  # 2 seconds at 100 Hz
    d_model = 256
    d_state = 32
    n_layers = 6
    expand_factor = 2
    conv_size = 4
    dropout = 0.15  # More dropout for Challenge 2

    # Spatial processing
    spatial_kernel_size = 3
    n_spatial_filters = 128

    # Aggregation
    aggregation_method = "attention"  # "mean", "attention", or "lstm"
    n_aggregation_layers = 2

    # Training
    batch_size = 32  # Smaller batch due to multiple windows per subject
    learning_rate = 0.0003
    weight_decay = 0.0001
    n_epochs = 100  # More epochs for Challenge 2
    early_stopping_patience = 15
    warmup_epochs = 3
    grad_clip = 1.0

    # Data loading
    num_workers = 8
    window_size_seconds = 2
    window_stride_seconds = 1  # 50% overlap
    max_windows_per_subject = 200  # Limit for memory
    sfreq = 100

    # Tasks to include (all available tasks for maximum coverage)
    tasks = [
        "contrastChangeDetection",
        "DespicableMe", "DiaryOfAWimpyKid", "ThePresent",
        "FunwithFractals",
        "RestingState",
        "surroundSupp",
        "symbolSearch"
    ]

    # Releases
    train_releases = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"]
    test_release = "R5"

    # Full dataset (not mini)
    use_mini = False

    # Validation
    val_split = 0.2  # 80/20 subject-level split
    seed = 2025

    # Tracking
    use_wandb = True
    experiment_name = f"mamba_c2_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Resume from checkpoint
    resume = False  # Set to True to resume from existing best_model.pt

    # Output
    checkpoint_dir = CACHE_PATH / "mamba_checkpoints" / experiment_name

cfg = Config()
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Set seeds
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

print(f"Configuration:")
print(f"  HBN_ROOT: {cfg.HBN_ROOT}")
print(f"  Training releases: {cfg.train_releases}")
print(f"  Tasks: {len(cfg.tasks)} tasks")
print(f"  Using mini: {cfg.use_mini}")
print(f"  Max windows per subject: {cfg.max_windows_per_subject}")

# %% Load and prepare data using UniversalCacheManager
print("\n" + "="*50)
print("Loading HBN data for Challenge 2 with caching...")
print("="*50)

from cerebro.data.unified_cache import UniversalCacheManager

# Initialize cache manager
cache_mgr = UniversalCacheManager(
    cache_root=str(cfg.CACHE_PATH / "challenge2"),
    preprocessing_params={
        "sfreq": cfg.sfreq,
        "bandpass": None,
        "n_channels": cfg.n_channels,
        "standardize": False
    },
    data_dir=str(cfg.HBN_ROOT)
)

# Build raw cache (downloads + preprocesses if needed)
print("Building/loading raw cache...")
cache_mgr.build_raw(
    dataset="hbn",
    releases=cfg.train_releases,
    tasks=cfg.tasks,
    mini=cfg.use_mini
)

# Windowing happens on-demand via get_windowed_dataset()
print("Windowed cache will be loaded on-demand...")

# Load p_factors directly from participants.tsv files (no redundant data loading!)
print("Loading p_factor targets from participants.tsv files...")

# Release to dataset mapping
RELEASE_TO_DATASET = {
    "R1": "ds005504-bdf", "R2": "ds005505-bdf", "R3": "ds005506-bdf",
    "R4": "ds005507-bdf", "R5": "ds005508-bdf", "R6": "ds005510-bdf",
    "R7": "ds005511-bdf", "R8": "ds005512-bdf", "R9": "ds005514-bdf",
    "R10": "ds005515-bdf", "R11": "ds005516-bdf"
}

subject_p_factors = {}
for release in cfg.train_releases:
    dataset_name = RELEASE_TO_DATASET[release]
    participants_file = cfg.HBN_ROOT / dataset_name / "participants.tsv"

    if not participants_file.exists():
        print(f"⚠️ Warning: {participants_file} not found, skipping {release}")
        continue

    # Read participants.tsv
    df = pd.read_csv(participants_file, sep='\t')

    # Extract p_factors (column name might be "p_factor" or "externalizing")
    for _, row in df.iterrows():
        subject = row['participant_id']

        # Try both column names
        p_factor = row.get('externalizing', row.get('p_factor', None))

        if p_factor is not None and pd.notna(p_factor):
            subject_p_factors[subject] = float(p_factor)

print(f"✅ Loaded p_factors for {len(subject_p_factors)} subjects")

# %% Subject-level dataset using cached windows
class SubjectAggregatedDataset(Dataset):
    """Dataset for Challenge 2 with subject-level aggregation from cached windows."""

    def __init__(
        self,
        cache_mgr,
        subject_p_factors: dict,
        dataset: str,
        releases: list,
        tasks: list,
        mini: bool,
        window_len: float,
        stride: float,
        max_windows_per_subject: int = 200,
        seed=None,
    ):
        self.rng = random.Random(seed)
        self.max_windows_per_subject = max_windows_per_subject
        self.subject_p_factors = subject_p_factors

        # Query all cached recordings
        all_windows_meta = cache_mgr.query_raw(
            dataset=dataset,
            releases=releases,
            tasks=tasks,
            mini=mini
        )

        # Filter by subjects with p_factors
        filtered_meta = all_windows_meta[all_windows_meta["subject"].isin(subject_p_factors.keys())]

        # Group windows by subject
        self.subject_window_indices = defaultdict(list)
        for idx, row in filtered_meta.iterrows():
            subject = row["subject"]
            self.subject_window_indices[subject].append(idx)

        self.subjects = list(self.subject_window_indices.keys())

        # Get windowed dataset for loading
        # Note: get_windowed_dataset expects recordings DataFrame from query_raw()
        self.windowed_dataset = cache_mgr.get_windowed_dataset(
            recordings=all_windows_meta,  # Pass the recordings DataFrame
            window_len_s=window_len,      # Note: parameter uses '_s' suffix
            stride_s=stride               # Note: parameter uses '_s' suffix
        )

        print(f"Dataset contains {len(self.subjects)} subjects with {len(filtered_meta)} windows")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        subject = self.subjects[index]
        p_factor = self.subject_p_factors[subject]

        # Get all window indices for this subject
        window_indices = self.subject_window_indices[subject]

        # Sample windows if we have too many
        if len(window_indices) > self.max_windows_per_subject:
            window_indices = self.rng.sample(window_indices, self.max_windows_per_subject)

        # Load windows from cache (zero-copy via memory-mapped arrays)
        windows = []
        for idx in window_indices:
            window, _ = self.windowed_dataset[idx]  # (n_channels, n_times)
            windows.append(window)

        # Convert to tensor
        if len(windows) > 0:
            windows_tensor = torch.stack([torch.from_numpy(w) for w in windows])
        else:
            # Return dummy data if no valid windows
            windows_tensor = torch.zeros(1, cfg.n_channels, int(cfg.window_size_seconds * cfg.sfreq), dtype=torch.float32)

        # Basic info
        infos = {
            "subject": subject,
            "n_windows": len(windows_tensor),
        }

        return windows_tensor, p_factor, infos

# %% Model components
class SubjectAggregator(nn.Module):
    """Aggregate multiple window embeddings into subject-level representation."""

    def __init__(self, d_model, method="attention", n_layers=2):
        super().__init__()
        self.method = method

        if method == "attention":
            # Multi-layer self-attention
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
                for _ in range(n_layers)
            ])
            self.norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(n_layers)
            ])

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, n_windows, d_model)
            mask: (batch, n_windows) - True for valid windows
        Returns:
            (batch, d_model) aggregated features
        """
        if self.method == "mean":
            if mask is not None:
                x = x * mask.unsqueeze(-1)
                lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
                return x.sum(dim=1) / lengths
            else:
                return x.mean(dim=1)

        elif self.method == "attention":
            # Create attention mask
            attn_mask = ~mask if mask is not None else None

            # Apply attention layers
            for attn, norm in zip(self.attention_layers, self.norms):
                residual = x
                x, _ = attn(x, x, x, key_padding_mask=attn_mask)
                x = norm(x + residual)

            # Pool over windows
            if mask is not None:
                x = x * mask.unsqueeze(-1)
                lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
                return x.sum(dim=1) / lengths
            else:
                return x.mean(dim=1)


class MambaChallenge2Model(nn.Module):
    """Mamba2 model for Challenge 2 with subject-level aggregation."""

    def __init__(self, cfg):
        super().__init__()

        # Spatial encoder (reuse from Challenge 1)
        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(cfg.n_channels, cfg.n_spatial_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg.n_spatial_filters),
            nn.GELU(),
            nn.Conv1d(cfg.n_spatial_filters, cfg.d_model, kernel_size=1),
            nn.Dropout(cfg.dropout)
        )

        # Mamba2 blocks
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

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)
        ])

        # Window-level pooling
        self.window_pooler = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Subject-level aggregation
        self.subject_aggregator = SubjectAggregator(
            cfg.d_model,
            method=cfg.aggregation_method,
            n_layers=cfg.n_aggregation_layers
        )

        # Regression head for p_factor
        self.regression_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.d_model // 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 4, 1)
        )

    def forward_window(self, x):
        """Process a single window through Mamba."""
        # Spatial encoding
        x = self.spatial_encoder(x)  # (batch, d_model, n_times)
        x = x.transpose(1, 2)  # (batch, n_times, d_model)

        # Mamba blocks
        for i, (mamba, norm) in enumerate(zip(self.mamba_blocks, self.layer_norms)):
            residual = x
            x = mamba(x)
            x = norm(x + residual)

        # Pool across time
        x = x.transpose(1, 2)  # (batch, d_model, n_times)
        x = self.window_pooler(x)  # (batch, d_model)

        return x

    def forward(self, windows):
        """
        Args:
            windows: (batch, n_windows, n_channels, n_times)
        Returns:
            (batch, 1) p_factor predictions
        """
        batch_size, n_windows, n_channels, n_times = windows.shape

        # Process each window
        window_features = []
        for i in range(n_windows):
            window = windows[:, i, :, :]
            features = self.forward_window(window)
            window_features.append(features)

        # Stack window features
        window_features = torch.stack(window_features, dim=1)  # (batch, n_windows, d_model)

        # Create mask for valid windows
        mask = (windows.sum(dim=(2, 3)) != 0)  # (batch, n_windows)

        # Aggregate across windows
        subject_features = self.subject_aggregator(window_features, mask)

        # Predict p_factor
        output = self.regression_head(subject_features)

        return output

# %% Custom collate function
def collate_subject_windows(batch):
    """Custom collate for variable number of windows per subject."""
    windows_list = []
    targets = []
    infos_list = []

    max_windows = max(len(windows) for windows, _, _ in batch)

    # Pad windows to same length
    for windows, target, info in batch:
        n_windows = len(windows)
        if n_windows < max_windows:
            # Pad with zeros
            padding = torch.zeros(
                max_windows - n_windows,
                windows.shape[1],
                windows.shape[2],
                dtype=windows.dtype
            )
            windows = torch.cat([windows, padding], dim=0)

        windows_list.append(windows)
        targets.append(target)
        infos_list.append(info)

    # Stack
    windows_tensor = torch.stack(windows_list)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    return windows_tensor, targets_tensor, infos_list

# %% Prepare datasets
print("\n" + "="*50)
print("Preparing train/val split at subject level...")
print("="*50)

# Split subjects
all_subjects = list(subject_p_factors.keys())
random.Random(cfg.seed).shuffle(all_subjects)
n_val = int(len(all_subjects) * cfg.val_split)
val_subjects = set(all_subjects[:n_val])
train_subjects = set(all_subjects[n_val:])

print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")

# Create cache-based datasets
train_dataset = SubjectAggregatedDataset(
    cache_mgr=cache_mgr,
    subject_p_factors={s: p for s, p in subject_p_factors.items() if s in train_subjects},
    dataset="hbn",
    releases=cfg.train_releases,
    tasks=cfg.tasks,
    mini=cfg.use_mini,
    window_len=cfg.window_size_seconds,
    stride=cfg.window_stride_seconds,
    max_windows_per_subject=cfg.max_windows_per_subject,
    seed=cfg.seed
)

val_dataset = SubjectAggregatedDataset(
    cache_mgr=cache_mgr,
    subject_p_factors={s: p for s, p in subject_p_factors.items() if s in val_subjects},
    dataset="hbn",
    releases=cfg.train_releases,
    tasks=cfg.tasks,
    mini=cfg.use_mini,
    window_len=cfg.window_size_seconds,
    stride=cfg.window_stride_seconds,
    max_windows_per_subject=cfg.max_windows_per_subject,
    seed=cfg.seed
)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_subject_windows,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
    collate_fn=collate_subject_windows,
)

# %% Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️ Using device: {device}")

model = MambaChallenge2Model(cfg).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Model parameters: {total_params:,}")

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
scaler = GradScaler()

# Initialize wandb
if cfg.use_wandb:
    wandb.init(
        project="cerebro-mamba",
        name=cfg.experiment_name,
        config=vars(cfg),
        tags=["mamba2", "challenge2", "full_hbn", "p_factor"]
    )

# %% Training utilities
def calculate_nrmse(y_true, y_pred):
    """Calculate Normalized RMSE."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    y_range = y_true.max() - y_true.min()
    nrmse = rmse / y_range if y_range > 0 else rmse
    return nrmse

def train_epoch(model, loader, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    losses = []
    predictions = []
    targets = []

    pbar = tqdm(loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        windows, y, infos = batch
        windows = windows.to(device)
        y = y.to(device)

        # Mixed precision
        with autocast():
            y_pred = model(windows)
            loss = F.mse_loss(y_pred, y)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Track
        losses.append(loss.item())
        with torch.no_grad():
            predictions.extend(y_pred.detach().cpu().numpy())
            targets.extend(y.detach().cpu().numpy())

        pbar.set_postfix({"loss": np.mean(losses[-10:]) if len(losses) > 10 else np.mean(losses)})

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
    subjects = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            windows, y, infos = batch
            windows = windows.to(device)
            y = y.to(device)

            with autocast():
                y_pred = model(windows)
                loss = F.mse_loss(y_pred, y)

            losses.append(loss.item())
            predictions.extend(y_pred.cpu().numpy())
            targets.extend(y.cpu().numpy())
            subjects.extend([info["subject"] for info in infos])

    # Calculate metrics
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    nrmse = calculate_nrmse(targets, predictions)

    return np.mean(losses), nrmse, predictions, targets, subjects

# %% Training loop
print("\n" + "="*50)
print("🚀 Starting Challenge 2 training on full HBN dataset...")
print("="*50)

best_val_nrmse = float('inf')
patience_counter = 0
training_history = {
    'train_loss': [], 'val_loss': [],
    'train_nrmse': [], 'val_nrmse': []
}
start_epoch = 0

# Check for existing checkpoint to resume from
resume_checkpoint = cfg.checkpoint_dir / "best_model.pt"
if resume_checkpoint.exists() and cfg.get('resume', False):
    print(f"\n🔄 Resuming from checkpoint: {resume_checkpoint}")
    checkpoint = torch.load(resume_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_nrmse = checkpoint['val_nrmse']
    if 'history' in checkpoint:
        training_history = checkpoint['history']
    print(f"   ✅ Resuming from epoch {start_epoch}, best NRMSE: {best_val_nrmse:.4f}\n")
elif resume_checkpoint.exists():
    print(f"ℹ️  Checkpoint exists but resume=False. Starting fresh training.")
    print(f"   To resume, set cfg.resume=True or add 'resume: true' to config.\n")

for epoch in range(start_epoch, cfg.n_epochs):
    print(f"\n📅 Epoch {epoch+1}/{cfg.n_epochs}")
    print("-" * 40)

    # Train
    train_loss, train_nrmse = train_epoch(model, train_loader, optimizer, scaler, device)

    # Validate
    val_loss, val_nrmse, val_preds, val_trues, val_subjs = validate(model, val_loader, device)

    # Scheduler step
    scheduler.step(val_loss)

    # Track history
    training_history['train_loss'].append(train_loss)
    training_history['val_loss'].append(val_loss)
    training_history['train_nrmse'].append(train_nrmse)
    training_history['val_nrmse'].append(val_nrmse)

    print(f"📈 Train - Loss: {train_loss:.4f}, NRMSE: {train_nrmse:.4f}")
    print(f"📊 Val   - Loss: {val_loss:.4f}, NRMSE: {val_nrmse:.4f}")

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
        best_val_nrmse = val_nrmse
        patience_counter = 0

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_nrmse': val_nrmse,
            'val_predictions': val_preds,
            'val_targets': val_trues,
            'val_subjects': val_subjs,
            'config': vars(cfg),
            'history': training_history
        }
        torch.save(checkpoint, cfg.checkpoint_dir / "best_model.pt")
        print(f"✅ Saved best model with Val NRMSE: {val_nrmse:.4f}")
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= cfg.early_stopping_patience:
        print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
        break

    # Report progress
    if val_nrmse < 1.2:
        print(f"🎯 Good progress! Val NRMSE: {val_nrmse:.4f}")
    if val_nrmse < 1.0:
        print(f"🏆 EXCELLENT! Achieved sub-1.0 NRMSE: {val_nrmse:.4f}")

# %% Final results and visualization
print("\n" + "="*50)
print("🏁 Training Complete!")
print("="*50)
print(f"Best Val NRMSE: {best_val_nrmse:.4f}")

# Load best checkpoint for analysis
checkpoint = torch.load(cfg.checkpoint_dir / "best_model.pt")
best_val_preds = checkpoint['val_predictions']
best_val_trues = checkpoint['val_targets']

# Calculate correlation
from scipy import stats
correlation, p_value = stats.pearsonr(best_val_trues, best_val_preds)
print(f"Pearson correlation: {correlation:.4f} (p={p_value:.4e})")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss curves
axes[0, 0].plot(training_history['train_loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(training_history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].set_title('Training Progress - Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# NRMSE curves
axes[0, 1].plot(training_history['train_nrmse'], label='Train NRMSE', linewidth=2)
axes[0, 1].plot(training_history['val_nrmse'], label='Val NRMSE', linewidth=2)
axes[0, 1].axhline(y=1.2, color='orange', linestyle='--', label='Target (1.2)', alpha=0.5)
axes[0, 1].axhline(y=1.0, color='g', linestyle='--', label='Excellent (1.0)', alpha=0.5)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('NRMSE')
axes[0, 1].set_title('Training Progress - NRMSE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Scatter plot
axes[1, 0].scatter(best_val_trues, best_val_preds, alpha=0.6)
axes[1, 0].plot([best_val_trues.min(), best_val_trues.max()],
                 [best_val_trues.min(), best_val_trues.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('True P-Factor')
axes[1, 0].set_ylabel('Predicted P-Factor')
axes[1, 0].set_title(f'P-Factor Predictions (r={correlation:.3f})')
axes[1, 0].grid(True, alpha=0.3)

# Residuals
residuals = best_val_preds - best_val_trues
axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Residuals (Pred - True)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title(f'Residual Distribution\nMean: {np.mean(residuals):.3f}, Std: {np.std(residuals):.3f}')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(cfg.checkpoint_dir / "training_results.png", dpi=150)
plt.show()

# Save final metrics
final_metrics = {
    'best_val_nrmse': best_val_nrmse,
    'correlation': float(correlation),
    'p_value': float(p_value),
    'n_train_subjects': len(train_dataset),
    'n_val_subjects': len(val_dataset),
    'total_epochs': len(training_history['train_loss']),
    'model_params': total_params,
    'config': vars(cfg)
}

import json
with open(cfg.checkpoint_dir / "metrics.json", 'w') as f:
    json.dump(final_metrics, f, indent=2, default=str)

print(f"\n📁 All results saved to: {cfg.checkpoint_dir}")
print("\n✨ Challenge 2 Full Dataset Training Complete!")