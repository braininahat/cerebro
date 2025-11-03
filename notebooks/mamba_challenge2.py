# %% [markdown]
# # Challenge 2: Externalizing Factor Prediction with Mamba2
#
# Direct supervision using Mamba2 SSM for predicting the externalizing psychopathology factor (p_factor).
# This notebook aggregates features across multiple tasks and windows to create subject-level predictions.

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
from joblib import Parallel, delayed

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
import mne

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% config
class Config:
    # Data
    DATA_DIR = Path("~/mne_data/eeg2025_competition").expanduser()
    CACHE_DIR = Path("/home/varun/repos/cerebro/cache/mamba_runs")

    # Model architecture (same as Challenge 1)
    n_channels = 129
    n_times = 200  # 2 seconds at 100 Hz
    d_model = 256
    d_state = 16
    n_layers = 4
    expand_factor = 2
    conv_size = 4
    dropout = 0.1

    # Spatial processing
    spatial_kernel_size = 3
    n_spatial_filters = 64

    # Aggregation
    aggregation_method = "attention"  # "mean", "attention", or "lstm"
    n_aggregation_layers = 2

    # Training
    batch_size = 32  # Smaller batch since we aggregate multiple windows
    learning_rate = 0.0005  # Lower LR for more stable training
    weight_decay = 0.00001
    n_epochs = 150
    early_stopping_patience = 15
    warmup_steps = 500
    grad_clip = 1.0

    # Data loading
    num_workers = 4
    window_size_seconds = 4
    window_stride_seconds = 2
    crop_size_seconds = 2
    sfreq = 100
    max_windows_per_subject = 100  # Limit for memory

    # Tasks to include (all available tasks)
    tasks = [
        "contrastChangeDetection",
        "DespicableMe", "DiaryOfAWimpyKid", "ThePresent",  # Movies
        "FunwithFractals",
        "RestingState",
        "seqLearning6target", "seqLearning8target",
        "surroundSupp",
        "symbolSearch"
    ]

    # Validation
    val_split = 0.2
    seed = 42

    # Tracking
    use_wandb = False
    experiment_name = "mamba_challenge2"

cfg = Config()
cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set seeds
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

# %% data loading
print("Loading HBN data for Challenge 2...")

# Load releases (excluding R5 for validation)
release_list = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"]
# For quick testing
# release_list = ["R1", "R2"]  # Uncomment for quick testing

# Load all tasks for each release
all_datasets_dict = defaultdict(list)

for release in release_list:
    for task in cfg.tasks:
        try:
            dataset = EEGChallengeDataset(
                release=release,
                task=task,
                mini=True,  # Using mini for faster iteration
                description_fields=[
                    "subject", "session", "run", "task",
                    "age", "gender", "sex", "p_factor", "externalizing"
                ],
                cache_dir=cfg.DATA_DIR,
            )
            all_datasets_dict[task].append(dataset)
            print(f"Loaded {release} - {task}")
        except Exception as e:
            print(f"Could not load {release} - {task}: {e}")
            continue

# Combine all datasets
all_datasets_list = []
for task, datasets in all_datasets_dict.items():
    all_datasets_list.extend(datasets)

all_datasets = BaseConcatDataset(all_datasets_list)
print(f"Loaded {len(all_datasets.datasets)} recordings across {len(cfg.tasks)} tasks")

# %% filter and prepare data
# Excluded subjects from Challenge 1
excluded_subjects = [
    "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
    "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"
]

# Filter datasets: valid subjects, sufficient length, has p_factor
filtered_datasets = []
subject_p_factors = {}

for ds in all_datasets.datasets:
    subject = ds.description.subject

    # Check p_factor (use 'externalizing' if available, otherwise 'p_factor')
    p_factor = ds.description.get("externalizing", ds.description.get("p_factor", None))

    if (subject not in excluded_subjects and
        ds.raw.n_times >= 4 * cfg.sfreq and
        len(ds.raw.ch_names) == 129 and
        p_factor is not None and not math.isnan(p_factor)):

        filtered_datasets.append(ds)
        subject_p_factors[subject] = float(p_factor)

filtered_datasets = BaseConcatDataset(filtered_datasets)
print(f"After filtering: {len(filtered_datasets.datasets)} recordings")
print(f"Unique subjects with p_factor: {len(subject_p_factors)}")

# %% dataset wrapper
class MambaChallenge2Dataset(Dataset):
    """Dataset for Challenge 2 with subject-level aggregation."""

    def __init__(
        self,
        datasets: BaseConcatDataset,
        subject_p_factors: dict,
        crop_size_samples: int,
        max_windows_per_subject: int = 100,
        seed=None,
    ):
        self.rng = random.Random(seed)
        self.crop_size_samples = crop_size_samples
        self.max_windows_per_subject = max_windows_per_subject

        # Group datasets by subject
        self.subject_datasets = defaultdict(list)
        for ds in datasets.datasets:
            subject = ds.description["subject"]
            if subject in subject_p_factors:
                self.subject_datasets[subject].append(ds)

        # Create subject list
        self.subjects = list(self.subject_datasets.keys())
        self.subject_p_factors = subject_p_factors

        print(f"Dataset contains {len(self.subjects)} subjects")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        subject = self.subjects[index]
        p_factor = self.subject_p_factors[subject]

        # Get all recordings for this subject
        subject_recordings = self.subject_datasets[subject]

        # Extract windows from all recordings
        all_windows = []
        all_tasks = []

        for recording in subject_recordings:
            # Create windows for this recording
            try:
                windows = create_fixed_length_windows(
                    BaseConcatDataset([recording]),
                    window_size_samples=4 * cfg.sfreq,
                    window_stride_samples=2 * cfg.sfreq,
                    drop_last_window=True,
                )

                # Extract window data
                for win_ds in windows.datasets:
                    for i in range(min(len(win_ds), 10)):  # Limit windows per recording
                        X, _, _ = win_ds[i]
                        # Random crop
                        if X.shape[1] >= self.crop_size_samples:
                            start = self.rng.randint(0, X.shape[1] - self.crop_size_samples)
                            X = X[:, start:start + self.crop_size_samples]
                        else:
                            # Pad if necessary
                            X = X[:, :self.crop_size_samples]

                        all_windows.append(X)
                        all_tasks.append(recording.description["task"])

            except Exception as e:
                # Skip problematic recordings
                continue

        # Limit total windows
        if len(all_windows) > self.max_windows_per_subject:
            indices = self.rng.sample(range(len(all_windows)), self.max_windows_per_subject)
            all_windows = [all_windows[i] for i in indices]
            all_tasks = [all_tasks[i] for i in indices]

        # Convert to tensor
        if len(all_windows) > 0:
            windows_tensor = torch.stack([torch.tensor(w, dtype=torch.float32) for w in all_windows])
        else:
            # Return dummy data if no valid windows
            windows_tensor = torch.zeros(1, cfg.n_channels, self.crop_size_samples, dtype=torch.float32)

        # Get demographic info
        demo = subject_recordings[0].description
        infos = {
            "subject": subject,
            "sex": demo.get("sex", "unknown"),
            "age": float(demo.get("age", 0)),
            "n_windows": len(windows_tensor),
            "tasks": all_tasks
        }

        return windows_tensor, p_factor, infos

# %% model definition with aggregation
class SubjectAggregator(nn.Module):
    """Aggregate multiple window embeddings into subject-level representation."""

    def __init__(self, d_model, method="attention"):
        super().__init__()
        self.method = method

        if method == "attention":
            # Self-attention aggregation
            self.attention = nn.MultiheadAttention(
                d_model, num_heads=8, batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)
        elif method == "lstm":
            # LSTM aggregation
            self.lstm = nn.LSTM(
                d_model, d_model // 2,
                num_layers=2, batch_first=True, bidirectional=True
            )
        # For "mean", no additional parameters needed

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
                # Masked mean
                x = x * mask.unsqueeze(-1)
                lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
                return x.sum(dim=1) / lengths
            else:
                return x.mean(dim=1)

        elif self.method == "attention":
            # Create attention mask
            if mask is not None:
                attn_mask = ~mask  # Invert for PyTorch attention

            # Self-attention
            attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask if mask is not None else None)
            x = self.norm(x + attn_out)

            # Pool over windows
            if mask is not None:
                x = x * mask.unsqueeze(-1)
                lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
                return x.sum(dim=1) / lengths
            else:
                return x.mean(dim=1)

        elif self.method == "lstm":
            # Pack sequences if masked
            if mask is not None:
                lengths = mask.sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                output, (h_n, _) = self.lstm(packed)
                # Use final hidden state
                return h_n[-2:].transpose(0, 1).flatten(1)  # Combine bidirectional
            else:
                output, (h_n, _) = self.lstm(x)
                return h_n[-2:].transpose(0, 1).flatten(1)


class MambaEEGChallenge2Model(nn.Module):
    """Mamba2 model for Challenge 2 with subject-level aggregation."""

    def __init__(self, cfg):
        super().__init__()

        # Reuse spatial encoder from Challenge 1
        from mamba_challenge1 import SpatialChannelEncoder

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
                conv_init=None,
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
            cfg.d_model, method=cfg.aggregation_method
        )

        # Regression head for p_factor
        self.regression_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1)
        )

    def forward_window(self, x):
        """Process a single window through Mamba."""
        # Encode spatial relationships
        x = self.spatial_encoder(x)  # (batch, n_times, d_model)

        # Pass through Mamba blocks
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

        # Create mask for valid windows (non-zero)
        mask = (windows.sum(dim=(2, 3)) != 0)  # (batch, n_windows)

        # Aggregate across windows
        subject_features = self.subject_aggregator(window_features, mask)

        # Predict p_factor
        output = self.regression_head(subject_features)

        return output

# %% custom collate function
def collate_challenge2(batch):
    """Custom collate for variable number of windows per subject."""
    windows_list = []
    targets = []
    infos_list = []

    max_windows = 0
    for windows, target, info in batch:
        max_windows = max(max_windows, len(windows))

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
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    return windows_tensor, targets_tensor, infos_list

# %% prepare data
print("\nPreparing train/val split at subject level...")

# Split subjects
all_subjects = list(subject_p_factors.keys())
random.Random(cfg.seed).shuffle(all_subjects)
n_val = int(len(all_subjects) * cfg.val_split)
val_subjects = set(all_subjects[:n_val])
train_subjects = set(all_subjects[n_val:])

print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")

# Create train/val datasets
train_dataset = MambaChallenge2Dataset(
    datasets=filtered_datasets,
    subject_p_factors={s: p for s, p in subject_p_factors.items() if s in train_subjects},
    crop_size_samples=cfg.crop_size_seconds * cfg.sfreq,
    max_windows_per_subject=cfg.max_windows_per_subject,
    seed=cfg.seed
)

val_dataset = MambaChallenge2Dataset(
    datasets=filtered_datasets,
    subject_p_factors={s: p for s, p in subject_p_factors.items() if s in val_subjects},
    crop_size_samples=cfg.crop_size_seconds * cfg.sfreq,
    max_windows_per_subject=cfg.max_windows_per_subject,
    seed=cfg.seed
)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# %% data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_challenge2,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
    collate_fn=collate_challenge2,
)

# %% initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import spatial encoder
import sys
sys.path.append(str(Path(__file__).parent))
from mamba_challenge1 import SpatialChannelEncoder

model = MambaEEGChallenge2Model(cfg).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        config=cfg.__dict__
    )

# %% training utilities
def calculate_nrmse(y_true, y_pred):
    """Calculate Normalized RMSE."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    y_range = y_true.max() - y_true.min()
    nrmse = rmse / y_range if y_range > 0 else rmse
    return nrmse

# %% training loop
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
        y = y.to(device).unsqueeze(1)

        # Mixed precision
        with autocast():
            y_pred = model(windows)
            loss = F.mse_loss(y_pred, y)

        # Backward
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

        pbar.set_postfix({"loss": np.mean(losses[-10:])})

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
            y = y.to(device).unsqueeze(1)

            with autocast():
                y_pred = model(windows)
                loss = F.mse_loss(y_pred, y)

            losses.append(loss.item())
            predictions.extend(y_pred.cpu().numpy())
            targets.extend(y.cpu().numpy())
            subjects.extend([info["subject"] for info in infos])

    # Calculate NRMSE
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    nrmse = calculate_nrmse(targets, predictions)

    return np.mean(losses), nrmse, predictions, targets, subjects

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
    val_loss, val_nrmse, val_preds, val_trues, val_subjs = validate(model, val_loader, device)

    # Scheduler step
    scheduler.step(val_loss)

    # Track
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_nrmses.append(train_nrmse)
    val_nrmses.append(val_nrmse)

    print(f"Train Loss: {train_loss:.4f}, Train NRMSE: {train_nrmse:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val NRMSE: {val_nrmse:.4f}")

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

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_nrmse': val_nrmse,
            'val_predictions': val_preds,
            'val_targets': val_trues,
            'val_subjects': val_subjs,
            'config': cfg.__dict__,
        }, cfg.CACHE_DIR / "best_mamba_challenge2.pt")

        print(f"✓ Saved best model with Val NRMSE: {val_nrmse:.4f}")
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= cfg.early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print(f"\n✅ Training complete! Best Val NRMSE: {best_val_nrmse:.4f}")

# %% visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss curves
axes[0, 0].plot(train_losses, label='Train Loss')
axes[0, 0].plot(val_losses, label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].set_title('Training Progress')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# NRMSE curves
axes[0, 1].plot(train_nrmses, label='Train NRMSE')
axes[0, 1].plot(val_nrmses, label='Val NRMSE')
axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Target (1.0)')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('NRMSE')
axes[0, 1].set_title('NRMSE Progress')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Load best checkpoint for predictions plot
checkpoint = torch.load(cfg.CACHE_DIR / "best_mamba_challenge2.pt")
best_val_preds = checkpoint['val_predictions']
best_val_trues = checkpoint['val_targets']

# Scatter plot
axes[1, 0].scatter(best_val_trues, best_val_preds, alpha=0.6)
axes[1, 0].plot([best_val_trues.min(), best_val_trues.max()],
                 [best_val_trues.min(), best_val_trues.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('True P-Factor')
axes[1, 0].set_ylabel('Predicted P-Factor')
axes[1, 0].set_title(f'Challenge 2: P-Factor Predictions\nNRMSE = {best_val_nrmse:.4f}')
axes[1, 0].grid(True, alpha=0.3)

# Residuals plot
residuals = best_val_preds - best_val_trues
axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Residuals (Pred - True)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title(f'Residual Distribution\nMean: {np.mean(residuals):.3f}, Std: {np.std(residuals):.3f}')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(cfg.CACHE_DIR / "training_curves_challenge2.png", dpi=150)
plt.show()

# %% subject-level analysis
# Analyze predictions by subject characteristics
print("\n📊 Subject-Level Analysis:")

# Calculate correlation
from scipy import stats
correlation, p_value = stats.pearsonr(best_val_trues, best_val_preds)
print(f"Pearson correlation: {correlation:.4f} (p={p_value:.4e})")

# Calculate R-squared
from sklearn.metrics import r2_score
r2 = r2_score(best_val_trues, best_val_preds)
print(f"R-squared: {r2:.4f}")

# %% save final results
results = {
    'val_nrmse': best_val_nrmse,
    'correlation': correlation,
    'r2_score': r2,
    'n_subjects': len(val_dataset),
    'n_train_subjects': len(train_dataset),
    'config': cfg.__dict__,
}

import json
with open(cfg.CACHE_DIR / "challenge2_results.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n✨ Challenge 2 notebook complete!")
print(f"Model and results saved in {cfg.CACHE_DIR}")

if best_val_nrmse < 1.0:
    print("🎉 EXCELLENT! Val NRMSE < 1.0 achieved for Challenge 2!")
    print("Consider contrastive pretraining to further improve results.")
elif best_val_nrmse < 1.2:
    print("✓ Good progress! Val NRMSE < 1.2. Consider:")
    print("  - Increasing model capacity")
    print("  - Including more tasks/windows")
    print("  - Different aggregation strategies")
else:
    print("⚠️ Val NRMSE > 1.2. Consider:")
    print("  - Checking data quality and p_factor distribution")
    print("  - Adjusting hyperparameters")
    print("  - Using pretrained features")