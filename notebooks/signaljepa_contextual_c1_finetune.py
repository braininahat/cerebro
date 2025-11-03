# %% [markdown]
# # SignalJEPA Contextual Finetuning: Challenge 1 (Response Time Prediction)
#
# **Transfer learning workflow:**
# 1. Load pretrained MAE+Aux encoder from `best_mae_1_aux_1/best_pretrain.pt`
# 2. Transfer to SignalJEPA_Contextual (full encoder: feature + pos + transformer)
# 3. Finetune on Challenge 1 with stimulus-locked windowing
#
# **Challenge 1 task:**
# - **Input**: 2s EEG windows starting 0.5s after stimulus onset (Contrast Change Detection)
# - **Target**: Response time from stimulus (rt_from_stimulus)
# - **Metric**: NRMSE (normalized RMSE, competition primary metric)
#
# **Training strategy:**
# - **Warmup (epochs 0-4)**: Freeze encoder, train task head only (lr=1e-3)
# - **Finetuning (epochs 5+)**: Unfreeze encoder, dual LR (encoder=1e-4, head=1e-3)
# - **Early stopping**: Patience=15 epochs on validation NRMSE

# %% Imports
from pathlib import Path
import os
import random
from collections import defaultdict
from typing import Optional
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import r2_score, mean_absolute_error as mae

# Braindecode imports
from braindecode.models import SignalJEPA, SignalJEPA_Contextual
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events

# EEGDash imports
from eegdash import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    keep_only_recordings_with,
    add_extras_columns
)

# Cerebro imports
from cerebro.utils.electrode_locations import load_hbn_chs_info

# Optional: WandB logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  WandB not available. Install with: uv pip install wandb")

# Set seeds
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# %% Configuration
class Config:
    """Configuration for SignalJEPA Contextual finetuning on Challenge 1."""

    # Paths
    HBN_ROOT = Path(os.getenv("HBN_ROOT", "/media/varun/OS/Users/varun/DATASETS/HBN"))
    CACHE_PATH = Path(os.getenv("CACHE_PATH", "/home/varun/repos/cerebro/cache"))

    # Pretrained checkpoint
    pretrained_checkpoint = CACHE_PATH / "signaljepa_pretrain/best_mae_aux_1/best_pretrain.pt"

    # Model architecture (must match pretraining config)
    n_channels = 129
    n_times = 200  # 2.0s at 100 Hz
    sfreq = 100
    input_window_seconds = 2.0
    transformer_d_model = 64
    transformer_num_encoder_layers = 12
    transformer_nhead = 8
    dropout = 0.1
    n_spat_filters = 4  # Spatial filters for task head

    # Training hyperparameters
    batch_size = 128
    warmup_epochs = 5  # Freeze encoder for first 5 epochs
    n_epochs = 100
    encoder_lr = 1e-4  # Lower LR for pretrained encoder
    new_layers_lr = 1e-3  # Higher LR for new task head
    weight_decay = 1e-4
    early_stopping_patience = 15
    grad_clip = 1.0

    # Data configuration
    train_releases = ["R11", "R2", "R3", "R4", "R7", "R8", "R9", "R10"]
    val_frac = 0.1
    seed = 2025
    num_workers = 8

    # Challenge 1 specific: stimulus-locked windowing
    shift_after_stim = 0.5  # Start window 0.5s after stimulus onset
    window_len = 2.0  # 2.0s windows
    epoch_len_s = 2.0

    # Excluded subjects (from competition guidelines)
    excluded_subjects = [
        "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
        "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
        "NDARBA381JGH"
    ]

    # Tracking
    use_wandb = False
    experiment_name = f"signaljepa_contextual_c1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = CACHE_PATH / "signaljepa_finetune" / experiment_name

cfg = Config()
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

print(f"Configuration:")
print(f"  HBN_ROOT: {cfg.HBN_ROOT}")
print(f"  Pretrained checkpoint: {cfg.pretrained_checkpoint}")
print(f"  Output directory: {cfg.checkpoint_dir}")
print(f"  Train releases: {cfg.train_releases}")

# %% Load Pretrained Checkpoint and Transfer to Contextual

print("\n" + "="*60)
print("Loading Pretrained Checkpoint")
print("="*60)

# Load electrode locations
print("📍 Loading electrode locations...")
chs_info = load_hbn_chs_info()

# Load pretrained checkpoint
print(f"\n📦 Loading checkpoint: {cfg.pretrained_checkpoint}")
checkpoint = torch.load(cfg.pretrained_checkpoint, map_location=device, weights_only=False)
print(f"   Checkpoint trained for {checkpoint.get('epoch', 'unknown')} epochs")
print(f"   Available keys: {list(checkpoint.keys())}")

# Extract encoder state dict
if 'encoder_state_dict' in checkpoint:
    encoder_state = checkpoint['encoder_state_dict']
    print(f"   ✅ Found 'encoder_state_dict' key")
else:
    raise KeyError(f"Could not find 'encoder_state_dict' in checkpoint. Available keys: {list(checkpoint.keys())}")

# Create base SignalJEPA model (must match pretraining architecture)
print(f"\n🏗️  Creating base SignalJEPA model...")
print(f"   Architecture: d_model={cfg.transformer_d_model}, num_layers={cfg.transformer_num_encoder_layers}")

base_model = SignalJEPA(
    n_chans=cfg.n_channels,
    n_times=cfg.n_times,
    sfreq=cfg.sfreq,
    input_window_seconds=cfg.input_window_seconds,
    chs_info=chs_info,
    transformer__d_model=cfg.transformer_d_model,
    transformer__num_encoder_layers=cfg.transformer_num_encoder_layers,
    transformer__nhead=cfg.transformer_nhead,
    drop_prob=cfg.dropout,
    n_outputs=None  # No final layer for pretraining
)

# Load pretrained weights into base model
print(f"   Loading pretrained encoder weights...")
base_model.load_state_dict(encoder_state, strict=False)
print(f"   ✅ Pretrained weights loaded")

# Transfer to SignalJEPA_Contextual (full encoder transfer)
print(f"\n🔄 Transferring to SignalJEPA_Contextual...")
print(f"   This transfers: feature_encoder + pos_encoder + transformer")
print(f"   New task head: n_outputs=1 (regression), n_spat_filters={cfg.n_spat_filters}")

model = SignalJEPA_Contextual.from_pretrained(
    base_model,
    n_outputs=1,  # Regression (response time)
    n_spat_filters=cfg.n_spat_filters
)
model = model.to(device)

# Print parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 Model parameters:")
print(f"   Total: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")

# %% Load Challenge 1 Data (Stimulus-Locked Windowing)

print("\n" + "="*60)
print("Loading Challenge 1 Data (Stimulus-Locked Windows)")
print("="*60)

print(f"\n📊 Loading data from {len(cfg.train_releases)} releases...")
print(f"   Releases: {cfg.train_releases}")

all_windows = []

for release_idx, release in enumerate(cfg.train_releases):
    print(f"\n[{release_idx+1}/{len(cfg.train_releases)}] Loading {release}...")

    # Load dataset
    dataset = EEGChallengeDataset(
        task="contrastChangeDetection",
        release=release,
        cache_dir=cfg.HBN_ROOT,
        mini=False
    )
    print(f"   Loaded {len(dataset.datasets)} recordings")

    # Preprocess: annotate trials with response time target
    print(f"   Annotating trials with response times...")
    preprocessors = [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus",
            epoch_length=cfg.epoch_len_s,
            require_stimulus=True,
            require_response=True,
            apply_on_array=False
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False)
    ]
    preprocess(dataset, preprocessors, n_jobs=-1)

    # Keep only recordings with stimulus anchors
    dataset = keep_only_recordings_with("stimulus_anchor", dataset)
    print(f"   {len(dataset.datasets)} recordings have stimulus anchors")

    # Create stimulus-locked windows
    # Window starts 0.5s after stimulus, lasts 2.0s
    windows = create_windows_from_events(
        dataset,
        mapping={"stimulus_anchor": 0},
        trial_start_offset_samples=int(cfg.shift_after_stim * cfg.sfreq),  # +0.5s
        trial_stop_offset_samples=int((cfg.shift_after_stim + cfg.window_len) * cfg.sfreq),  # +2.5s
        window_size_samples=int(cfg.epoch_len_s * cfg.sfreq),  # 200 samples (2s)
        window_stride_samples=cfg.sfreq,  # 1s stride
        preload=True
    )

    # Add metadata (target, rt, correct, etc.)
    windows = add_extras_columns(
        windows,
        dataset,
        desc="stimulus_anchor",
        keys=("target", "rt_from_stimulus", "rt_from_trialstart",
              "stimulus_onset", "response_onset", "correct", "response_type")
    )

    print(f"   Created {len(windows)} stimulus-locked windows")
    all_windows.extend(windows.datasets)

# Combine all windows
all_windows = BaseConcatDataset(all_windows)
print(f"\n✅ Total windows: {len(all_windows):,}")

# %% Subject-Level Train/Val Split (Prevent Data Leakage)

print("\n" + "="*60)
print("Splitting Data (Subject-Level)")
print("="*60)

# Get metadata
metadata = all_windows.get_metadata()
print(f"   Total windows: {len(metadata)}")

# Get unique subjects (excluding problematic ones)
subjects = metadata["subject"].unique()
subjects = [s for s in subjects if s not in cfg.excluded_subjects]
print(f"   Subjects: {len(subjects)} (after excluding {len(cfg.excluded_subjects)} problematic subjects)")

# Subject-level split (prevents leakage - same subject not in both train/val)
train_subjects, val_subjects = train_test_split(
    subjects,
    test_size=cfg.val_frac,
    random_state=cfg.seed,
    shuffle=True
)

print(f"   Train subjects: {len(train_subjects)}")
print(f"   Val subjects: {len(val_subjects)}")

# Split windows by subject
subject_split = all_windows.split("subject")
train_windows = BaseConcatDataset([
    subject_split[s] for s in train_subjects if s in subject_split
])
val_windows = BaseConcatDataset([
    subject_split[s] for s in val_subjects if s in subject_split
])

print(f"\n✅ Split complete:")
print(f"   Train windows: {len(train_windows):,}")
print(f"   Val windows: {len(val_windows):,}")

# Create dataloaders
train_loader = DataLoader(
    train_windows,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_windows,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True
)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# %% Training Setup (Dual LR, Warmup, Metrics)

print("\n" + "="*60)
print("Training Setup")
print("="*60)

# Separate encoder parameters from task head parameters
encoder_params = (
    list(model.feature_encoder.parameters()) +
    list(model.pos_encoder.parameters()) +
    list(model.transformer.parameters())
)
new_params = list(model.final_layer.parameters())

print(f"\n🔧 Optimizer configuration:")
print(f"   Encoder parameters: {sum(p.numel() for p in encoder_params):,} (lr={cfg.encoder_lr})")
print(f"   Task head parameters: {sum(p.numel() for p in new_params):,} (lr={cfg.new_layers_lr})")

# Dual learning rate optimizer
optimizer = torch.optim.AdamW([
    {'params': encoder_params, 'lr': cfg.encoder_lr, 'weight_decay': cfg.weight_decay},
    {'params': new_params, 'lr': cfg.new_layers_lr, 'weight_decay': cfg.weight_decay}
])

# Cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cfg.n_epochs
)

# Loss function
loss_fn = nn.MSELoss()

# Encoder freezing/unfreezing helpers
def freeze_encoder(model):
    """Freeze encoder parameters for warmup."""
    for param in model.feature_encoder.parameters():
        param.requires_grad = False
    for param in model.pos_encoder.parameters():
        param.requires_grad = False
    for param in model.transformer.parameters():
        param.requires_grad = False

def unfreeze_encoder(model):
    """Unfreeze encoder parameters for full finetuning."""
    for param in model.feature_encoder.parameters():
        param.requires_grad = True
    for param in model.pos_encoder.parameters():
        param.requires_grad = True
    for param in model.transformer.parameters():
        param.requires_grad = True

# Competition metrics (matching local_scoring.py)
def compute_metrics(y_true, y_pred):
    """Compute competition metrics: RMSE, NRMSE (primary), R², MAE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse_val = rmse(y_true, y_pred)
    nrmse_val = rmse_val / y_true.std()  # Normalized by std (competition metric)
    r2_val = r2_score(y_true, y_pred)
    mae_val = mae(y_true, y_pred)

    return {
        'rmse': rmse_val,
        'nrmse': nrmse_val,  # PRIMARY COMPETITION METRIC
        'r2': r2_val,
        'mae': mae_val
    }

# WandB initialization (optional)
if cfg.use_wandb and WANDB_AVAILABLE:
    wandb.init(
        project="signaljepa-finetune",
        name=cfg.experiment_name,
        config=vars(cfg)
    )
    print("\n📊 WandB logging enabled")
else:
    print("\n⚠️  WandB logging disabled")

# %% Training Loop (Warmup + Full Finetuning)

print("\n" + "="*60)
print("Training Loop")
print("="*60)

# Training state
best_val_nrmse = float('inf')
best_epoch = 0
patience_counter = 0
training_history = defaultdict(list)

for epoch in range(cfg.n_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{cfg.n_epochs}")
    print(f"{'='*60}")

    # Warmup: freeze encoder for first 5 epochs
    if epoch == 0:
        freeze_encoder(model)
        print("🧊 Encoder FROZEN for warmup (epochs 1-5)")
        print("   Training task head only with lr=1e-3")
    elif epoch == cfg.warmup_epochs:
        unfreeze_encoder(model)
        print("🔥 Encoder UNFROZEN - full finetuning begins")
        print(f"   Dual LR: encoder={cfg.encoder_lr}, head={cfg.new_layers_lr}")

    # ===== Training Phase =====
    model.train()
    train_loss = []
    train_preds = []
    train_targets = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for batch in progress_bar:
        # Unpack batch (braindecode returns X, y, window_inds)
        X, y = batch[0].to(device).float(), batch[1].to(device).float()

        # Forward pass
        optimizer.zero_grad()
        preds = model(X)  # (batch, 1)
        loss = loss_fn(preds, y.unsqueeze(1) if y.dim() == 1 else y)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Track metrics
        train_loss.append(loss.item())
        train_preds.extend(preds.detach().cpu().numpy().flatten())
        train_targets.extend(y.detach().cpu().numpy().flatten())

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    # Compute train metrics
    train_metrics = compute_metrics(train_targets, train_preds)

    # ===== Validation Phase =====
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        for batch in progress_bar:
            X, y = batch[0].to(device).float(), batch[1].to(device).float()
            preds = model(X)

            val_preds.extend(preds.cpu().numpy().flatten())
            val_targets.extend(y.cpu().numpy().flatten())

    # Compute val metrics
    val_metrics = compute_metrics(val_targets, val_preds)

    # Print epoch summary
    print(f"\n📊 Epoch {epoch+1} Summary:")
    print(f"   Train: NRMSE={train_metrics['nrmse']:.4f}, RMSE={train_metrics['rmse']:.4f}, R²={train_metrics['r2']:.4f}")
    print(f"   Val:   NRMSE={val_metrics['nrmse']:.4f}, RMSE={val_metrics['rmse']:.4f}, R²={val_metrics['r2']:.4f}")

    # Track history
    training_history['train_nrmse'].append(train_metrics['nrmse'])
    training_history['val_nrmse'].append(val_metrics['nrmse'])
    training_history['train_rmse'].append(train_metrics['rmse'])
    training_history['val_rmse'].append(val_metrics['rmse'])
    training_history['train_r2'].append(train_metrics['r2'])
    training_history['val_r2'].append(val_metrics['r2'])

    # WandB logging
    if cfg.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'epoch': epoch + 1,
            'train/nrmse': train_metrics['nrmse'],
            'train/rmse': train_metrics['rmse'],
            'train/r2': train_metrics['r2'],
            'val/nrmse': val_metrics['nrmse'],
            'val/rmse': val_metrics['rmse'],
            'val/r2': val_metrics['r2'],
            'lr': optimizer.param_groups[0]['lr']
        })

    # Save best model (based on validation NRMSE)
    if val_metrics['nrmse'] < best_val_nrmse:
        best_val_nrmse = val_metrics['nrmse']
        best_epoch = epoch + 1
        patience_counter = 0

        # Save checkpoint
        checkpoint_path = cfg.checkpoint_dir / "best_model.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_nrmse': val_metrics['nrmse'],
            'val_rmse': val_metrics['rmse'],
            'val_r2': val_metrics['r2'],
            'train_nrmse': train_metrics['nrmse'],
            'config': vars(cfg)
        }, checkpoint_path)

        print(f"   ✅ New best model saved! (Val NRMSE: {best_val_nrmse:.4f})")
    else:
        patience_counter += 1
        print(f"   No improvement for {patience_counter} epochs (best: {best_val_nrmse:.4f} @ epoch {best_epoch})")

        # Early stopping
        if patience_counter >= cfg.early_stopping_patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            print(f"   Best Val NRMSE: {best_val_nrmse:.4f} (epoch {best_epoch})")
            break

    # Step scheduler
    scheduler.step()

print(f"\n{'='*60}")
print("Training Complete")
print(f"{'='*60}")
print(f"Best Val NRMSE: {best_val_nrmse:.4f} (epoch {best_epoch})")

# %% Evaluation on Best Model

print("\n" + "="*60)
print("Evaluating Best Model")
print("="*60)

# Load best checkpoint
best_checkpoint = torch.load(cfg.checkpoint_dir / "best_model.pt", map_location=device)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()

print(f"✅ Loaded best model from epoch {best_checkpoint['epoch']}")
print(f"   Val NRMSE: {best_checkpoint['val_nrmse']:.4f}")

# Evaluate on validation set
val_preds = []
val_targets = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        X, y = batch[0].to(device).float(), batch[1].to(device).float()
        preds = model(X)

        val_preds.extend(preds.cpu().numpy().flatten())
        val_targets.extend(y.cpu().numpy().flatten())

# Compute final metrics
final_metrics = compute_metrics(val_targets, val_preds)

print(f"\n📊 Final Validation Metrics:")
print(f"   NRMSE: {final_metrics['nrmse']:.4f}  ← Competition metric")
print(f"   RMSE:  {final_metrics['rmse']:.4f}")
print(f"   R²:    {final_metrics['r2']:.4f}")
print(f"   MAE:   {final_metrics['mae']:.4f}")

# %% Visualization

print("\n" + "="*60)
print("Creating Visualizations")
print("="*60)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training curves
ax = axes[0]
ax.plot(training_history['train_nrmse'], label='Train NRMSE', linewidth=2)
ax.plot(training_history['val_nrmse'], label='Val NRMSE', linewidth=2)
ax.axvline(cfg.warmup_epochs, color='red', linestyle='--', alpha=0.5, label='Encoder Unfrozen')
ax.axhline(best_val_nrmse, color='green', linestyle='--', alpha=0.5, label=f'Best ({best_val_nrmse:.4f})')
ax.set_xlabel('Epoch')
ax.set_ylabel('NRMSE')
ax.set_title('Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: R² curves
ax = axes[1]
ax.plot(training_history['train_r2'], label='Train R²', linewidth=2)
ax.plot(training_history['val_r2'], label='Val R²', linewidth=2)
ax.axvline(cfg.warmup_epochs, color='red', linestyle='--', alpha=0.5, label='Encoder Unfrozen')
ax.set_xlabel('Epoch')
ax.set_ylabel('R²')
ax.set_title('R² Score Curves')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Predictions vs Ground Truth
ax = axes[2]
ax.scatter(val_targets, val_preds, alpha=0.3, s=10)
min_val = min(min(val_targets), min(val_preds))
max_val = max(max(val_targets), max(val_preds))
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Ground Truth (Response Time, s)')
ax.set_ylabel('Predicted (Response Time, s)')
ax.set_title(f'Predictions vs Ground Truth\n(NRMSE={final_metrics["nrmse"]:.4f}, R²={final_metrics["r2"]:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(cfg.checkpoint_dir / "training_curves.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved plot: {cfg.checkpoint_dir / 'training_curves.png'}")

if cfg.use_wandb and WANDB_AVAILABLE:
    wandb.log({"training_curves": wandb.Image(fig)})

plt.show()

print(f"\n{'='*60}")
print("✅ Finetuning Complete!")
print(f"{'='*60}")
print(f"Best model: {cfg.checkpoint_dir / 'best_model.pt'}")
print(f"Val NRMSE: {best_val_nrmse:.4f} (competition metric)")
print(f"Val R²: {final_metrics['r2']:.4f}")
