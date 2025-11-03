# %% [markdown]
# # SignalJEPA Contextual Finetuning: Challenge 2 (Externalizing Factor Prediction)
#
# **Transfer learning workflow:**
# 1. Load pretrained MAE+Aux encoder from `best_mae_1_aux_1/best_pretrain.pt`
# 2. Transfer to SignalJEPA_Contextual (full encoder: feature + pos + transformer)
# 3. Finetune on Challenge 2 with fixed-length windowing
#
# **Challenge 2 task:**
# - **Input**: 2s EEG windows (randomly cropped from 4s windows, multi-task data)
# - **Target**: Externalizing factor (psychopathology score from CBCL)
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
import math
from collections import defaultdict
from typing import Optional
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import r2_score, mean_absolute_error as mae

# Braindecode imports
from braindecode.models import SignalJEPA, SignalJEPA_Contextual
from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.preprocessing import create_fixed_length_windows

# EEGDash imports
from eegdash import EEGChallengeDataset

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
    """Configuration for SignalJEPA Contextual finetuning on Challenge 2."""

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
    n_epochs = 150  # Longer training for Challenge 2
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

    # Challenge 2 specific: fixed-length windowing with random crops
    window_size_s = 4.0  # 4s windows
    window_stride_s = 2.0  # 2s stride (50% overlap)
    crop_size_s = 2.0  # Random 2s crops during training

    # Excluded subjects (from competition guidelines)
    excluded_subjects = [
        "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
        "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
        "NDARBA381JGH"
    ]

    # Tracking
    use_wandb = False
    experiment_name = f"signaljepa_contextual_c2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = CACHE_PATH / "signaljepa_finetune" / experiment_name

cfg = Config()
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

print(f"Configuration:")
print(f"  HBN_ROOT: {cfg.HBN_ROOT}")
print(f"  Pretrained checkpoint: {cfg.pretrained_checkpoint}")
print(f"  Output directory: {cfg.checkpoint_dir}")
print(f"  Train releases: {cfg.train_releases}")

# %% Dataset Wrapper for Random Crops
class DatasetWrapper(BaseDataset):
    """Wraps windowed dataset to add random 2s crops from 4s windows."""

    def __init__(self, dataset, crop_size_samples=200, target_name="externalizing", seed=2025):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name = target_name
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get original window
        X, _, crop_inds = self.dataset[index]

        # Get target (externalizing factor)
        target = float(self.dataset.description[self.target_name])

        # Random crop to 2s (200 samples)
        i_window, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples, f"Window too short: {i_stop - i_start} < {self.crop_size_samples}"

        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        X = X[:, start_offset:start_offset + self.crop_size_samples]

        return X, target, crop_inds, {}

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

# Extract encoder state dict
if 'encoder_state_dict' in checkpoint:
    encoder_state = checkpoint['encoder_state_dict']
    print(f"   ✅ Found 'encoder_state_dict' key")
else:
    raise KeyError(f"Could not find 'encoder_state_dict' in checkpoint.")

# Create base SignalJEPA model
print(f"\n🏗️  Creating base SignalJEPA model...")
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
    n_outputs=None
)

# Load pretrained weights
print(f"   Loading pretrained encoder weights...")
base_model.load_state_dict(encoder_state, strict=False)
print(f"   ✅ Pretrained weights loaded")

# Transfer to SignalJEPA_Contextual
print(f"\n🔄 Transferring to SignalJEPA_Contextual...")
model = SignalJEPA_Contextual.from_pretrained(
    base_model,
    n_outputs=1,  # Regression (externalizing factor)
    n_spat_filters=cfg.n_spat_filters
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 Model parameters: {total_params:,} (trainable: {trainable_params:,})")

# %% Load Challenge 2 Data (Fixed-Length Windowing)

print("\n" + "="*60)
print("Loading Challenge 2 Data (Fixed-Length Windows)")
print("="*60)

print(f"\n📊 Loading data from {len(cfg.train_releases)} releases...")

all_windows = []

for release_idx, release in enumerate(cfg.train_releases):
    print(f"\n[{release_idx+1}/{len(cfg.train_releases)}] Loading {release}...")

    # Load dataset with externalizing field
    dataset = EEGChallengeDataset(
        task="contrastChangeDetection",  # You can add more tasks here
        release=release,
        cache_dir=cfg.HBN_ROOT,
        mini=False,
        description_fields=["subject", "externalizing"]
    )
    print(f"   Loaded {len(dataset.datasets)} recordings")

    # Filter: sufficient length + valid externalizing score + correct channels
    filtered_datasets = []
    for ds in dataset.datasets:
        # Check conditions
        has_sufficient_length = ds.raw.n_times >= int(cfg.window_size_s * cfg.sfreq)
        has_valid_externalizing = not math.isnan(ds.description["externalizing"])
        has_correct_channels = len(ds.raw.ch_names) == 129
        is_not_excluded = ds.description["subject"] not in cfg.excluded_subjects

        if has_sufficient_length and has_valid_externalizing and has_correct_channels and is_not_excluded:
            filtered_datasets.append(ds)

    dataset = BaseConcatDataset(filtered_datasets)
    print(f"   {len(dataset.datasets)} recordings after filtering")

    if len(dataset.datasets) == 0:
        print(f"   ⚠️  No valid recordings, skipping release")
        continue

    # Create 4s windows with 2s stride
    windows = create_fixed_length_windows(
        dataset,
        window_size_samples=int(cfg.window_size_s * cfg.sfreq),
        window_stride_samples=int(cfg.window_stride_s * cfg.sfreq),
        drop_last_window=True
    )

    print(f"   Created {len(windows)} 4s windows")
    all_windows.extend(windows.datasets)

# Combine all windows
all_windows = BaseConcatDataset(all_windows)
print(f"\n✅ Total 4s windows: {len(all_windows):,}")

# Wrap with DatasetWrapper for random 2s crops
print(f"\n🔄 Wrapping windows with random 2s crop wrapper...")
wrapped_windows = BaseConcatDataset([
    DatasetWrapper(ds, crop_size_samples=int(cfg.crop_size_s * cfg.sfreq), seed=cfg.seed)
    for ds in all_windows.datasets
])
print(f"   ✅ Wrapped {len(wrapped_windows)} windows")

# %% Subject-Level Train/Val Split

print("\n" + "="*60)
print("Splitting Data (Subject-Level)")
print("="*60)

# Get unique subjects
subjects = list(set([
    ds.dataset.description["subject"]
    for ds in wrapped_windows.datasets
]))
print(f"   Total subjects: {len(subjects)}")

# Subject-level split
train_subjects, val_subjects = train_test_split(
    subjects,
    test_size=cfg.val_frac,
    random_state=cfg.seed,
    shuffle=True
)

print(f"   Train subjects: {len(train_subjects)}")
print(f"   Val subjects: {len(val_subjects)}")

# Split windows by subject
train_windows = BaseConcatDataset([
    ds for ds in wrapped_windows.datasets
    if ds.dataset.description["subject"] in train_subjects
])
val_windows = BaseConcatDataset([
    ds for ds in wrapped_windows.datasets
    if ds.dataset.description["subject"] in val_subjects
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

# %% Training Setup

print("\n" + "="*60)
print("Training Setup")
print("="*60)

# Separate encoder from task head
encoder_params = (
    list(model.feature_encoder.parameters()) +
    list(model.pos_encoder.parameters()) +
    list(model.transformer.parameters())
)
new_params = list(model.final_layer.parameters())

print(f"\n🔧 Optimizer configuration:")
print(f"   Encoder: {sum(p.numel() for p in encoder_params):,} params (lr={cfg.encoder_lr})")
print(f"   Task head: {sum(p.numel() for p in new_params):,} params (lr={cfg.new_layers_lr})")

# Dual learning rate optimizer
optimizer = torch.optim.AdamW([
    {'params': encoder_params, 'lr': cfg.encoder_lr, 'weight_decay': cfg.weight_decay},
    {'params': new_params, 'lr': cfg.new_layers_lr, 'weight_decay': cfg.weight_decay}
])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
loss_fn = nn.MSELoss()

# Encoder freezing helpers
def freeze_encoder(model):
    for param in model.feature_encoder.parameters(): param.requires_grad = False
    for param in model.pos_encoder.parameters(): param.requires_grad = False
    for param in model.transformer.parameters(): param.requires_grad = False

def unfreeze_encoder(model):
    for param in model.feature_encoder.parameters(): param.requires_grad = True
    for param in model.pos_encoder.parameters(): param.requires_grad = True
    for param in model.transformer.parameters(): param.requires_grad = True

# Competition metrics
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse_val = rmse(y_true, y_pred)
    nrmse_val = rmse_val / y_true.std()
    r2_val = r2_score(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    return {'rmse': rmse_val, 'nrmse': nrmse_val, 'r2': r2_val, 'mae': mae_val}

# WandB
if cfg.use_wandb and WANDB_AVAILABLE:
    wandb.init(project="signaljepa-finetune", name=cfg.experiment_name, config=vars(cfg))

# %% Training Loop

print("\n" + "="*60)
print("Training Loop")
print("="*60)

best_val_nrmse = float('inf')
best_epoch = 0
patience_counter = 0
training_history = defaultdict(list)

for epoch in range(cfg.n_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{cfg.n_epochs}")
    print(f"{'='*60}")

    # Warmup management
    if epoch == 0:
        freeze_encoder(model)
        print("🧊 Encoder FROZEN for warmup")
    elif epoch == cfg.warmup_epochs:
        unfreeze_encoder(model)
        print("🔥 Encoder UNFROZEN - full finetuning begins")

    # Training
    model.train()
    train_loss, train_preds, train_targets = [], [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        X, y = batch[0].to(device).float(), batch[1].to(device).float()

        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y.unsqueeze(1) if y.dim() == 1 else y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        train_loss.append(loss.item())
        train_preds.extend(preds.detach().cpu().numpy().flatten())
        train_targets.extend(y.detach().cpu().numpy().flatten())

    train_metrics = compute_metrics(train_targets, train_preds)

    # Validation
    model.eval()
    val_preds, val_targets = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            X, y = batch[0].to(device).float(), batch[1].to(device).float()
            preds = model(X)
            val_preds.extend(preds.cpu().numpy().flatten())
            val_targets.extend(y.cpu().numpy().flatten())

    val_metrics = compute_metrics(val_targets, val_preds)

    print(f"\n📊 Epoch {epoch+1}:")
    print(f"   Train: NRMSE={train_metrics['nrmse']:.4f}, R²={train_metrics['r2']:.4f}")
    print(f"   Val:   NRMSE={val_metrics['nrmse']:.4f}, R²={val_metrics['r2']:.4f}")

    # Track history
    for key in ['nrmse', 'rmse', 'r2']:
        training_history[f'train_{key}'].append(train_metrics[key])
        training_history[f'val_{key}'].append(val_metrics[key])

    # WandB logging
    if cfg.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'epoch': epoch + 1,
            'train/nrmse': train_metrics['nrmse'],
            'val/nrmse': val_metrics['nrmse'],
            'train/r2': train_metrics['r2'],
            'val/r2': val_metrics['r2']
        })

    # Save best model
    if val_metrics['nrmse'] < best_val_nrmse:
        best_val_nrmse = val_metrics['nrmse']
        best_epoch = epoch + 1
        patience_counter = 0

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_nrmse': val_metrics['nrmse'],
            'val_r2': val_metrics['r2'],
            'config': vars(cfg)
        }, cfg.checkpoint_dir / "best_model.pt")

        print(f"   ✅ New best: Val NRMSE={best_val_nrmse:.4f}")
    else:
        patience_counter += 1
        print(f"   No improvement ({patience_counter}/{cfg.early_stopping_patience})")

        if patience_counter >= cfg.early_stopping_patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            break

    scheduler.step()

print(f"\n✅ Training complete. Best Val NRMSE: {best_val_nrmse:.4f} (epoch {best_epoch})")

# %% Evaluation

print("\n" + "="*60)
print("Final Evaluation")
print("="*60)

best_checkpoint = torch.load(cfg.checkpoint_dir / "best_model.pt", map_location=device)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()

val_preds, val_targets = [], []
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        X, y = batch[0].to(device).float(), batch[1].to(device).float()
        preds = model(X)
        val_preds.extend(preds.cpu().numpy().flatten())
        val_targets.extend(y.cpu().numpy().flatten())

final_metrics = compute_metrics(val_targets, val_preds)

print(f"\n📊 Final Validation Metrics:")
print(f"   NRMSE: {final_metrics['nrmse']:.4f}  ← Competition metric")
print(f"   RMSE:  {final_metrics['rmse']:.4f}")
print(f"   R²:    {final_metrics['r2']:.4f}")
print(f"   MAE:   {final_metrics['mae']:.4f}")

# %% Visualization

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training curves
axes[0].plot(training_history['train_nrmse'], label='Train', linewidth=2)
axes[0].plot(training_history['val_nrmse'], label='Val', linewidth=2)
axes[0].axvline(cfg.warmup_epochs, color='r', linestyle='--', alpha=0.5, label='Unfrozen')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('NRMSE')
axes[0].set_title('Training Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# R² curves
axes[1].plot(training_history['train_r2'], label='Train', linewidth=2)
axes[1].plot(training_history['val_r2'], label='Val', linewidth=2)
axes[1].axvline(cfg.warmup_epochs, color='r', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('R²')
axes[1].set_title('R² Curves')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Predictions
axes[2].scatter(val_targets, val_preds, alpha=0.3, s=10)
min_val = min(min(val_targets), min(val_preds))
max_val = max(max(val_targets), max(val_preds))
axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[2].set_xlabel('Ground Truth (Externalizing)')
axes[2].set_ylabel('Predicted')
axes[2].set_title(f'Predictions (NRMSE={final_metrics["nrmse"]:.4f})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(cfg.checkpoint_dir / "training_curves.png", dpi=300)
print(f"✅ Saved: {cfg.checkpoint_dir / 'training_curves.png'}")

if cfg.use_wandb and WANDB_AVAILABLE:
    wandb.log({"training_curves": wandb.Image(fig)})

plt.show()
