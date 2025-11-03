# %% [markdown]
# # SignalJEPA PostLocal Finetuning: Challenge 1 (Response Time Prediction)
#
# **Transfer learning workflow:**
# 1. Load pretrained MAE+Aux encoder from `best_mae_1_aux_1/best_pretrain.pt`
# 2. Transfer to SignalJEPA_PostLocal (feature encoder only, transformer unused)
# 3. Finetune on Challenge 1 with stimulus-locked windowing
#
# **PostLocal architecture:**
# - **Transfers**: feature_encoder only (lighter than Contextual)
# - **Skips**: pos_encoder + transformer (not used in forward pass)
# - **Use case**: Faster training/inference, baseline comparison
#
# **Challenge 1 task:**
# - **Input**: 2s EEG windows starting 0.5s after stimulus onset
# - **Target**: Response time from stimulus (rt_from_stimulus)
# - **Metric**: NRMSE (competition primary metric)
#
# **Training strategy:**
# - **Warmup (epochs 0-4)**: Freeze encoder, train head only (lr=1e-3)
# - **Finetuning (epochs 5+)**: Unfreeze encoder, dual LR (encoder=1e-4, head=1e-3)

# %% Imports
from pathlib import Path
import os
import random
from collections import defaultdict
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
from braindecode.models import SignalJEPA, SignalJEPA_PostLocal
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

# Optional: WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Set seeds
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# %% Configuration
class Config:
    """Configuration for SignalJEPA PostLocal finetuning on Challenge 1."""

    HBN_ROOT = Path(os.getenv("HBN_ROOT", "/media/varun/OS/Users/varun/DATASETS/HBN"))
    CACHE_PATH = Path(os.getenv("CACHE_PATH", "/home/varun/repos/cerebro/cache"))

    pretrained_checkpoint = CACHE_PATH / "signaljepa_pretrain/best_mae_1_aux_1/best_pretrain.pt"

    # Model architecture (must match pretraining)
    n_channels = 129
    n_times = 200
    sfreq = 100
    input_window_seconds = 2.0
    transformer_d_model = 64
    transformer_num_encoder_layers = 12
    transformer_nhead = 8
    dropout = 0.1
    n_spat_filters = 4

    # Training hyperparameters
    batch_size = 128
    warmup_epochs = 5
    n_epochs = 100
    encoder_lr = 1e-4  # Lower for pretrained encoder
    new_layers_lr = 1e-3  # Higher for new head
    weight_decay = 1e-4
    early_stopping_patience = 15
    grad_clip = 1.0

    # Data
    train_releases = ["R11", "R2", "R3", "R4", "R7", "R8", "R9", "R10"]
    val_frac = 0.1
    seed = 2025
    num_workers = 8

    # Challenge 1: stimulus-locked windowing
    shift_after_stim = 0.5
    window_len = 2.0
    epoch_len_s = 2.0

    excluded_subjects = [
        "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
        "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
        "NDARBA381JGH"
    ]

    use_wandb = False
    experiment_name = f"signaljepa_postlocal_c1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = CACHE_PATH / "signaljepa_finetune" / experiment_name

cfg = Config()
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

print(f"Configuration:")
print(f"  Architecture: PostLocal (feature_encoder only)")
print(f"  Pretrained checkpoint: {cfg.pretrained_checkpoint}")
print(f"  Output: {cfg.checkpoint_dir}")

# %% Load Pretrained and Transfer to PostLocal

print("\n" + "="*60)
print("Loading Pretrained Checkpoint → PostLocal")
print("="*60)

# Load electrode locations
chs_info = load_hbn_chs_info()

# Load checkpoint
checkpoint = torch.load(cfg.pretrained_checkpoint, map_location=device, weights_only=False)
encoder_state = checkpoint['encoder_state_dict']
print(f"✅ Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")

# Create base SignalJEPA
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
base_model.load_state_dict(encoder_state, strict=False)

# Transfer to PostLocal (feature_encoder only)
print(f"\n🔄 Transferring to SignalJEPA_PostLocal...")
print(f"   This transfers: feature_encoder ONLY (transformer unused)")

model = SignalJEPA_PostLocal.from_pretrained(
    base_model,
    n_outputs=1,
    n_spat_filters=cfg.n_spat_filters
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 Model: {total_params:,} params ({trainable_params:,} trainable)")
print(f"   (Note: Lighter than Contextual - no transformer)")

# %% Load Challenge 1 Data

print("\n" + "="*60)
print("Loading Challenge 1 Data (Stimulus-Locked)")
print("="*60)

all_windows = []

for release_idx, release in enumerate(cfg.train_releases):
    print(f"[{release_idx+1}/{len(cfg.train_releases)}] {release}...", end=" ")

    dataset = EEGChallengeDataset(
        task="contrastChangeDetection",
        release=release,
        cache_dir=cfg.HBN_ROOT,
        mini=False
    )

    # Annotate trials
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

    # Keep recordings with stimulus anchors
    dataset = keep_only_recordings_with("stimulus_anchor", dataset)

    # Create stimulus-locked windows
    windows = create_windows_from_events(
        dataset,
        mapping={"stimulus_anchor": 0},
        trial_start_offset_samples=int(cfg.shift_after_stim * cfg.sfreq),
        trial_stop_offset_samples=int((cfg.shift_after_stim + cfg.window_len) * cfg.sfreq),
        window_size_samples=int(cfg.epoch_len_s * cfg.sfreq),
        window_stride_samples=cfg.sfreq,
        preload=True
    )

    windows = add_extras_columns(
        windows, dataset, desc="stimulus_anchor",
        keys=("target", "rt_from_stimulus", "correct")
    )

    print(f"{len(windows)} windows")
    all_windows.extend(windows.datasets)

all_windows = BaseConcatDataset(all_windows)
print(f"\n✅ Total: {len(all_windows):,} windows")

# %% Subject-Level Split

print("\n" + "="*60)
print("Splitting Data (Subject-Level)")
print("="*60)

metadata = all_windows.get_metadata()
subjects = [s for s in metadata["subject"].unique() if s not in cfg.excluded_subjects]

train_subj, val_subj = train_test_split(
    subjects, test_size=cfg.val_frac, random_state=cfg.seed, shuffle=True
)

subject_split = all_windows.split("subject")
train_windows = BaseConcatDataset([subject_split[s] for s in train_subj if s in subject_split])
val_windows = BaseConcatDataset([subject_split[s] for s in val_subj if s in subject_split])

print(f"Train: {len(train_windows):,} windows ({len(train_subj)} subjects)")
print(f"Val: {len(val_windows):,} windows ({len(val_subj)} subjects)")

train_loader = DataLoader(train_windows, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
val_loader = DataLoader(val_windows, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

# %% Training Setup

print("\n" + "="*60)
print("Training Setup")
print("="*60)

# Separate encoder from head
# PostLocal: feature_encoder only (no pos_encoder or transformer)
encoder_params = list(model.feature_encoder.parameters())
new_params = list(model.final_layer.parameters())

print(f"Encoder: {sum(p.numel() for p in encoder_params):,} params (lr={cfg.encoder_lr})")
print(f"Head: {sum(p.numel() for p in new_params):,} params (lr={cfg.new_layers_lr})")

optimizer = torch.optim.AdamW([
    {'params': encoder_params, 'lr': cfg.encoder_lr, 'weight_decay': cfg.weight_decay},
    {'params': new_params, 'lr': cfg.new_layers_lr, 'weight_decay': cfg.weight_decay}
])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
loss_fn = nn.MSELoss()

def freeze_encoder(model):
    for param in model.feature_encoder.parameters(): param.requires_grad = False

def unfreeze_encoder(model):
    for param in model.feature_encoder.parameters(): param.requires_grad = True

def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse_val = rmse(y_true, y_pred)
    nrmse_val = rmse_val / y_true.std()
    return {
        'rmse': rmse_val,
        'nrmse': nrmse_val,
        'r2': r2_score(y_true, y_pred),
        'mae': mae(y_true, y_pred)
    }

if cfg.use_wandb and WANDB_AVAILABLE:
    wandb.init(project="signaljepa-finetune", name=cfg.experiment_name, config=vars(cfg))

# %% Training Loop

print("\n" + "="*60)
print("Training")
print("="*60)

best_val_nrmse = float('inf')
best_epoch = 0
patience_counter = 0
history = defaultdict(list)

for epoch in range(cfg.n_epochs):
    print(f"\nEpoch {epoch+1}/{cfg.n_epochs}")

    if epoch == 0:
        freeze_encoder(model)
        print("🧊 Encoder frozen (warmup)")
    elif epoch == cfg.warmup_epochs:
        unfreeze_encoder(model)
        print("🔥 Encoder unfrozen")

    # Train
    model.train()
    train_preds, train_targets = [], []
    for batch in tqdm(train_loader, desc="Train"):
        X, y = batch[0].to(device).float(), batch[1].to(device).float()
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y.unsqueeze(1) if y.dim() == 1 else y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        train_preds.extend(preds.detach().cpu().numpy().flatten())
        train_targets.extend(y.detach().cpu().numpy().flatten())

    train_metrics = compute_metrics(train_targets, train_preds)

    # Val
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val"):
            X, y = batch[0].to(device).float(), batch[1].to(device).float()
            preds = model(X)
            val_preds.extend(preds.cpu().numpy().flatten())
            val_targets.extend(y.cpu().numpy().flatten())

    val_metrics = compute_metrics(val_targets, val_preds)

    print(f"Train NRMSE={train_metrics['nrmse']:.4f}, Val NRMSE={val_metrics['nrmse']:.4f}, Val R²={val_metrics['r2']:.4f}")

    # Track
    for k in ['nrmse', 'rmse', 'r2']:
        history[f'train_{k}'].append(train_metrics[k])
        history[f'val_{k}'].append(val_metrics[k])

    if cfg.use_wandb and WANDB_AVAILABLE:
        wandb.log({'epoch': epoch+1, 'train/nrmse': train_metrics['nrmse'], 'val/nrmse': val_metrics['nrmse'], 'val/r2': val_metrics['r2']})

    # Save best
    if val_metrics['nrmse'] < best_val_nrmse:
        best_val_nrmse = val_metrics['nrmse']
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'val_nrmse': val_metrics['nrmse'],
            'val_r2': val_metrics['r2'],
            'config': vars(cfg)
        }, cfg.checkpoint_dir / "best_model.pt")
        print(f"✅ Best: {best_val_nrmse:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= cfg.early_stopping_patience:
            print(f"🛑 Early stop")
            break

    scheduler.step()

print(f"\n✅ Done. Best NRMSE={best_val_nrmse:.4f} @ epoch {best_epoch}")

# %% Evaluation

print("\n" + "="*60)
print("Final Evaluation")
print("="*60)

model.load_state_dict(torch.load(cfg.checkpoint_dir / "best_model.pt", map_location=device)['model_state_dict'])
model.eval()

val_preds, val_targets = [], []
with torch.no_grad():
    for batch in tqdm(val_loader):
        X, y = batch[0].to(device).float(), batch[1].to(device).float()
        preds = model(X)
        val_preds.extend(preds.cpu().numpy().flatten())
        val_targets.extend(y.cpu().numpy().flatten())

final = compute_metrics(val_targets, val_preds)
print(f"\n📊 Final Metrics:")
print(f"   NRMSE: {final['nrmse']:.4f}  ← Competition")
print(f"   RMSE:  {final['rmse']:.4f}")
print(f"   R²:    {final['r2']:.4f}")
print(f"   MAE:   {final['mae']:.4f}")

# %% Visualization

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history['train_nrmse'], label='Train', linewidth=2)
axes[0].plot(history['val_nrmse'], label='Val', linewidth=2)
axes[0].axvline(cfg.warmup_epochs, color='r', linestyle='--', alpha=0.5, label='Unfrozen')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('NRMSE')
axes[0].set_title('Training Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(val_targets, val_preds, alpha=0.3, s=10)
min_v, max_v = min(min(val_targets), min(val_preds)), max(max(val_targets), max(val_preds))
axes[1].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
axes[1].set_xlabel('Ground Truth (RT)')
axes[1].set_ylabel('Predicted')
axes[1].set_title(f'PostLocal: NRMSE={final["nrmse"]:.4f}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(cfg.checkpoint_dir / "results.png", dpi=300)
print(f"✅ Saved: {cfg.checkpoint_dir / 'results.png'}")

if cfg.use_wandb and WANDB_AVAILABLE:
    wandb.log({"results": wandb.Image(fig)})

plt.show()
