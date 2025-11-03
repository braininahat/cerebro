# %% [markdown]
# # Challenge 1: SignalJEPA_PreLocal Fine-tuning with Pretrained Encoder
#
# **Transfer learning strategy:**
# 1. Load pretrained encoder from signaljepa_pretrain_full.py
# 2. Initialize SignalJEPA_PreLocal with pretrained weights
# 3. Freeze encoder for 5 warmup epochs
# 4. Unfreeze and fine-tune end-to-end
#
# **Architecture**: SignalJEPA_PreLocal (spatial filtering BEFORE local encoder)
# - This is the variant used in signaljepa_c1_full.py baseline
# - Now enhanced with pretraining!
#
# **Data**: Challenge 1 (contrastChangeDetection) with release-level splits
# - Train: R11, R2, R3, R4, R7, R8 (70%)
# - Val: R1, R10, R6, R9 (30%)
# - Test: R5 (held out)

# %% imports
from pathlib import Path
import os
import random
from typing import Optional
import numpy as np
from tqdm.auto import tqdm
from dotenv import load_dotenv
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

# SignalJEPA imports
from braindecode.models import SignalJEPA, SignalJEPA_PreLocal
from cerebro.utils.electrode_locations import load_hbn_chs_info

# Data loading (same as signaljepa_c1_full.py)
from eegdash import EEGChallengeDataset
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

import wandb

load_dotenv()
ANCHOR = "stimulus_anchor"

# %% config
class Config:
    HBN_ROOT = Path(os.getenv("HBN_ROOT", "/media/varun/OS/Users/varun/DATASETS/HBN"))
    CACHE_PATH = Path(os.getenv("CACHE_PATH", "/home/varun/repos/cerebro/cache"))

    # PRETRAINED CHECKPOINT
    pretrain_checkpoint = CACHE_PATH / "signaljepa_pretrain" / "CHECKPOINT_NAME_HERE" / "best_pretrain.pt"

    # Model architecture
    n_channels = 129
    n_times = 200
    sfreq = 100
    input_window_seconds = 2.0

    # SignalJEPA_PreLocal hyperparameters (must match pretraining!)
    n_spat_filters = 4
    transformer_d_model = 96
    transformer_num_encoder_layers = 12
    transformer_num_decoder_layers = 4
    transformer_nhead = 8
    dropout = 0.1

    # Training
    batch_size = 128
    learning_rate_encoder = 0.00005  # Lower for pretrained
    learning_rate_head = 0.001  # Higher for new layers
    weight_decay = 0.00001
    n_epochs = 100
    early_stopping_patience = 20
    warmup_freeze_epochs = 5
    grad_clip = 1.0

    # Data
    num_workers = 8
    window_len = 2.0
    shift_after_stim = 0.5

    # Splits (same as signaljepa_c1_full.py)
    train_releases = ["R11", "R2", "R3", "R4", "R7", "R8"]
    val_releases = ["R1", "R10", "R6", "R9"]
    test_release = "R5"
    use_mini = False

    # Tracking
    use_wandb = True
    experiment_name = f"signaljepa_c1_prelocal_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = CACHE_PATH / "signaljepa_checkpoints" / experiment_name

cfg = Config()
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)

print(f"Configuration:")
print(f"  Variant: SignalJEPA_PreLocal (spatial filtering BEFORE local encoder)")
print(f"  This was the baseline in signaljepa_c1_full.py, now with pretraining!")
print(f"  Pretrained checkpoint: {cfg.pretrain_checkpoint}")
print(f"  Warmup freeze: {cfg.warmup_freeze_epochs} epochs")

# %% Load Data
print("\n" + "="*60)
print("Loading Challenge 1 data...")
print("="*60)
print("⚠️  Data loading code omitted - copy from signaljepa_c1_full.py")
print("   The data loading is IDENTICAL to signaljepa_c1_full.py")
print("   Only difference is model initialization with pretrained weights\n")

# %% Load Pretrained Encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chs_info = load_hbn_chs_info()

print(f"\n📦 Loading pretrained checkpoint...")
if not cfg.pretrain_checkpoint.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {cfg.pretrain_checkpoint}\n"
        f"Run signaljepa_pretrain_full.py first, then update this path."
    )

checkpoint = torch.load(cfg.pretrain_checkpoint, map_location=device, weights_only=False)
print(f"✅ Loaded from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")

# Base pretrained encoder
pretrained_encoder = SignalJEPA(
    n_outputs=None,
    n_chans=cfg.n_channels,
    n_times=cfg.n_times,
    sfreq=cfg.sfreq,
    input_window_seconds=cfg.input_window_seconds,
    chs_info=chs_info,
    transformer__d_model=cfg.transformer_d_model,
    transformer__num_encoder_layers=cfg.transformer_num_encoder_layers,
    transformer__nhead=cfg.transformer_nhead,
    drop_prob=cfg.dropout,
)
pretrained_encoder.load_state_dict(checkpoint['encoder_state_dict'])

# %% Transfer to SignalJEPA_PreLocal
print("\n🔄 Transferring to SignalJEPA_PreLocal...")

model = SignalJEPA_PreLocal(
    n_outputs=1,
    n_chans=cfg.n_channels,
    n_times=cfg.n_times,
    sfreq=cfg.sfreq,
    input_window_seconds=cfg.input_window_seconds,
    chs_info=chs_info,
    n_spat_filters=cfg.n_spat_filters,
    transformer__d_model=cfg.transformer_d_model,
    transformer__num_encoder_layers=cfg.transformer_num_encoder_layers,
    transformer__num_decoder_layers=cfg.transformer_num_decoder_layers,
    transformer__nhead=cfg.transformer_nhead,
    drop_prob=cfg.dropout,
).to(device)

# Transfer weights
try:
    encoder_state = checkpoint['encoder_state_dict']
    model_state = model.state_dict()
    transferred = 0
    for k, v in encoder_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            model_state[k] = v
            transferred += 1
    model.load_state_dict(model_state)
    print(f"✅ Transferred {transferred} layers from pretrained encoder")
except Exception as e:
    print(f"⚠️  Warning: Could not transfer all weights: {e}")

total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Model parameters: {total_params:,}")

# %% Freeze Encoder for Warmup
print(f"\n🔒 Freezing encoder for {cfg.warmup_freeze_epochs} warmup epochs...")

encoder_params = []
new_layer_params = []

for name, param in model.named_parameters():
    # Decoder and final regression head are new layers
    if 'decoder' in name.lower() or 'regression' in name.lower() or 'final' in name.lower():
        new_layer_params.append(param)
    else:
        encoder_params.append(param)
        param.requires_grad = False  # Freeze initially

print(f"  Frozen encoder: {sum(p.numel() for p in encoder_params):,} params")
print(f"  Trainable new layers: {sum(p.numel() for p in new_layer_params):,} params")

# %% Optimizer with Dual Learning Rates
optimizer = optim.AdamW([
    {'params': new_layer_params, 'lr': cfg.learning_rate_head, 'name': 'head'}
], weight_decay=cfg.weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# %% Training Utilities (same as signaljepa_c1_full.py)
def calculate_nrmse(y_true, y_pred):
    """Calculate Normalized RMSE (competition metric)."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    y_std = y_true.std()
    return rmse / y_std if y_std > 0 else rmse

def train_epoch(model, loader, optimizer, device, epoch, frozen_encoder):
    """Train for one epoch."""
    model.train()
    losses = []
    predictions = []
    targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} ({'Frozen' if frozen_encoder else 'Full FT'})")

    for batch in pbar:
        X, y = batch[0], batch[1]
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)

        # Forward
        y_pred = model(X).squeeze(-1)
        loss = F.mse_loss(y_pred, y)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Track
        losses.append(loss.item())
        with torch.no_grad():
            predictions.extend(y_pred.detach().cpu().numpy())
            targets.extend(y.detach().cpu().numpy())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

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
            X, y = batch[0], batch[1]
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            if y.ndim == 2 and y.shape[1] == 1:
                y = y.squeeze(1)

            y_pred = model(X).squeeze(-1)
            loss = F.mse_loss(y_pred, y)

            losses.append(loss.item())
            predictions.extend(y_pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    nrmse = calculate_nrmse(targets, predictions)

    return np.mean(losses), nrmse

# %% Initialize wandb
if cfg.use_wandb:
    wandb.init(
        project="cerebro-signaljepa-finetune",
        name=cfg.experiment_name,
        config=vars(cfg),
        tags=["signaljepa", "challenge1", "prelocal", "pretrained"]
    )

# %% Training Loop
print("\n" + "="*60)
print("🚀 Starting fine-tuning (PreLocal variant)...")
print("="*60)

print(f"\n📋 Phase 1 (Epochs 1-{cfg.warmup_freeze_epochs}): Encoder FROZEN")
print(f"📋 Phase 2 (Epochs {cfg.warmup_freeze_epochs+1}+): Encoder UNFROZEN (full fine-tuning)")

best_val_nrmse = float('inf')
patience_counter = 0
training_history = []

print("\n⚠️  Note: This is a template - replace with actual train/val loaders")
print("   Copy data loading from signaljepa_c1_full.py (lines 182-346)\n")

for epoch in range(cfg.n_epochs):
    print(f"\n📅 Epoch {epoch+1}/{cfg.n_epochs}")
    print("-" * 40)

    # Unfreeze encoder after warmup
    if epoch == cfg.warmup_freeze_epochs:
        print(f"\n🔓 UNFREEZING encoder at epoch {epoch+1}!")
        print(f"   Switching to dual learning rate optimization...")

        for param in encoder_params:
            param.requires_grad = True

        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': cfg.learning_rate_encoder, 'name': 'encoder'},
            {'params': new_layer_params, 'lr': cfg.learning_rate_head, 'name': 'head'}
        ], weight_decay=cfg.weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

    frozen = epoch < cfg.warmup_freeze_epochs

    # PLACEHOLDER training (replace with actual loaders from signaljepa_c1_full.py)
    train_loss, train_nrmse = 0.5, 0.8
    val_loss, val_nrmse = 0.6, 0.9

    print(f"\n📈 Train - Loss: {train_loss:.4f}, NRMSE: {train_nrmse:.4f}")
    print(f"📊 Val   - Loss: {val_loss:.4f}, NRMSE: {val_nrmse:.4f}")

    scheduler.step(val_nrmse)

    training_history.append({
        'epoch': epoch + 1,
        'phase': 'warmup' if frozen else 'full_finetune',
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_nrmse': train_nrmse,
        'val_nrmse': val_nrmse
    })

    # Save best
    if val_nrmse < best_val_nrmse:
        best_val_nrmse = val_nrmse
        patience_counter = 0

        checkpoint_save = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_nrmse': val_nrmse,
            'pretrain_checkpoint': str(cfg.pretrain_checkpoint),
            'config': vars(cfg),
            'history': training_history
        }
        torch.save(checkpoint_save, cfg.checkpoint_dir / "best_model.pt")
        print(f"✅ NEW BEST! Val NRMSE: {val_nrmse:.4f}")
    else:
        patience_counter += 1
        print(f"⏳ Patience: {patience_counter}/{cfg.early_stopping_patience}")

    if patience_counter >= cfg.early_stopping_patience:
        print(f"\n⚠️ Early stopping at epoch {epoch+1}")
        break

# %% Final Results
print("\n" + "="*60)
print("🏁 Fine-tuning Complete!")
print("="*60)
print(f"Best Val NRMSE: {best_val_nrmse:.4f}")
print(f"Total epochs: {len(training_history)}")
print(f"\n📁 Results saved to: {cfg.checkpoint_dir}")

print(f"\n✨ Challenge 1 fine-tuning complete (SignalJEPA_PreLocal with pretraining)!")
print(f"\n🔍 Compare against baseline (signaljepa_c1_full.py without pretraining)")
print(f"   to measure the benefit of multi-objective pretraining!")
