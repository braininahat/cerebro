# %% [markdown]
# # SignalJEPA PostLocal Finetuning: Challenge 2 (Externalizing Factor Prediction)
#
# **Transfer learning workflow:**
# 1. Load pretrained MAE+Aux encoder from `best_mae_1_aux_1/best_pretrain.pt`
# 2. Transfer to SignalJEPA_PostLocal (feature encoder only, transformer unused)
# 3. Finetune on Challenge 2 with fixed-length windowing
#
# **PostLocal architecture:**
# - **Transfers**: feature_encoder only (lighter than Contextual)
# - **Skips**: pos_encoder + transformer
# - **Use case**: Faster baseline, compare vs Contextual
#
# **Challenge 2 task:**
# - **Input**: 2s random crops from 4s windows (multi-task EEG)
# - **Target**: Externalizing factor (psychopathology)
# - **Metric**: NRMSE

# %% Imports
from pathlib import Path
import os
import random
import math
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

from braindecode.models import SignalJEPA, SignalJEPA_PostLocal
from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.preprocessing import create_fixed_length_windows
from eegdash import EEGChallengeDataset
from cerebro.utils.electrode_locations import load_hbn_chs_info

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

# %% Configuration
class Config:
    HBN_ROOT = Path(os.getenv("HBN_ROOT", "/media/varun/OS/Users/varun/DATASETS/HBN"))
    CACHE_PATH = Path(os.getenv("CACHE_PATH", "/home/varun/repos/cerebro/cache"))
    pretrained_checkpoint = CACHE_PATH / "signaljepa_pretrain/best_mae_1_aux_1/best_pretrain.pt"

    # Model
    n_channels = 129
    n_times = 200
    sfreq = 100
    input_window_seconds = 2.0
    transformer_d_model = 64
    transformer_num_encoder_layers = 12
    transformer_nhead = 8
    dropout = 0.1
    n_spat_filters = 4

    # Training
    batch_size = 128
    warmup_epochs = 5
    n_epochs = 150  # Longer for C2
    encoder_lr = 1e-4
    new_layers_lr = 1e-3
    weight_decay = 1e-4
    early_stopping_patience = 15
    grad_clip = 1.0

    # Data
    train_releases = ["R11", "R2", "R3", "R4", "R7", "R8", "R9", "R10"]
    val_frac = 0.1
    seed = 2025
    num_workers = 8

    # Challenge 2: fixed windows + random crops
    window_size_s = 4.0
    window_stride_s = 2.0
    crop_size_s = 2.0

    excluded_subjects = [
        "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
        "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
        "NDARBA381JGH"
    ]

    use_wandb = False
    experiment_name = f"signaljepa_postlocal_c2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = CACHE_PATH / "signaljepa_finetune" / experiment_name

cfg = Config()
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

# %% Dataset Wrapper
class DatasetWrapper(BaseDataset):
    """Wrap 4s windows with random 2s crops."""
    def __init__(self, dataset, crop_size_samples=200, target_name="externalizing", seed=2025):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name = target_name
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]
        target = float(self.dataset.description[self.target_name])
        i_window, i_start, i_stop = crop_inds
        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        X = X[:, start_offset:start_offset + self.crop_size_samples]
        return X, target, crop_inds, {}

# %% Load Pretrained → PostLocal

print("\n" + "="*60)
print("Loading Pretrained → PostLocal")
print("="*60)

chs_info = load_hbn_chs_info()

checkpoint = torch.load(cfg.pretrained_checkpoint, map_location=device, weights_only=False)
encoder_state = checkpoint['encoder_state_dict']
print(f"✅ Checkpoint loaded (epoch {checkpoint.get('epoch', '?')})")

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

print(f"\n🔄 Transferring to PostLocal (feature_encoder only)...")
model = SignalJEPA_PostLocal.from_pretrained(base_model, n_outputs=1, n_spat_filters=cfg.n_spat_filters)
model = model.to(device)

print(f"📊 Model: {sum(p.numel() for p in model.parameters()):,} params")

# %% Load Challenge 2 Data

print("\n" + "="*60)
print("Loading Challenge 2 Data")
print("="*60)

all_windows = []

for i, release in enumerate(cfg.train_releases):
    print(f"[{i+1}/{len(cfg.train_releases)}] {release}...", end=" ")

    dataset = EEGChallengeDataset(
        task="contrastChangeDetection",
        release=release,
        cache_dir=cfg.HBN_ROOT,
        mini=False,
        description_fields=["subject", "externalizing"]
    )

    # Filter
    filtered = [
        ds for ds in dataset.datasets
        if ds.raw.n_times >= int(cfg.window_size_s * cfg.sfreq)
        and not math.isnan(ds.description["externalizing"])
        and len(ds.raw.ch_names) == 129
        and ds.description["subject"] not in cfg.excluded_subjects
    ]
    dataset = BaseConcatDataset(filtered)

    if len(dataset.datasets) == 0:
        print("skipped")
        continue

    # Create 4s windows
    windows = create_fixed_length_windows(
        dataset,
        window_size_samples=int(cfg.window_size_s * cfg.sfreq),
        window_stride_samples=int(cfg.window_stride_s * cfg.sfreq),
        drop_last_window=True
    )

    print(f"{len(windows)} windows")
    all_windows.extend(windows.datasets)

all_windows = BaseConcatDataset(all_windows)
print(f"\n✅ Total: {len(all_windows):,} 4s windows")

# Wrap for random 2s crops
wrapped = BaseConcatDataset([
    DatasetWrapper(ds, crop_size_samples=int(cfg.crop_size_s * cfg.sfreq), seed=cfg.seed)
    for ds in all_windows.datasets
])
print(f"✅ Wrapped: {len(wrapped):,} (with random 2s crops)")

# %% Subject Split

print("\n" + "="*60)
print("Splitting (Subject-Level)")
print("="*60)

subjects = list(set([ds.dataset.description["subject"] for ds in wrapped.datasets]))
train_subj, val_subj = train_test_split(subjects, test_size=cfg.val_frac, random_state=cfg.seed, shuffle=True)

train_windows = BaseConcatDataset([ds for ds in wrapped.datasets if ds.dataset.description["subject"] in train_subj])
val_windows = BaseConcatDataset([ds for ds in wrapped.datasets if ds.dataset.description["subject"] in val_subj])

print(f"Train: {len(train_windows):,} | Val: {len(val_windows):,}")

train_loader = DataLoader(train_windows, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
val_loader = DataLoader(val_windows, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

# %% Training Setup

encoder_params = list(model.feature_encoder.parameters())
new_params = list(model.final_layer.parameters())

optimizer = torch.optim.AdamW([
    {'params': encoder_params, 'lr': cfg.encoder_lr, 'weight_decay': cfg.weight_decay},
    {'params': new_params, 'lr': cfg.new_layers_lr, 'weight_decay': cfg.weight_decay}
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
loss_fn = nn.MSELoss()

def freeze_encoder(m):
    for p in m.feature_encoder.parameters(): p.requires_grad = False

def unfreeze_encoder(m):
    for p in m.feature_encoder.parameters(): p.requires_grad = True

def compute_metrics(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    r = rmse(yt, yp)
    return {'rmse': r, 'nrmse': r/yt.std(), 'r2': r2_score(yt, yp), 'mae': mae(yt, yp)}

if cfg.use_wandb and WANDB_AVAILABLE:
    wandb.init(project="signaljepa-finetune", name=cfg.experiment_name, config=vars(cfg))

# %% Training Loop

print("\n" + "="*60)
print("Training")
print("="*60)

best_val_nrmse = float('inf')
best_epoch = 0
patience = 0
history = defaultdict(list)

for epoch in range(cfg.n_epochs):
    print(f"\nEpoch {epoch+1}/{cfg.n_epochs}")

    if epoch == 0:
        freeze_encoder(model)
        print("🧊 Frozen")
    elif epoch == cfg.warmup_epochs:
        unfreeze_encoder(model)
        print("🔥 Unfrozen")

    # Train
    model.train()
    tr_preds, tr_targets = [], []
    for batch in tqdm(train_loader, desc="Train"):
        X, y = batch[0].to(device).float(), batch[1].to(device).float()
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y.unsqueeze(1) if y.dim() == 1 else y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        tr_preds.extend(preds.detach().cpu().numpy().flatten())
        tr_targets.extend(y.detach().cpu().numpy().flatten())

    tr_m = compute_metrics(tr_targets, tr_preds)

    # Val
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val"):
            X, y = batch[0].to(device).float(), batch[1].to(device).float()
            preds = model(X)
            val_preds.extend(preds.cpu().numpy().flatten())
            val_targets.extend(y.cpu().numpy().flatten())

    val_m = compute_metrics(val_targets, val_preds)

    print(f"Train NRMSE={tr_m['nrmse']:.4f}, Val NRMSE={val_m['nrmse']:.4f}, R²={val_m['r2']:.4f}")

    for k in ['nrmse', 'rmse', 'r2']:
        history[f'train_{k}'].append(tr_m[k])
        history[f'val_{k}'].append(val_m[k])

    if cfg.use_wandb and WANDB_AVAILABLE:
        wandb.log({'epoch': epoch+1, 'train/nrmse': tr_m['nrmse'], 'val/nrmse': val_m['nrmse'], 'val/r2': val_m['r2']})

    if val_m['nrmse'] < best_val_nrmse:
        best_val_nrmse = val_m['nrmse']
        best_epoch = epoch + 1
        patience = 0
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'val_nrmse': val_m['nrmse'],
            'val_r2': val_m['r2'],
            'config': vars(cfg)
        }, cfg.checkpoint_dir / "best_model.pt")
        print(f"✅ Best: {best_val_nrmse:.4f}")
    else:
        patience += 1
        if patience >= cfg.early_stopping_patience:
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
print(f"\n📊 Final:")
print(f"   NRMSE: {final['nrmse']:.4f}  ← Competition")
print(f"   RMSE:  {final['rmse']:.4f}")
print(f"   R²:    {final['r2']:.4f}")

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
axes[0].set_title('PostLocal C2: Training')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(val_targets, val_preds, alpha=0.3, s=10)
mn, mx = min(min(val_targets), min(val_preds)), max(max(val_targets), max(val_preds))
axes[1].plot([mn, mx], [mn, mx], 'r--', linewidth=2)
axes[1].set_xlabel('Ground Truth (Externalizing)')
axes[1].set_ylabel('Predicted')
axes[1].set_title(f'NRMSE={final["nrmse"]:.4f}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(cfg.checkpoint_dir / "results.png", dpi=300)
print(f"✅ Saved: {cfg.checkpoint_dir / 'results.png'}")

if cfg.use_wandb and WANDB_AVAILABLE:
    wandb.log({"results": wandb.Image(fig)})

plt.show()
