# stage1_mae_labram_refactored_v2.py
"""
Improved Route-B LaBraM-style pretraining for EEG-2025
Stages:
1) tokenizer  : VQ-SNP tokenizer pretraining with improved architecture
2) dump_codes : offline code extraction to .npz shards
3) masked     : spatially-aware masked code modeling
4) finetune   : progressive unfreezing for downstream tasks

Improvements:
- Uses LaBraM encoder architecture
- Gumbel-Softmax VQ option
- Frequency band-based FFT targets
- Spatial masking for masked modeling
- Progressive fine-tuning support
"""

from joblib import Parallel, delayed
from collections import Counter
from typing import Optional, Tuple
from pathlib import Path
import argparse
import glob
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets import BaseConcatDataset
from eegdash import EEGChallengeDataset

# Import improved modules (use v2 versions if available, fall back to original)
try:
    from tokenizer_vq_snp_v2 import TokenizerVQSNP
    print("Using improved TokenizerVQSNP v2")
except ImportError:
    from tokenizer_vq_snp import TokenizerVQSNP
    print("Using original TokenizerVQSNP")

try:
    from masked_code_model_v2 import SpatialMaskedCodeModel as MaskedCodeModel
    print("Using improved SpatialMaskedCodeModel")
except ImportError:
    from masked_code_model import MaskedCodeModel
    print("Using original MaskedCodeModel")

# For debugging CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# =============================
# Configuration
# =============================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class Config:
    """Centralized configuration for Route-B pretraining"""

    # Data paths
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
    CACHE_DIR = os.path.join(DATA_DIR, "mini")

    # Releases
    RELEASES = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"]

    # Passive tasks for pre-training
    PASSIVE_TASKS = [
        "restingState",
        "surroundSupp",
        "despicableMe",
        "thePresent",
        "diaryOfAWimpyKid",
        "funwithFractals",
    ]

    # Excluded subjects
    EXCLUDED_SUBJECTS = [
        "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
        "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
        "NDARBA381JGH"
    ]

    # Preprocessing
    SFREQ = 100
    L_FREQ = 0.5
    H_FREQ = 40.0
    MIN_DURATION_S = 4.0
    N_CHANNELS = 129

    # Windowing
    WINDOW_SIZE_S = 4.0
    WINDOW_STRIDE_S = 2.0
    CROP_SIZE_S = 2.0

    # Patch spec
    PATCH_LEN = 20
    N_FFT_BINS = 5  # Changed to 5 frequency bands for band-based FFT
    USE_BAND_FFT = True  # Use frequency bands instead of raw FFT

    # Model sizes
    TOK_DIM = 256
    CODEBOOK_SIZE = 8192
    VQ_TYPE = "gumbel"  # "gumbel" or "ema"
    TEMPERATURE = 1.0
    KL_WEIGHT = 0.01

    MC_D = 256
    MC_LAYERS = 8
    MC_HEADS = 8
    MASK_RATIO = 0.5
    MASK_STRATEGY = "spatial"  # "spatial", "channel", or "random"

    # Training
    BATCH_SIZE = 128
    NUM_WORKERS = 8
    MAX_EPOCHS = 100
    LR = 1e-3
    VAL_RATIO = 0.1

    # I/O
    ROOT_DIR = "./checkpoints/routeB"
    TOK_DIR = os.path.join(ROOT_DIR, "tokenizer")
    CODES_DIR = os.path.join(ROOT_DIR, "codes_passive")
    MC_DIR = os.path.join(ROOT_DIR, "masked")
    FT_DIR = os.path.join(ROOT_DIR, "finetune")

    LOG_DIR = "./logs/routeB"

    # WandB
    WANDB_PROJECT = "eeg-challenge-2025"
    WANDB_ENTITY = None
    USE_WANDB = False

    # Seeds
    SEED = 42


# =============================
# Data Loading & Preprocessing
# =============================
def load_passive_task_data(config: Config, use_mini: bool = True) -> BaseConcatDataset:
    """Load all passive task datasets from specified releases"""
    all_datasets_list = []
    for release in config.RELEASES:
        for task in config.PASSIVE_TASKS:
            try:
                print(f"Loading {release}/{task}...")
                dataset = EEGChallengeDataset(
                    task=task,
                    release=release,
                    cache_dir=config.CACHE_DIR,
                    mini=use_mini,
                    description_fields=[
                        "subject", "session", "run", "task", "age", "sex"],
                )
                all_datasets_list.append(dataset)
                print(f"  ✓ Loaded {len(dataset.datasets)} recordings")
            except Exception as e:
                print(f"  ✗ Failed to load {release}/{task}: {e}")
                continue

    if not all_datasets_list:
        raise ValueError("No datasets were loaded successfully!")

    all_datasets = BaseConcatDataset(all_datasets_list)
    print(
        f"\n{'='*60}\nTotal recordings loaded: {len(all_datasets.datasets)}\n{'='*60}")
    return all_datasets


def filter_datasets(datasets: BaseConcatDataset, config: Config) -> BaseConcatDataset:
    """Apply quality filters to datasets"""
    print(f"Filtering {len(datasets.datasets)} datasets...")

    def check_dataset(ds):
        try:
            # Check subject exclusion
            subj = ds.description.get('subject', '')
            if subj in config.EXCLUDED_SUBJECTS:
                return None

            # Check duration
            raw = ds.raw
            if raw is None:
                return None

            dur = raw.times[-1]
            if dur < config.MIN_DURATION_S:
                return None

            # Check channels
            if len(raw.ch_names) != config.N_CHANNELS:
                return None

            return ds
        except:
            return None

    # Filter in parallel
    print("  Running quality checks...")
    filtered = Parallel(n_jobs=min(config.NUM_WORKERS, 4), backend='threading')(
        delayed(check_dataset)(ds) for ds in datasets.datasets
    )

    filtered = [ds for ds in filtered if ds is not None]
    filtered_datasets = BaseConcatDataset(filtered)
    print(
        f"✓ Kept {len(filtered_datasets.datasets)}/{len(datasets.datasets)} recordings")

    task_counts = Counter(
        [ds.description.task for ds in filtered_datasets.datasets])
    print("\nTask distribution:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    return filtered_datasets


def create_windows(datasets: BaseConcatDataset, config: Config) -> BaseConcatDataset:
    """Create fixed-length windows from continuous data"""
    print("\nCreating fixed-length windows...")
    window_size_samples = int(config.WINDOW_SIZE_S * config.SFREQ)
    window_stride_samples = int(config.WINDOW_STRIDE_S * config.SFREQ)

    windows = create_fixed_length_windows(
        datasets,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=True,
        preload=False
    )

    print(f"✓ Created {len(windows.datasets)} windows")
    return windows


def split_by_subject(
    windows_dataset: BaseConcatDataset,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[BaseConcatDataset, BaseConcatDataset]:
    """Split windows by subject to avoid leakage"""
    subjects = [ds.description['subject'] for ds in windows_dataset.datasets]
    unique_subjects = list(set(subjects))

    rng = random.Random(seed)
    rng.shuffle(unique_subjects)

    n_val = max(1, int(len(unique_subjects) * val_ratio))
    val_subjects = set(unique_subjects[:n_val])

    train_indices = [i for i, s in enumerate(
        subjects) if s not in val_subjects]
    val_indices = [i for i, s in enumerate(subjects) if s in val_subjects]

    train_ds = BaseConcatDataset(
        [windows_dataset.datasets[i] for i in train_indices])
    val_ds = BaseConcatDataset(
        [windows_dataset.datasets[i] for i in val_indices])

    print(
        f"\nSplit: {len(train_ds.datasets)} train, {len(val_ds.datasets)} val windows")
    print(
        f"       {len(unique_subjects) - n_val} train subjects, {n_val} val subjects")

    return train_ds, val_ds


# =============================
# Dataset with Random Cropping
# =============================
class CroppedWindowDataset(Dataset):
    """Wraps windows dataset with random temporal cropping"""

    def __init__(self, windows_dataset: BaseConcatDataset, crop_len: int, sfreq: int = 100):
        self.windows_dataset = windows_dataset
        self.crop_len = crop_len
        self.sfreq = sfreq

    def __len__(self):
        return len(self.windows_dataset.datasets)

    def __getitem__(self, idx):
        ds = self.windows_dataset.datasets[idx]
        x, y, _ = ds[0]  # (C, T)
        C, T = x.shape

        # Random crop
        if T > self.crop_len:
            start = np.random.randint(0, T - self.crop_len + 1)
            x_crop = x[:, start:start + self.crop_len]
        else:
            x_crop = x[:, :self.crop_len]
            if x_crop.shape[1] < self.crop_len:
                # Pad if needed
                pad = self.crop_len - x_crop.shape[1]
                x_crop = np.pad(x_crop, ((0, 0), (0, pad)), mode='constant')

        return torch.from_numpy(x_crop).float()


def collate_float(batch):
    return torch.stack(batch, dim=0)


# =============================
# Codes shards dataset
# =============================
class CodesShardDataset(Dataset):
    """Dataset that loads .npz code shards"""

    def __init__(self, shard_dir: str):
        self.shard_files = sorted(
            glob.glob(os.path.join(shard_dir, "shard_*.npz")))
        if not self.shard_files:
            raise ValueError(f"No shard files found in {shard_dir}")
        print(f"Found {len(self.shard_files)} shard files")

        # Preload all shards (if they fit in memory)
        self.codes_list = []
        for f in self.shard_files:
            data = np.load(f)
            codes = torch.from_numpy(data['codes']).long()  # (N, C, P)
            for i in range(len(codes)):
                self.codes_list.append(codes[i])

        print(f"Loaded {len(self.codes_list)} code samples")

    def __len__(self):
        return len(self.codes_list)

    def __getitem__(self, idx):
        return {"codes": self.codes_list[idx]}


def collate_codes(batch):
    codes = torch.stack([b["codes"] for b in batch], dim=0)
    return {"codes": codes}


# =============================
# Trainers / Runners
# =============================
def setup_logger_and_callbacks(run_name: str, monitor: str, ckpt_dir: str, use_wandb: bool):
    """Setup logging and callbacks"""
    os.makedirs(ckpt_dir, exist_ok=True)

    if use_wandb:
        logger = WandbLogger(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            name=run_name,
            save_dir=Config.LOG_DIR
        )
    else:
        logger = None

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{run_name}-{{epoch:02d}}-{{{monitor}:.4f}}",
        monitor=monitor,
        mode='min',
        save_top_k=3,
        save_last=True
    )

    early_stop = EarlyStopping(
        monitor=monitor,
        patience=15,
        mode='min',
        verbose=True
    )

    callbacks = [ckpt_cb, early_stop]

    return logger, callbacks, ckpt_cb


def train_tokenizer(train_loader, val_loader, config: Config):
    """Train VQ-SNP tokenizer"""
    print("\n" + "="*60)
    print("Stage 1: Training VQ-SNP Tokenizer")
    print("="*60)

    crop_len = int(config.CROP_SIZE_S * config.SFREQ)

    model = TokenizerVQSNP(
        n_chans=config.N_CHANNELS,
        crop_len=crop_len,
        patch_len=config.PATCH_LEN,
        dim=config.TOK_DIM,
        num_codes=config.CODEBOOK_SIZE,
        n_fft_bins=config.N_FFT_BINS,
        sfreq=config.SFREQ,
        use_band_fft=config.USE_BAND_FFT,
        vq_type=config.VQ_TYPE,
        lr=config.LR,
        temperature=config.TEMPERATURE,
        kl_weight=config.KL_WEIGHT,
    )

    logger, callbacks, ckpt_cb = setup_logger_and_callbacks(
        run_name="tokenizer_vqsnp",
        monitor="val_tok_loss",
        ckpt_dir=config.TOK_DIR,
        use_wandb=config.USE_WANDB
    )

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    trainer.fit(model, train_loader, val_loader)

    best_ckpt = ckpt_cb.best_model_path
    print(f"✓ Tokenizer training complete: {best_ckpt}")
    return best_ckpt


@torch.inference_mode()
def dump_codes(tokenizer_ckpt: str, windows_dataset: BaseConcatDataset, config: Config):
    """Extract codes offline and save to .npz shards"""
    print("\n" + "="*60)
    print("Stage 2: Dumping codes to shards")
    print("="*60)

    os.makedirs(config.CODES_DIR, exist_ok=True)

    # Load tokenizer
    model = TokenizerVQSNP.load_from_checkpoint(tokenizer_ckpt)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    crop_len = int(config.CROP_SIZE_S * config.SFREQ)
    dataset = CroppedWindowDataset(
        windows_dataset, crop_len=crop_len, sfreq=config.SFREQ)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                        shuffle=False, collate_fn=collate_float)

    codes_all = []
    for batch in loader:
        x = batch.to(model.device)
        _, _, _, codes, _ = model(x)  # codes: (B, C, P)
        codes_all.append(codes.cpu())

    codes_all = torch.cat(codes_all, dim=0)  # (N, C, P)
    print(f"Extracted {len(codes_all)} code samples, shape: {codes_all.shape}")

    # Save in shards
    shard_size = 10000
    n_shards = (len(codes_all) + shard_size - 1) // shard_size

    for i in range(n_shards):
        start = i * shard_size
        end = min((i + 1) * shard_size, len(codes_all))
        shard = codes_all[start:end]

        shard_path = os.path.join(config.CODES_DIR, f"shard_{i:04d}.npz")
        np.savez_compressed(shard_path, codes=shard.numpy())
        print(f"  Saved {shard_path}: {len(shard)} samples")

    print(f"✓ Code dumping complete: {n_shards} shards")


def train_masked_code(codes_train_loader, codes_val_loader, config: Config):
    """Train masked code modeling"""
    print("\n" + "="*60)
    print("Stage 3: Training Masked Code Model")
    print("="*60)

    model = MaskedCodeModel(
        num_codes=config.CODEBOOK_SIZE,
        C=config.N_CHANNELS,
        P=int(config.CROP_SIZE_S * config.SFREQ) // config.PATCH_LEN,
        D=config.MC_D,
        n_layers=config.MC_LAYERS,
        n_heads=config.MC_HEADS,
        mask_ratio=config.MASK_RATIO,
        lr=config.LR,
        mask_strategy=config.MASK_STRATEGY,
    )

    logger, callbacks, ckpt_cb = setup_logger_and_callbacks(
        run_name="masked_code",
        monitor="val_mc_loss",
        ckpt_dir=config.MC_DIR,
        use_wandb=config.USE_WANDB
    )

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    trainer.fit(model, codes_train_loader, codes_val_loader)

    best_ckpt = ckpt_cb.best_model_path
    print(f"✓ Masked code training complete: {best_ckpt}")
    return best_ckpt


# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all",
                        choices=["tokenizer", "dump_codes", "masked", "all"],
                        help="Which stage to run")
    parser.add_argument("--use_mini", action="store_true", default=True,
                        help="Use mini dataset")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None,
                        help="Path to tokenizer checkpoint (for dump_codes/masked stages)")
    args = parser.parse_args()

    # Set seeds
    pl.seed_everything(Config.SEED)

    # Load data
    print("Loading passive task data...")
    all_datasets = load_passive_task_data(Config, use_mini=args.use_mini)

    # Filter
    filtered_datasets = filter_datasets(all_datasets, Config)

    # Create windows
    windows_dataset = create_windows(filtered_datasets, Config)

    # Split
    train_windows, val_windows = split_by_subject(
        windows_dataset, val_ratio=Config.VAL_RATIO, seed=Config.SEED
    )

    # Stage 1: Tokenizer
    if args.stage in ["tokenizer", "all"]:
        crop_len = int(Config.CROP_SIZE_S * Config.SFREQ)

        train_ds = CroppedWindowDataset(
            train_windows, crop_len=crop_len, sfreq=Config.SFREQ)
        val_ds = CroppedWindowDataset(
            val_windows, crop_len=crop_len, sfreq=Config.SFREQ)

        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                                  num_workers=Config.NUM_WORKERS, shuffle=True,
                                  collate_fn=collate_float, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE,
                                num_workers=Config.NUM_WORKERS, shuffle=False,
                                collate_fn=collate_float, pin_memory=True)

        tokenizer_ckpt = train_tokenizer(train_loader, val_loader, Config)
    else:
        tokenizer_ckpt = args.tokenizer_ckpt
        if tokenizer_ckpt is None:
            raise ValueError("Must provide --tokenizer_ckpt for this stage")

    # Stage 2: Dump codes
    if args.stage in ["dump_codes", "all"]:
        dump_codes(tokenizer_ckpt, train_windows, Config)

    # Stage 3: Masked code modeling
    if args.stage in ["masked", "all"]:
        codes_train_ds = CodesShardDataset(Config.CODES_DIR)
        # For validation, we could create a separate val codes directory, or just use train for now
        codes_val_ds = codes_train_ds  # Simplified for now

        codes_train_loader = DataLoader(codes_train_ds, batch_size=Config.BATCH_SIZE,
                                        num_workers=Config.NUM_WORKERS, shuffle=True,
                                        collate_fn=collate_codes, pin_memory=True)
        codes_val_loader = DataLoader(codes_val_ds, batch_size=Config.BATCH_SIZE,
                                      num_workers=Config.NUM_WORKERS, shuffle=False,
                                      collate_fn=collate_codes, pin_memory=True)

        masked_ckpt = train_masked_code(
            codes_train_loader, codes_val_loader, Config)

    print("\n" + "="*60)
    print("✓ All stages complete!")
    print("="*60)


if __name__ == "__main__":
    main()
