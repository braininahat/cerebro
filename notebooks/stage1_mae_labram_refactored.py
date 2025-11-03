"""
Route-B LaBraM-style pretraining for EEG-2025
Stages:
  1) tokenizer  : VQ-SNP tokenizer pretraining (predict spectrum amp + sin/cos phase)
  2) dump_codes : offline code extraction to .npz shards (C,P) integer codes
  3) masked     : masked code modeling transformer (predict code IDs for masked (channel,patch))
"""

from joblib import Parallel, delayed
from masked_code_model import MaskedCodeModel
from tokenizer_vq_snp import TokenizerVQSNP
from eegdash import EEGChallengeDataset
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.preprocessing import preprocess, Preprocessor
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import os
import glob
import argparse
from pathlib import Path
from typing import Optional, Tuple
from collections import Counter
import random

# For debugging CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Braindecode / EEGDash

# Your modules


# =============================
# Configuration
# =============================
# Use TF32 where applicable (matmul/conv) for speed/accuracy balance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # Hopper likes this


class Config:
    """Centralized configuration for Route-B pretraining"""

    # Data paths
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
    CACHE_DIR = os.path.join(DATA_DIR, "mini")  # set to "full" when ready

    # Releases (you can include R5 later if you don't need it as a warmup holdout)
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

    # Excluded subjects (known issues)
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
    CROP_SIZE_S = 2.0  # 2s crop used by both tokenizer and code dumper

    # Patch spec (used by tokenizer & masked-code model)
    PATCH_LEN = 20  # 2s @100Hz → 200 samples; 200/20 = P=10 patches
    N_FFT_BINS = 11  # number of frequency bins supervised (amp/sin/cos)

    # Model sizes
    TOK_DIM = 256           # tokenizer latent dim
    CODEBOOK_SIZE = 8192    # VQ codes
    MC_D = 256              # masked-code model width
    MC_LAYERS = 8
    MC_HEADS = 8
    MASK_RATIO = 0.5        # fraction of (channel,patch) positions masked

    # Training
    BATCH_SIZE = 128        # practical default; scale up if VRAM allows
    NUM_WORKERS = 8
    MAX_EPOCHS = 100
    LR = 1e-3
    VAL_RATIO = 0.1

    # I/O
    ROOT_DIR = "./checkpoints/routeB"
    TOK_DIR = os.path.join(ROOT_DIR, "tokenizer")
    CODES_DIR = os.path.join(ROOT_DIR, "codes_passive")  # shards written here
    MC_DIR = os.path.join(ROOT_DIR, "masked")

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
        """Check if dataset passes filters (returns ds or None)"""
        try:
            # Quick checks first (no data access)
            if ds.description.subject in config.EXCLUDED_SUBJECTS:
                return None

            # Only access raw data if needed (this is the slow part)
            # Use lazy evaluation - only load what's necessary
            if hasattr(ds.raw, 'n_times'):
                n_times = ds.raw.n_times
            else:
                # Fallback: try to get from info
                n_times = ds.raw.info.get(
                    'sfreq', config.SFREQ) * ds.raw.times[-1]

            if n_times < config.MIN_DURATION_S * config.SFREQ:
                return None

            if len(ds.raw.ch_names) != config.N_CHANNELS:
                return None

            return ds
        except Exception as e:
            print(
                f"  Warning: Failed to filter: {e}")
            return None

    # Filter in parallel for speed
    print("  Running quality checks (this may take a moment)...")
    filtered = Parallel(n_jobs=min(config.NUM_WORKERS, 4), backend='threading')(
        delayed(check_dataset)(ds) for ds in datasets.datasets
    )

    # Remove None values
    filtered = [ds for ds in filtered if ds is not None]

    filtered_datasets = BaseConcatDataset(filtered)
    print(
        f"✓ Kept {len(filtered_datasets.datasets)}/{len(datasets.datasets)} recordings")

    task_counts = Counter(
        [ds.description.task for ds in filtered_datasets.datasets])
    print("\nTask distribution:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} recordings")

    return filtered_datasets


def create_windows(datasets: BaseConcatDataset, config: Config) -> BaseConcatDataset:
    """Create fixed-length windows from continuous data"""
    windows_dataset = create_fixed_length_windows(
        datasets,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=int(config.WINDOW_SIZE_S * config.SFREQ),
        window_stride_samples=int(config.WINDOW_STRIDE_S * config.SFREQ),
        drop_last_window=True,
        preload=False,
    )
    print(f"Total windows created: {len(windows_dataset)}")
    return windows_dataset


def split_by_subject(
    windows_dataset: BaseConcatDataset,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[BaseConcatDataset, BaseConcatDataset]:
    """Split dataset by subject to prevent leakage"""
    subjects = [ds.description["subject"] for ds in windows_dataset.datasets]
    unique_subjects = sorted(set(subjects))
    rng = random.Random(seed)
    rng.shuffle(unique_subjects)
    n_val = max(1, int(round(len(unique_subjects) * val_ratio)))

    val_subjects = set(unique_subjects[:n_val])
    train_subjects = set(unique_subjects[n_val:])

    print(
        f"\nSubject split: train={len(train_subjects)}  val={len(val_subjects)}")

    train_list, val_list = [], []
    for ds in windows_dataset.datasets:
        subject = ds.description["subject"]
        (train_list if subject in train_subjects else val_list).append(ds)

    train_set = BaseConcatDataset(train_list)
    val_set = BaseConcatDataset(val_list)
    print(
        f"  Training windows: {len(train_set)} | Validation windows: {len(val_set)}")
    return train_set, val_set


# =============================
# Dataset with Random Cropping (2s)
# =============================
class CroppedWindowDataset(Dataset):
    """Wraps a windowed dataset and applies random 2s crop."""

    def __init__(self, windows_dataset: BaseConcatDataset, crop_size_samples: int, seed: Optional[int] = None):
        self.windows_dataset = windows_dataset
        self.crop_size_samples = crop_size_samples
        self.rng = random.Random(seed)
        if len(windows_dataset) == 0:
            raise ValueError("Empty dataset provided!")
        first_X, _, _ = windows_dataset[0]
        if first_X.shape[1] < crop_size_samples:
            raise ValueError("Window shorter than crop size.")
        self.valid_indices = list(range(len(windows_dataset)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        actual_idx = self.valid_indices[idx]
        X, _, _ = self.windows_dataset[actual_idx]  # (C,T)
        C, T = X.shape
        if T > self.crop_size_samples:
            start = self.rng.randint(0, T - self.crop_size_samples)
            X = X[:, start:start + self.crop_size_samples]
        return torch.as_tensor(X, dtype=torch.float32)


def collate_float(batch):
    return torch.stack(batch, dim=0).float()  # (B,C,T)


# =============================
# Codes shards dataset (for masked-code stage)
# =============================
class CodesShardDataset(Dataset):
    """
    Reads *.npz shards with 'codes' arrays of shape (B,C,P) integers.
    Each item returns {"codes": (C,P) LongTensor}.
    """

    def __init__(self, shards_dir: str):
        self.files = sorted(glob.glob(os.path.join(shards_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz shards found in {shards_dir}")
        self._index = []  # (file_idx, item_idx)
        for fi, f in enumerate(self.files):
            with np.load(f) as npz:
                n = npz["codes"].shape[0]
            self._index.extend([(fi, i) for i in range(n)])

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fi, i = self._index[idx]
        with np.load(self.files[fi]) as npz:
            codes = npz["codes"][i]  # (C,P)
        return {"codes": torch.as_tensor(codes, dtype=torch.long)}


def collate_codes(batch):
    codes = [b["codes"] for b in batch]
    return {"codes": torch.stack(codes, dim=0).long()}  # (B,C,P)


# =============================
# Trainers / Runners for each stage
# =============================
def setup_logger_and_callbacks(run_name: str, monitor: str, ckpt_dir: str, use_wandb: bool):
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f'{run_name}-' + '{epoch:02d}-{' + monitor + ':.4f}',
        monitor=monitor,
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    es_cb = EarlyStopping(monitor=monitor, patience=10,
                          mode='min', verbose=True)

    if use_wandb:
        logger = WandbLogger(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            name=run_name,
            save_dir=Config.LOG_DIR,
            log_model=True,
        )
    else:
        logger = False

    return logger, [ckpt_cb, es_cb], ckpt_cb


def train_tokenizer(train_loader, val_loader, crop_len):
    print("\n=== Stage 1: Tokenizer (VQ-SNP) ===")
    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = TokenizerVQSNP(
        n_chans=Config.N_CHANNELS,
        crop_len=crop_len,
        patch_len=Config.PATCH_LEN,
        dim=Config.TOK_DIM,
        num_codes=Config.CODEBOOK_SIZE,
        n_fft_bins=Config.N_FFT_BINS,
        lr=Config.LR,
    )

    logger, callbacks, ckpt_cb = setup_logger_and_callbacks(
        run_name="tokenizer_vq_snp",
        monitor="tok_loss",
        ckpt_dir=Config.TOK_DIR,
        use_wandb=Config.USE_WANDB,
    )

    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic='warn',
        enable_progress_bar=True,  # Ensure progress bar is enabled
        enable_model_summary=True,  # Show model summary
    )

    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    best = ckpt_cb.best_model_path
    print(f"✓ Tokenizer best checkpoint: {best}")
    return best


@torch.inference_mode()
def dump_codes(ckpt_path: str, loader, out_dir: str):
    print("\n=== Stage 2: Dump codes ===")
    os.makedirs(out_dir, exist_ok=True)

    # load tokenizer
    model = TokenizerVQSNP.load_from_checkpoint(ckpt_path)
    model.eval().cuda() if torch.cuda.is_available() else model.eval()

    shard_idx = 0
    for xb in loader:
        xb = xb.cuda() if torch.cuda.is_available() else xb
        _, _, _, codes, _ = model(xb)  # (B,C,P) ints
        codes = codes.cpu().numpy()
        np.savez_compressed(os.path.join(
            out_dir, f"shard_{shard_idx:06d}.npz"), codes=codes)
        shard_idx += 1
    print(f"✓ Wrote {shard_idx} shards to {out_dir}")


def train_masked_code(codes_train_loader, codes_val_loader):
    print("\n=== Stage 3: Masked code modeling ===")
    model = MaskedCodeModel(
        num_codes=Config.CODEBOOK_SIZE,
        C=Config.N_CHANNELS,
        P=int(Config.CROP_SIZE_S * Config.SFREQ //
              Config.PATCH_LEN),  # e.g. 200/20=10
        D=Config.MC_D,
        L=Config.MC_LAYERS,
        H=Config.MC_HEADS,
        mask_ratio=Config.MASK_RATIO,
        lr=Config.LR,
    )

    logger, callbacks, ckpt_cb = setup_logger_and_callbacks(
        run_name="masked_code_model",
        monitor="mc_loss",
        ckpt_dir=Config.MC_DIR,
        use_wandb=Config.USE_WANDB,
    )

    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic='warn',
    )

    trainer.fit(model, codes_train_loader, codes_val_loader)
    best = ckpt_cb.best_model_path
    print(f"✓ Masked-code best checkpoint: {best}")
    return best


# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True,
                        choices=["tokenizer", "dump_codes", "masked"],
                        help="Which stage to run")
    parser.add_argument("--use_mini", action="store_true",
                        help="Use the MINI cache split")
    parser.add_argument("--tokenizer-ckpt", type=str, default=None,
                        help="Path to tokenizer checkpoint (required for dump_codes)")
    parser.add_argument("--codes-dir", type=str, default=Config.CODES_DIR,
                        help="Directory to read/write code shards")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    args = parser.parse_args()

    pl.seed_everything(Config.SEED)

    # Make dirs
    os.makedirs(Config.ROOT_DIR, exist_ok=True)
    os.makedirs(Config.TOK_DIR, exist_ok=True)
    os.makedirs(Config.MC_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    # Common: load passive → filter → windows → split by subject
    print("\n1) Loading passive datasets…")
    all_datasets = load_passive_task_data(
        Config, use_mini=args.use_mini if hasattr(args, "use_mini") else True)
    print("\n2) Filtering…")
    all_datasets = filter_datasets(all_datasets, Config)
    print("\n3) Creating windows…")
    windows = create_windows(all_datasets, Config)
    print("\n4) Subject split…")
    train_win, val_win = split_by_subject(
        windows, val_ratio=Config.VAL_RATIO, seed=Config.SEED)

    crop_len = int(Config.CROP_SIZE_S * Config.SFREQ)
    train_set = CroppedWindowDataset(
        train_win, crop_size_samples=crop_len, seed=Config.SEED)
    val_set = CroppedWindowDataset(
        val_win,   crop_size_samples=crop_len, seed=Config.SEED+1)

    if args.stage == "tokenizer":
        # Reduce num_workers for tokenizer to avoid deadlocks with persistent_workers
        # Tokenizer training is GPU-bound (FFT ops), so fewer workers is fine
        tok_workers = min(4, Config.NUM_WORKERS)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=tok_workers, pin_memory=True,
                                  drop_last=True, collate_fn=collate_float,
                                  persistent_workers=False)  # Disable to avoid deadlocks
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=tok_workers, pin_memory=True,
                                drop_last=False, collate_fn=collate_float,
                                persistent_workers=False)

        print(f"✓ Using {tok_workers} workers for tokenizer dataloaders")

        best_tok = train_tokenizer(train_loader, val_loader, crop_len)

        if Config.USE_WANDB:
            wandb.finish()

    elif args.stage == "dump_codes":
        if not args.tokenizer_ckpt:
            raise ValueError("--tokenizer-ckpt is required for dump_codes")
        # dump with SAME crops/splits
        dump_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=Config.NUM_WORKERS, pin_memory=True,
                                 drop_last=False, collate_fn=collate_float,
                                 persistent_workers=Config.NUM_WORKERS > 0)
        dump_loader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                     num_workers=Config.NUM_WORKERS, pin_memory=True,
                                     drop_last=False, collate_fn=collate_float,
                                     persistent_workers=Config.NUM_WORKERS > 0)

        dump_codes(args.tokenizer_ckpt, dump_loader,
                   os.path.join(args.codes_dir, "train"))
        dump_codes(args.tokenizer_ckpt, dump_loader_val,
                   os.path.join(args.codes_dir, "val"))

    elif args.stage == "masked":
        # Train masked-code model on dumped shards
        train_codes = CodesShardDataset(os.path.join(args.codes_dir, "train"))
        val_codes = CodesShardDataset(os.path.join(args.codes_dir, "val"))

        train_loader = DataLoader(train_codes, batch_size=args.batch_size, shuffle=True,
                                  num_workers=Config.NUM_WORKERS, pin_memory=True,
                                  drop_last=True, collate_fn=collate_codes,
                                  persistent_workers=Config.NUM_WORKERS > 0)
        val_loader = DataLoader(val_codes, batch_size=args.batch_size, shuffle=False,
                                num_workers=Config.NUM_WORKERS, pin_memory=True,
                                drop_last=False, collate_fn=collate_codes,
                                persistent_workers=Config.NUM_WORKERS > 0)

        best_mc = train_masked_code(train_loader, val_loader)

        if Config.USE_WANDB:
            wandb.finish()

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
