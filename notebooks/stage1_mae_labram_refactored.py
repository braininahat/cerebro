"""
Stage-1 MAE Pre-training for EEG-2025 using LabRAM
Refactored version with proper dataset structure and PyTorch Lightning
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from collections import Counter
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
import wandb

# Braindecode / EEGDash
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets import BaseConcatDataset
from braindecode.models import Labram
from eegdash import EEGChallengeDataset
from joblib import Parallel, delayed


# =============================
# Configuration
# =============================
class Config:
    """Centralized configuration for Stage 1 MAE pre-training"""

    # Data paths
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
    # Use "full" for complete dataset
    CACHE_DIR = os.path.join(DATA_DIR, "mini")

    # Releases (exclude R5 for testing)
    RELEASES = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"]
    # RELEASES = ["R1", "R2"]

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
    SFREQ = 100  # Target sampling frequency
    L_FREQ = 0.5
    H_FREQ = 40.0
    MIN_DURATION_S = 4.0
    N_CHANNELS = 129  # Expected channel count

    # Windowing
    WINDOW_SIZE_S = 4.0
    WINDOW_STRIDE_S = 2.0
    CROP_SIZE_S = 2.0  # Random crop size for training

    # Model architecture
    EMB_SIZE = 256
    N_LAYERS = 10
    N_HEADS = 8
    PATCH_LEN = 20  # 2s @ 100Hz = 200 samples; 200/20 = 10 patches
    MASK_RATIO = 0.5

    # Training
    # Initial batch size (will be tuned if AUTO_BATCH_SIZE=True)
    BATCH_SIZE = 4096
    AUTO_BATCH_SIZE = False  # Automatically find optimal batch size
    NUM_WORKERS = 8
    MAX_EPOCHS = 100
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    VAL_RATIO = 0.1

    # Checkpointing
    CHECKPOINT_DIR = "./checkpoints/stage1_mae"
    LOG_DIR = "./logs/stage1_mae"

    # Weights & Biases
    WANDB_PROJECT = "eeg-challenge-2025"
    WANDB_ENTITY = None  # Set to your wandb username/team, or None for default
    WANDB_NAME = "stage1_mae_labram"  # Run name
    USE_WANDB = True  # Enable/disable wandb logging

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
    print(f"\n{'='*60}")
    print(f"Total recordings loaded: {len(all_datasets.datasets)}")
    print(f"{'='*60}")

    return all_datasets


def filter_datasets(datasets: BaseConcatDataset, config: Config) -> BaseConcatDataset:
    """Apply quality filters to datasets"""

    filtered = [
        ds for ds in datasets.datasets
        if ds.description.subject not in config.EXCLUDED_SUBJECTS
        and ds.raw.n_times >= config.MIN_DURATION_S * config.SFREQ
        and len(ds.raw.ch_names) == config.N_CHANNELS
    ]

    filtered_datasets = BaseConcatDataset(filtered)
    print(f"Recordings after filtering: {len(filtered_datasets.datasets)}")

    # Show task distribution
    task_counts = Counter(
        [ds.description.task for ds in filtered_datasets.datasets])
    print("\nTask distribution:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} recordings")

    return filtered_datasets


def apply_preprocessing(datasets: BaseConcatDataset, config: Config) -> None:
    """Apply standard EEG preprocessing pipeline"""

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False,
                     stim=False, eog=False, ecg=False),
        Preprocessor(lambda data: data.set_eeg_reference(
            ref_channels='average'), apply_on_array=False),
        Preprocessor('filter', l_freq=config.L_FREQ,
                     h_freq=config.H_FREQ, method='iir'),
        Preprocessor('resample', sfreq=config.SFREQ),
    ]

    print("Applying preprocessing...")
    preprocess(datasets, preprocessors, n_jobs=-1)
    print("✓ Preprocessing complete")


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
    """Split dataset by subject to prevent data leakage"""

    # Get unique subjects
    subjects = [ds.description["subject"] for ds in windows_dataset.datasets]
    unique_subjects = sorted(set(subjects))

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(unique_subjects)
    n_val = max(1, int(round(len(unique_subjects) * val_ratio)))

    val_subjects = set(unique_subjects[:n_val])
    train_subjects = set(unique_subjects[n_val:])

    print(f"\nSubject split:")
    print(f"  Training subjects: {len(train_subjects)}")
    print(f"  Validation subjects: {len(val_subjects)}")

    # Split datasets
    train_list, val_list = [], []
    for ds in windows_dataset.datasets:
        subject = ds.description["subject"]
        if subject in train_subjects:
            train_list.append(ds)
        else:
            val_list.append(ds)

    train_set = BaseConcatDataset(train_list)
    val_set = BaseConcatDataset(val_list)

    print(f"  Training windows: {len(train_set)}")
    print(f"  Validation windows: {len(val_set)}")

    return train_set, val_set


# =============================
# Dataset with Random Cropping
# =============================
class CroppedWindowDataset(Dataset):
    """
    Wraps a windowed dataset and applies random temporal cropping.

    This is useful for data augmentation - each 4s window is randomly
    cropped to 2s during training.
    """

    def __init__(
        self,
        windows_dataset: BaseConcatDataset,
        crop_size_samples: int,
        seed: Optional[int] = None,
        validate_all: bool = False,
    ):
        """
        Parameters
        ----------
        windows_dataset : BaseConcatDataset
            Source windows dataset
        crop_size_samples : int
            Size of random crop in samples
        seed : int, optional
            Random seed for reproducibility
        validate_all : bool, default=False
            If True, validate all windows (SLOW!). If False, only check first window.
            Set to False when using create_fixed_length_windows (all windows same size).
        """
        self.windows_dataset = windows_dataset
        self.crop_size_samples = crop_size_samples
        self.rng = random.Random(seed)

        if len(windows_dataset) == 0:
            raise ValueError("Empty dataset provided!")

        if validate_all:
            # SLOW: Validate every window (use only if windows have variable sizes)
            print(
                f"  Validating all {len(windows_dataset)} windows (this may take a while)...")
            self.valid_indices = []
            for idx in range(len(windows_dataset)):
                X, _, _ = windows_dataset[idx]
                if X.shape[1] >= crop_size_samples:
                    self.valid_indices.append(idx)

            if not self.valid_indices:
                raise ValueError(
                    "No windows are large enough for the specified crop size!")

            print(
                f"  ✓ Found {len(self.valid_indices)}/{len(windows_dataset)} valid windows")
        else:
            # FAST: Only validate first window (assumes all windows have same size)
            # This is safe when using create_fixed_length_windows()
            print(f"  Fast initialization: validating first window only...")

            first_X, _, _ = windows_dataset[0]
            window_size = first_X.shape[1]

            if window_size < crop_size_samples:
                raise ValueError(
                    f"Window size ({window_size}) < crop size ({crop_size_samples})! "
                    f"Adjust CROP_SIZE_S or WINDOW_SIZE_S in Config."
                )

            # Assume all windows have the same size
            self.valid_indices = list(range(len(windows_dataset)))
            print(
                f"  ✓ All {len(self.valid_indices)} windows assumed valid (window_size={window_size})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns
        -------
        X : torch.Tensor
            Cropped EEG window of shape (n_channels, crop_size_samples)
        """
        actual_idx = self.valid_indices[idx]
        X, _, _ = self.windows_dataset[actual_idx]

        # X shape: (n_channels, n_times)
        n_channels, n_times = X.shape

        # Random crop
        if n_times > self.crop_size_samples:
            max_start = n_times - self.crop_size_samples
            start = self.rng.randint(0, max_start)
            X_cropped = X[:, start:start + self.crop_size_samples]
        else:
            X_cropped = X

        # Convert to tensor if needed
        if isinstance(X_cropped, np.ndarray):
            X_cropped = torch.from_numpy(X_cropped).float()
        else:
            X_cropped = X_cropped.float()

        return X_cropped


# =============================
# Patchify & Masking Utils
# =============================
class TimePatchify(nn.Module):
    """Splits time series into non-overlapping patches"""

    def __init__(self, patch_len: int):
        super().__init__()
        self.patch_len = patch_len

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        patches : (B, P, C, L) where P = T // L
        """
        B, C, T = x.shape
        assert T % self.patch_len == 0, f"T={T} must be divisible by patch_len={self.patch_len}"
        P = T // self.patch_len
        return x.view(B, C, P, self.patch_len).permute(0, 2, 1, 3).contiguous()

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patches : (B, P, C, L)

        Returns
        -------
        x : (B, C, T) where T = P * L
        """
        B, P, C, L = patches.shape
        return patches.permute(0, 2, 1, 3).contiguous().view(B, C, P * L)


def make_random_patch_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device = None
) -> torch.BoolTensor:
    """
    Create random binary mask for patches.

    Returns
    -------
    mask : (B, P) BoolTensor where True = masked
    """
    num_keep = max(1, int(round((1.0 - mask_ratio) * num_patches)))

    masks = []
    for _ in range(batch_size):
        idx = torch.randperm(num_patches)
        mask = torch.ones(num_patches, dtype=torch.bool)
        mask[idx[:num_keep]] = False  # False = keep, True = mask
        masks.append(mask)

    mask_tensor = torch.stack(masks, dim=0)
    if device is not None:
        mask_tensor = mask_tensor.to(device)

    return mask_tensor


# =============================
# MAE Lightning Module
# =============================
class MAELabRAM(pl.LightningModule):
    """
    Masked Autoencoder with LabRAM encoder for EEG pre-training.

    Architecture:
    1. Patchify input (C, T) -> (P, C, L)
    2. Mask random patches
    3. Encode with LabRAM -> (P, D) tokens
    4. Decode to reconstruct patches
    5. Compute loss only on masked patches
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: float,
        emb_size: int = 256,
        n_layers: int = 10,
        n_heads: int = 8,
        patch_len: int = 20,
        mask_ratio: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,  # For tuner compatibility
    ):
        super().__init__()
        self.save_hyperparameters()

        # Validate patch divisibility
        assert n_times % patch_len == 0, \
            f"n_times={n_times} must be divisible by patch_len={patch_len}"

        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = sfreq
        self.patch_len = patch_len
        self.num_patches = n_times // patch_len
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size  # Store for tuner

        # Encoder: LabRAM in tokenizer mode
        # Note: n_outputs is required by LabRAM but we won't use the final classification layer
        # We'll extract features from the transformer before the final layer
        self.encoder = Labram(
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            input_window_seconds=n_times / sfreq,
            neural_tokenizer=True,
            emb_size=emb_size,
            n_layers=n_layers,
            att_num_heads=n_heads,
            use_mean_pooling=False,  # Keep all tokens
            n_outputs=0,  # Set to emb_size; we'll bypass the final layer
        )

        # Decoder: token -> patch reconstruction
        self.decoder = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_chans * patch_len),
        )

        # Token projection layer (if LabRAM outputs different dimension)
        # Initialize as Identity to allow checkpoint loading, will be replaced
        # with Linear projection if needed during first forward pass
        self.token_projection = nn.Identity()
        self._projection_initialized = False

        # Patchify helper
        self.patchify = TimePatchify(patch_len)

    def encode_tokens(self, x_masked: torch.Tensor) -> torch.Tensor:
        """
        Return (B, P, D) time-patch tokens by pooling over channels.
        Assumes LaBraM tokenizer emits one token per (channel × time-patch).
        """
        # Try to get raw patch tokens from LaBraM
        # Some braindecode versions expose forward_features(..., return_patch_tokens=True).
        # Fall back to self.encoder(x) if needed.
        try:
            tokens = self.encoder.forward_features(
                x_masked, return_patch_tokens=True
            )  # expected (B, N_tok, D)
        except TypeError:
            # may already be tokens (B, N_tok, D)
            tokens = self.encoder(x_masked)

        if tokens.dim() != 3:
            raise RuntimeError(
                f"Unexpected token shape from LaBraM: {tokens.shape}")

        B, N_tok, D = tokens.shape
        P = self.num_patches
        C = self.n_chans
        expected = C * P

        # First batch: emit a helpful message once
        if not getattr(self, "_shape_logged", False):
            self._shape_logged = True
            self.print(f"[LaBraM] tokens shape: (B={B}, N_tok={N_tok}, D={D}), "
                       f"expected C*P={expected} with C={C}, P={P}")

        if N_tok == expected:
            # Reshape to (B, C, P, D) and pool across channels → (B, P, D)
            tokens = tokens.view(B, C, P, D).mean(dim=1)
            return tokens  # (B, P, D)

        # Some builds produce an extra CLS or slightly different ordering.
        # If N_tok == expected + 1, drop the first token (assume CLS).
        if N_tok == expected + 1:
            tokens = tokens[:, 1:, :]  # drop CLS
            tokens = tokens.view(B, C, P, D).mean(dim=1)
            return tokens

        # Fallback: interpolate along the token axis to exactly P time tokens.
        # This keeps you moving even if the internal layout differs.
        self.print(f"[LaBraM] Warning: N_tok ({N_tok}) != C*P ({expected}). "
                   f"Falling back to interpolation to P={P} time tokens.")
        tokens = F.interpolate(tokens.transpose(1, 2),
                               size=P, mode="linear", align_corners=False).transpose(1, 2)
        return tokens  # (B, P, D)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode and decode.

        Parameters
        ----------
        x_masked : (B, C, T) masked input

        Returns
        -------
        pred_patches : (B, P, C, L) reconstructed patches
        """
        # Encode
        tokens = self.encode_tokens(x_masked)  # (B, P, D)

        # Decode
        pred_flat = self.decoder(tokens)  # (B, P, C*L)

        # Reshape to patches
        B = pred_flat.size(0)
        pred_patches = pred_flat.view(
            B, self.num_patches, self.n_chans, self.patch_len)

        return pred_patches

    def compute_loss(
        self,
        pred_patches: torch.Tensor,
        target_patches: torch.Tensor,
        mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Compute MSE loss only on masked patches.

        Parameters
        ----------
        pred_patches : (B, P, C, L)
        target_patches : (B, P, C, L)
        mask : (B, P) where True = masked

        Returns
        -------
        loss : scalar
        """
        # Expand mask to match patch dimensions
        mask_expanded = mask[:, :, None, None]  # (B, P, 1, 1)

        # Compute squared error
        squared_error = (pred_patches - target_patches) ** 2

        # Apply mask and average
        masked_error = squared_error * mask_expanded
        num_masked = mask.sum() * self.n_chans * self.patch_len

        loss = masked_error.sum() / (num_masked + 1e-8)

        return loss

    def training_step(self, batch, batch_idx):
        """Training step"""
        x = batch  # (B, C, T)

        # Patchify
        target_patches = self.patchify.patchify(x)  # (B, P, C, L)

        # Create random mask
        mask = make_random_patch_mask(
            x.size(0),
            self.num_patches,
            self.mask_ratio,
            device=self.device
        )

        # Apply mask (zero out masked patches)
        masked_patches = target_patches.clone()
        masked_patches[mask] = 0.0
        x_masked = self.patchify.unpatchify(masked_patches)

        # Forward pass
        pred_patches = self.forward(x_masked)

        # Compute loss
        loss = self.compute_loss(pred_patches, target_patches, mask)

        # Log
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=x.size(0))
        self.log('train_mask_ratio', mask.float().mean(), on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x = batch  # (B, C, T)

        # Patchify
        target_patches = self.patchify.patchify(x)

        # Create random mask
        mask = make_random_patch_mask(
            x.size(0),
            self.num_patches,
            self.mask_ratio,
            device=self.device
        )

        # Apply mask
        masked_patches = target_patches.clone()
        masked_patches[mask] = 0.0
        x_masked = self.patchify.unpatchify(masked_patches)

        # Forward pass
        pred_patches = self.forward(x_masked)

        # Compute loss
        loss = self.compute_loss(pred_patches, target_patches, mask)

        # Log
        self.log('val_loss', loss, on_epoch=True, prog_bar=True,
                 batch_size=x.size(0))

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }


# =============================
# Collate Function
# =============================
def collate_fn(batch):
    """Stack tensors from batch"""
    tensors = []
    for x in batch:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        tensors.append(x.float())
    return torch.stack(tensors, dim=0)


# =============================
# Data Module for Batch Size Tuning
# =============================
class EEGDataModule(pl.LightningDataModule):
    """
    LightningDataModule wrapper for EEG datasets.
    Required for automatic batch size finding.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )


# =============================
# Main Training Pipeline
# =============================
def main():
    """Main training pipeline"""

    # Set seeds
    pl.seed_everything(Config.SEED)

    # Configure TF32 precision (suppresses PyTorch 2.9+ deprecation warning)
    if torch.cuda.is_available():
        # Use TF32 for better performance on Ampere+ GPUs (A100, RTX 30xx, etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Create output directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("Stage 1: MAE Pre-training with LabRAM")
    print("="*60)

    # -------------------------
    # 1. Load Data
    # -------------------------
    print("\n1. Loading passive task data...")
    all_datasets = load_passive_task_data(Config, use_mini=True)

    # -------------------------
    # 2. Filter
    # -------------------------
    print("\n2. Filtering datasets...")
    all_datasets = filter_datasets(all_datasets, Config)

    # -------------------------
    # 3. Preprocess
    # -------------------------
    # print("\n3. Preprocessing...")
    # apply_preprocessing(all_datasets, Config)

    # -------------------------
    # 4. Create Windows
    # -------------------------
    print("\n4. Creating windows...")
    windows_dataset = create_windows(all_datasets, Config)

    # -------------------------
    # 5. Split by Subject
    # -------------------------
    print("\n5. Splitting by subject...")
    train_windows, val_windows = split_by_subject(
        windows_dataset,
        val_ratio=Config.VAL_RATIO,
        seed=Config.SEED
    )

    # -------------------------
    # 6. Create Cropped Datasets
    # -------------------------
    print("\n6. Creating cropped datasets...")
    crop_samples = int(Config.CROP_SIZE_S * Config.SFREQ)

    train_dataset = CroppedWindowDataset(
        train_windows,
        crop_size_samples=crop_samples,
        seed=Config.SEED
    )

    val_dataset = CroppedWindowDataset(
        val_windows,
        crop_size_samples=crop_samples,
        seed=Config.SEED + 1
    )

    # -------------------------
    # 7. Create DataModule
    # -------------------------
    print("\n7. Creating data module...")
    datamodule = EEGDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Create initial dataloaders to show stats
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print(f"  Initial batch size: {Config.BATCH_SIZE}")
    print(f"  Train batches per epoch: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # -------------------------
    # 8. Initialize Model
    # -------------------------
    print("\n8. Initializing model...")

    # Get dimensions from sample
    sample_x = train_dataset[0]
    n_chans = sample_x.shape[0]
    n_times = sample_x.shape[1]

    print(f"  Input shape: ({n_chans} channels, {n_times} time points)")
    print(f"  Patches: {n_times // Config.PATCH_LEN}")

    model = MAELabRAM(
        n_chans=n_chans,
        n_times=n_times,
        sfreq=Config.SFREQ,
        emb_size=Config.EMB_SIZE,
        n_layers=Config.N_LAYERS,
        n_heads=Config.N_HEADS,
        patch_len=Config.PATCH_LEN,
        mask_ratio=Config.MASK_RATIO,
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY,
        batch_size=Config.BATCH_SIZE,  # For auto-tuning
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # -------------------------
    # 9. Setup Callbacks
    # -------------------------
    print("\n9. Setting up callbacks...")

    checkpoint_callback = ModelCheckpoint(
        dirpath=Config.CHECKPOINT_DIR,
        filename='mae-labram-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True,
    )

    # -------------------------
    # 10. Setup Logger
    # -------------------------
    # 10. Setup Logger
    # -------------------------
    print("\n9. Setting up logger...")

    # Weights & Biases Logger
    if Config.USE_WANDB:
        logger = WandbLogger(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            name=Config.WANDB_NAME,
            save_dir=Config.LOG_DIR,
            log_model=True,  # Log model checkpoints to wandb
            config={
                # Data config
                "releases": Config.RELEASES,
                "passive_tasks": Config.PASSIVE_TASKS,
                "sfreq": Config.SFREQ,
                "window_size_s": Config.WINDOW_SIZE_S,
                "window_stride_s": Config.WINDOW_STRIDE_S,
                "crop_size_s": Config.CROP_SIZE_S,
                "val_ratio": Config.VAL_RATIO,

                # Model config
                "emb_size": Config.EMB_SIZE,
                "n_layers": Config.N_LAYERS,
                "n_heads": Config.N_HEADS,
                "patch_len": Config.PATCH_LEN,
                "mask_ratio": Config.MASK_RATIO,

                # Training config
                "batch_size": Config.BATCH_SIZE,
                "auto_batch_size": Config.AUTO_BATCH_SIZE,
                "max_epochs": Config.MAX_EPOCHS,
                "lr": Config.LR,
                "weight_decay": Config.WEIGHT_DECAY,
                "num_workers": Config.NUM_WORKERS,
                "seed": Config.SEED,
            }
        )
        print(f"  ✓ WandB logging enabled (project: {Config.WANDB_PROJECT})")
    else:
        logger = False  # Disable logging
        print("  ✓ Logging disabled")

    # -------------------------
    # 11. Create Trainer
    # -------------------------
    print("\n10. Creating trainer...")

    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        log_every_n_steps=1,  # Log every step (useful for small datasets)
        gradient_clip_val=1.0,
        deterministic='warn',  # Use 'warn' instead of True to allow non-deterministic ops
    )

    # -------------------------
    # 11.5. Auto Batch Size Tuning (Optional)
    # -------------------------
    if Config.AUTO_BATCH_SIZE and torch.cuda.is_available():
        print("\n10.5. Running automatic batch size finder...")
        print(
            "  This will test different batch sizes to find the optimal one for your GPU.")

        # Create a tuner
        tuner = Tuner(trainer)

        # Run batch size finder
        # This will update both model.hparams.batch_size and datamodule.batch_size
        tuner.scale_batch_size(
            model,
            datamodule=datamodule,
            mode='power',  # 'power' tries powers of 2, 'binsearch' does binary search
            init_val=Config.BATCH_SIZE,  # Starting batch size
            max_trials=10,  # Maximum number of trials
        )

        # Get the optimal batch size found
        optimal_batch_size = model.hparams.get('batch_size', Config.BATCH_SIZE)
        print(f"  ✓ Optimal batch size found: {optimal_batch_size}")
        print(
            f"  ✓ Train batches per epoch: {len(datamodule.train_dataloader())}")
        print(f"  ✓ Val batches: {len(datamodule.val_dataloader())}")
    else:
        if Config.AUTO_BATCH_SIZE:
            print("\n10.5. Skipping batch size tuning (no GPU available)")
        else:
            print(f"\n10.5. Using configured batch size: {Config.BATCH_SIZE}")

    # -------------------------
    # 12. Train
    # -------------------------
    print("\n11. Starting training...")
    print("="*60)

    trainer.fit(model, datamodule=datamodule)

    # -------------------------
    # 13. Save Final Encoder
    # -------------------------
    print("\n12. Saving final encoder...")

    encoder_path = os.path.join(Config.CHECKPOINT_DIR, "stage1_mae_encoder.pt")
    torch.save({
        'encoder_state_dict': model.encoder.state_dict(),
        'hparams': model.hparams,
        'config': {
            'n_chans': n_chans,
            'n_times': n_times,
            'sfreq': Config.SFREQ,
            'emb_size': Config.EMB_SIZE,
            'n_layers': Config.N_LAYERS,
            'n_heads': Config.N_HEADS,
        }
    }, encoder_path)

    print(f"✓ Encoder saved to: {encoder_path}")
    print(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"✓ Best val_loss: {checkpoint_callback.best_model_score:.4f}")

    # -------------------------
    # 14. Finish Logging
    # -------------------------
    if Config.USE_WANDB:
        wandb.finish()
        print("\n✓ WandB run finished")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
