from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from braindecode.datasets import BaseConcatDataset
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from typing import Tuple
import random
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm
from braindecode.datasets import BaseConcatDataset
from eegdash import EEGChallengeDataset, EEGDash
from braindecode.preprocessing import create_fixed_length_windows
import argparse
import os
import sys
import traceback
import pickle
import pytorch_lightning as pl
from timm.models import create_model
import modeling_vqnsp
import modeling_pretrain
from modeling_pretrain import MEMPretrainModule

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class Config:
    """Centralized configuration for Route-B pretraining"""

    # Data paths
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
    CACHE_DIR = os.path.join(DATA_DIR, "cache")

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
    INPUT_SIZE = int(SFREQ * CROP_SIZE_S)  # 200 samples

    # Patch spec
    PATCH_LEN = 20
    N_FFT_BINS = 5  # Changed to 5 frequency bands for band-based FFT
    USE_BAND_FFT = True  # Use frequency bands instead of raw FFT

    # Model sizes
    TOK_DIM = 256
    CODEBOOK_SIZE = 8192
    CODEBOOK_EMD_DIM = 64
    EMA_DECAY = 0.99
    VQ_TYPE = "gumbel"  # "gumbel" or "ema"
    TEMPERATURE = 1.0
    KL_WEIGHT = 0.01

    MC_D = 256
    MC_LAYERS = 8
    MC_HEADS = 8
    MASK_RATIO = 0.5
    MASK_STRATEGY = "spatial"  # "spatial", "channel", or "random"

    # Training
    BATCH_SIZE = 256
    NUM_WORKERS = 8
    MAX_EPOCHS = 100
    LR = 1e-3
    VAL_RATIO = 0.1

    # I/O
    ROOT_DIR = "./checkpoints/labram"
    TOK_DIR = os.path.join(ROOT_DIR, "tokenizer")
    MEM_DIR = os.path.join(ROOT_DIR, "mem_pretrain")
    CODES_DIR = os.path.join(ROOT_DIR, "codes_passive")
    MC_DIR = os.path.join(ROOT_DIR, "masked")
    FT_DIR = os.path.join(ROOT_DIR, "finetune")

    LOG_DIR = "./logs/labram"

    # WandB
    WANDB_PROJECT = "eeg-challenge-2025"
    WANDB_ENTITY = None
    USE_WANDB = True

    # Seeds
    SEED = 42


class CroppedWindowDataset(Dataset):
    """Wraps a windows dataset with random or deterministic temporal cropping."""

    def __init__(self, windows_dataset: BaseConcatDataset, crop_len: int, sfreq: int = 100, mode: str = "train"):
        """
        Parameters
        ----------
        windows_dataset : BaseConcatDataset
            The input dataset of EEG windows.
        crop_len : int
            Crop length in samples (e.g., 2 * sfreq for 2-second crops).
        sfreq : int, optional
            Sampling frequency, default 100 Hz.
        mode : {"train", "val", "test"}
            Controls whether cropping is random or deterministic (center crop).
        """
        self.windows_dataset = windows_dataset
        self.crop_len = crop_len
        self.sfreq = sfreq
        self.mode = mode.lower()
        assert self.mode in {"train", "val",
                             "test"}, f"Invalid mode: {self.mode}"

    def __len__(self):
        return len(self.windows_dataset.datasets)

    def __getitem__(self, idx):
        ds = self.windows_dataset.datasets[idx]
        x, y, _ = ds[0]  # (C, T)
        C, T = x.shape

        if T > self.crop_len:
            if self.mode == "train":
                # Random crop during training
                start = np.random.randint(0, T - self.crop_len + 1)
            else:
                # Deterministic crop (center)
                start = (T - self.crop_len) // 2
            x_crop = x[:, start:start + self.crop_len]
        else:
            # Pad if too short
            x_crop = x[:, :self.crop_len]
            if x_crop.shape[1] < self.crop_len:
                pad = self.crop_len - x_crop.shape[1]
                x_crop = np.pad(x_crop, ((0, 0), (0, pad)), mode="constant")

        return torch.from_numpy(x_crop).float()


def collate_float(batch):
    return torch.stack(batch, dim=0)


def load_passive_task_data(config: Config, use_mini: bool = True) -> BaseConcatDataset:
    """Load all passive task datasets from specified releases with filtering at load time

    OPTIMIZATION: Uses eegdash's built-in query parameter to filter subjects at load time,
    which is much faster than loading all data and filtering afterward.

    For even more optimization, you could use EEGDash().find() to query metadata first
    and only load recordings that meet all criteria (duration, channels, etc.).
    """
    all_datasets_list = []

    # Calculate total combinations for progress bar
    total_combinations = len(config.RELEASES) * len(config.PASSIVE_TASKS)

    with tqdm(total=total_combinations, desc="Loading datasets", unit="dataset") as pbar:
        for release in config.RELEASES:
            for task in config.PASSIVE_TASKS:
                try:
                    pbar.set_postfix_str(f"{release}/{task}")
                    dataset = EEGChallengeDataset(
                        task=task,
                        release=release,
                        cache_dir=config.DATA_DIR,
                        mini=use_mini,
                        description_fields=[
                            "subject", "session", "run", "task", "age", "sex"],
                    )
                    all_datasets_list.append(dataset)
                    pbar.set_postfix_str(
                        f"{release}/{task} ✓ {len(dataset.datasets)} recordings")
                except Exception as e:
                    pbar.set_postfix_str(f"{release}/{task} ✗ Failed")
                finally:
                    pbar.update(1)

    if not all_datasets_list:
        raise ValueError("No datasets were loaded successfully!")

    all_datasets = BaseConcatDataset(all_datasets_list)
    print(
        f"\n{'='*60}\nTotal recordings loaded: {len(all_datasets.datasets)}\n{'='*60}")
    return all_datasets


def filter_datasets(datasets: BaseConcatDataset, config: Config) -> BaseConcatDataset:
    """Apply quality filters to datasets - optimized version using metadata"""
    print(f"Filtering {len(datasets.datasets)} datasets...")

    def check_dataset(ds):
        try:
            # Subject exclusion is now handled at load time, but double-check
            subj = ds.description.get('subject', '')
            if subj in config.EXCLUDED_SUBJECTS:
                return None

            # Access raw metadata without triggering full S3 download
            # The raw object is lazy-loaded but basic info should be available
            raw = ds.raw
            if raw is None:
                return None

            # Quick checks on metadata (doesn't download data)
            dur = raw.times[-1]
            if dur < config.MIN_DURATION_S:
                return None

            # Check channels
            if len(raw.ch_names) != config.N_CHANNELS:
                return None

            return ds
        except Exception:
            # Silently skip datasets that fail - they're likely corrupted
            return None

    # Use threading for I/O-bound metadata checks with progress bar
    print("  Checking durations and channels...")
    filtered = Parallel(n_jobs=-1, backend='threading')(
        delayed(check_dataset)(ds) for ds in tqdm(
            datasets.datasets,
            desc="  Filtering datasets",
            unit="recording",
            ncols=80
        )
    )

    filtered = [ds for ds in filtered if ds is not None]
    filtered_datasets = BaseConcatDataset(filtered)
    print(
        f"\n✓ Kept {len(filtered_datasets.datasets)}/{len(datasets.datasets)} recordings")

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

    model = create_model(
        "vqnsp_encoder_base_decoder_3x100x12",
        pretrained=False,
        as_tokenizer=False,
        n_code=config.CODEBOOK_SIZE,
        code_dim=config.CODEBOOK_EMD_DIM,
        EEG_size=config.INPUT_SIZE,
        decay=config.EMA_DECAY,
        quantize_kmeans_init=True,
        n_chans_hint=129,                # <-- add this to match your montage
        max_time_window_hint=2
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
        precision="32-true",
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    print("About to call trainer.fit()...")
    sys.stdout.flush()  # Force flush to ensure print appears

    try:
        trainer.fit(model, train_loader, val_loader)
        print("trainer.fit() completed successfully!")
        sys.stdout.flush()
    except KeyboardInterrupt:
        print("Training interrupted by user")
        raise
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR in trainer.fit():")
        print(f"{'='*60}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print(f"{'='*60}")
        print("\nFULL STACK TRACE:")
        print(f"{'='*60}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

    best_ckpt = ckpt_cb.best_model_path
    print(f"✓ Tokenizer training complete: {best_ckpt}")
    return best_ckpt


def pre_train(train_loader, val_loader, config: Config,
              tokenizer_ckpt: str = "best_tokenizer.ckpt"):
    cb_path = os.path.join(config.TOK_DIR, tokenizer_ckpt)

    vq_model = create_model(
        "vqnsp_encoder_base_decoder_3x100x12",
        pretrained=True,
        pretrained_weight=cb_path,
        as_tokenizer=True,
        n_code=config.CODEBOOK_SIZE,
        code_dim=config.CODEBOOK_EMD_DIM,
        EEG_size=config.INPUT_SIZE,
        decay=config.EMA_DECAY,
        n_chans_hint=129,                # <-- add this to match your montage
        max_time_window_hint=2).eval()

    labram_model = create_model(
        "labram_base_patch100_200_8k_vocab",
        pretrained=False,
        drop_path_rate=0.1,
        drop_block_rate=None,
        use_shared_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        vocab_size=config.CODEBOOK_SIZE,
        n_chans_hint=config.N_CHANNELS,     # 129
        max_time_window_hint=2,
    )

    model = MEMPretrainModule(
        labram_model,
        tokenizer=vq_model,
        patch_size=100,
    )

    logger, callbacks, ckpt_cb = setup_logger_and_callbacks(
        run_name="pre-train_labram",
        monitor="val_mem_loss",
        ckpt_dir=config.MEM_DIR,
        use_wandb=config.USE_WANDB
    )

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision="32-true",
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    print("About to call trainer.fit()...")
    sys.stdout.flush()  # Force flush to ensure print appears

    try:
        trainer.fit(model, train_loader, val_loader)
        print("trainer.fit() completed successfully!")
        sys.stdout.flush()
    except KeyboardInterrupt:
        print("Training interrupted by user")
        raise
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR in trainer.fit():")
        print(f"{'='*60}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print(f"{'='*60}")
        print("\nFULL STACK TRACE:")
        print(f"{'='*60}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

    best_ckpt = ckpt_cb.best_model_path
    print(f"✓ Tokenizer training complete: {best_ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mini", action="store_true", default=False,
                        help="Use mini dataset")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None,
                        help="Path to tokenizer checkpoint (for dump_codes/masked stages)")
    parser.add_argument("--skip_cache", action="store_true", default=False,
                        help="Skip loading from cache and reprocess data")
    parser.add_argument("--stage", type=str, default="tokenizer",
                        choices=["tokenizer", "pre-train", "finetune"],
                        help="Training stage to run")
    args = parser.parse_args()

    # Set seeds
    pl.seed_everything(Config.SEED)

    # Cache file paths
    dataset_type = "mini" if args.use_mini else "full"
    cache_file = os.path.join(
        Config.CACHE_DIR, f"filtered_datasets_{dataset_type}.pkl")
    metadata_cache_file = os.path.join(
        Config.CACHE_DIR, f"recording_metadata_{dataset_type}.pkl")

    if os.path.exists(cache_file) and not args.skip_cache:
        print(f"\n{'='*60}")
        print(f"Loading filtered datasets from cache:")
        print(f"  {cache_file}")
        print(f"{'='*60}")
        try:
            with open(cache_file, 'rb') as f:
                filtered_datasets = pickle.load(f)
            print(
                f"✓ Loaded {len(filtered_datasets.datasets)} filtered recordings from cache")
        except Exception as e:
            print(f"✗ Failed to load cache: {e}")
            print("  Reprocessing data...")
            filtered_datasets = None
    else:
        filtered_datasets = None

    if filtered_datasets is None:
        # Load data
        print("Loading passive task data...")
        all_datasets = load_passive_task_data(Config, use_mini=args.use_mini)

        # Filter
        filtered_datasets = filter_datasets(all_datasets, Config)

        # Save filtered data for future runs
        print(f"\n{'='*60}")
        print(f"Saving filtered datasets to cache:")
        print(f"  {cache_file}")
        print(f"{'='*60}")
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(filtered_datasets, f)
            print(f"✓ Cache saved successfully")
        except Exception as e:
            print(f"✗ Failed to save cache: {e}")
            print("  Continuing without cache...")

    # Create windows (fast, don't cache this)
    windows_dataset = create_windows(filtered_datasets, Config)

    # Split
    train_windows, val_windows = split_by_subject(
        windows_dataset, val_ratio=Config.VAL_RATIO, seed=Config.SEED
    )

    crop_len = int(Config.CROP_SIZE_S * Config.SFREQ)

    train_ds = CroppedWindowDataset(
        train_windows, crop_len=crop_len, sfreq=Config.SFREQ, mode="train")
    val_ds = CroppedWindowDataset(
        val_windows, crop_len=crop_len, sfreq=Config.SFREQ, mode="val")

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              num_workers=Config.NUM_WORKERS, shuffle=True,
                              collate_fn=collate_float, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE,
                            num_workers=Config.NUM_WORKERS, shuffle=False,
                            collate_fn=collate_float, pin_memory=True)

    if args.stage == "tokenizer":
        tokenizer_ckpt = train_tokenizer(train_loader, val_loader, Config)
        print(f"Tokenizer checkpoint saved at: {tokenizer_ckpt}")

    if args.stage == "pre-train":
        encoder_ckpt = pre_train(train_loader, val_loader, Config)
        print(f"Encoder checkpoint saved at: {encoder_ckpt}")


if __name__ == "__main__":
    # Set up custom exception handler to catch all unhandled exceptions
    def exception_handler(exc_type, exc_value, exc_traceback):
        print(f"\n{'='*60}")
        print("UNHANDLED EXCEPTION:")
        print(f"{'='*60}")
        print(f"Type: {exc_type.__name__}")
        print(f"Value: {exc_value}")
        print(f"{'='*60}")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.stdout.flush()

    sys.excepthook = exception_handler

    try:
        main()
        print("\nScript completed successfully!")
    except Exception as e:
        print(f"\n{'='*60}")
        print("EXCEPTION IN MAIN:")
        print(f"{'='*60}")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {e}")
        print(f"{'='*60}")
        print("\nFULL STACK TRACE:")
        print(f"{'='*60}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
