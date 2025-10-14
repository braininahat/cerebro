# %% [markdown]
# # Challenge 2: Supervised P-Factor Prediction with PyTorch Lightning
#
# This notebook trains an EEGNeX model on multi-task EEG data to predict
# the externalizing factor (p_factor) - a psychopathology biomarker.
#
# ## Pipeline Overview
# 1. Load multi-task EEG data (movies, resting state, CCD, etc.)
# 2. Create 4s fixed-length windows with 2s stride
# 3. Wrap with DatasetWrapper for random 2s crops (data augmentation)
# 4. Split at subject level (train/val/test)
# 5. Train EEGNeX with PyTorch Lightning
# 6. Save weights for submission
#
# ## Key Differences from Challenge 1
# - **Task**: Multi-task (movies, resting, etc.) vs CCD only
# - **Windowing**: Fixed 4s → random 2s crops vs stimulus-locked
# - **Loss**: MAE (L1) vs MSE
# - **Label**: externalizing (per-subject, constant) vs rt_from_stimulus (per-trial)

import math
import os
import random
import warnings

# %% Setup and imports
from pathlib import Path

import lightning as L
import torch
from braindecode.datasets import BaseConcatDataset, BaseDataset, EEGWindowsDataset
from braindecode.models import EEGNeX
from braindecode.preprocessing import create_fixed_length_windows
from dotenv import load_dotenv
from eegdash.dataset import EEGChallengeDataset
from joblib import Parallel, delayed
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader

# Suppress EEGChallengeDataset warning
warnings.filterwarnings(
    "ignore", category=UserWarning, module="eegdash.dataset.dataset"
)

# Reproducibility
L.seed_everything(42)

# Lightning will auto-detect hardware
print("Trainer will auto-detect available accelerator (GPU/CPU/TPU/MPS)")

# %% Configuration
# Anchor paths to repo root for consistent output location
REPO_ROOT = Path(__file__).resolve().parent.parent

# Load data paths from environment
load_dotenv()
DATA_ROOT = Path(os.getenv("EEG2025_DATA_ROOT", str(REPO_ROOT / "data"))).resolve()
MINI_DIR = (DATA_ROOT / "mini").resolve()
FULL_DIR = (DATA_ROOT / "full").resolve()

# Data settings
USE_MINI = False  # Set to True for quick testing
DATA_DIR = MINI_DIR if USE_MINI else FULL_DIR
RELEASES = (
    ["R5"]
    if USE_MINI
    else ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"]
)

# Tasks to include (adjust based on what you want to use for pretraining)
TASKS = [
    "contrastChangeDetection",
    "restingState",
    "despicableMe",
    "thePresent",
    "diaryOfAWimpyKid",
    "funwithFractals",
    # Add more if needed
]

# Excluded subjects (from startkit)
EXCLUDED_SUBJECTS = [
    "NDARWV769JM7",
    "NDARME789TD2",
    "NDARUA442ZVF",
    "NDARJP304NK1",
    "NDARTY128YLU",
    "NDARDW550GU6",
    "NDARLD243KRE",
    "NDARUJ292JXV",
    "NDARBA381JGH",
]

# Windowing parameters
SFREQ = 100  # Hz
WINDOW_SIZE_S = 4.0  # 4 second windows
WINDOW_STRIDE_S = 2.0  # 2 second stride
CROP_SIZE_S = 2.0  # Random crop to 2 seconds

# Training parameters
BATCH_SIZE = 128
EPOCHS = 2 if USE_MINI else 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10

# Splits
VAL_FRAC = 0.1
TEST_FRAC = 0.1
SEED = 2025

# Output (anchored to repo root for consistent location)
OUTPUT_ROOT = REPO_ROOT / "outputs"
OUTPUT_DIR = OUTPUT_ROOT / "challenge2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Data: {DATA_DIR}")
print(f"  Releases: {RELEASES}")
print(f"  Tasks: {TASKS}")
print(f"  Mini mode: {USE_MINI}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")

# %% Load Challenge 2 data (multi-task)
print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

# Load all tasks from all releases
all_datasets_list = []
for release in RELEASES:
    for task in TASKS:
        try:
            print(f"Loading {release}/{task}...")
            dataset = EEGChallengeDataset(
                task=task,
                release=release,
                cache_dir=DATA_DIR,
                mini=USE_MINI,
                description_fields=[
                    "subject",
                    "session",
                    "run",
                    "task",
                    "age",
                    "sex",
                    "p_factor",
                ],
            )
            all_datasets_list.append(dataset)
        except Exception as e:
            print(f"  Warning: Failed to load {release}/{task}: {e}")
            continue

# Combine datasets
all_datasets = BaseConcatDataset(all_datasets_list)
print(f"Total recordings loaded: {len(all_datasets.datasets)}")

# Preload raws in parallel (speeds up preprocessing)
print("Preloading raw data...")
raws = Parallel(n_jobs=os.cpu_count())(
    delayed(lambda d: d.raw)(d) for d in all_datasets.datasets
)
print("Done loading raw data")

# %% Filter recordings
print("\n" + "=" * 60)
print("FILTERING RECORDINGS")
print("=" * 60)

# Filter criteria:
# - Exclude specific subjects
# - Duration >= 4s (need at least one window)
# - Exactly 129 channels
# - Valid p_factor (not NaN)

filtered_datasets = [
    ds
    for ds in all_datasets.datasets
    if ds.description.subject not in EXCLUDED_SUBJECTS
    and ds.raw.n_times >= WINDOW_SIZE_S * SFREQ
    and len(ds.raw.ch_names) == 129
    and not math.isnan(ds.description.get("p_factor", float("nan")))
]

all_datasets = BaseConcatDataset(filtered_datasets)
print(f"Recordings after filtering: {len(all_datasets.datasets)}")

# Inspect p_factor distribution
p_factors = [ds.description["p_factor"] for ds in all_datasets.datasets]
print(f"\nP-Factor Statistics:")
print(f"  Mean: {sum(p_factors) / len(p_factors):.4f}")
print(f"  Min: {min(p_factors):.4f}")
print(f"  Max: {max(p_factors):.4f}")

# %% Create fixed-length windows
print("\n" + "=" * 60)
print("CREATING WINDOWS")
print("=" * 60)

print(f"Window size: {WINDOW_SIZE_S}s ({int(WINDOW_SIZE_S * SFREQ)} samples)")
print(f"Window stride: {WINDOW_STRIDE_S}s ({int(WINDOW_STRIDE_S * SFREQ)} samples)")
print(f"Crop size: {CROP_SIZE_S}s ({int(CROP_SIZE_S * SFREQ)} samples)")

# Create 4s windows with 2s stride
windows_ds = create_fixed_length_windows(
    all_datasets,
    window_size_samples=int(WINDOW_SIZE_S * SFREQ),
    window_stride_samples=int(WINDOW_STRIDE_S * SFREQ),
    drop_last_window=True,
    preload=True,
)

print(f"Created {len(windows_ds)} windows")


# %% Define DatasetWrapper for random cropping
class DatasetWrapper(BaseDataset):
    """Wrapper that randomly crops 2s from 4s windows for Challenge 2.

    Provides data augmentation by randomly selecting a 2s crop from each 4s window.
    This matches the startkit's Challenge 2 approach.
    """

    def __init__(
        self,
        dataset: EEGWindowsDataset,
        crop_size_samples: int,
        target_name: str = "p_factor",
        seed=None,
    ):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name = target_name
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]

        # Get target (p_factor is per-subject, constant across windows)
        target = self.dataset.get_metadata().iloc[index][self.target_name]
        target = float(target)

        # Additional information
        metadata = self.dataset.get_metadata().iloc[index]
        infos = {
            "subject": metadata["subject"],
            "sex": metadata.get("sex", ""),
            "age": float(metadata.get("age", 0)),
            "task": metadata["task"],
            "session": metadata.get("session", ""),
            "run": metadata.get("run", ""),
        }

        # Randomly crop the signal to the desired length
        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples, f"{i_stop=} {i_start=}"

        # Random offset for cropping
        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        i_start_crop = i_start + start_offset
        i_stop_crop = i_start_crop + self.crop_size_samples

        # Crop X
        X = X[:, start_offset : start_offset + self.crop_size_samples]

        return X, target, (i_window_in_trial, i_start_crop, i_stop_crop), infos


# %% Wrap datasets with DatasetWrapper
print("\n" + "=" * 60)
print("WRAPPING WITH DATASETWARAPPER")
print("=" * 60)

wrapped_windows_ds = BaseConcatDataset(
    [
        DatasetWrapper(ds, crop_size_samples=int(CROP_SIZE_S * SFREQ), seed=SEED + i)
        for i, ds in enumerate(windows_ds.datasets)
    ]
)

print(f"Wrapped {len(wrapped_windows_ds)} windows")

# Test a sample
sample = wrapped_windows_ds[0]
X_sample, y_sample, crop_inds_sample, infos_sample = sample
print(f"\nSample output:")
print(f"  X shape: {X_sample.shape}")
print(f"  y (p_factor): {y_sample:.4f}")
print(f"  Subject: {infos_sample['subject']}")
print(f"  Task: {infos_sample['task']}")

# %% Split data at subject level
print("\n" + "=" * 60)
print("SPLITTING DATA")
print("=" * 60)

metadata = wrapped_windows_ds.get_metadata()
subjects = metadata["subject"].unique()
print(f"Total subjects: {len(subjects)}")

# Split: train / (val + test)
train_subj, valid_test_subj = train_test_split(
    subjects,
    test_size=(VAL_FRAC + TEST_FRAC),
    random_state=check_random_state(SEED),
    shuffle=True,
)

# Split: val / test
valid_subj, test_subj = train_test_split(
    valid_test_subj,
    test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
    random_state=check_random_state(SEED + 1),
    shuffle=True,
)

# Sanity check
assert (set(valid_subj) | set(test_subj) | set(train_subj)) == set(subjects)

print(f"Train subjects: {len(train_subj)}")
print(f"Val subjects: {len(valid_subj)}")
print(f"Test subjects: {len(test_subj)}")

# Create splits
subject_split = wrapped_windows_ds.split("subject")
train_set = BaseConcatDataset(
    [subject_split[s] for s in train_subj if s in subject_split]
)
val_set = BaseConcatDataset(
    [subject_split[s] for s in valid_subj if s in subject_split]
)
test_set = BaseConcatDataset(
    [subject_split[s] for s in test_subj if s in subject_split]
)

print(f"\nWindow counts:")
print(f"  Train: {len(train_set)}")
print(f"  Val: {len(val_set)}")
print(f"  Test: {len(test_set)}")


# %% Define LightningDataModule
class Challenge2DataModule(L.LightningDataModule):
    """DataModule for Challenge 2 (multi-task, p_factor prediction)"""

    def __init__(self, train_set, val_set, test_set, batch_size=128, num_workers=0):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# %% Define LightningModule
class Challenge2Module(L.LightningModule):
    """LightningModule for Challenge 2 (p_factor prediction from multi-task EEG)"""

    def __init__(self, lr=1e-3, weight_decay=1e-5, epochs=100):
        super().__init__()
        self.save_hyperparameters()

        # Model (same architecture as C1)
        self.model = EEGNeX(
            n_chans=129,
            n_outputs=1,
            n_times=200,  # 2 seconds at 100 Hz
            sfreq=100,
        )

        # Loss: MAE (L1) instead of MSE
        self.loss_fn = torch.nn.L1Loss()

        # Metrics storage
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Batch: (X, y, crop_inds, infos)
        X, y, crop_inds, infos = batch
        X = X.float()
        y = y.float().unsqueeze(1)

        # Forward
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)

        # Log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y, crop_inds, infos = batch
        X = X.float()
        y = y.float().unsqueeze(1)

        # Forward
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)

        # Store for NRMSE computation
        self.val_preds.append(y_pred.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        # Log
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        # Compute NRMSE (normalized RMSE)
        all_preds = torch.cat(self.val_preds, dim=0).squeeze()
        all_targets = torch.cat(self.val_targets, dim=0).squeeze()

        # RMSE
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2))

        # Normalize by target standard deviation
        target_std = torch.std(all_targets)
        nrmse = rmse / target_std if target_std > 0 else rmse

        # Log
        self.log("val_rmse", rmse, prog_bar=False)
        self.log("val_nrmse", nrmse, prog_bar=True)

        # Clear for next epoch
        self.val_preds.clear()
        self.val_targets.clear()

    def test_step(self, batch, batch_idx):
        X, y, crop_inds, infos = batch
        X = X.float()
        y = y.float().unsqueeze(1)

        # Forward
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)

        # Store for NRMSE computation
        self.test_preds.append(y_pred.detach().cpu())
        self.test_targets.append(y.detach().cpu())

        # Log
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        # Compute NRMSE
        all_preds = torch.cat(self.test_preds, dim=0).squeeze()
        all_targets = torch.cat(self.test_targets, dim=0).squeeze()

        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2))
        target_std = torch.std(all_targets)
        nrmse = rmse / target_std if target_std > 0 else rmse

        self.log("test_rmse", rmse)
        self.log("test_nrmse", nrmse)

        print(f"\nTest Results:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  NRMSE: {nrmse:.6f}")

        # Clear
        self.test_preds.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
        )
        return [optimizer], [scheduler]


# %% Setup callbacks and logger
print("\n" + "=" * 60)
print("SETUP TRAINING")
print("=" * 60)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=OUTPUT_DIR / "checkpoints",
    filename="challenge2-{epoch:02d}-{val_nrmse:.4f}",
    monitor="val_nrmse",
    mode="min",
    save_top_k=3,
    save_last=True,
)

early_stop_callback = EarlyStopping(
    monitor="val_nrmse",
    patience=EARLY_STOPPING_PATIENCE,
    mode="min",
    verbose=True,
)

# Logger
wandb_logger = WandbLogger(
    project="eeg-foundation-challenge",
    name=f"challenge2_baseline_{'mini' if USE_MINI else 'full'}",
    tags=["challenge2", "supervised", "eegnex", "baseline"],
    save_dir=OUTPUT_DIR,
)

print("Callbacks and logger configured")

# %% Initialize model and datamodule
datamodule = Challenge2DataModule(
    train_set=train_set,
    val_set=val_set,
    test_set=test_set,
    batch_size=BATCH_SIZE,
    num_workers=0,  # Set to >0 for parallel data loading
)

model = Challenge2Module(
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS,
)

print(f"\nModel: {model.model.__class__.__name__}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% Setup Trainer
trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",  # Auto-detect best available hardware
    devices=1,  # Use single device
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=wandb_logger,
    gradient_clip_val=1.0,
    precision="16-mixed",
    deterministic=True,
    log_every_n_steps=10,
)

print("\nTrainer configured:")
print(f"  Max epochs: {EPOCHS}")
print(f"  Accelerator: auto-detected ({trainer.accelerator.__class__.__name__})")
print(f"  Precision: {trainer.precision}")
print(f"  Gradient clip: 1.0")

# %% Train model
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

trainer.fit(model, datamodule)

print("\nTraining complete!")
print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
print(f"Best val NRMSE: {checkpoint_callback.best_model_score:.6f}")

# %% Test model
print("\n" + "=" * 60)
print("TESTING")
print("=" * 60)

trainer.test(model, datamodule, ckpt_path="best")

# %% Save weights for submission
print("\n" + "=" * 60)
print("SAVING FOR SUBMISSION")
print("=" * 60)

# Load best checkpoint
best_model = Challenge2Module.load_from_checkpoint(checkpoint_callback.best_model_path)

# Save model state dict (compatible with submission.py)
weights_path = OUTPUT_DIR / "weights_challenge_2.pt"
torch.save(best_model.model.state_dict(), weights_path)

print(f"Model weights saved to: {weights_path}")
print(f"Use this file in submission.py's get_model_challenge_2() method")

# %% Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Experiment: Challenge 2 Baseline (EEGNeX)")
print(
    f"Data: {len(RELEASES)} releases, {len(TASKS)} tasks, {'mini' if USE_MINI else 'full'} dataset"
)
print(f"Training windows: {len(train_set)}")
print(f"Validation windows: {len(val_set)}")
print(f"Test windows: {len(test_set)}")
print(f"Best val NRMSE: {checkpoint_callback.best_model_score:.6f}")
print(f"Weights saved: {weights_path}")
print(f"Checkpoints: {OUTPUT_DIR / 'checkpoints'}")
print("=" * 60)
