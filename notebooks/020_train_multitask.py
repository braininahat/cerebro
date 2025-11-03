# %% [markdown]
# # Multitask Training: Shared Encoder + Dual Heads with PyTorch Lightning
#
# This notebook trains a shared EEGNeX encoder with dual task-specific heads
# to simultaneously predict:
# - Challenge 1: Response time (RT) from CCD task
# - Challenge 2: Externalizing factor (p_factor) from multi-task EEG
#
# ## Pipeline Overview
# 1. Load both C1 (CCD) and C2 (multi-task) datasets
# 2. Setup CombinedLoader to alternate between C1 and C2 batches
# 3. Train shared encoder + dual heads with frozen encoder phase
# 4. Monitor overall score: 0.3 × NRMSE_C1 + 0.7 × NRMSE_C2
# 5. Save encoder and heads for submission
#
# ## Key Features
# - **Shared encoder**: EEGNeX(n_outputs=128) outputs embeddings
# - **Dual heads**: Linear(128, 1) for each task
# - **Frozen phase**: Encoder frozen for first 5 epochs (only train heads)
# - **Alternating batches**: CombinedLoader cycles between C1 and C2
# - **Competition metric**: Overall score matches leaderboard (30% C1 + 70% C2)

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
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    create_windows_from_events,
    preprocess,
)
from dotenv import load_dotenv
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
    keep_only_recordings_with,
)
from joblib import Parallel, delayed
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import CombinedLoader
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
DATA_ROOT = Path(os.getenv("HBN_ROOT", str(REPO_ROOT / "data"))).resolve()
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

# Excluded subjects
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
EPOCH_LEN_S = 2.0
ANCHOR = "stimulus_anchor"
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0

# C2 windowing
WINDOW_SIZE_S = 4.0
WINDOW_STRIDE_S = 2.0
CROP_SIZE_S = 2.0

# C2 tasks
C2_TASKS = [
    "contrastChangeDetection",
    "restingState",
    "despicableMe",
    "thePresent",
    "diaryOfAWimpyKid",
    "funwithFractals",
]

# Training parameters
BATCH_SIZE = 128
EPOCHS = 2 if USE_MINI else 50  # Fewer epochs for fine-tuning
LR = 1e-4  # Lower LR for multitask
WEIGHT_DECAY = 1e-5
FREEZE_ENCODER_EPOCHS = 5  # Freeze encoder for first N epochs
EARLY_STOPPING_PATIENCE = 10

# Competition weights
WEIGHT_C1 = 0.3
WEIGHT_C2 = 0.7

# Splits
VAL_FRAC = 0.1
TEST_FRAC = 0.1
SEED = 2025

# Output (anchored to repo root for consistent location)
OUTPUT_ROOT = REPO_ROOT / "outputs"
OUTPUT_DIR = OUTPUT_ROOT / "multitask"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Data: {DATA_DIR}")
print(f"  Releases: {RELEASES}")
print(f"  Mini mode: {USE_MINI}")
print(f"  Epochs: {EPOCHS}")
print(f"  Freeze encoder epochs: {FREEZE_ENCODER_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Competition weights: C1={WEIGHT_C1}, C2={WEIGHT_C2}")

# %% Load Challenge 1 data (CCD task)
print("\n" + "=" * 60)
print("LOADING CHALLENGE 1 DATA")
print("=" * 60)

# Load CCD task
c1_datasets_list = []
for release in RELEASES:
    print(f"Loading C1 from {release}...")
    dataset = EEGChallengeDataset(
        task="contrastChangeDetection",
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
    c1_datasets_list.append(dataset)

dataset_ccd = BaseConcatDataset(c1_datasets_list)
print(f"C1 recordings: {len(dataset_ccd.datasets)}")

# Preload raws
print("Preloading C1 raw data...")
raws_c1 = Parallel(n_jobs=os.cpu_count())(
    delayed(lambda d: d.raw)(d) for d in dataset_ccd.datasets
)

# Preprocess
transformation_offline = [
    Preprocessor(
        annotate_trials_with_target,
        target_field="rt_from_stimulus",
        epoch_length=EPOCH_LEN_S,
        require_stimulus=True,
        require_response=True,
        apply_on_array=False,
    ),
    Preprocessor(add_aux_anchors, apply_on_array=False),
]
preprocess(dataset_ccd, transformation_offline, n_jobs=1)
dataset_ccd = keep_only_recordings_with(ANCHOR, dataset_ccd)

# Create windows
c1_windows = create_windows_from_events(
    dataset_ccd,
    mapping={ANCHOR: 0},
    trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
    trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
    window_size_samples=int(EPOCH_LEN_S * SFREQ),
    window_stride_samples=SFREQ,
    preload=True,
)
c1_windows = add_extras_columns(
    c1_windows,
    dataset_ccd,
    desc=ANCHOR,
    keys=(
        "target",
        "rt_from_stimulus",
        "rt_from_trialstart",
        "stimulus_onset",
        "response_onset",
        "correct",
        "response_type",
    ),
)

print(f"C1 windows: {len(c1_windows)}")

# %% Load Challenge 2 data (multi-task)
print("\n" + "=" * 60)
print("LOADING CHALLENGE 2 DATA")
print("=" * 60)

# Load all tasks
c2_datasets_list = []
for release in RELEASES:
    for task in C2_TASKS:
        try:
            print(f"Loading C2 from {release}/{task}...")
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
            c2_datasets_list.append(dataset)
        except Exception as e:
            print(f"  Warning: Failed to load {release}/{task}: {e}")
            continue

c2_all_datasets = BaseConcatDataset(c2_datasets_list)
print(f"C2 recordings (before filtering): {len(c2_all_datasets.datasets)}")

# Preload raws
print("Preloading C2 raw data...")
raws_c2 = Parallel(n_jobs=os.cpu_count())(
    delayed(lambda d: d.raw)(d) for d in c2_all_datasets.datasets
)

# Filter
c2_filtered = [
    ds
    for ds in c2_all_datasets.datasets
    if ds.description.subject not in EXCLUDED_SUBJECTS
    and ds.raw.n_times >= WINDOW_SIZE_S * SFREQ
    and len(ds.raw.ch_names) == 129
    and not math.isnan(ds.description.get("p_factor", float("nan")))
]
c2_all_datasets = BaseConcatDataset(c2_filtered)
print(f"C2 recordings (after filtering): {len(c2_all_datasets.datasets)}")

# Create windows
c2_windows = create_fixed_length_windows(
    c2_all_datasets,
    window_size_samples=int(WINDOW_SIZE_S * SFREQ),
    window_stride_samples=int(WINDOW_STRIDE_S * SFREQ),
    drop_last_window=True,
    preload=True,
)


# Wrap with DatasetWrapper
class DatasetWrapper(BaseDataset):
    """Random crops 2s from 4s windows for C2"""

    def __init__(self, dataset, crop_size_samples, target_name="p_factor", seed=None):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name = target_name
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]
        metadata = self.dataset.get_metadata().iloc[index]
        target = float(metadata[self.target_name])
        infos = {
            "subject": metadata["subject"],
            "sex": metadata.get("sex", ""),
            "age": float(metadata.get("age", 0)),
            "task": metadata["task"],
            "session": metadata.get("session", ""),
            "run": metadata.get("run", ""),
        }

        i_window_in_trial, i_start, i_stop = crop_inds
        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        X = X[:, start_offset : start_offset + self.crop_size_samples]
        i_start_crop = i_start + start_offset
        i_stop_crop = i_start_crop + self.crop_size_samples

        return X, target, (i_window_in_trial, i_start_crop, i_stop_crop), infos


c2_windows = BaseConcatDataset(
    [
        DatasetWrapper(ds, crop_size_samples=int(CROP_SIZE_S * SFREQ), seed=SEED + i)
        for i, ds in enumerate(c2_windows.datasets)
    ]
)

print(f"C2 windows: {len(c2_windows)}")

# %% Split data at subject level
print("\n" + "=" * 60)
print("SPLITTING DATA")
print("=" * 60)

# C1 splits
c1_metadata = c1_windows.get_metadata()
c1_subjects = [s for s in c1_metadata["subject"].unique() if s not in EXCLUDED_SUBJECTS]
c1_train_subj, c1_valid_test_subj = train_test_split(
    c1_subjects,
    test_size=(VAL_FRAC + TEST_FRAC),
    random_state=check_random_state(SEED),
    shuffle=True,
)
c1_valid_subj, c1_test_subj = train_test_split(
    c1_valid_test_subj,
    test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
    random_state=check_random_state(SEED + 1),
    shuffle=True,
)

c1_subject_split = c1_windows.split("subject")
c1_train_set = BaseConcatDataset(
    [c1_subject_split[s] for s in c1_train_subj if s in c1_subject_split]
)
c1_val_set = BaseConcatDataset(
    [c1_subject_split[s] for s in c1_valid_subj if s in c1_subject_split]
)
c1_test_set = BaseConcatDataset(
    [c1_subject_split[s] for s in c1_test_subj if s in c1_subject_split]
)

print(
    f"C1 splits: Train={len(c1_train_set)}, Val={len(c1_val_set)}, Test={len(c1_test_set)}"
)

# C2 splits
c2_metadata = c2_windows.get_metadata()
c2_subjects = c2_metadata["subject"].unique()
c2_train_subj, c2_valid_test_subj = train_test_split(
    c2_subjects,
    test_size=(VAL_FRAC + TEST_FRAC),
    random_state=check_random_state(SEED),
    shuffle=True,
)
c2_valid_subj, c2_test_subj = train_test_split(
    c2_valid_test_subj,
    test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
    random_state=check_random_state(SEED + 1),
    shuffle=True,
)

c2_subject_split = c2_windows.split("subject")
c2_train_set = BaseConcatDataset(
    [c2_subject_split[s] for s in c2_train_subj if s in c2_subject_split]
)
c2_val_set = BaseConcatDataset(
    [c2_subject_split[s] for s in c2_valid_subj if s in c2_subject_split]
)
c2_test_set = BaseConcatDataset(
    [c2_subject_split[s] for s in c2_test_subj if s in c2_subject_split]
)

print(
    f"C2 splits: Train={len(c2_train_set)}, Val={len(c2_val_set)}, Test={len(c2_test_set)}"
)


# %% Define MultitaskDataModule
class MultitaskDataModule(L.LightningDataModule):
    """DataModule for multitask training with CombinedLoader"""

    def __init__(
        self,
        c1_train,
        c1_val,
        c1_test,
        c2_train,
        c2_val,
        c2_test,
        batch_size=128,
        num_workers=0,
    ):
        super().__init__()
        self.c1_train = c1_train
        self.c1_val = c1_val
        self.c1_test = c1_test
        self.c2_train = c2_train
        self.c2_val = c2_val
        self.c2_test = c2_test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        # Combine C1 and C2 loaders with alternating strategy
        c1_loader = DataLoader(
            self.c1_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
        c2_loader = DataLoader(
            self.c2_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

        # CombinedLoader alternates batches and cycles shorter dataset
        return CombinedLoader({"c1": c1_loader, "c2": c2_loader}, mode="max_size_cycle")

    def val_dataloader(self):
        # Return list of loaders for separate validation
        c1_loader = DataLoader(
            self.c1_val, batch_size=self.batch_size, num_workers=self.num_workers
        )
        c2_loader = DataLoader(
            self.c2_val, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return [c1_loader, c2_loader]

    def test_dataloader(self):
        # Return list of loaders for separate testing
        c1_loader = DataLoader(
            self.c1_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
        c2_loader = DataLoader(
            self.c2_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return [c1_loader, c2_loader]


# %% Define MultitaskModule
class MultitaskModule(L.LightningModule):
    """Multitask module: shared encoder + dual heads for C1 and C2"""

    def __init__(
        self,
        lr=1e-4,
        weight_decay=1e-5,
        freeze_encoder_epochs=5,
        epochs=50,
        weight_c1=0.3,
        weight_c2=0.7,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Shared encoder (output embeddings, not predictions)
        self.encoder = EEGNeX(
            n_chans=129,
            n_outputs=128,  # Embedding dimension
            n_times=200,
            sfreq=100,
        )

        # Task-specific heads
        self.head_c1 = torch.nn.Linear(128, 1)  # RT prediction
        self.head_c2 = torch.nn.Linear(128, 1)  # p_factor prediction

        # Losses
        self.loss_c1 = torch.nn.MSELoss()
        self.loss_c2 = torch.nn.L1Loss()

        # Metrics storage
        self.val_c1_preds = []
        self.val_c1_targets = []
        self.val_c2_preds = []
        self.val_c2_targets = []

    def forward(self, x, task="c1"):
        embeddings = self.encoder(x)
        if task == "c1":
            return self.head_c1(embeddings)
        else:
            return self.head_c2(embeddings)

    def on_train_epoch_start(self):
        # Unfreeze encoder after freeze_encoder_epochs
        if self.current_epoch == self.hparams.freeze_encoder_epochs:
            print(f"\n{'='*60}")
            print(f"UNFREEZING ENCODER at epoch {self.current_epoch}")
            print(f"{'='*60}\n")
            for param in self.encoder.parameters():
                param.requires_grad = True

    def training_step(self, batch, batch_idx):
        # Batch is dict: {"c1": c1_batch, "c2": c2_batch}
        c1_batch = batch["c1"]
        c2_batch = batch["c2"]

        # C1 forward
        X_c1, y_c1 = c1_batch[0], c1_batch[1]
        X_c1 = X_c1.float()
        y_c1 = y_c1.float().unsqueeze(1)
        y_pred_c1 = self(X_c1, task="c1")
        loss_c1 = self.loss_c1(y_pred_c1, y_c1)

        # C2 forward
        X_c2, y_c2 = c2_batch[0], c2_batch[1]
        X_c2 = X_c2.float()
        y_c2 = y_c2.float().unsqueeze(1)
        y_pred_c2 = self(X_c2, task="c2")
        loss_c2 = self.loss_c2(y_pred_c2, y_c2)

        # Combined loss (weighted)
        loss = self.hparams.weight_c1 * loss_c1 + self.hparams.weight_c2 * loss_c2

        # Log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_c1", loss_c1, on_step=False, on_epoch=True)
        self.log("train_loss_c2", loss_c2, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # dataloader_idx: 0=C1, 1=C2
        if dataloader_idx == 0:  # C1
            X, y = batch[0], batch[1]
            X = X.float()
            y = y.float().unsqueeze(1)
            y_pred = self(X, task="c1")
            loss = self.loss_c1(y_pred, y)

            self.val_c1_preds.append(y_pred.detach().cpu())
            self.val_c1_targets.append(y.detach().cpu())

            self.log("val_loss_c1", loss, on_step=False, on_epoch=True)
        else:  # C2
            X, y = batch[0], batch[1]
            X = X.float()
            y = y.float().unsqueeze(1)
            y_pred = self(X, task="c2")
            loss = self.loss_c2(y_pred, y)

            self.val_c2_preds.append(y_pred.detach().cpu())
            self.val_c2_targets.append(y.detach().cpu())

            self.log("val_loss_c2", loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        # Compute NRMSE for both tasks
        # C1
        all_preds_c1 = torch.cat(self.val_c1_preds, dim=0).squeeze()
        all_targets_c1 = torch.cat(self.val_c1_targets, dim=0).squeeze()
        rmse_c1 = torch.sqrt(torch.mean((all_preds_c1 - all_targets_c1) ** 2))
        nrmse_c1 = rmse_c1 / torch.std(all_targets_c1)

        # C2
        all_preds_c2 = torch.cat(self.val_c2_preds, dim=0).squeeze()
        all_targets_c2 = torch.cat(self.val_c2_targets, dim=0).squeeze()
        rmse_c2 = torch.sqrt(torch.mean((all_preds_c2 - all_targets_c2) ** 2))
        nrmse_c2 = rmse_c2 / torch.std(all_targets_c2)

        # Overall score (competition metric)
        overall_score = (
            self.hparams.weight_c1 * nrmse_c1 + self.hparams.weight_c2 * nrmse_c2
        )

        # Log
        self.log("val_nrmse_c1", nrmse_c1, prog_bar=False)
        self.log("val_nrmse_c2", nrmse_c2, prog_bar=False)
        self.log("val_overall_score", overall_score, prog_bar=True)

        # Clear
        self.val_c1_preds.clear()
        self.val_c1_targets.clear()
        self.val_c2_preds.clear()
        self.val_c2_targets.clear()

    def configure_optimizers(self):
        # Initially freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

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

checkpoint_callback = ModelCheckpoint(
    dirpath=OUTPUT_DIR / "checkpoints",
    filename="multitask-{epoch:02d}-{val_overall_score:.4f}",
    monitor="val_overall_score",
    mode="min",
    save_top_k=3,
    save_last=True,
)

early_stop_callback = EarlyStopping(
    monitor="val_overall_score",
    patience=EARLY_STOPPING_PATIENCE,
    mode="min",
    verbose=True,
)

wandb_logger = WandbLogger(
    project="eeg-foundation-challenge",
    name=f"multitask_{'mini' if USE_MINI else 'full'}",
    tags=["multitask", "shared_encoder", "dual_heads"],
    save_dir=OUTPUT_DIR,
)

print("Callbacks and logger configured")

# %% Initialize model and datamodule
datamodule = MultitaskDataModule(
    c1_train=c1_train_set,
    c1_val=c1_val_set,
    c1_test=c1_test_set,
    c2_train=c2_train_set,
    c2_val=c2_val_set,
    c2_test=c2_test_set,
    batch_size=BATCH_SIZE,
    num_workers=0,
)

model = MultitaskModule(
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    freeze_encoder_epochs=FREEZE_ENCODER_EPOCHS,
    epochs=EPOCHS,
    weight_c1=WEIGHT_C1,
    weight_c2=WEIGHT_C2,
)

print(f"\nModel architecture:")
print(f"  Encoder: {model.encoder.__class__.__name__} → 128 embeddings")
print(f"  Head C1: Linear(128, 1)")
print(f"  Head C2: Linear(128, 1)")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

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
print(f"  Freeze encoder: {FREEZE_ENCODER_EPOCHS} epochs")
print(f"  Accelerator: auto-detected ({trainer.accelerator.__class__.__name__})")

# %% Train model
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)
print("NOTE: Encoder is frozen for first 5 epochs (only heads trained)")
print("      Encoder unfreezes at epoch 5 for end-to-end fine-tuning")
print("=" * 60 + "\n")

trainer.fit(model, datamodule)

print("\nTraining complete!")
print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
print(f"Best val overall score: {checkpoint_callback.best_model_score:.6f}")

# %% Save weights for submission
print("\n" + "=" * 60)
print("SAVING FOR SUBMISSION")
print("=" * 60)

# Load best checkpoint
best_model = MultitaskModule.load_from_checkpoint(checkpoint_callback.best_model_path)

# Save encoder and heads separately
encoder_path = OUTPUT_DIR / "encoder.pt"
head_c1_path = OUTPUT_DIR / "head_c1.pt"
head_c2_path = OUTPUT_DIR / "head_c2.pt"

torch.save(best_model.encoder.state_dict(), encoder_path)
torch.save(best_model.head_c1.state_dict(), head_c1_path)
torch.save(best_model.head_c2.state_dict(), head_c2_path)

print(f"Encoder saved: {encoder_path}")
print(f"Head C1 saved: {head_c1_path}")
print(f"Head C2 saved: {head_c2_path}")

# Also create combined weights for submission.py compatibility
# For C1: encoder + head_c1
print("\nCreating combined weights for submission...")

# C1 combined model
c1_combined = EEGNeX(n_chans=129, n_outputs=128, n_times=200, sfreq=100)
c1_combined.load_state_dict(best_model.encoder.state_dict())
# Need to add head - this requires modifying model architecture
# For now, save encoder and heads separately and document assembly in submission.py

weights_c1_path = OUTPUT_DIR / "weights_challenge_1.pt"
weights_c2_path = OUTPUT_DIR / "weights_challenge_2.pt"

# Save as dict with encoder + head
torch.save(
    {
        "encoder": best_model.encoder.state_dict(),
        "head": best_model.head_c1.state_dict(),
    },
    weights_c1_path,
)

torch.save(
    {
        "encoder": best_model.encoder.state_dict(),
        "head": best_model.head_c2.state_dict(),
    },
    weights_c2_path,
)

print(f"Combined weights C1: {weights_c1_path}")
print(f"Combined weights C2: {weights_c2_path}")

# %% Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Experiment: Multitask (Shared Encoder + Dual Heads)")
print(f"Data: {len(RELEASES)} releases, {'mini' if USE_MINI else 'full'} dataset")
print(f"C1 training windows: {len(c1_train_set)}")
print(f"C2 training windows: {len(c2_train_set)}")
print(f"Best val overall score: {checkpoint_callback.best_model_score:.6f}")
print(f"Weights saved:")
print(f"  - {weights_c1_path}")
print(f"  - {weights_c2_path}")
print(f"  - {encoder_path} (encoder only)")
print(f"  - {head_c1_path} (C1 head only)")
print(f"  - {head_c2_path} (C2 head only)")
print(f"Checkpoints: {OUTPUT_DIR / 'checkpoints'}")
print("=" * 60)
print("\nNOTE: submission.py needs to be updated to load encoder + head architecture")
print("      See OUTPUT_DIR for saved model components")
