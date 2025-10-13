# %% [markdown]
# # Challenge 1: Supervised Response Time Prediction with PyTorch Lightning
#
# This notebook trains an EEGNeX model on the Contrast Change Detection (CCD) task
# to predict response time (RT) from stimulus-locked EEG windows.
#
# ## Pipeline Overview
# 1. Load CCD task data from EEGChallengeDataset
# 2. Preprocess: annotate trials with target, add anchors
# 3. Create stimulus-locked windows [stim+0.5s, stim+2.5s] → (129, 200)
# 4. Split at subject level (train/val/test)
# 5. Train EEGNeX with PyTorch Lightning
# 6. Save weights for submission
#
# ## Key Differences from Challenge 2
# - **Windowing**: Stimulus-locked (event-based) vs fixed-length
# - **Loss**: MSE vs MAE
# - **Label**: rt_from_stimulus (per-trial) vs externalizing (per-subject)

# %% Setup and imports
from pathlib import Path
import os
import torch
import warnings
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from braindecode.models import EEGNeX
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from dotenv import load_dotenv
import pickle

# Suppress EEGChallengeDataset warning
warnings.filterwarnings("ignore", category=UserWarning, module="eegdash.dataset.dataset")

# Reproducibility
L.seed_everything(42)

# Lightning will auto-detect hardware
print("Trainer will auto-detect available accelerator (GPU/CPU/TPU/MPS)")

# %% Configuration
# Load data paths from environment
load_dotenv()
DATA_ROOT = Path(os.getenv("EEG2025_DATA_ROOT", "../data")).resolve()
MINI_DIR = (DATA_ROOT / "mini").resolve()
FULL_DIR = (DATA_ROOT / "full").resolve()

# Load wandb config from environment
WANDB_TEAM = os.getenv("WANDB_TEAM", "ubcse-eeg2025")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "eeg2025")

# Data settings
USE_MINI = True  # Set to True for quick testing
DATA_DIR = MINI_DIR if USE_MINI else FULL_DIR
RELEASES = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"]  # Always use full release list

# Excluded subjects (from startkit)
EXCLUDED_SUBJECTS = [
    "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
    "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"
]

# Windowing parameters
EPOCH_LEN_S = 2.0
SFREQ = 100  # Hz
ANCHOR = "stimulus_anchor"
SHIFT_AFTER_STIM = 0.5  # Start window 0.5s after stimulus
WINDOW_LEN = 2.0

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

# Output
OUTPUT_DIR = Path("outputs/challenge1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Data: {DATA_DIR}")
print(f"  Releases: {RELEASES}")
print(f"  Mini mode: {USE_MINI}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")

# %% Load and preprocess data (with caching)
# Setup cache directory
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Create descriptive cache key
releases_str = "_".join(RELEASES)
windowing_params = f"shift{int(SHIFT_AFTER_STIM*10)}_len{int(WINDOW_LEN*10)}"
cache_key = f"windows_{releases_str}_{windowing_params}_mini{USE_MINI}.pkl"
cache_path = CACHE_DIR / cache_key

# Try loading from cache first
if cache_path.exists():
    print("\n" + "="*60)
    print("LOADING FROM CACHE")
    print("="*60)
    print(f"✓ Loading cached windows from: {cache_key}")
    with open(cache_path, "rb") as f:
        single_windows = pickle.load(f)
    print(f"✓ Loaded {len(single_windows)} windows from cache")
else:
    # Cache miss - run full pipeline
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    # Load all releases
    all_datasets_list = []
    for release in RELEASES:
        print(f"Loading {release}...")
        dataset = EEGChallengeDataset(
            task="contrastChangeDetection",
            release=release,
            cache_dir=DATA_DIR,
            mini=USE_MINI,
            description_fields=[
                "subject", "session", "run", "task", "age", "sex", "p_factor",
            ],
        )
        all_datasets_list.append(dataset)

    # Combine datasets
    dataset_ccd = BaseConcatDataset(all_datasets_list)
    print(f"Total recordings: {len(dataset_ccd.datasets)}")

    # Preload raws in parallel (speeds up preprocessing)
    print("Preloading raw data...")
    raws = Parallel(n_jobs=os.cpu_count())(
        delayed(lambda d: d.raw)(d) for d in dataset_ccd.datasets
    )
    print("Done loading raw data")

    # %% Preprocess: Annotate trials with target RT
    print("\n" + "="*60)
    print("PREPROCESSING")
    print("="*60)

    # Apply preprocessing transformations
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

    print("Annotating trials with target RT...")
    preprocess(dataset_ccd, transformation_offline, n_jobs=-1)

    # Keep only recordings with stimulus anchors
    dataset_ccd = keep_only_recordings_with(ANCHOR, dataset_ccd)
    print(f"Recordings with stimulus anchors: {len(dataset_ccd.datasets)}")

    # %% Create stimulus-locked windows
    print("\n" + "="*60)
    print("CREATING WINDOWS")
    print("="*60)

    print(f"Window: [{SHIFT_AFTER_STIM}s, {SHIFT_AFTER_STIM + WINDOW_LEN}s] relative to stimulus")
    print(f"Window length: {WINDOW_LEN}s ({int(WINDOW_LEN * SFREQ)} samples)")

    # Create single-interval windows (stim-locked, long enough to include the response)
    single_windows = create_windows_from_events(
        dataset_ccd,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),  # +0.5s
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),  # +2.5s
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )

    print(f"Created {len(single_windows)} windows")

    # Inject metadata into windows
    single_windows = add_extras_columns(
        single_windows,
        dataset_ccd,
        desc=ANCHOR,
        keys=("target", "rt_from_stimulus", "rt_from_trialstart",
              "stimulus_onset", "response_onset", "correct", "response_type")
    )

    # Save to cache
    print(f"⚠ Caching windows to: {cache_key}")
    with open(cache_path, "wb") as f:
        pickle.dump(single_windows, f)
    print(f"⚠ Delete cache if windowing params change: rm {cache_path}")

# %% Inspect metadata
metadata = single_windows.get_metadata()
print(f"\nMetadata columns: {list(metadata.columns)}")
print(f"\nSample metadata:")
print(metadata[["subject", "target", "rt_from_stimulus", "correct"]].head())

# RT distribution statistics
print(f"\nResponse Time Statistics:")
print(f"  Mean: {metadata['target'].mean():.4f}s")
print(f"  Std: {metadata['target'].std():.4f}s")
print(f"  Min: {metadata['target'].min():.4f}s")
print(f"  Max: {metadata['target'].max():.4f}s")

# %% Split data at subject level
print("\n" + "="*60)
print("SPLITTING DATA")
print("="*60)

subjects = metadata["subject"].unique()
subjects = [s for s in subjects if s not in EXCLUDED_SUBJECTS]
print(f"Total subjects (after exclusion): {len(subjects)}")

# Split: train / (val + test)
train_subj, valid_test_subj = train_test_split(
    subjects,
    test_size=(VAL_FRAC + TEST_FRAC),
    random_state=check_random_state(SEED),
    shuffle=True
)

# Split: val / test
valid_subj, test_subj = train_test_split(
    valid_test_subj,
    test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
    random_state=check_random_state(SEED + 1),
    shuffle=True
)

# Sanity check
assert (set(valid_subj) | set(test_subj) | set(train_subj)) == set(subjects)

print(f"Train subjects: {len(train_subj)}")
print(f"Val subjects: {len(valid_subj)}")
print(f"Test subjects: {len(test_subj)}")

# Create splits
subject_split = single_windows.split("subject")
train_set = BaseConcatDataset([subject_split[s] for s in train_subj if s in subject_split])
val_set = BaseConcatDataset([subject_split[s] for s in valid_subj if s in subject_split])
test_set = BaseConcatDataset([subject_split[s] for s in test_subj if s in subject_split])

print(f"\nWindow counts:")
print(f"  Train: {len(train_set)}")
print(f"  Val: {len(val_set)}")
print(f"  Test: {len(test_set)}")

# %% Define LightningDataModule
class Challenge1DataModule(L.LightningDataModule):
    """DataModule for Challenge 1 (CCD task, RT prediction)"""

    def __init__(self, train_set, val_set, test_set, batch_size=128, num_workers=4):
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
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

# %% Define LightningModule
class Challenge1Module(L.LightningModule):
    """LightningModule for Challenge 1 (RT prediction from CCD task)"""

    def __init__(self, lr=1e-3, weight_decay=1e-5, epochs=100):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = EEGNeX(
            n_chans=129,
            n_outputs=1,
            n_times=200,  # 2 seconds at 100 Hz
            sfreq=100,
        )

        # Loss
        self.loss_fn = torch.nn.MSELoss()

        # Metrics storage
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Batch: (X, y, ...)
        X, y = batch[0], batch[1]
        X = X.float()
        y = y.float().unsqueeze(1)

        # Forward
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)

        # Log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch[0], batch[1]
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
        X, y = batch[0], batch[1]
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
print("\n" + "="*60)
print("SETUP TRAINING")
print("="*60)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=OUTPUT_DIR / "checkpoints",
    filename="challenge1-{epoch:02d}-{val_nrmse:.4f}",
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

# Progress bar with persistence
progress_bar = TQDMProgressBar(leave=True)  # Keep bars after epoch completion

# Logger
wandb_logger = WandbLogger(
    entity=WANDB_TEAM,
    project=WANDB_PROJECT,
    name=f"challenge1_baseline_{'mini' if USE_MINI else 'full'}",
    tags=["challenge1", "supervised", "eegnex", "baseline"],
    save_dir=OUTPUT_DIR,
)

print("Callbacks and logger configured")

# %% Initialize model and datamodule
datamodule = Challenge1DataModule(
    train_set=train_set,
    val_set=val_set,
    test_set=test_set,
    batch_size=BATCH_SIZE,
    num_workers=4,  # Parallel data loading (increase to 8 if you have 16+ cores)
)

model = Challenge1Module(
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
    devices=1,           # Use single device
    callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
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
print("\n" + "="*60)
print("TRAINING")
print("="*60)

trainer.fit(model, datamodule)

print("\nTraining complete!")
print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
print(f"Best val NRMSE: {checkpoint_callback.best_model_score:.6f}")

# %% Test model
print("\n" + "="*60)
print("TESTING")
print("="*60)

trainer.test(model, datamodule, ckpt_path="best")

# %% Save weights for submission
print("\n" + "="*60)
print("SAVING FOR SUBMISSION")
print("="*60)

# Load best checkpoint
best_model = Challenge1Module.load_from_checkpoint(checkpoint_callback.best_model_path)

# Save model state dict (compatible with submission.py)
weights_path = OUTPUT_DIR / "weights_challenge_1.pt"
torch.save(best_model.model.state_dict(), weights_path)

print(f"Model weights saved to: {weights_path}")
print(f"Use this file in submission.py's get_model_challenge_1() method")

# %% Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Experiment: Challenge 1 Baseline (EEGNeX)")
print(f"Data: {len(RELEASES)} releases, {'mini' if USE_MINI else 'full'} dataset")
print(f"Training windows: {len(train_set)}")
print(f"Validation windows: {len(val_set)}")
print(f"Test windows: {len(test_set)}")
print(f"Best val NRMSE: {checkpoint_callback.best_model_score:.6f}")
print(f"Weights saved: {weights_path}")
print(f"Checkpoints: {OUTPUT_DIR / 'checkpoints'}")
print("="*60)
