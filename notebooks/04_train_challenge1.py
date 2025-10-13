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
from rich.console import Console
from rich.logging import RichHandler
import logging
from datetime import datetime

# Suppress EEGChallengeDataset warning
warnings.filterwarnings("ignore", category=UserWarning, module="eegdash.dataset.dataset")

# Capture other warnings to log
logging.captureWarnings(True)

# Enable Tensor Cores on RTX 4090 for faster matmul operations
torch.set_float32_matmul_precision('high')  # Trades minimal precision for 30-50% speedup

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

# Load wandb config from environment
WANDB_TEAM = os.getenv("WANDB_TEAM", "ubcse-eeg2025")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "eeg2025")

# Data settings
USE_MINI = False  # Set to True for quick testing
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
BATCH_SIZE = 1024
EPOCHS = 2 if USE_MINI else 100
LR = 1e-3  # Will be overridden by LR finder
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10
PRECISION = "32"  # Options: "32" (full), "16-mixed" (fast), "bf16-mixed" (stable+fast, requires Ampere+)

# Hyperparameter tuning switches
RUN_LR_FINDER = True
RUN_BATCH_SIZE_FINDER = True

# Splits
VAL_FRAC = 0.1
TEST_FRAC = 0.1
SEED = 2025

# Output (organized by run timestamp)
OUTPUT_ROOT = REPO_ROOT / "outputs"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_ROOT / "challenge1" / timestamp
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Subdirectories
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
LOG_DIR = RUN_DIR
CACHE_DIR = OUTPUT_ROOT / "challenge1" / "cache"  # Shared across runs
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging to file and console
log_file = LOG_DIR / f"train_mini{USE_MINI}.log"

# Configure logging with Rich handler for console + file handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True, markup=True),  # Pretty console output
        logging.FileHandler(log_file, mode='w'),  # File output
    ]
)

console = Console()
logger = logging.getLogger("cerebro")

# Log configuration
logger.info(f"[bold green]Logging to:[/bold green] {log_file}")
logger.info(f"\n[bold]Configuration:[/bold]")
logger.info(f"  Data: {DATA_DIR}")
logger.info(f"  Releases: {RELEASES}")
logger.info(f"  Mini mode: {USE_MINI}")
logger.info(f"  Epochs: {EPOCHS}")
logger.info(f"  Batch size: {BATCH_SIZE}")

# %% Load and preprocess data (with caching)
# Create descriptive cache key
releases_str = "_".join(RELEASES)
windowing_params = f"shift{int(SHIFT_AFTER_STIM*10)}_len{int(WINDOW_LEN*10)}"
cache_key = f"windows_{releases_str}_{windowing_params}_mini{USE_MINI}.pkl"
cache_path = CACHE_DIR / cache_key

# Try loading from cache first
if cache_path.exists():
    logger.info("\n" + "="*60)
    logger.info("[bold cyan]LOADING FROM CACHE[/bold cyan]")
    logger.info("="*60)
    logger.info(f"[green]✓[/green] Loading cached windows from: {cache_key}")
    with open(cache_path, "rb") as f:
        single_windows = pickle.load(f)
    logger.info(f"[green]✓[/green] Loaded {len(single_windows)} windows from cache")
else:
    # Cache miss - run full pipeline
    logger.info("\n" + "="*60)
    logger.info("[bold cyan]LOADING DATA[/bold cyan]")
    logger.info("="*60)

    # Load all releases
    all_datasets_list = []
    for release in RELEASES:
        logger.info(f"Loading {release}...")
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
    logger.info(f"[bold]Total recordings:[/bold] {len(dataset_ccd.datasets)}")

    # Preload raws in parallel (speeds up preprocessing)
    logger.info("Preloading raw data...")
    raws = Parallel(n_jobs=os.cpu_count())(
        delayed(lambda d: d.raw)(d) for d in dataset_ccd.datasets
    )
    logger.info("[green]Done loading raw data[/green]")

    # %% Preprocess: Annotate trials with target RT
    logger.info("\n" + "="*60)
    logger.info("[bold cyan]PREPROCESSING[/bold cyan]")
    logger.info("="*60)

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

    logger.info("Annotating trials with target RT...")
    preprocess(dataset_ccd, transformation_offline, n_jobs=-1)

    # Keep only recordings with stimulus anchors
    dataset_ccd = keep_only_recordings_with(ANCHOR, dataset_ccd)
    logger.info(f"[bold]Recordings with stimulus anchors:[/bold] {len(dataset_ccd.datasets)}")

    # %% Create stimulus-locked windows
    logger.info("\n" + "="*60)
    logger.info("[bold cyan]CREATING WINDOWS[/bold cyan]")
    logger.info("="*60)

    logger.info(f"Window: [{SHIFT_AFTER_STIM}s, {SHIFT_AFTER_STIM + WINDOW_LEN}s] relative to stimulus")
    logger.info(f"Window length: {WINDOW_LEN}s ({int(WINDOW_LEN * SFREQ)} samples)")

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

    logger.info(f"[bold]Created {len(single_windows)} windows[/bold]")

    # Inject metadata into windows
    single_windows = add_extras_columns(
        single_windows,
        dataset_ccd,
        desc=ANCHOR,
        keys=("target", "rt_from_stimulus", "rt_from_trialstart",
              "stimulus_onset", "response_onset", "correct", "response_type")
    )

    # Save to cache
    logger.info(f"[yellow]⚠[/yellow] Caching windows to: {cache_key}")
    with open(cache_path, "wb") as f:
        pickle.dump(single_windows, f)
    logger.info(f"[yellow]⚠[/yellow] Delete cache if windowing params change: rm {cache_path}")

# %% Inspect metadata
metadata = single_windows.get_metadata()
logger.info(f"\n[bold]Metadata columns:[/bold] {len(list(metadata.columns))} columns")
logger.info(f"\n[bold]Sample metadata:[/bold]")
logger.info(metadata[["subject", "target", "rt_from_stimulus", "correct"]].head().to_string())

# RT distribution statistics
logger.info(f"\n[bold]Response Time Statistics:[/bold]")
logger.info(f"  Mean: {metadata['target'].mean():.4f}s")
logger.info(f"  Std: {metadata['target'].std():.4f}s")
logger.info(f"  Min: {metadata['target'].min():.4f}s")
logger.info(f"  Max: {metadata['target'].max():.4f}s")

# %% Split data at subject level
logger.info("\n" + "="*60)
logger.info("[bold cyan]SPLITTING DATA[/bold cyan]")
logger.info("="*60)

subjects = metadata["subject"].unique()
subjects = [s for s in subjects if s not in EXCLUDED_SUBJECTS]
logger.info(f"[bold]Total subjects (after exclusion):[/bold] {len(subjects)}")

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

logger.info(f"Train subjects: {len(train_subj)}")
logger.info(f"Val subjects: {len(valid_subj)}")
logger.info(f"Test subjects: {len(test_subj)}")

# Create splits
subject_split = single_windows.split("subject")
train_set = BaseConcatDataset([subject_split[s] for s in train_subj if s in subject_split])
val_set = BaseConcatDataset([subject_split[s] for s in valid_subj if s in subject_split])
test_set = BaseConcatDataset([subject_split[s] for s in test_subj if s in subject_split])

logger.info(f"\n[bold]Window counts:[/bold]")
logger.info(f"  Train: {len(train_set)}")
logger.info(f"  Val: {len(val_set)}")
logger.info(f"  Test: {len(test_set)}")

# %% Define LightningDataModule
class Challenge1DataModule(L.LightningDataModule):
    """DataModule for Challenge 1 (CCD task, RT prediction)"""

    def __init__(self, train_set, val_set, test_set, batch_size=128, num_workers=8):
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
        y = y.float().view(-1, 1)

        # Forward
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)

        # Log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch[0], batch[1]
        X = X.float()
        y = y.float().view(-1, 1)

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
        y = y.float().view(-1, 1)

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
logger.info("\n" + "="*60)
logger.info("[bold cyan]SETUP TRAINING[/bold cyan]")
logger.info("="*60)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
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
    name=f"challenge1_bs{BATCH_SIZE}_{timestamp}",
    tags=["challenge1", "supervised", "eegnex", "baseline"],
    save_dir=RUN_DIR,
    log_model="all",  # Upload all checkpoints as wandb artifacts
)

logger.info("[green]Callbacks and logger configured[/green]")

# %% Initialize model and datamodule
# Initialize with small batch size if scaler will run, otherwise use configured value
initial_batch_size = 32 if RUN_BATCH_SIZE_FINDER else BATCH_SIZE

datamodule = Challenge1DataModule(
    train_set=train_set,
    val_set=val_set,
    test_set=test_set,
    batch_size=initial_batch_size,
    num_workers=8,  # Parallel data loading (increase to 8 if you have 16+ cores)
)

model = Challenge1Module(
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS,
)

logger.info(f"\n[bold]Model:[/bold] {model.model.__class__.__name__}")
logger.info(f"[bold]Total parameters:[/bold] {sum(p.numel() for p in model.parameters()):,}")

# %% Setup Trainer
trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",  # Auto-detect best available hardware
    devices=1,           # Use single device
    callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
    logger=wandb_logger,
    gradient_clip_val=1.0,
    precision=PRECISION,
    deterministic=True,
    log_every_n_steps=10,
)

logger.info("\n[bold]Trainer configured:[/bold]")
logger.info(f"  Max epochs: {EPOCHS}")
logger.info(f"  Accelerator: auto-detected ({trainer.accelerator.__class__.__name__})")
logger.info(f"  Precision: {trainer.precision}")
logger.info(f"  Gradient clip: 1.0")

# %% Optional: Auto-tune hyperparameters with Lightning Tuner
# Separate controls for tuner components
RUN_LR_FINDER = True
RUN_BATCH_SIZE_FINDER = False  # Disabled, using manual batch_size=2048

if RUN_LR_FINDER or RUN_BATCH_SIZE_FINDER:
    from lightning.pytorch.tuner import Tuner

    logger.info("\n" + "="*60)
    logger.info("[bold cyan]TUNING HYPERPARAMETERS[/bold cyan]")
    logger.info("="*60)

    tuner = Tuner(trainer)

    # Batch size finder (RUN FIRST - LR depends on batch size)
    if RUN_BATCH_SIZE_FINDER:
        logger.info("\n[bold]Running batch size scaler...[/bold]")
        logger.info(f"[yellow]Original batch size:[/yellow] {BATCH_SIZE}")

        tuner.scale_batch_size(
            model,
            datamodule,
            mode="power",
            steps_per_trial=3,
            init_val=32,
            max_trials=6  # Caps at 1024 (32*2^5), won't try 4096+
        )

        new_batch_size = datamodule.batch_size
        logger.info(f"[green]Optimal batch size:[/green] {new_batch_size}")
        logger.info(f"[bold]Updated datamodule batch_size to:[/bold] {new_batch_size}")

    # Learning rate finder (RUN SECOND - uses finalized batch size)
    if RUN_LR_FINDER:
        logger.info("\n[bold]Running learning rate finder...[/bold]")
        lr_finder = tuner.lr_find(
            model,
            datamodule,
            min_lr=1e-6,
            max_lr=1e-1,
            num_training=1000,
            mode="exponential",
            attr_name="lr"
        )

        suggested_lr = lr_finder.suggestion()
        logger.info(f"[green]Suggested LR:[/green] {suggested_lr:.6f}")
        logger.info(f"[yellow]Original LR:[/yellow] {LR:.6f}")

        fig = lr_finder.plot(suggest=True)
        lr_plot_path = RUN_DIR / "lr_finder_plot.png"
        fig.savefig(lr_plot_path)
        logger.info(f"[green]LR finder plot saved to:[/green] {lr_plot_path}")

        model.hparams.lr = suggested_lr
        logger.info(f"[bold]Updated model LR to:[/bold] {suggested_lr:.6f}")

    logger.info("\n[bold green]Tuning complete![/bold green]")
    logger.info("="*60)

# %% Train model
logger.info("\n" + "="*60)
logger.info("[bold cyan]TRAINING[/bold cyan]")
logger.info("="*60)

trainer.fit(model, datamodule)

logger.info("\n[bold green]Training complete![/bold green]")
logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
logger.info(f"[bold]Best val NRMSE:[/bold] {checkpoint_callback.best_model_score:.6f}")

# %% Test model
logger.info("\n" + "="*60)
logger.info("[bold cyan]TESTING[/bold cyan]")
logger.info("="*60)

trainer.test(model, datamodule, ckpt_path="best")

# %% Save weights for submission
logger.info("\n" + "="*60)
logger.info("[bold cyan]SAVING FOR SUBMISSION[/bold cyan]")
logger.info("="*60)

# Load best checkpoint
best_model = Challenge1Module.load_from_checkpoint(checkpoint_callback.best_model_path)

# Save model state dict (compatible with submission.py)
weights_path = RUN_DIR / "weights_challenge_1.pt"
torch.save(best_model.model.state_dict(), weights_path)

logger.info(f"[green]Model weights saved to:[/green] {weights_path}")
logger.info(f"Use this file in submission.py's get_model_challenge_1() method")

# %% Summary
logger.info("\n" + "="*60)
logger.info("[bold cyan]SUMMARY[/bold cyan]")
logger.info("="*60)
logger.info(f"Experiment: Challenge 1 Baseline (EEGNeX)")
logger.info(f"Data: {len(RELEASES)} releases, {'mini' if USE_MINI else 'full'} dataset")
logger.info(f"Training windows: {len(train_set)}")
logger.info(f"Validation windows: {len(val_set)}")
logger.info(f"Test windows: {len(test_set)}")
logger.info(f"[bold]Best val NRMSE:[/bold] {checkpoint_callback.best_model_score:.6f}")
logger.info(f"Weights saved: {weights_path}")
logger.info(f"Checkpoints: {CHECKPOINT_DIR}")
logger.info("="*60)
