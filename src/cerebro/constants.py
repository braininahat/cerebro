import os
from pathlib import Path

# Repository root (two levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_data_root() -> Path:
    """Resolve the cache directory for EEG2025 datasets.

    Priority:
        1. `EEG2025_DATA_DIR` environment variable (expanded/resolved)
        2. `<repo>/data` relative to the project root
    """
    env_path = os.getenv("EEG2025_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (_REPO_ROOT / "data").resolve()


# Data paths
MINI_DATASET_ROOT = _resolve_data_root()

# EEG parameters
N_CHANS = 129
SFREQ = 100  # Sampling frequency in Hz
EPOCH_LEN_S = 2.0  # Epoch length in seconds
N_TIMES = int(EPOCH_LEN_S * SFREQ)  # Number of time points

# Windowing parameters
ANCHOR = "stimulus_anchor"
SHIFT_AFTER_STIM = 0.5  # Shift after stimulus in seconds
WINDOW_LEN = 2.0  # Window length in seconds

# Data split fractions
TRAIN_FRAC = 0.8
VALID_FRAC = 0.1
TEST_FRAC = 0.1
SEED = 2025

# Subjects to remove (known issues)
SUBJECTS_TO_REMOVE = [
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

# Training hyperparameters
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 8
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_N_EPOCHS = 100
DEFAULT_EARLY_STOPPING_PATIENCE = 50

# Device
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# W&B configuration
WANDB_PROJECT = "eeg2025"
WANDB_ENTITY = "ubcse-eeg2025"
