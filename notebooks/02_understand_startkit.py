# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Understand Startkit Code
#
# Walk through the competition startkit code to understand:
# - Exact preprocessing pipeline for Challenge 1
# - Exact windowing strategy for Challenge 2
# - Data loader patterns
# - Evaluation metrics
#
# **Timeline**: Day 1-2
#
# **Goals**:
# - Replicate startkit preprocessing exactly
# - Understand annotate_trials_with_target
# - Understand DatasetWrapper for Challenge 2
# - Understand local scoring mechanism

# %%
import sys
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

# Add startkit to path
startkit_path = Path("..") / "startkit"
sys.path.insert(0, str(startkit_path))

from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    create_windows_from_events,
    preprocess,
)

# Import from startkit
from eegdash import EEGChallengeDataset

# Startkit imports (from challenge_1.py and challenge_2.py)
from eegdash.hbn.windows import (
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
    keep_only_recordings_with,
)

# Suppress EEGChallengeDataset warning
warnings.filterwarnings("ignore", category=UserWarning, module="eegdash.dataset.dataset")

# %%
DATA_DIR = Path("..") / "data" / "full"
SFREQ = 100
BATCH_SIZE = 1
EPOCH_LEN_S = 2.0

# %% [markdown]
# ## Challenge 1: Step-by-Step Walkthrough

# %% [markdown]
# ### Step 1: Load Dataset
#
# Load R5 contrastChangeDetection task

# %%
print("=" * 60)
print("CHALLENGE 1: Response Time Prediction")
print("=" * 60)

dataset_1 = EEGChallengeDataset(
    release="R5",
    mini=True,  # Start with mini for speed
    query=dict(task="contrastChangeDetection"),
    cache_dir=str(DATA_DIR),
)

print(f"\nStep 1: Loaded {len(dataset_1.datasets)} recordings")

# Inspect first recording
rec0 = dataset_1.datasets[0]
print(f"\nFirst recording:")
print(f"  Subject: {rec0.description['subject']}")
print(f"  Duration: {rec0.raw.times[-1]:.1f} s")
print(f"  Sfreq: {rec0.raw.info['sfreq']} Hz")
print(f"  Channels: {len(rec0.raw.ch_names)}")

# %% [markdown]
# ### Step 2: Annotate Trials with Target
#
# **This is the KEY preprocessing step for Challenge 1!**
#
# `annotate_trials_with_target` does:
# 1. Finds stimulus events (right_target, left_target)
# 2. Finds corresponding response events (right_buttonPress, left_buttonPress)
# 3. Calculates RT = response_time - stimulus_time
# 4. Stores as 'rt_from_stimulus' in extras
#
# Parameters:
# - `target_field`: What to extract (rt_from_stimulus, rt_from_trialstart, etc.)
# - `epoch_length`: Length of trial in seconds
# - `require_stimulus`: Only keep trials with stimulus
# - `require_response`: Only keep trials with response

# %%
print("\nStep 2: Annotate trials with RT")

preprocessors = [
    Preprocessor(
        annotate_trials_with_target,
        apply_on_array=False,  # Operates on Raw object, not numpy array
        target_field="rt_from_stimulus",
        epoch_length=EPOCH_LEN_S,
        require_stimulus=True,
        require_response=True,
    ),
    Preprocessor(add_aux_anchors, apply_on_array=False),
]

# Apply preprocessors
preprocess(dataset_1, preprocessors, n_jobs=1)  # n_jobs=1 for debugging

print("✓ Preprocessing complete")

# Examine what was added
rec0_after = dataset_1.datasets[0]
print(f"\nAfter preprocessing, extras contains:")
for key in rec0_after.raw.annotations.description[:5]:
    print(f"  - {key}")

# %% [markdown]
# ### Step 3: Keep Only Valid Recordings
#
# Filter to recordings that have 'stimulus_anchor' annotations

# %%
print("\nStep 3: Filter to valid recordings")

dataset_2 = keep_only_recordings_with("stimulus_anchor", dataset_1)

print(f"Before filter: {len(dataset_1.datasets)} recordings")
print(f"After filter: {len(dataset_2.datasets)} recordings")

# %% [markdown]
# ### Step 4: Create Windows from Events
#
# **This creates the actual training windows!**
#
# Window specification:
# - Event: 'stimulus_anchor' (marks stimulus onset)
# - Start: stimulus + 0.5s (50 samples @ 100Hz)
# - Stop: stimulus + 2.5s (250 samples @ 100Hz)
# - Window size: 2.0s (200 samples)
# - Stride: 1.0s (100 samples) - allows multiple windows per trial
#
# Result: Each window is 2s of EEG starting 0.5s after stimulus

# %%
print("\nStep 4: Create windows from stimulus events")

SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0

dataset_3 = create_windows_from_events(
    dataset_2,
    mapping={"stimulus_anchor": 0},  # Map stimulus_anchor to class 0
    trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),  # +0.5s
    trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),  # +2.5s
    window_size_samples=int(EPOCH_LEN_S * SFREQ),  # 2.0s = 200 samples
    window_stride_samples=SFREQ,  # 1.0s = 100 samples
    preload=True,
)

print(f"Created {len(dataset_3.datasets)} window datasets")

# Inspect first window dataset
win0 = dataset_3.datasets[0]
print(f"\nFirst window dataset:")
print(f"  Number of windows: {len(win0)}")
print(f"  Window shape: {win0[0][0].shape}")  # Should be (129, 200)

# %% [markdown]
# ### Step 5: Add Target Labels to Windows
#
# Bring the 'rt_from_stimulus' from the original recording into window metadata

# %%
print("\nStep 5: Add RT labels to windows")

dataset_4 = add_extras_columns(
    dataset_3,
    dataset_2,
    desc="stimulus_anchor",
    keys=(
        "target",
        "rt_from_stimulus",  # This is our label!
        "rt_from_trialstart",
        "stimulus_onset",
        "response_onset",
        "correct",
        "response_type",
    ),
)

print(f"✓ Added target labels")

# Access a single window with its label
window_ds = dataset_4.datasets[0]
X, y, window_inds = window_ds[0]  # Get first window

print(f"\nFirst window:")
print(f"  X shape: {X.shape}")  # (129, 200)
print(f"  y (class): {y}")  # Class label (not used for regression)
print(f"  window_inds: {window_inds}")  # (window_idx, start_sample, stop_sample)

# Get RT from description
rt = window_ds.description["rt_from_stimulus"]
print(f"  RT (label): {rt:.3f} s")

# %% [markdown]
# ### Step 6: Create DataLoader
#
# Standard PyTorch DataLoader with sequential sampling

# %%
print("\nStep 6: Create DataLoader")

dataloader_1 = DataLoader(
    dataset_4,
    batch_size=BATCH_SIZE,
    sampler=SequentialSampler(dataset_4),
    shuffle=False,
    drop_last=False,
)

print(f"DataLoader created with {len(dataloader_1)} batches")

# Iterate through first few batches
print(f"\nFirst 3 batches:")
for i, batch in enumerate(dataloader_1):
    if i >= 3:
        break

    X, y, infos = batch
    print(f"  Batch {i}: X.shape={X.shape}, y.shape={y.shape}")
    # Note: y here is class label, not RT
    # RT must be extracted from window_ds.description["rt_from_stimulus"]

# %% [markdown]
# ## Challenge 1 Complete Pipeline Summary
#
# ```python
# # 1. Load dataset
# dataset = EEGChallengeDataset(release="R5", task="contrastChangeDetection")
#
# # 2. Annotate with RT
# preprocess(dataset, [
#     Preprocessor(annotate_trials_with_target, target_field="rt_from_stimulus"),
#     Preprocessor(add_aux_anchors)
# ])
#
# # 3. Filter valid recordings
# dataset = keep_only_recordings_with("stimulus_anchor", dataset)
#
# # 4. Create windows (0.5-2.5s after stimulus)
# windows = create_windows_from_events(
#     dataset,
#     mapping={"stimulus_anchor": 0},
#     trial_start_offset_samples=50,   # +0.5s
#     trial_stop_offset_samples=250,   # +2.5s
#     window_size_samples=200,         # 2.0s
#     window_stride_samples=100,       # 1.0s
# )
#
# # 5. Add RT labels
# windows = add_extras_columns(windows, dataset, desc="stimulus_anchor",
#                              keys=("rt_from_stimulus",))
#
# # 6. DataLoader
# loader = DataLoader(windows, batch_size=128)
# ```

# %% [markdown]
# ## Challenge 2: Step-by-Step Walkthrough

# %% [markdown]
# ### Step 1: Load Dataset with p_factor
#
# Load any task, but must request 'externalizing' field

# %%
print("\n" + "=" * 60)
print("CHALLENGE 2: P-Factor Prediction")
print("=" * 60)

import math

dataset_c2 = EEGChallengeDataset(
    release="R5",
    mini=True,
    query=dict(task="contrastChangeDetection"),  # Can use any task
    description_fields=["externalizing"],  # Request p_factor field
    cache_dir=str(DATA_DIR),
)

print(f"\nStep 1: Loaded {len(dataset_c2.datasets)} recordings")

# Check p_factor availability
rec0_c2 = dataset_c2.datasets[0]
p_factor = rec0_c2.description.get("externalizing", float("nan"))
print(f"\nFirst recording:")
print(f"  Subject: {rec0_c2.description['subject']}")
print(f"  p_factor: {p_factor}")

# %% [markdown]
# ### Step 2: Filter Valid Recordings
#
# Keep only recordings with:
# - At least 4 seconds of data (for 4s windows)
# - Valid p_factor (not NaN)

# %%
print("\nStep 2: Filter valid recordings")

dataset_c2_filtered = BaseConcatDataset(
    [
        ds
        for ds in dataset_c2.datasets
        if ds.raw.n_times >= 4 * SFREQ  # At least 4s
        and not math.isnan(ds.description.get("externalizing", float("nan")))
    ]
)

print(f"Before filter: {len(dataset_c2.datasets)} recordings")
print(f"After filter: {len(dataset_c2_filtered.datasets)} recordings")

# %% [markdown]
# ### Step 3: Create Fixed Windows
#
# Create 4s windows with 2s stride (50% overlap)

# %%
print("\nStep 3: Create 4s windows with 2s stride")

dataset_c2_windows = create_fixed_length_windows(
    dataset_c2_filtered,
    window_size_samples=4 * SFREQ,  # 4s = 400 samples
    window_stride_samples=2 * SFREQ,  # 2s = 200 samples
    drop_last_window=True,  # Drop incomplete windows
)

print(f"Created {len(dataset_c2_windows.datasets)} window datasets")

# %% [markdown]
# ### Step 4: Wrap with DatasetWrapper
#
# **This is the magic for Challenge 2!**
#
# DatasetWrapper does:
# 1. Takes 4s windows
# 2. During training: randomly crops 2s from the 4s window
# 3. Returns (X: 2s crop, y: p_factor)
#
# This provides data augmentation and ensures consistent 2s windows

# %%
print("\nStep 4: Wrap with DatasetWrapper for random cropping")

import random

# Define DatasetWrapper (from startkit/local_scoring.py)
from braindecode.datasets.base import BaseDataset


class DatasetWrapper(BaseDataset):
    def __init__(
        self, dataset, crop_size_samples: int, target_name="externalizing", seed=None
    ):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name = target_name
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]

        # Get p_factor from description
        target = self.dataset.description[self.target_name]
        target = float(target)

        # Randomly crop to 2s
        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples

        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        i_start = i_start + start_offset
        i_stop = i_start + self.crop_size_samples

        X = X[:, start_offset : start_offset + self.crop_size_samples]

        return X, target, (i_window_in_trial, i_start, i_stop), {}


# Wrap each window dataset
dataset_c2_wrapped = BaseConcatDataset(
    [
        DatasetWrapper(ds, crop_size_samples=2 * SFREQ, seed=42)
        for ds in dataset_c2_windows.datasets
    ]
)

print(f"Wrapped {len(dataset_c2_wrapped.datasets)} datasets")

# Test a single sample
wrap0 = dataset_c2_wrapped.datasets[0]
X, y, crop_inds, _ = wrap0[0]

print(f"\nFirst wrapped sample:")
print(f"  X shape: {X.shape}")  # Should be (129, 200) - 2s crop
print(f"  y (p_factor): {y:.3f}")

# %% [markdown]
# ## Challenge 2 Complete Pipeline Summary
#
# ```python
# # 1. Load dataset with p_factor
# dataset = EEGChallengeDataset(
#     release="R5",
#     task="contrastChangeDetection",  # Or any task
#     description_fields=["externalizing"]
# )
#
# # 2. Filter valid recordings
# dataset = BaseConcatDataset([
#     ds for ds in dataset.datasets
#     if ds.raw.n_times >= 4*SFREQ and not math.isnan(ds.description["externalizing"])
# ])
#
# # 3. Create 4s windows with 2s stride
# windows = create_fixed_length_windows(
#     dataset,
#     window_size_samples=4 * SFREQ,
#     window_stride_samples=2 * SFREQ,
#     drop_last_window=True
# )
#
# # 4. Wrap for random 2s crops
# wrapped = BaseConcatDataset([
#     DatasetWrapper(ds, crop_size_samples=2 * SFREQ)
#     for ds in windows.datasets
# ])
#
# # 5. DataLoader
# loader = DataLoader(wrapped, batch_size=128)
# ```

# %% [markdown]
# ## Local Scoring
#
# Understand how startkit/local_scoring.py evaluates submissions

# %%
print("\n" + "=" * 60)
print("LOCAL SCORING")
print("=" * 60)

from sklearn.metrics import root_mean_squared_error as rmse


def nrmse(y_trues, y_preds):
    """Normalized RMSE using standard deviation"""
    return rmse(y_trues, y_preds) / y_trues.std()


# Simulate predictions
np.random.seed(42)
y_true_c1 = np.random.randn(100) * 0.1 + 0.5  # RT around 0.5s
y_pred_c1 = y_true_c1 + np.random.randn(100) * 0.05  # Add noise

y_true_c2 = np.random.randn(100)  # p_factor around 0
y_pred_c2 = y_true_c2 + np.random.randn(100) * 0.3

# Calculate scores
nrmse_c1 = nrmse(y_true_c1, y_pred_c1)
nrmse_c2 = nrmse(y_true_c2, y_pred_c2)

overall = 0.3 * nrmse_c1 + 0.7 * nrmse_c2

print(f"\nExample Scoring:")
print(f"  Challenge 1 NRMSE: {nrmse_c1:.4f}")
print(f"  Challenge 2 NRMSE: {nrmse_c2:.4f}")
print(f"  Overall Score: {overall:.4f}")
print(f"\n  Formula: 0.3 × NRMSE_C1 + 0.7 × NRMSE_C2")

# %% [markdown]
# ## Summary: Key Takeaways
#
# ### Challenge 1:
# 1. **Preprocessing**: `annotate_trials_with_target` to calculate RT
# 2. **Windowing**: `create_windows_from_events` with +0.5s offset, 2s duration
# 3. **Labels**: Extract `rt_from_stimulus` from window description
# 4. **Loss**: MSE
# 5. **Metric**: NRMSE (30% of overall score)
#
# ### Challenge 2:
# 1. **Preprocessing**: Filter for valid p_factor (not NaN)
# 2. **Windowing**: Fixed 4s windows, 2s stride
# 3. **Data augmentation**: Random 2s crops via DatasetWrapper
# 4. **Labels**: `externalizing` field from description (subject-level)
# 5. **Loss**: MAE/L1
# 6. **Metric**: NRMSE (70% of overall score)
#
# ### Implementation Next Steps:
# 1. Create `src/data/challenge1.py` replicating this pipeline
# 2. Create `src/data/challenge2.py` replicating this pipeline
# 3. Create `src/data/movies.py` for contrastive learning
# 4. Integrate with Hydra configs
# 5. Test with EEGNeX baseline

# %%
print("\n" + "=" * 60)
print("STARTKIT WALKTHROUGH COMPLETE")
print("=" * 60)
print("\nReady to implement Dataset classes in src/data/!")
