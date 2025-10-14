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
# # Explore HBN-EEG Data Structure
#
# Deep dive into the Healthy Brain Network EEG dataset structure to understand:
# - BIDS organization
# - Event annotations for each task
# - Windowing strategies needed for challenges
# - Subject metadata and targets
#
# **Timeline**: Day 1
#
# **Goals**:
# - Understand event structure for each task
# - Map events to time windows for challenges
# - Understand p_factor distribution and missingness
# - Plan dataset implementations

import json

# %%
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from eegdash import EEGChallengeDataset

# Suppress EEGChallengeDataset warning
warnings.filterwarnings(
    "ignore", category=UserWarning, module="eegdash.dataset.dataset"
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %% [markdown]
# ## 1. Load R5 Dataset (Competition Test Set)

# %%
# Load R5 contrastChangeDetection (Challenge 1 target task)
r5_ccd = EEGChallengeDataset(
    release="R5",
    task="contrastChangeDetection",
    cache_dir=str(DATA_DIR),
    mini=False,
    description_fields=["p_factor", "age", "sex"],
)

print(f"R5 CCD task: {len(r5_ccd.datasets)} recordings")

# Load first subject for exploration
sample_recording = r5_ccd.datasets[0]
print(f"\nSample subject: {sample_recording.description['subject']}")
print(f"Sampling rate: {sample_recording.raw.info['sfreq']} Hz")
print(f"Channels: {len(sample_recording.raw.ch_names)}")
print(f"Duration: {sample_recording.raw.times[-1]:.1f} s")

# %% [markdown]
# ## 2. Understand BIDS Structure
#
# HBN-EEG uses BIDS format with:
# - `participants.tsv`: Subject metadata
# - `task-*_eeg.json`: Task-level metadata (sampling rate, reference, etc.)
# - `task-*_events.json`: Event code descriptions
# - `sub-*/eeg/*.bdf`: Raw EEG data
# - `sub-*/eeg/*_events.tsv`: Event markers per recording

# %%
r5_path = DATA_DIR / "ds005510-bdf"  # R5

# Load and examine metadata files
with open(r5_path / "dataset_description.json") as f:
    dataset_desc = json.load(f)

print("Dataset Description:")
for key, value in dataset_desc.items():
    if key not in ["GeneratedBy", "SourceDatesets"]:
        print(f"  {key}: {value}")

# %% [markdown]
# ## 3. Contrast Change Detection Task Events
#
# **Critical for Challenge 1!**
#
# Event types:
# - `right_target` / `left_target`: Stimulus presentation
# - `right_buttonPress` / `left_buttonPress`: Subject response
# - `feedback`: Smiley (correct) or sad face (incorrect)
#
# **Our goal**: Predict response time (RT) from EEG window 0.5-2.5s after stimulus

# %%
# Load event descriptions
with open(r5_path / "task-contrastChangeDetection_events.json") as f:
    ccd_events = json.load(f)

print("CCD Event Types:")
for event_type in ccd_events["value"]["Levels"].keys():
    desc = ccd_events["value"]["Levels"][event_type]
    print(f"  {event_type}: {desc}")

# %% [markdown]
# ## 4. Examine Events from Sample Recording

# %%
# Get events from MNE Raw object
events, event_id = mne.events_from_annotations(sample_recording.raw)

print(f"\nTotal events: {len(events)}")
print(f"\nEvent types found:")
for name, code in event_id.items():
    count = np.sum(events[:, 2] == code)
    print(f"  {name} (code {code}): {count} occurrences")

# Show first 10 events
print(f"\nFirst 10 events:")
print("Sample | Time (s) | Code | Type")
print("-" * 50)
for i in range(min(10, len(events))):
    sample_idx = events[i, 0]
    time_s = sample_idx / sample_recording.raw.info["sfreq"]
    code = events[i, 2]
    # Find event name
    event_name = [name for name, c in event_id.items() if c == code][0]
    print(f"{sample_idx:6d} | {time_s:7.2f} | {code:4d} | {event_name}")

# %% [markdown]
# ## 5. Visualize Event Timeline
#
# Understand the temporal structure of trials

# %%
# Extract stimulus and response events
stim_events = events[
    [name for name in event_id.keys() if "target" in name and "buttonPress" not in name]
]
response_events = events[[name for name in event_id.keys() if "buttonPress" in name]]

# Create timeline visualization
fig, ax = plt.subplots(figsize=(14, 4))

# Plot stimulus onsets
stim_times = (
    events[
        [
            any(
                name in event_id and event_id[name] == code
                for name in ["right_target", "left_target"]
            )
            for code in events[:, 2]
        ],
        0,
    ]
    / sample_recording.raw.info["sfreq"]
)

if len(stim_times) > 0:
    ax.scatter(
        stim_times,
        [1] * len(stim_times),
        c="blue",
        s=100,
        marker="|",
        label="Stimulus",
        alpha=0.7,
    )

# Plot response onsets
resp_times = (
    events[
        [
            any(
                name in event_id and event_id[name] == code
                for name in ["right_buttonPress", "left_buttonPress"]
            )
            for code in events[:, 2]
        ],
        0,
    ]
    / sample_recording.raw.info["sfreq"]
)

if len(resp_times) > 0:
    ax.scatter(
        resp_times,
        [1] * len(resp_times),
        c="red",
        s=100,
        marker="|",
        label="Response",
        alpha=0.7,
    )

ax.set_xlabel("Time (s)")
ax.set_yticks([])
ax.set_title(f'Event Timeline - {sample_recording.description["subject"]}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Calculate Response Times
#
# **This is what Challenge 1 predicts!**
#
# RT = time(buttonPress) - time(stimulus)

# %%
# Find stimulus-response pairs
stim_codes = [event_id.get("right_target", -1), event_id.get("left_target", -1)]
resp_codes = [
    event_id.get("right_buttonPress", -1),
    event_id.get("left_buttonPress", -1),
]

# Extract stimulus times
stim_events_filtered = events[np.isin(events[:, 2], stim_codes)]
stim_times = stim_events_filtered[:, 0] / sample_recording.raw.info["sfreq"]

# Extract response times
resp_events_filtered = events[np.isin(events[:, 2], resp_codes)]
resp_times = resp_events_filtered[:, 0] / sample_recording.raw.info["sfreq"]

# Compute RT for each stimulus (find next response after each stimulus)
rts = []
for stim_time in stim_times:
    # Find next response after this stimulus
    next_responses = resp_times[resp_times > stim_time]
    if len(next_responses) > 0:
        rt = next_responses[0] - stim_time
        rts.append(rt)

rts = np.array(rts)

print(f"\nResponse Times (RT):")
print(f"  N trials with RT: {len(rts)}")
print(f"  Mean RT: {rts.mean():.3f} s")
print(f"  Std RT: {rts.std():.3f} s")
print(f"  Min RT: {rts.min():.3f} s")
print(f"  Max RT: {rts.max():.3f} s")

# Plot RT distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(rts, bins=30, edgecolor="black", alpha=0.7)
ax.set_xlabel("Response Time (seconds)")
ax.set_ylabel("Count")
ax.set_title(f'RT Distribution - {sample_recording.description["subject"]}')
ax.axvline(rts.mean(), color="red", linestyle="--", label=f"Mean: {rts.mean():.3f}s")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Understand Windowing for Challenge 1
#
# From startkit, the windowing strategy is:
# 1. Find stimulus onset (right_target or left_target)
# 2. Extract window from +0.5s to +2.5s after stimulus (2 seconds total)
# 3. Label: rt_from_stimulus (the RT we calculated above)
#
# **Window**: `[stim_time + 0.5s, stim_time + 2.5s]` → shape (129, 200) @ 100Hz

# %%
SFREQ = 100
SHIFT_AFTER_STIM = 0.5  # seconds
WINDOW_LEN = 2.0  # seconds
WINDOW_SAMPLES = int(WINDOW_LEN * SFREQ)  # 200 samples

print(f"Challenge 1 Windowing:")
print(f"  Sampling rate: {SFREQ} Hz")
print(f"  Shift after stimulus: {SHIFT_AFTER_STIM} s")
print(f"  Window length: {WINDOW_LEN} s ({WINDOW_SAMPLES} samples)")
print(
    f"  Window: [stim + {SHIFT_AFTER_STIM}s, stim + {SHIFT_AFTER_STIM + WINDOW_LEN}s]"
)

# Example: Extract window for first stimulus
if len(stim_times) > 0:
    first_stim_time = stim_times[0]
    window_start = first_stim_time + SHIFT_AFTER_STIM
    window_end = window_start + WINDOW_LEN

    print(f"\nExample first trial:")
    print(f"  Stimulus time: {first_stim_time:.3f} s")
    print(f"  Window: [{window_start:.3f}, {window_end:.3f}] s")
    print(f"  RT for this trial: {rts[0]:.3f} s")

    # Extract EEG data for this window
    window_start_sample = int(window_start * SFREQ)
    window_end_sample = int(window_end * SFREQ)

    data, times = sample_recording.raw[:, window_start_sample:window_end_sample]
    print(f"  EEG data shape: {data.shape}")  # Should be (129, 200)

# %% [markdown]
# ## 8. Participants.tsv - Challenge 2 Target
#
# **p_factor** (externalizing psychopathology factor) is Challenge 2's target

# %%
participants_df = pd.read_csv(r5_path / "participants.tsv", sep="\t")

print(f"Participants: {len(participants_df)}")
print(f"\nColumns: {list(participants_df.columns)}")

# Analyze p_factor
print(f"\np_factor Statistics:")
print(participants_df["p_factor"].describe())

# Count missing values
n_missing = participants_df["p_factor"].isna().sum()
n_valid = len(participants_df) - n_missing
print(
    f"\nValid p_factor: {n_valid} / {len(participants_df)} ({100*n_valid/len(participants_df):.1f}%)"
)

# Visualize distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(
    participants_df["p_factor"].dropna(), bins=30, edgecolor="black", alpha=0.7
)
axes[0].set_xlabel("p_factor")
axes[0].set_ylabel("Count")
axes[0].set_title("P-Factor Distribution")
axes[0].axvline(
    participants_df["p_factor"].mean(),
    color="red",
    linestyle="--",
    label=f'Mean: {participants_df["p_factor"].mean():.3f}',
)
axes[0].legend()

# Box plot with age groups
participants_df["age_group"] = pd.cut(
    participants_df["age"], bins=[0, 10, 15, 25], labels=["<10", "10-15", "15+"]
)
participants_df.boxplot(column="p_factor", by="age_group", ax=axes[1])
axes[1].set_xlabel("Age Group")
axes[1].set_ylabel("p_factor")
axes[1].set_title("P-Factor by Age")
plt.suptitle("")  # Remove automatic title

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Task Availability Across Subjects
#
# Challenge 2 can use data from ANY task. Let's see which tasks are most available.

# %%
task_cols = [
    "RestingState",
    "DespicableMe",
    "ThePresent",
    "FunwithFractals",
    "DiaryOfAWimpyKid",
    "contrastChangeDetection",
    "surroundSupp",
    "symbolSearch",
    "seqLearning6target",
    "seqLearning8target",
]

# Count availability
task_availability = {}
for task in task_cols:
    if task in participants_df.columns:
        available = (participants_df[task] == "available").sum()
        task_availability[task] = available

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
tasks_sorted = sorted(task_availability.items(), key=lambda x: x[1], reverse=True)
task_names = [t[0] for t in tasks_sorted]
task_counts = [t[1] for t in tasks_sorted]

ax.barh(task_names, task_counts, color="steelblue", alpha=0.7)
ax.set_xlabel("Number of Subjects")
ax.set_title(f"Task Availability (N={len(participants_df)} subjects)")
ax.grid(True, axis="x", alpha=0.3)

for i, count in enumerate(task_counts):
    ax.text(count + 2, i, str(count), va="center")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Movie Tasks for Contrastive Learning
#
# Load movie tasks to understand data for Phase 2 (contrastive pretraining)

# %%
movie_tasks = ["DespicableMe", "ThePresent", "FunwithFractals", "DiaryOfAWimpyKid"]

movie_info = []
for movie_task in movie_tasks:
    try:
        movie_ds = EEGChallengeDataset(
            release="R5", task=movie_task, cache_dir=str(DATA_DIR), mini=False
        )
        n_subjects = len(movie_ds.datasets)

        # Get duration from first subject
        if n_subjects > 0:
            duration = movie_ds.datasets[0].raw.times[-1]
        else:
            duration = 0

        movie_info.append(
            {"task": movie_task, "n_subjects": n_subjects, "duration_s": duration}
        )
        print(f"✓ {movie_task}: {n_subjects} subjects, {duration:.1f}s")
    except Exception as e:
        print(f"✗ {movie_task}: Error - {e}")

movie_df = pd.DataFrame(movie_info)

# %% [markdown]
# ## 11. Movie Events
#
# Movies only have simple start/stop events. We'll need to create sliding windows.

# %%
if len(movie_df) > 0:
    # Load sample movie recording
    movie_ds = EEGChallengeDataset(
        release="R5", task="DespicableMe", cache_dir=str(DATA_DIR), mini=False
    )

    sample_movie = movie_ds.datasets[0]

    # Get events
    events, event_id = mne.events_from_annotations(sample_movie.raw)

    print(f"\nDespicableMe - Sample Subject:")
    print(f"  Duration: {sample_movie.raw.times[-1]:.1f} s")
    print(f"  Events: {len(events)}")
    print(f"  Event types: {list(event_id.keys())}")

    print(f"\nFor contrastive learning:")
    print(f"  - Window size: 2s (200 samples)")
    print(f"  - Stride: 1s (overlap)")
    print(f"  - Expected windows: ~{int(sample_movie.raw.times[-1] - 2)}")

# %% [markdown]
# ## 12. Challenge 2 Windowing Strategy
#
# From startkit:
# 1. Create 4s windows with 2s stride from any task
# 2. During training: random 2s crops from 4s windows
# 3. Label: p_factor (subject-level, constant across windows)

# %%
WINDOW_SIZE_C2 = 4.0  # seconds
STRIDE_C2 = 2.0  # seconds
CROP_SIZE_C2 = 2.0  # seconds

print(f"Challenge 2 Windowing:")
print(f"  Window size: {WINDOW_SIZE_C2} s ({int(WINDOW_SIZE_C2 * SFREQ)} samples)")
print(f"  Stride: {STRIDE_C2} s ({int(STRIDE_C2 * SFREQ)} samples)")
print(f"  Crop size: {CROP_SIZE_C2} s ({int(CROP_SIZE_C2 * SFREQ)} samples)")
print(f"\nFor a {sample_recording.raw.times[-1]:.1f}s recording:")
n_windows = int((sample_recording.raw.times[-1] - WINDOW_SIZE_C2) / STRIDE_C2) + 1
print(f"  Number of 4s windows: ~{n_windows}")

# %% [markdown]
# ## Summary
#
# ### Challenge 1 (RT Prediction):
# - **Task**: contrastChangeDetection
# - **Events**: right_target, left_target (stimuli)
# - **Window**: [stim + 0.5s, stim + 2.5s] → (129, 200)
# - **Target**: rt_from_stimulus (response time in seconds)
# - **Loss**: MSE
# - **Metric**: NRMSE
#
# ### Challenge 2 (P-Factor Prediction):
# - **Tasks**: Any (CCD, resting, movies, etc.)
# - **Window**: 4s windows, 2s stride → random 2s crops → (129, 200)
# - **Target**: p_factor (subject-level trait)
# - **Loss**: MAE/L1
# - **Metric**: NRMSE
#
# ### Movie Contrastive Pretraining:
# - **Tasks**: DespicableMe, ThePresent, FunwithFractals, DiaryOfAWimpyKid
# - **Window**: 2s sliding windows, 1s stride
# - **Positive pairs**: Same movie, same timestamp, different subjects
# - **Negative pairs**: Different movies
# - **Loss**: InfoNCE
#
# **Next**: Implement these windowing strategies in Dataset classes!

# %%
print("\n" + "=" * 60)
print("EXPLORATION COMPLETE")
print("=" * 60)
print("\nKey insights:")
print("  - Challenge 1: Stimulus-locked windows (0.5-2.5s after target)")
print("  - Challenge 2: Fixed windows + random crops from any task")
print("  - Movies: Sliding windows for contrastive pairs")
print("  - p_factor: Subject-level target, some missing values")
print("\nReady to implement Dataset classes!")
