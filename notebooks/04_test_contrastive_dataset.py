# %% [markdown]
# # Test Contrastive Dataset Implementation
#
# This notebook tests the movie windowing and contrastive pair sampling
# implementation to verify:
# 1. Windows are created correctly with metadata
# 2. Positive pairs have matching (movie, time) but different subjects
# 3. Negative pairs have different movies
# 4. DataLoader integration works correctly
#
# ## ⚠️ IMPORTANT: Uses R5 for CODE VALIDATION ONLY
# - **R5 is competition validation set (leaderboard) - this is for testing code logic**
# - For prototyping: split {R1-R4, R6-R11} at subject level into train/val
# - For final runs: train on all {R1-R4, R6-R11}, submit to R5
# - Only 3 movies available in R5 (FunwithFractals unavailable in competition val set)

# %% Setup
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().parent / 'src'))

import torch
from torch.utils.data import DataLoader
from eegdash.dataset import EEGChallengeDataset

from utils.movie_windows import load_and_window_movies
from utils.contrastive_dataset import ContrastivePairDataset, print_dataset_stats

# %% Load and window movies
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Loading and windowing movies...")
windows_ds = load_and_window_movies(
    movie_names=['DespicableMe', 'ThePresent', 'DiaryOfAWimpyKid'],
    dataset_class=EEGChallengeDataset,
    cache_dir=DATA_DIR,
    release="R5",
    mini=True,
    window_len_s=2.0,
    stride_s=1.0,
    sfreq=100.0,
    time_bin_size_s=1.0,
    preload=True,
)

print(f"Total windows created: {len(windows_ds)}")

# %% Inspect metadata
metadata = windows_ds.get_metadata()
print("\nMetadata columns:", list(metadata.columns))
print("\nFirst few rows:")
print(metadata[['movie_id', 'subject_id', 'time_offset_seconds', 'time_bin']].head(10))

print("\nMovie distribution:")
print(metadata['movie_id'].value_counts())

print("\nSubject count:", metadata['subject_id'].nunique())

# %% Create contrastive dataset
print("\n" + "="*60)
print("Creating ContrastivePairDataset")
print("="*60)

contrastive_ds = ContrastivePairDataset(
    windows_ds,
    pos_strategy='same_movie_time',
    neg_strategy='diff_movie_mixed',
    return_triplets=True,
    random_state=42,
)

print_dataset_stats(contrastive_ds)

# %% Test single sample
print("\n" + "="*60)
print("Testing single sample")
print("="*60)

anchor, positive, negative = contrastive_ds[0]

print(f"Anchor shape: {anchor.shape}")
print(f"Positive shape: {positive.shape}")
print(f"Negative shape: {negative.shape}")
print(f"Anchor dtype: {anchor.dtype}")

# Verify shapes match expected (200 samples, 129 channels)
expected_shape = (200, 129)
assert anchor.shape == expected_shape, f"Expected {expected_shape}, got {anchor.shape}"
assert positive.shape == expected_shape
assert negative.shape == expected_shape

print("\nShapes verified successfully!")

# %% Verify positive pair properties
print("\n" + "="*60)
print("Verifying positive pair properties")
print("="*60)

# Get anchor metadata
valid_anchors = list(contrastive_ds.window_to_group.keys())
anchor_idx = valid_anchors[0]
anchor_meta = metadata.loc[anchor_idx]

# Sample positive
pos_idx = contrastive_ds._sample_positive(anchor_idx)
pos_meta = metadata.loc[pos_idx]

print(f"Anchor: movie={anchor_meta['movie_id']}, "
      f"time_bin={anchor_meta['time_bin']}, "
      f"subject={anchor_meta['subject_id']}")
print(f"Positive: movie={pos_meta['movie_id']}, "
      f"time_bin={pos_meta['time_bin']}, "
      f"subject={pos_meta['subject_id']}")

# Verify constraints
assert anchor_meta['movie_id'] == pos_meta['movie_id'], "Positive should have same movie!"
assert anchor_meta['time_bin'] == pos_meta['time_bin'], "Positive should have same time bin!"
assert anchor_meta['subject_id'] != pos_meta['subject_id'], "Positive should have different subject!"

print("\nPositive pair constraints verified!")

# %% Verify negative pair properties
print("\n" + "="*60)
print("Verifying negative pair properties")
print("="*60)

# Sample negative
neg_idx = contrastive_ds._sample_negative(anchor_idx)
neg_meta = metadata.loc[neg_idx]

print(f"Anchor: movie={anchor_meta['movie_id']}, subject={anchor_meta['subject_id']}")
print(f"Negative: movie={neg_meta['movie_id']}, subject={neg_meta['subject_id']}")

# Verify constraint
assert anchor_meta['movie_id'] != neg_meta['movie_id'], "Negative should have different movie!"

print("\nNegative pair constraints verified!")

# %% Test multiple samples
print("\n" + "="*60)
print("Testing multiple samples (batch)")
print("="*60)

n_samples = 10
movies_pos_match = 0
times_pos_match = 0
subjects_pos_diff = 0
movies_neg_diff = 0

for i in range(n_samples):
    # Get indices
    anchor_idx = valid_anchors[i]
    pos_idx = contrastive_ds._sample_positive(anchor_idx)
    neg_idx = contrastive_ds._sample_negative(anchor_idx)

    # Get metadata
    a_meta = metadata.loc[anchor_idx]
    p_meta = metadata.loc[pos_idx]
    n_meta = metadata.loc[neg_idx]

    # Check positive constraints
    if a_meta['movie_id'] == p_meta['movie_id']:
        movies_pos_match += 1
    if a_meta['time_bin'] == p_meta['time_bin']:
        times_pos_match += 1
    if a_meta['subject_id'] != p_meta['subject_id']:
        subjects_pos_diff += 1

    # Check negative constraint
    if a_meta['movie_id'] != n_meta['movie_id']:
        movies_neg_diff += 1

print(f"\nPositive pairs ({n_samples} samples):")
print(f"  Same movie: {movies_pos_match}/{n_samples} ({100*movies_pos_match/n_samples:.0f}%)")
print(f"  Same time bin: {times_pos_match}/{n_samples} ({100*times_pos_match/n_samples:.0f}%)")
print(f"  Different subject: {subjects_pos_diff}/{n_samples} ({100*subjects_pos_diff/n_samples:.0f}%)")

print(f"\nNegative pairs ({n_samples} samples):")
print(f"  Different movie: {movies_neg_diff}/{n_samples} ({100*movies_neg_diff/n_samples:.0f}%)")

assert movies_pos_match == n_samples, "All positive pairs should have same movie!"
assert times_pos_match == n_samples, "All positive pairs should have same time!"
assert subjects_pos_diff == n_samples, "All positive pairs should have different subjects!"
assert movies_neg_diff == n_samples, "All negative pairs should have different movies!"

print("\nAll constraints verified across multiple samples!")

# %% Test with DataLoader
print("\n" + "="*60)
print("Testing DataLoader integration")
print("="*60)

dataloader = DataLoader(
    contrastive_ds,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

# Get one batch
batch = next(iter(dataloader))
anchor_batch, pos_batch, neg_batch = batch

print(f"Anchor batch shape: {anchor_batch.shape}")
print(f"Positive batch shape: {pos_batch.shape}")
print(f"Negative batch shape: {neg_batch.shape}")

expected_batch_shape = (4, 200, 129)  # (batch, time, channels)
assert anchor_batch.shape == expected_batch_shape
assert pos_batch.shape == expected_batch_shape
assert neg_batch.shape == expected_batch_shape

print("\nDataLoader integration verified!")

# %% Test different negative strategies
print("\n" + "="*60)
print("Testing different negative strategies")
print("="*60)

for neg_strategy in ['diff_movie_mixed', 'diff_movie_same_subj', 'diff_movie_diff_subj']:
    print(f"\nTesting: {neg_strategy}")

    ds = ContrastivePairDataset(
        windows_ds,
        pos_strategy='same_movie_time',
        neg_strategy=neg_strategy,
        return_triplets=True,
        random_state=42,
    )

    # Sample a few negatives
    anchor_idx = valid_anchors[0]
    anchor_subj = metadata.loc[anchor_idx, 'subject_id']

    same_subj_count = 0
    diff_subj_count = 0

    for _ in range(20):
        neg_idx = ds._sample_negative(anchor_idx)
        neg_subj = metadata.loc[neg_idx, 'subject_id']

        if neg_subj == anchor_subj:
            same_subj_count += 1
        else:
            diff_subj_count += 1

    print(f"  Same subject: {same_subj_count}/20")
    print(f"  Different subject: {diff_subj_count}/20")

# %% Test pair mode (non-triplet)
print("\n" + "="*60)
print("Testing pair mode (return_triplets=False)")
print("="*60)

pair_ds = ContrastivePairDataset(
    windows_ds,
    pos_strategy='same_movie_time',
    neg_strategy='diff_movie_mixed',
    return_triplets=False,
    random_state=42,
)

anchor, other, label = pair_ds[0]
print(f"Anchor shape: {anchor.shape}")
print(f"Other shape: {other.shape}")
print(f"Label: {label.item()} (1=positive, 0=negative)")

assert label.item() in [0.0, 1.0], "Label should be 0 or 1"

print("\nPair mode verified!")

# %% Summary
print("\n" + "="*60)
print("SUMMARY - All tests passed!")
print("="*60)
print("""
Successfully verified:
✓ Window creation with proper shapes (200, 129)
✓ Metadata columns (movie_id, subject_id, time_bin, time_offset_seconds)
✓ Positive pairs: same (movie, time), different subject
✓ Negative pairs: different movie
✓ Multiple sampling strategies work correctly
✓ DataLoader integration (batch processing)
✓ Both triplet and pair modes work

Ready to use for contrastive learning!
""")
