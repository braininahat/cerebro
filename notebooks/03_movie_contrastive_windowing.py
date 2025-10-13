# %% [markdown]
# # Movie Task Windowing for Contrastive Learning
#
# This notebook explores how to create windows from movie task EEG data
# and prepare them for contrastive learning.
#
# ## Key differences from CCD task:
# - Movies have NO trial structure (only video_start/stop markers)
# - Must use `create_fixed_length_windows()` instead of `create_windows_from_events()`
# - Need to add custom metadata: movie_id, time_offset, subject_id
#
# ## ⚠️ IMPORTANT: This notebook uses R5 (competition validation set) for EXPLORATION ONLY
# - **R5 is the competition validation set (leaderboard) - NEVER train on it!**
# - During prototyping: split {R1-R4, R6-R11} into train/val at subject level
# - During final runs: train on ALL {R1-R4, R6-R11}, submit to R5
# - This notebook demonstrates with R5/mini for code validation only

# %% Setup and imports
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from braindecode.preprocessing import preprocess, Preprocessor, create_fixed_length_windows
from braindecode.datasets import BaseConcatDataset
from eegdash.dataset import EEGChallengeDataset

# Suppress EEGChallengeDataset warning
warnings.filterwarnings("ignore", category=UserWarning, module="eegdash.dataset.dataset")

# Data directories
DATA_ROOT = Path("/home/varun/repos/cerebro/data")
MINI_DIR = DATA_ROOT / "mini"
FULL_DIR = DATA_ROOT / "full"

# Toggle between mini/full dataset
USE_MINI = True  # Set to False for full dataset
DATA_DIR = MINI_DIR if USE_MINI else FULL_DIR

# Ensure directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% Load movie task data
# Start with one movie to understand structure
print("Loading DespicableMe movie task...")
dataset_movie = EEGChallengeDataset(
    task="DespicableMe",
    release="R5",
    cache_dir=DATA_DIR,
    mini=USE_MINI
)

print(f"Number of recordings: {len(dataset_movie.datasets)}")
print(f"First recording: {dataset_movie.datasets[0].description}")

# %% Inspect raw data structure
raw = dataset_movie.datasets[0].raw
print(f"Channels: {len(raw.ch_names)}")
print(f"Duration: {raw.times[-1]:.2f} seconds")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"\nAnnotations:")
for ann in raw.annotations:
    print(f"  {ann['onset']:.2f}s - {ann['description']}")

# %% Create fixed-length windows
# Parameters matching CCD task setup
WINDOW_LEN_S = 2.0
STRIDE_S = 1.0
SFREQ = 100  # Hz

print(f"\nCreating windows:")
print(f"  Window length: {WINDOW_LEN_S}s")
print(f"  Stride: {STRIDE_S}s")

# Create windows from continuous data (no event anchoring needed)
windows = create_fixed_length_windows(
    dataset_movie,
    start_offset_samples=0,  # Start from beginning
    stop_offset_samples=None,  # Go to end
    window_size_samples=int(WINDOW_LEN_S * SFREQ),
    window_stride_samples=int(STRIDE_S * SFREQ),
    drop_last_window=True,  # Drop incomplete windows at end
    preload=True,
)

print(f"\nCreated {len(windows)} windows")

# %% Inspect window metadata
metadata = windows.get_metadata()
print(f"\nWindow metadata columns: {list(metadata.columns)}")
print(f"\nFirst few rows:")
print(metadata.head(10))

# %% Extract subject IDs from description
# Need to parse BIDS-style subject IDs
sample_desc = windows.datasets[0].description
print(f"\nSample description: {sample_desc}")
print("Need to extract subject ID from description field")

# %% Add custom metadata columns
# We need to add:
# 1. movie_id (constant for this dataset)
# 2. time_offset_seconds (relative to video_start)
# 3. subject_id (extracted from description)

def extract_subject_id(description):
    """Extract subject ID from BIDS description."""
    # Description is a pandas Series with subject field
    try:
        return description['subject']
    except (KeyError, TypeError):
        # Fallback: try to parse string if description is string-like
        if isinstance(description, str) and 'sub-' in description:
            import re
            match = re.search(r'sub-([A-Z0-9]+)', description)
            if match:
                return match.group(1)
        return None

# Test extraction
for ds in windows.datasets[:3]:
    subj = extract_subject_id(ds.description)
    print(f"Description: {ds.description} -> Subject: {subj}")

# %% Compute time offsets relative to video_start
# Each window's i_start_in_trial tells us the sample index
# We need to convert to seconds and account for video_start timing

def get_video_start_time(raw):
    """Get video_start annotation onset time."""
    for ann in raw.annotations:
        if 'video_start' in ann['description']:
            return ann['onset']
    return 0.0  # Fallback if not found

# Check video_start times
for i, ds in enumerate(dataset_movie.datasets[:3]):
    vstart = get_video_start_time(ds.raw)
    print(f"Recording {i}: video_start at {vstart:.2f}s")

# %% Add metadata to windows
print("\nAdding custom metadata...")

# Iterate through each recording's windows
for win_ds, base_ds in zip(windows.datasets, dataset_movie.datasets):
    # Get base info
    subject_id = extract_subject_id(base_ds.description)
    video_start = get_video_start_time(base_ds.raw)

    # Get metadata for this recording's windows
    md = win_ds.metadata.copy()

    # Add movie_id
    md['movie_id'] = 'DespicableMe'

    # Add subject_id
    md['subject_id'] = subject_id

    # Compute time_offset_seconds
    # i_start_in_trial is in samples, convert to seconds
    # Offset from video_start, not recording start
    md['time_offset_seconds'] = (md['i_start_in_trial'] / SFREQ) + video_start

    # Update metadata
    win_ds.metadata = md

# %% Verify metadata
metadata = windows.get_metadata()
print(f"\nUpdated metadata columns: {list(metadata.columns)}")
print(f"\nSample with new columns:")
print(metadata[['subject_id', 'movie_id', 'time_offset_seconds', 'i_window_in_trial']].head(10))

# %% Explore contrastive pair sampling strategy
print("\n" + "="*60)
print("CONTRASTIVE PAIR SAMPLING STRATEGY")
print("="*60)

# Group windows by (movie_id, time_bin) to find positive pairs
# Use 1-second bins to allow some temporal flexibility
TIME_BIN_SIZE_S = 1.0

metadata['time_bin'] = (metadata['time_offset_seconds'] // TIME_BIN_SIZE_S).astype(int)

print(f"\nTotal windows: {len(metadata)}")
print(f"Unique subjects: {metadata['subject_id'].nunique()}")
print(f"Unique time bins: {metadata['time_bin'].nunique()}")

# %% Find potential positive pairs
# For each time bin, how many different subjects have windows?
time_bin_subjects = metadata.groupby('time_bin')['subject_id'].nunique()
print(f"\nTime bins with multiple subjects (for positive pairs):")
multi_subject_bins = time_bin_subjects[time_bin_subjects > 1]
print(f"  Count: {len(multi_subject_bins)}")
print(f"  Mean subjects per bin: {multi_subject_bins.mean():.1f}")
print(f"  Max subjects in a bin: {multi_subject_bins.max()}")

# %% Example positive pair
# Take a time bin with multiple subjects
if len(multi_subject_bins) > 0:
    example_bin = multi_subject_bins.idxmax()  # bin with most subjects
    bin_windows = metadata[metadata['time_bin'] == example_bin]
    print(f"\nExample time bin {example_bin}:")
    print(f"  Subjects: {bin_windows['subject_id'].tolist()}")
    print(f"  Window indices: {bin_windows.index.tolist()[:5]}...")
else:
    print("\nNo time bins with multiple subjects found (likely due to mini dataset)")

# %% Negative pair strategy
# For negative pairs, we need windows from DIFFERENT movies
# Since we only loaded one movie, let's simulate the logic

print("\nNegative pair strategy:")
print("  - Different movie (e.g., DespicableMe vs ThePresent)")
print("  - Any time offset")
print("  - Same OR different subject (mixed)")
print("\nTo test this, we need to load multiple movie datasets...")

# %% Load multiple movies for contrastive pairs
print("\n" + "="*60)
print("LOADING MULTIPLE MOVIES")
print("="*60)

movies = ['DespicableMe', 'ThePresent', 'DiaryOfAWimpyKid', 'FunwithFractals']
all_windows = []

for movie_name in movies:
    print(f"\nLoading {movie_name}...")

    # Load dataset
    ds = EEGChallengeDataset(
        task=movie_name,
        release="R5",
        cache_dir=DATA_DIR,
        mini=USE_MINI
    )

    print(f"  Recordings: {len(ds.datasets)}")

    # Create windows
    wins = create_fixed_length_windows(
        ds,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=int(WINDOW_LEN_S * SFREQ),
        window_stride_samples=int(STRIDE_S * SFREQ),
        drop_last_window=True,
        preload=True,
    )

    # Add metadata
    for win_ds, base_ds in zip(wins.datasets, ds.datasets):
        subject_id = extract_subject_id(base_ds.description)
        video_start = get_video_start_time(base_ds.raw)

        md = win_ds.metadata.copy()
        md['movie_id'] = movie_name
        md['subject_id'] = subject_id
        md['time_offset_seconds'] = (md['i_start_in_trial'] / SFREQ) + video_start
        md['time_bin'] = (md['time_offset_seconds'] // TIME_BIN_SIZE_S).astype(int)

        win_ds.metadata = md

    all_windows.append(wins)
    print(f"  Created {len(wins)} windows")

# Concatenate all movie windows
all_windows_concat = BaseConcatDataset(
    [ds for movie_wins in all_windows for ds in movie_wins.datasets]
)

print(f"\nTotal windows across all movies: {len(all_windows_concat)}")

# Get combined metadata (needed for later analysis)
all_metadata = all_windows_concat.get_metadata()

# %% Movie Duration Statistics
print("\n" + "="*60)
print("MOVIE DURATION STATISTICS")
print("="*60)

# Collect duration statistics for each movie
duration_stats = []
for movie_name in movies:
    print(f"\n{movie_name}:")

    # Load dataset
    ds = EEGChallengeDataset(
        task=movie_name,
        release="R5",
        cache_dir=DATA_DIR,
        mini=USE_MINI
    )

    # Get durations from each recording
    durations = []
    for base_ds in ds.datasets:
        duration_s = base_ds.raw.times[-1]
        durations.append(duration_s)

    # Compute statistics
    durations = np.array(durations)
    stats = {
        'movie': movie_name,
        'n_recordings': len(durations),
        'min_duration': durations.min(),
        'max_duration': durations.max(),
        'mean_duration': durations.mean(),
        'median_duration': np.median(durations),
        'std_duration': durations.std(),
        'total_duration': durations.sum(),
    }
    duration_stats.append(stats)

    print(f"  Recordings: {stats['n_recordings']}")
    print(f"  Duration range: {stats['min_duration']:.1f}s - {stats['max_duration']:.1f}s")
    print(f"  Mean ± Std: {stats['mean_duration']:.1f}s ± {stats['std_duration']:.1f}s")
    print(f"  Median: {stats['median_duration']:.1f}s")
    print(f"  Total across all subjects: {stats['total_duration']:.1f}s ({stats['total_duration']/60:.1f} minutes)")

# Create summary table
duration_df = pd.DataFrame(duration_stats)
print("\n" + "="*60)
print("DURATION SUMMARY TABLE")
print("="*60)
print(duration_df.to_string(index=False))

# %% Window Count Comparison: Different Stride Values
print("\n" + "="*60)
print("WINDOW COUNT COMPARISON: STRIDE SENSITIVITY")
print("="*60)

# Test different stride values
stride_configs = [
    {'name': '2s/1s (50% overlap)', 'window_s': 2.0, 'stride_s': 1.0},
    {'name': '2s/2s (non-overlapping)', 'window_s': 2.0, 'stride_s': 2.0},
    {'name': '4s/2s (50% overlap)', 'window_s': 4.0, 'stride_s': 2.0},
    {'name': '4s/4s (non-overlapping)', 'window_s': 4.0, 'stride_s': 4.0},
]

window_count_results = []
for config in stride_configs:
    print(f"\nConfig: {config['name']}")
    print(f"  Window: {config['window_s']}s, Stride: {config['stride_s']}s")

    total_windows = 0
    for movie_name in movies:
        # Load dataset
        ds = EEGChallengeDataset(
            task=movie_name,
            release="R5",
            cache_dir=DATA_DIR,
            mini=USE_MINI
        )

        # Create windows with this config
        wins = create_fixed_length_windows(
            ds,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=int(config['window_s'] * SFREQ),
            window_stride_samples=int(config['stride_s'] * SFREQ),
            drop_last_window=True,
            preload=False,  # Don't preload for speed
        )

        n_windows = len(wins)
        total_windows += n_windows
        print(f"    {movie_name}: {n_windows} windows")

    window_count_results.append({
        'config': config['name'],
        'window_s': config['window_s'],
        'stride_s': config['stride_s'],
        'total_windows_r5': total_windows,
        'windows_per_movie_avg': total_windows / len(movies),
    })

# Summary table
window_count_df = pd.DataFrame(window_count_results)
print("\n" + "="*60)
print("WINDOW COUNT SUMMARY")
print("="*60)
print(window_count_df.to_string(index=False))

# %% Perfect Alignment Analysis
print("\n" + "="*60)
print("PERFECT ALIGNMENT ANALYSIS (No Time Binning)")
print("="*60)

# Use the already-loaded all_metadata from earlier
print("\nRecall: Current approach uses TIME_BIN_SIZE_S = 1.0")
print("This allows windows within ±1s to be considered 'same time'")
print("\nLet's analyze what happens with PERFECT alignment (exact timestamps only):\n")

# For perfect alignment, we need windows starting at EXACTLY the same time
# This means identical time_offset_seconds values
perfect_alignment_groups = all_metadata.groupby(['movie_id', 'time_offset_seconds'])['subject_id'].nunique()
perfect_pos_pairs = perfect_alignment_groups[perfect_alignment_groups > 1]

print(f"Perfect alignment statistics:")
print(f"  Total (movie, exact_time) combinations: {len(perfect_alignment_groups)}")
print(f"  Combinations with 2+ subjects: {len(perfect_pos_pairs)}")
print(f"  Percentage valid for positive pairs: {100*len(perfect_pos_pairs)/len(perfect_alignment_groups):.1f}%")
print(f"  Mean subjects per valid group: {perfect_pos_pairs.mean():.2f}")
print(f"  Max subjects in a group: {perfect_pos_pairs.max()}")

# Compare with binned approach (already computed earlier)
print(f"\nComparison with TIME_BIN_SIZE_S = 1.0 (current approach):")
# Recompute binned stats for comparison
binned_groups = all_metadata.groupby(['movie_id', 'time_bin'])['subject_id'].nunique()
binned_pos_pairs = binned_groups[binned_groups > 1]
print(f"  Total (movie, time_bin) combinations: {len(binned_groups)}")
print(f"  Combinations with 2+ subjects: {len(binned_pos_pairs)}")
print(f"  Percentage valid for positive pairs: {100*len(binned_pos_pairs)/len(binned_groups):.1f}%")

print(f"\n⚠️  Impact of perfect alignment:")
print(f"  Positive pair groups: {len(perfect_pos_pairs)} vs {len(binned_pos_pairs)} (binned)")
print(f"  Reduction: {100*(1 - len(perfect_pos_pairs)/len(binned_pos_pairs)):.1f}%")

# Analyze per-movie perfect alignment
print(f"\nPer-movie perfect alignment:")
for movie_name in movies:
    movie_data = all_metadata[all_metadata['movie_id'] == movie_name]
    movie_perfect = movie_data.groupby('time_offset_seconds')['subject_id'].nunique()
    movie_perfect_valid = movie_perfect[movie_perfect > 1]
    print(f"  {movie_name}:")
    print(f"    Total exact timestamps: {len(movie_perfect)}")
    print(f"    Valid for pos pairs (2+ subjects): {len(movie_perfect_valid)}")
    print(f"    Percentage: {100*len(movie_perfect_valid)/len(movie_perfect):.1f}%")

# %% Full Training Dataset Projection (R1-R4, R6-R11)
print("\n" + "="*60)
print("FULL TRAINING DATASET PROJECTION")
print("="*60)

print("\nProjecting from R5 mini to R1-R4, R6-R11 (10 training releases):")
print("  - R5 mini: 1 release, ~20 subjects/movie")
print("  - Training: 10 releases (R1, R2, R3, R4, R6, R7, R8, R9, R10, R11)")
print("  - Assumption: Linear scaling (conservative estimate)")

# Scaling factor
N_TRAINING_RELEASES = 10  # R1-R4, R6-R11
N_R5_RELEASES = 1
SCALE_FACTOR = N_TRAINING_RELEASES / N_R5_RELEASES

print(f"\nScaling factor: {SCALE_FACTOR}x")

# Project window counts for different configs
print("\n" + "="*60)
print("PROJECTED WINDOW COUNTS (R1-R4, R6-R11)")
print("="*60)

projection_results = []
for config in stride_configs:
    r5_windows = window_count_df[window_count_df['config'] == config['name']]['total_windows_r5'].values[0]
    projected_windows = int(r5_windows * SCALE_FACTOR)

    projection_results.append({
        'config': config['name'],
        'r5_mini_windows': r5_windows,
        'projected_train_windows': projected_windows,
        'scale_factor': f"{SCALE_FACTOR}x",
    })

    print(f"\n{config['name']}:")
    print(f"  R5 mini: {r5_windows:,} windows")
    print(f"  Projected training: {projected_windows:,} windows")

projection_df = pd.DataFrame(projection_results)
print("\n" + "="*60)
print("PROJECTION SUMMARY TABLE")
print("="*60)
print(projection_df.to_string(index=False))

# Project positive pair counts
print("\n" + "="*60)
print("PROJECTED POSITIVE PAIR AVAILABILITY")
print("="*60)

# Current R5 mini stats (using 2s/1s config, perfect alignment)
r5_total_windows = len(all_metadata)
r5_perfect_pos_groups = len(perfect_pos_pairs)
r5_binned_pos_groups = len(binned_pos_pairs)

# Projected training stats
projected_total_windows = int(r5_total_windows * SCALE_FACTOR)
projected_perfect_pos_groups = int(r5_perfect_pos_groups * SCALE_FACTOR)
projected_binned_pos_groups = int(r5_binned_pos_groups * SCALE_FACTOR)

summary_table = pd.DataFrame([
    {
        'Dataset': 'R5 mini (current)',
        'Releases': 1,
        'Subjects/movie (avg)': len(all_metadata) // (len(movies) * (len(all_metadata) // len(movies) // 20)),
        'Total Windows': f"{r5_total_windows:,}",
        'Perfect Pos Groups': f"{r5_perfect_pos_groups:,}",
        'Binned Pos Groups': f"{r5_binned_pos_groups:,}",
    },
    {
        'Dataset': 'R1-R4, R6-R11 (projected)',
        'Releases': N_TRAINING_RELEASES,
        'Subjects/movie (avg)': '~200 (est)',
        'Total Windows': f"{projected_total_windows:,}",
        'Perfect Pos Groups': f"{projected_perfect_pos_groups:,}",
        'Binned Pos Groups': f"{projected_binned_pos_groups:,}",
    }
])

print(summary_table.to_string(index=False))

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("""
1. PERFECT ALIGNMENT (exact timestamps):
   - Provides highest-quality positive pairs (true neural correlation)
   - Reduces positive pair count vs binned approach
   - Projected training dataset: ~{:,} perfect alignment groups

2. WINDOW CONFIGURATION TRADEOFF:
   - 2s/1s stride: Maximum sample count, but 50% overlap (correlation)
   - 2s/2s stride: Non-overlapping, less correlation, fewer samples
   - 4s windows: Better for downstream C2 transfer, but need random crops

3. RECOMMENDED APPROACH FOR CONTRASTIVE PRETRAINING:
   - Start with 2s/2s stride (non-overlapping) for clean contrastive signal
   - Use PERFECT alignment (no time binning) to preserve ISC precision
   - Expected training set: ~{:,} windows from 10 releases
   - Validate ISC quality during training to ensure alignment effectiveness
""".format(projected_perfect_pos_groups, projected_total_windows))

# %% Analyze multi-movie metadata
print(f"\nCombined metadata shape: {all_metadata.shape}")
print(f"\nMovies: {all_metadata['movie_id'].value_counts()}")
print(f"Unique subjects: {all_metadata['subject_id'].nunique()}")

# %% Subject overlap across movies
# Do subjects watch multiple movies?
subject_movies = all_metadata.groupby('subject_id')['movie_id'].unique()
multi_movie_subjects = subject_movies[subject_movies.apply(len) > 1]
print(f"\nSubjects who watched multiple movies: {len(multi_movie_subjects)}")
if len(multi_movie_subjects) > 0:
    print("Examples:")
    for subj, movies in list(multi_movie_subjects.items())[:5]:
        print(f"  {subj}: {movies}")

# %% Positive pair availability
# For each (movie, time_bin), count subjects
pair_availability = all_metadata.groupby(['movie_id', 'time_bin'])['subject_id'].nunique()
valid_pos_pairs = pair_availability[pair_availability > 1]
print(f"\n(Movie, time_bin) pairs with 2+ subjects: {len(valid_pos_pairs)}")
print(f"Total (movie, time_bin) pairs: {len(pair_availability)}")
print(f"Percentage valid for positive pairs: {100*len(valid_pos_pairs)/len(pair_availability):.1f}%")

# %% Negative pair availability
# Count windows per movie
movie_counts = all_metadata['movie_id'].value_counts()
print(f"\nWindows per movie:")
for movie, count in movie_counts.items():
    print(f"  {movie}: {count}")

print(f"\nFor any anchor window, we can sample negatives from {len(movies)-1} other movies")

# %% Summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total windows: {len(all_metadata)}")
print(f"Movies: {all_metadata['movie_id'].nunique()}")
print(f"Subjects: {all_metadata['subject_id'].nunique()}")
print(f"Window shape: ({int(WINDOW_LEN_S * SFREQ)}, {len(raw.ch_names)})")
print(f"\nPositive pairs:")
print(f"  Strategy: Same (movie, time_bin), different subject")
print(f"  Available: {len(valid_pos_pairs)} time bins")
print(f"\nNegative pairs:")
print(f"  Strategy: Different movie, any time, any subject")
print(f"  Available: ~{len(all_metadata)} windows per negative sample")

# %% Next steps
print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("""
1. Extract functions to src/utils/movie_windows.py:
   - create_movie_windows(concat_ds, window_len_s, stride_s, sfreq)
   - add_movie_metadata(windows_ds, movie_name)

2. Create src/utils/contrastive_dataset.py:
   - ContrastivePairDataset(windows_ds, pos_strategy, neg_strategy)
   - __getitem__ returns (anchor, positive, negative) triplets

3. Test with PyTorch DataLoader to verify shapes and sampling
""")
