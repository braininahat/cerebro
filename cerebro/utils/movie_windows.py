"""Utility functions for creating windows from movie task EEG data.

These functions handle movie tasks that lack trial structure and only have
video_start/stop markers. They use fixed-length windowing and add custom
metadata for contrastive learning.
"""

import re
from typing import Optional

import pandas as pd
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows


def extract_subject_id(description) -> Optional[str]:
    """Extract subject ID from BIDS description.

    Parameters
    ----------
    description : dict or str
        Dataset description, either as dict with 'subject' key or string
        containing 'sub-XXXXXXX' pattern.

    Returns
    -------
    str or None
        Subject ID (e.g., 'NDARXXXXXX') or None if not found.
    """
    if isinstance(description, dict) and "subject" in description:
        return description["subject"]

    if isinstance(description, str) and "sub-" in description:
        match = re.search(r"sub-([A-Z0-9]+)", description)
        if match:
            return match.group(1)

    return None


def get_video_start_time(raw) -> float:
    """Get video_start annotation onset time from Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG object with annotations.

    Returns
    -------
    float
        Onset time in seconds of video_start marker, or 0.0 if not found.
    """
    for ann in raw.annotations:
        if "video_start" in ann["description"]:
            return ann["onset"]
    return 0.0


def create_movie_windows(
    concat_ds: BaseConcatDataset,
    window_len_s: float = 2.0,
    stride_s: float = 1.0,
    sfreq: float = 100.0,
    preload: bool = True,
) -> BaseConcatDataset:
    """Create fixed-length windows from movie task dataset.

    Uses create_fixed_length_windows since movie tasks lack event markers.

    Parameters
    ----------
    concat_ds : BaseConcatDataset
        Input dataset containing movie task recordings.
    window_len_s : float, default=2.0
        Window length in seconds.
    stride_s : float, default=1.0
        Stride between windows in seconds.
    sfreq : float, default=100.0
        Sampling frequency in Hz.
    preload : bool, default=True
        Whether to preload data into memory.

    Returns
    -------
    BaseConcatDataset
        Windowed dataset with fixed-length segments.
    """
    windows = create_fixed_length_windows(
        concat_ds,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=int(window_len_s * sfreq),
        window_stride_samples=int(stride_s * sfreq),
        drop_last_window=True,
        preload=preload,
    )
    return windows


def add_movie_metadata(
    windows_ds: BaseConcatDataset,
    original_ds: BaseConcatDataset,
    movie_name: str,
    sfreq: float = 100.0,
    time_bin_size_s: float = 1.0,
) -> BaseConcatDataset:
    """Add movie-specific metadata to windowed dataset.

    Adds columns: movie_id, subject_id, time_offset_seconds, time_bin.

    Parameters
    ----------
    windows_ds : BaseConcatDataset
        Windowed dataset to augment with metadata.
    original_ds : BaseConcatDataset
        Original (non-windowed) dataset containing raw data.
    movie_name : str
        Name of the movie (e.g., 'DespicableMe', 'ThePresent').
    sfreq : float, default=100.0
        Sampling frequency in Hz.
    time_bin_size_s : float, default=1.0
        Time bin size in seconds for grouping windows.

    Returns
    -------
    BaseConcatDataset
        Dataset with augmented metadata.
    """
    for win_ds, base_ds in zip(windows_ds.datasets, original_ds.datasets):
        subject_id = extract_subject_id(base_ds.description)
        video_start = get_video_start_time(base_ds.raw)

        md = win_ds.metadata.copy()

        # Add movie identifier
        md["movie_id"] = movie_name

        # Add subject identifier
        md["subject_id"] = subject_id

        # Compute time offset from video start
        md["time_offset_seconds"] = (md["i_start_in_trial"] / sfreq) + video_start

        # Create time bins for grouping
        md["time_bin"] = (md["time_offset_seconds"] // time_bin_size_s).astype(int)

        win_ds.metadata = md

    return windows_ds


def load_and_window_movies(
    movie_names: list[str],
    dataset_class,
    cache_dir,
    release: str = "R5",
    mini: bool = True,
    window_len_s: float = 2.0,
    stride_s: float = 1.0,
    sfreq: float = 100.0,
    time_bin_size_s: float = 1.0,
    preload: bool = True,
) -> BaseConcatDataset:
    """Load multiple movies and create windowed dataset with metadata.

    Convenience function that loads multiple movies, creates windows,
    and adds metadata in one call.

    Parameters
    ----------
    movie_names : list of str
        Names of movies to load (e.g., ['DespicableMe', 'ThePresent']).
    dataset_class : class
        Dataset class to use (e.g., EEGChallengeDataset).
    cache_dir : Path or str
        Directory for caching downloaded data.
    release : str, default='R5'
        Dataset release version.
    mini : bool, default=True
        Whether to use mini dataset.
    window_len_s : float, default=2.0
        Window length in seconds.
    stride_s : float, default=1.0
        Stride between windows in seconds.
    sfreq : float, default=100.0
        Sampling frequency in Hz.
    time_bin_size_s : float, default=1.0
        Time bin size in seconds.
    preload : bool, default=True
        Whether to preload data.

    Returns
    -------
    BaseConcatDataset
        Combined windowed dataset from all movies with metadata.
    """
    all_windows = []

    for movie_name in movie_names:
        # Load dataset
        ds = dataset_class(
            task=movie_name,
            release=release,
            cache_dir=cache_dir,
            mini=mini,
        )

        # Create windows
        wins = create_movie_windows(
            ds,
            window_len_s=window_len_s,
            stride_s=stride_s,
            sfreq=sfreq,
            preload=preload,
        )

        # Add metadata
        wins = add_movie_metadata(
            wins,
            ds,
            movie_name=movie_name,
            sfreq=sfreq,
            time_bin_size_s=time_bin_size_s,
        )

        all_windows.append(wins)

    # Combine all movies
    combined = BaseConcatDataset(
        [ds for movie_wins in all_windows for ds in movie_wins.datasets]
    )

    return combined


def get_positive_pair_stats(windows_ds: BaseConcatDataset) -> pd.DataFrame:
    """Compute statistics about positive pair availability.

    Parameters
    ----------
    windows_ds : BaseConcatDataset
        Windowed dataset with movie_id, time_bin, subject_id metadata.

    Returns
    -------
    pd.DataFrame
        DataFrame with (movie_id, time_bin) as index and subject counts.
    """
    metadata = windows_ds.get_metadata()
    pair_availability = metadata.groupby(["movie_id", "time_bin"])[
        "subject_id"
    ].nunique()
    return pair_availability[pair_availability > 1]
