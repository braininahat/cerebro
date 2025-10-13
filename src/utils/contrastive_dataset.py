"""Contrastive learning dataset for movie task EEG data.

Provides PyTorch Dataset that samples positive and negative pairs for
contrastive learning objectives.
"""

from typing import Literal, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from braindecode.datasets import BaseConcatDataset


class ContrastivePairDataset(Dataset):
    """Dataset for sampling contrastive pairs from movie EEG windows.

    Positive pairs: Same (movie, time_bin), different subject.
    Negative pairs: Different movie, any time, any subject.

    Parameters
    ----------
    windows_ds : BaseConcatDataset
        Windowed dataset with metadata containing movie_id, time_bin, subject_id.
    pos_strategy : str, default='same_movie_time'
        Strategy for positive pairs. Currently only 'same_movie_time' supported.
    neg_strategy : str, default='diff_movie_mixed'
        Strategy for negative pairs. Options:
        - 'diff_movie_mixed': Different movie, any subject (same or different)
        - 'diff_movie_same_subj': Different movie, same subject only
        - 'diff_movie_diff_subj': Different movie, different subject only
    return_triplets : bool, default=True
        If True, return (anchor, positive, negative) triplets.
        If False, return pairs with labels: (anchor, other, label) where label=1 for pos, 0 for neg.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    metadata : pd.DataFrame
        Metadata from windows_ds with movie_id, time_bin, subject_id columns.
    pos_pair_groups : pd.DataFrame
        Precomputed groups of (movie, time_bin) with multiple subjects.
    """

    def __init__(
        self,
        windows_ds: BaseConcatDataset,
        pos_strategy: Literal['same_movie_time'] = 'same_movie_time',
        neg_strategy: Literal['diff_movie_mixed', 'diff_movie_same_subj', 'diff_movie_diff_subj'] = 'diff_movie_mixed',
        return_triplets: bool = True,
        random_state: Optional[int] = None,
    ):
        self.windows_ds = windows_ds
        self.pos_strategy = pos_strategy
        self.neg_strategy = neg_strategy
        self.return_triplets = return_triplets
        self.rng = np.random.RandomState(random_state)

        # Get metadata
        self.metadata = windows_ds.get_metadata().reset_index(drop=True)

        # Validate required columns
        required_cols = ['movie_id', 'time_bin', 'subject_id']
        missing = [c for c in required_cols if c not in self.metadata.columns]
        if missing:
            raise ValueError(f"Metadata missing required columns: {missing}")

        # Precompute positive pair groups
        self._build_positive_groups()

        # Precompute movie-wise indices for fast negative sampling
        self.movie_indices = {
            movie: self.metadata[self.metadata['movie_id'] == movie].index.tolist()
            for movie in self.metadata['movie_id'].unique()
        }

    def _build_positive_groups(self):
        """Build lookup table for positive pair sampling."""
        # Group by (movie, time_bin) and get indices where multiple subjects exist
        grouped = self.metadata.groupby(['movie_id', 'time_bin']).apply(
            lambda g: g.index.tolist() if g['subject_id'].nunique() > 1 else []
        )

        # Filter out empty groups
        self.pos_pair_groups = grouped[grouped.apply(len) > 0]

        if len(self.pos_pair_groups) == 0:
            raise ValueError(
                "No valid positive pairs found. Need multiple subjects per (movie, time_bin)."
            )

        # For each window, precompute which group it belongs to (if any)
        self.window_to_group = {}
        for (movie, time_bin), indices in self.pos_pair_groups.items():
            for idx in indices:
                self.window_to_group[idx] = (movie, time_bin)

    def __len__(self):
        """Return number of valid anchor windows (those with positive pairs)."""
        return len(self.window_to_group)

    def _get_valid_anchors(self):
        """Get list of valid anchor indices."""
        return list(self.window_to_group.keys())

    def _sample_positive(self, anchor_idx: int) -> int:
        """Sample a positive pair for the anchor.

        Parameters
        ----------
        anchor_idx : int
            Index of anchor window.

        Returns
        -------
        int
            Index of positive window (same movie+time, different subject).
        """
        group_key = self.window_to_group[anchor_idx]
        candidate_indices = self.pos_pair_groups[group_key]

        # Get anchor subject
        anchor_subject = self.metadata.loc[anchor_idx, 'subject_id']

        # Filter to different subjects
        different_subject_indices = [
            idx for idx in candidate_indices
            if self.metadata.loc[idx, 'subject_id'] != anchor_subject
        ]

        if len(different_subject_indices) == 0:
            # Fallback: if somehow no different subject, return random from group
            different_subject_indices = [
                idx for idx in candidate_indices if idx != anchor_idx
            ]

        return self.rng.choice(different_subject_indices)

    def _sample_negative(self, anchor_idx: int) -> int:
        """Sample a negative pair for the anchor.

        Parameters
        ----------
        anchor_idx : int
            Index of anchor window.

        Returns
        -------
        int
            Index of negative window (different movie).
        """
        anchor_movie = self.metadata.loc[anchor_idx, 'movie_id']
        anchor_subject = self.metadata.loc[anchor_idx, 'subject_id']

        # Get all movies except anchor's movie
        other_movies = [m for m in self.movie_indices.keys() if m != anchor_movie]

        if len(other_movies) == 0:
            raise ValueError("Need at least 2 different movies for negative sampling")

        # Sample a random movie
        neg_movie = self.rng.choice(other_movies)
        neg_candidates = self.movie_indices[neg_movie]

        # Apply subject constraint based on strategy
        if self.neg_strategy == 'diff_movie_same_subj':
            # Filter to same subject
            neg_candidates = [
                idx for idx in neg_candidates
                if self.metadata.loc[idx, 'subject_id'] == anchor_subject
            ]
            if len(neg_candidates) == 0:
                # Fallback: if subject didn't watch this movie, use any
                neg_candidates = self.movie_indices[neg_movie]

        elif self.neg_strategy == 'diff_movie_diff_subj':
            # Filter to different subject
            neg_candidates = [
                idx for idx in neg_candidates
                if self.metadata.loc[idx, 'subject_id'] != anchor_subject
            ]
            if len(neg_candidates) == 0:
                # Fallback: use any
                neg_candidates = self.movie_indices[neg_movie]

        # For 'diff_movie_mixed', use all candidates from different movie

        return self.rng.choice(neg_candidates)

    def __getitem__(self, idx: int):
        """Get a contrastive sample.

        Parameters
        ----------
        idx : int
            Index into valid anchor windows.

        Returns
        -------
        tuple
            If return_triplets=True: (anchor, positive, negative)
            If return_triplets=False: (anchor, other, label) where label=1 for pos, 0 for neg
        """
        # Map idx to actual anchor index
        valid_anchors = self._get_valid_anchors()
        anchor_idx = valid_anchors[idx]

        # Get anchor data
        anchor_data = self.windows_ds[anchor_idx][0]  # X only, ignore y

        # Sample positive
        pos_idx = self._sample_positive(anchor_idx)
        pos_data = self.windows_ds[pos_idx][0]

        if self.return_triplets:
            # Sample negative
            neg_idx = self._sample_negative(anchor_idx)
            neg_data = self.windows_ds[neg_idx][0]

            return (
                torch.as_tensor(anchor_data, dtype=torch.float32),
                torch.as_tensor(pos_data, dtype=torch.float32),
                torch.as_tensor(neg_data, dtype=torch.float32),
            )
        else:
            # Return pairs with labels (useful for NT-Xent loss)
            # Randomly decide whether to return positive or negative
            if self.rng.rand() < 0.5:
                other_data = pos_data
                label = 1.0
            else:
                neg_idx = self._sample_negative(anchor_idx)
                other_data = self.windows_ds[neg_idx][0]
                label = 0.0

            return (
                torch.as_tensor(anchor_data, dtype=torch.float32),
                torch.as_tensor(other_data, dtype=torch.float32),
                torch.as_tensor(label, dtype=torch.float32),
            )

    def get_stats(self) -> dict:
        """Get statistics about the dataset.

        Returns
        -------
        dict
            Dictionary with statistics:
            - total_windows: Total windows in dataset
            - valid_anchors: Windows that can serve as anchors (have pos pairs)
            - movies: Number of unique movies
            - subjects: Number of unique subjects
            - pos_groups: Number of (movie, time_bin) groups with 2+ subjects
        """
        return {
            'total_windows': len(self.metadata),
            'valid_anchors': len(self.window_to_group),
            'movies': self.metadata['movie_id'].nunique(),
            'subjects': self.metadata['subject_id'].nunique(),
            'pos_groups': len(self.pos_pair_groups),
        }


def print_dataset_stats(dataset: ContrastivePairDataset):
    """Print statistics about a ContrastivePairDataset.

    Parameters
    ----------
    dataset : ContrastivePairDataset
        Dataset to analyze.
    """
    stats = dataset.get_stats()
    print("Contrastive Dataset Statistics:")
    print(f"  Total windows: {stats['total_windows']}")
    print(f"  Valid anchors: {stats['valid_anchors']}")
    print(f"  Movies: {stats['movies']}")
    print(f"  Subjects: {stats['subjects']}")
    print(f"  Positive pair groups: {stats['pos_groups']}")
    print(f"  Positive strategy: {dataset.pos_strategy}")
    print(f"  Negative strategy: {dataset.neg_strategy}")
    print(f"  Return triplets: {dataset.return_triplets}")
