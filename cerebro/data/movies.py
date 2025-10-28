"""Movie DataModule for contrastive learning.

This module prepares movie-watching EEG data for contrastive learning.
It leverages the temporal alignment across subjects watching the same
movie to create positive pairs (same movie+time, different subjects)
and negative pairs (different movies).

Pipeline:
1. Load movie task recordings from multiple subjects
2. Create fixed-length windows with configurable stride
3. Generate positive/negative pairs based on temporal alignment
4. Split at subject level to prevent data leakage
"""

import logging
import pickle
from pathlib import Path
from typing import List, Literal, Optional

import lightning as L
import numpy as np
from braindecode.datasets import BaseConcatDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from cerebro.utils.contrastive_dataset import ContrastivePairDataset
from cerebro.utils.movie_windows import load_and_window_movies

logger = logging.getLogger(__name__)


class MovieDataModule(L.LightningDataModule):
    """DataModule for movie contrastive learning.

    Prepares (anchor, positive, negative) triplets from movie-watching
    EEG data. Positive pairs are from the same movie and time window
    but different subjects. Negative pairs are from different movies.

    Args:
        data_dir: Root directory containing EEG data
        releases: List of release names to use (e.g., ["R1", "R2", ...])
        movie_names: List of movie names to include (default: all 4 movies)
        batch_size: Batch size for DataLoader
        num_workers: Number of parallel data loading workers
        window_len_s: Length of EEG windows in seconds
        stride_s: Stride between windows in seconds (overlap = window_len - stride)
        time_bin_size_s: Size of time bins for positive pair matching
        pos_strategy: Strategy for positive pair sampling
        neg_strategy: Strategy for negative pair sampling
        return_triplets: If True return (anchor, pos, neg), else (anchor, other, label)
        val_frac: Fraction of subjects for validation
        test_frac: Fraction of subjects for test
        seed: Random seed for reproducible splits
        use_mini: If True, use mini subset for fast prototyping
        cache_dir: Directory for caching preprocessed windows

    Example:
        >>> datamodule = MovieDataModule(
        ...     data_dir="data",
        ...     releases=["R1", "R2"],
        ...     batch_size=256,
        ...     window_len_s=2.0,
        ...     stride_s=1.0  # 50% overlap
        ... )
        >>> datamodule.setup()
        >>> train_loader = datamodule.train_dataloader()
        >>> for anchor, positive, negative in train_loader:
        ...     # Train contrastive model
        ...     pass
    """

    def __init__(
        self,
        data_dir: str,
        releases: List[str],
        movie_names: Optional[List[str]] = None,
        batch_size: int = 256,
        num_workers: int = 8,
        window_len_s: float = 2.0,
        stride_s: float = 1.0,
        time_bin_size_s: float = 1.0,
        pos_strategy: Literal["same_movie_time"] = "same_movie_time",
        neg_strategy: Literal[
            "diff_movie_mixed", "diff_movie_same_subj", "diff_movie_diff_subj"
        ] = "diff_movie_mixed",
        return_triplets: bool = True,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 2025,
        use_mini: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Default movie names (all 4 movies from HBN)
        if movie_names is None:
            movie_names = [
                "DespicableMe",
                "ThePresent",
                "DiaryOfAWimpyKid",
                "FunwithFractals",
            ]
        self.movie_names = movie_names

        # Default cache directory
        if cache_dir is None:
            cache_dir = Path(data_dir).parent / "cache" / "movies"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Download data if needed (handled by EEGChallengeDataset)."""
        # EEGChallengeDataset handles downloading automatically
        # This method is for download logic only (no state assignments)
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare datasets for training/validation/testing.

        This method:
        1. Loads and windows movie recordings
        2. Splits subjects into train/val/test
        3. Creates ContrastivePairDatasets for each split
        """
        # Create cache key for this configuration
        cache_key = self._get_cache_key()
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # Try to load from cache
        if cache_file.exists() and not self.hparams.use_mini:
            logger.info(f"Loading cached windows from {cache_file}")
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
                train_windows = cache_data["train_windows"]
                val_windows = cache_data["val_windows"]
                test_windows = cache_data["test_windows"]
        else:
            # Load and window movie data
            logger.info("Loading and windowing movie data...")
            windows_ds = load_and_window_movies(
                movie_names=self.movie_names,
                dataset_class=None,  # Will use EEGChallengeDataset
                cache_dir=self.hparams.data_dir,
                releases=self.hparams.releases,
                mini=self.hparams.use_mini,
                window_len_s=self.hparams.window_len_s,
                stride_s=self.hparams.stride_s,
                sfreq=100.0,  # HBN standard sampling rate
                time_bin_size_s=self.hparams.time_bin_size_s,
                preload=True,
            )

            logger.info(f"Total windows created: {len(windows_ds)}")

            # Get metadata for subject-level splitting
            metadata = windows_ds.get_metadata()
            subjects = metadata["subject_id"].unique()
            logger.info(f"Total subjects: {len(subjects)}")

            # Subject-level train/val/test split
            # Handle test_frac=0.0 and val_frac=0.0 cases
            if self.hparams.test_frac > 0.0:
                # First split train+val from test
                train_val_subjects, test_subjects = train_test_split(
                    subjects,
                    test_size=self.hparams.test_frac,
                    random_state=self.hparams.seed,
                )

                # Then split train from val
                if self.hparams.val_frac > 0.0:
                    val_size_adjusted = self.hparams.val_frac / (
                        1 - self.hparams.test_frac
                    )
                    train_subjects, val_subjects = train_test_split(
                        train_val_subjects,
                        test_size=val_size_adjusted,
                        random_state=self.hparams.seed + 1,
                    )
                else:
                    # No validation split
                    train_subjects = train_val_subjects
                    val_subjects = np.array([])
            else:
                # No test split
                test_subjects = np.array([])
                if self.hparams.val_frac > 0.0:
                    # Split train from val
                    train_subjects, val_subjects = train_test_split(
                        subjects,
                        test_size=self.hparams.val_frac,
                        random_state=self.hparams.seed + 1,
                    )
                else:
                    # No validation split either - use all subjects for training
                    train_subjects = subjects
                    val_subjects = np.array([])

            logger.info(
                f"Subject split: {len(train_subjects)} train, "
                f"{len(val_subjects)} val, {len(test_subjects)} test"
            )

            # Split at dataset level (each dataset is one subject-movie recording)
            # Get dataset indices for each subject
            train_ds_indices = []
            val_ds_indices = []
            test_ds_indices = []

            for i, ds in enumerate(windows_ds.datasets):
                # Support both dict and pandas Series for description
                try:
                    if hasattr(ds.description, "get"):
                        subject_id = ds.description.get("subject")
                    else:
                        subject_id = ds.description["subject"]
                except (KeyError, AttributeError, TypeError):
                    subject_id = None

                if subject_id in train_subjects:
                    train_ds_indices.append(i)
                elif subject_id in val_subjects:
                    val_ds_indices.append(i)
                elif subject_id in test_subjects:
                    test_ds_indices.append(i)

            # Split windows dataset at dataset level
            if train_ds_indices:
                train_windows = windows_ds.split([train_ds_indices])["0"]
            else:
                raise ValueError("No training datasets found")

            # Only create val windows if val_frac > 0
            if len(val_subjects) > 0 and val_ds_indices:
                val_windows = windows_ds.split([val_ds_indices])["0"]
            else:
                # Use a small subset of train datasets for validation (required by Lightning)
                # For contrastive learning, need:
                # - At least 2 subjects for positive pairs
                # - At least 2 movies for negative sampling
                # Strategy: pick first 2 subjects from different movies if possible
                metadata_for_split = windows_ds.get_metadata()

                # Group datasets by movie
                movie_ds_map = {}
                for idx in train_ds_indices:
                    ds = windows_ds.datasets[idx]
                    movie = metadata_for_split.iloc[0]["movie_id"] if len(metadata_for_split) > 0 else "unknown"
                    # Get movie from this dataset's first window
                    ds_meta = ds.metadata if hasattr(ds, "metadata") else None
                    if ds_meta is not None and len(ds_meta) > 0:
                        movie = ds_meta["movie_id"].iloc[0]
                    if movie not in movie_ds_map:
                        movie_ds_map[movie] = []
                    movie_ds_map[movie].append(idx)

                # Take 2 datasets from each of first 2 movies (if available)
                val_ds_indices = []
                for movie, ds_list in list(movie_ds_map.items())[:2]:
                    val_ds_indices.extend(ds_list[:2])

                # Fallback: if not enough movies, just take first 4 datasets
                if len(val_ds_indices) < 2:
                    val_ds_indices = train_ds_indices[:4]

                val_windows = windows_ds.split([val_ds_indices])["0"]
                logger.info(
                    f"No validation subjects specified, using {len(val_ds_indices)} "
                    f"train datasets for validation (from {len(movie_ds_map)} movies)"
                )

            # Only create test windows if test_frac > 0
            if len(test_subjects) > 0 and test_ds_indices:
                test_windows = windows_ds.split([test_ds_indices])["0"]
            else:
                test_windows = None

            # Cache if not using mini
            if not self.hparams.use_mini:
                logger.info(f"Caching windows to {cache_file}")
                with open(cache_file, "wb") as f:
                    pickle.dump(
                        {
                            "train_windows": train_windows,
                            "val_windows": val_windows,
                            "test_windows": test_windows,
                        },
                        f,
                    )

        # Create ContrastivePairDatasets
        logger.info("Creating contrastive pair datasets...")

        self.train_dataset = ContrastivePairDataset(
            train_windows,
            pos_strategy=self.hparams.pos_strategy,
            neg_strategy=self.hparams.neg_strategy,
            return_triplets=self.hparams.return_triplets,
            random_state=self.hparams.seed,
        )

        self.val_dataset = ContrastivePairDataset(
            val_windows,
            pos_strategy=self.hparams.pos_strategy,
            neg_strategy=self.hparams.neg_strategy,
            return_triplets=self.hparams.return_triplets,
            random_state=self.hparams.seed + 100,
        )

        if test_windows is not None and len(test_windows) > 0:
            self.test_dataset = ContrastivePairDataset(
                test_windows,
                pos_strategy=self.hparams.pos_strategy,
                neg_strategy=self.hparams.neg_strategy,
                return_triplets=self.hparams.return_triplets,
                random_state=self.hparams.seed + 200,
            )

        # Log dataset statistics
        self._log_dataset_stats()

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test DataLoader if test set exists."""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def _get_cache_key(self) -> str:
        """Generate unique cache key based on configuration."""
        components = [
            "movies",
            *sorted(self.movie_names),
            *sorted(self.hparams.releases),
            f"w{self.hparams.window_len_s}",
            f"s{self.hparams.stride_s}",
            f"b{self.hparams.time_bin_size_s}",
            f"seed{self.hparams.seed}",
        ]
        return "_".join(components).replace(".", "p")

    def _log_dataset_stats(self) -> None:
        """Log statistics about the datasets."""
        if self.train_dataset is not None:
            train_stats = self.train_dataset.get_stats()
            logger.info(f"Train dataset: {train_stats}")

        if self.val_dataset is not None:
            val_stats = self.val_dataset.get_stats()
            logger.info(f"Val dataset: {val_stats}")

        if self.test_dataset is not None:
            test_stats = self.test_dataset.get_stats()
            logger.info(f"Test dataset: {test_stats}")