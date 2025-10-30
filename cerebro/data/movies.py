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
from eegdash import EEGChallengeDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from cerebro.data.cache_manager import GranularCacheManager
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
        test_release: Specific release to use as test set (e.g., "R5"), or None for no test
        seed: Random seed for reproducible splits
        use_mini: If True, use mini subset for fast prototyping
        cache_dir: Directory for caching preprocessed windows

    Example:
        >>> # Development mode with R5 test set
        >>> datamodule = MovieDataModule(
        ...     data_dir="data",
        ...     releases=["R1", "R2", "R5"],
        ...     test_release="R5",
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
        test_release: Optional[str] = None,  # Specific release for test (e.g., "R5")
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
            cache_dir = Path(data_dir) / "cache" / "movies"
        cache_root = Path(cache_dir)

        # Initialize granular cache manager
        # Cache key includes only preprocessing params (NOT releases or seed)
        self.cache_mgr = GranularCacheManager(
            cache_root=str(cache_root),
            preprocessing_params={
                "movies": sorted(self.movie_names),
                "window_len_s": window_len_s,
                "stride_s": stride_s,
                "time_bin_size_s": time_bin_size_s,
                "sfreq": 100.0,  # HBN standard
            }
        )

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
        """Prepare datasets for training/validation/testing with release-based test splitting.

        This method:
        1. Pops test release from releases list
        2. Loads and windows movie recordings per-release (with caching)
        3. Combines train/val releases and test releases separately
        4. Splits subjects from train/val based on val_frac
        5. Creates ContrastivePairDatasets for each split
        """
        logger.info("Loading and windowing movie data...")
        logger.info(f"Releases: {self.hparams.releases}")
        logger.info(f"Movies: {self.movie_names}")
        logger.info(f"Test release: {self.hparams.test_release}")

        # Step 1: Pop test release from releases list
        if self.hparams.test_release:
            if self.hparams.test_release not in self.hparams.releases:
                raise ValueError(
                    f"test_release '{self.hparams.test_release}' not in releases {self.hparams.releases}"
                )
            test_releases = [self.hparams.test_release]
            train_val_releases = [r for r in self.hparams.releases if r != self.hparams.test_release]
            logger.info(f"Using {self.hparams.test_release} as test set")
            logger.info(f"Train/val releases: {train_val_releases}")
        else:
            test_releases = []
            train_val_releases = self.hparams.releases
            logger.info("No test release specified - no test split")

        # Helper function to load and window releases
        def load_release_windows(release_list, desc="Loading"):
            """Load windowed datasets from list of releases."""
            all_windows_list = []

            if not release_list:
                return all_windows_list

            # Check cache status
            cached_releases = self.cache_mgr.get_cached_releases(release_list)
            missing_releases = self.cache_mgr.get_missing_releases(release_list)

            # Load cached releases
            for release in cached_releases:
                windows_ds = self.cache_mgr.load_release(release)
                all_windows_list.append(windows_ds)
                logger.info(f"  ✓ Loaded {len(windows_ds)} windows from {release} cache")

            # Load missing releases from scratch
            if missing_releases:
                logger.info(f"{desc}: {len(missing_releases)} missing releases...")

                for release in missing_releases:
                    logger.info(f"  Loading {release}...")
                    try:
                        # Load and window this release
                        windows_ds = load_and_window_movies(
                            movie_names=self.movie_names,
                            dataset_class=EEGChallengeDataset,
                            cache_dir=self.hparams.data_dir,
                            releases=[release],  # Single release at a time
                            mini=self.hparams.use_mini,
                            window_len_s=self.hparams.window_len_s,
                            stride_s=self.hparams.stride_s,
                            sfreq=100.0,  # HBN standard sampling rate
                            time_bin_size_s=self.hparams.time_bin_size_s,
                            preload=True,
                        )

                        logger.info(f"    Windowed: {len(windows_ds)} windows")

                        # Save to cache
                        self.cache_mgr.save_release(release, windows_ds)
                        self.cache_mgr.mark_complete(
                            release,
                            metadata={"n_windows": len(windows_ds), "n_datasets": len(windows_ds.datasets)}
                        )

                        # Add to collection
                        all_windows_list.append(windows_ds)

                    except Exception as e:
                        logger.warning(f"  ✗ {release}: {type(e).__name__}: {str(e)[:60]}")
                        self.cache_mgr.mark_failed(release, str(e))
                        continue

            return all_windows_list

        # Step 2: Load train/val and test releases separately
        logger.info("\n=== Loading train/val releases ===")
        train_val_windows_list = load_release_windows(train_val_releases, "Loading train/val")

        logger.info("\n=== Loading test releases ===")
        test_windows_list = load_release_windows(test_releases, "Loading test")

        if not train_val_windows_list:
            raise ValueError("No train/val movie windows loaded successfully!")

        # Combine train/val releases into single BaseConcatDataset
        train_val_windows = BaseConcatDataset(train_val_windows_list)
        logger.info(f"\nTrain/val windows: {len(train_val_windows)}")

        # Combine test releases if any
        if test_windows_list:
            test_windows_combined = BaseConcatDataset(test_windows_list)
            logger.info(f"Test windows: {len(test_windows_combined)}")
        else:
            test_windows_combined = None
            logger.info("Test windows: 0")

        # Step 3: Subject-level split for train/val
        metadata = train_val_windows.get_metadata()
        subjects = metadata["subject_id"].unique()
        logger.info(f"Train/val unique subjects: {len(subjects)}")

        if self.hparams.val_frac > 0.0 and len(subjects) > 1:
            # Split subjects into train and val
            train_subjects, val_subjects = train_test_split(
                subjects,
                test_size=self.hparams.val_frac,
                random_state=self.hparams.seed,
            )
        else:
            # No validation split - use all subjects for training
            train_subjects = subjects
            val_subjects = np.array([])
            logger.info("No validation split - using all train/val subjects for training")

        logger.info(
            f"Subject split: {len(train_subjects)} train, {len(val_subjects)} val"
        )

        # Split at dataset level (each dataset is one subject-movie recording)
        train_ds_indices = []
        val_ds_indices = []

        for i, ds in enumerate(train_val_windows.datasets):
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

        # Split windows dataset at dataset level
        if train_ds_indices:
            train_windows = train_val_windows.split([train_ds_indices])["0"]
        else:
            raise ValueError("No training datasets found")

        # Create val windows
        if len(val_subjects) > 0 and val_ds_indices:
            val_windows = train_val_windows.split([val_ds_indices])["0"]
        else:
            # Use a small subset of train datasets for validation (required by Lightning)
            # For contrastive learning, need:
            # - At least 2 subjects for positive pairs
            # - At least 2 movies for negative sampling
            # Strategy: pick first 2 subjects from different movies if possible
            metadata_for_split = train_val_windows.get_metadata()

            # Group datasets by movie
            movie_ds_map = {}
            for idx in train_ds_indices:
                ds = train_val_windows.datasets[idx]
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

            val_windows = train_val_windows.split([val_ds_indices])["0"]
            logger.info(
                f"No validation subjects specified, using {len(val_ds_indices)} "
                f"train datasets for validation (from {len(movie_ds_map)} movies)"
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

        # Create test dataset if test release specified
        if test_windows_combined is not None and len(test_windows_combined) > 0:
            self.test_dataset = ContrastivePairDataset(
                test_windows_combined,
                pos_strategy=self.hparams.pos_strategy,
                neg_strategy=self.hparams.neg_strategy,
                return_triplets=self.hparams.return_triplets,
                random_state=self.hparams.seed + 200,
            )
        else:
            self.test_dataset = None
            logger.info("No test dataset created")

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