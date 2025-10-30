"""JEPA Phase 1 pretraining data module.

Loads ALL tasks (CCD, movies, resting state, etc.) without labels.
No trial filtering - pure self-supervised learning on raw EEG.

From magnum_opus.md Phase 1:
- Use all available tasks
- Windows with optional random temporal crops (augmentation)
- Mix all tasks to learn general representations

MIGRATED TO UniversalCacheManager (Zarr-based lazy loading)
- Replaces old GranularCacheManager (pickle-based)
- Better memory efficiency via lazy loading
- Parallel cache building
- Fault tolerance with parquet manifests
"""

from pathlib import Path
from typing import Optional
import logging

import lightning as L
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from cerebro.data.unified_cache import UniversalCacheManager

logger = logging.getLogger(__name__)


class JEPAPretrainDataModule(L.LightningDataModule):
    """Data module for JEPA Phase 1 self-supervised pretraining.

    Loads ALL tasks from HBN dataset without labels.
    Creates fixed-length windows from all available recordings.

    Uses UniversalCacheManager for efficient Zarr-based caching with lazy loading.

    Args:
        data_dir: Root directory for EEG data
        releases: List of release IDs (default: all except R5)
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        window_length: Window length in seconds (default: 2.0 for S-JEPA)
        stride: Stride between windows (default: 1.0)
        crop_length: Random crop length in seconds (default: None, set to enable temporal augmentation)
        val_split: Validation split fraction (default: 0.2 subject-level)
        test_release: Specific release to use as test set (e.g., "R5"), or None for no test
        n_chans_select: Number of channels to use (default: 129)
        sfreq: Target sampling frequency (default: 100 Hz)
        mini: If True, use mini dataset for fast prototyping
        all_tasks: List of tasks to include (default: all available)
            Options: ["contrastChangeDetection", "restingState",
                     "DespicableMe", "ThePresent", "DiaryOfAWimpyKid",
                     "FunwithFractals", "surroundSupp"]
        seed: Random seed for subject-level splits (default: 42)

    Example:
        >>> # Development mode with R5 test set
        >>> dm = JEPAPretrainDataModule(
        ...     data_dir="./data",
        ...     releases=["R1", "R2", "R3", "R4", "R5"],
        ...     test_release="R5",
        ...     batch_size=256
        ... )
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        releases: Optional[list[str]] = None,
        batch_size: int = 256,
        num_workers: int = 8,
        window_length: float = 2.0,  # 2s windows for S-JEPA
        stride: float = 1.0,
        crop_length: Optional[float] = None,  # Optional temporal augmentation
        val_split: float = 0.2,  # Subject-level split
        test_release: Optional[str] = None,  # Specific release for test (e.g., "R5")
        n_chans_select: int = 129,
        sfreq: int = 100,
        mini: bool = False,
        all_tasks: Optional[list[str]] = None,
        seed: int = 42,  # Random seed for subject-level splits
    ):
        super().__init__()

        # Save all hyperparameters for Lightning compatibility
        self.save_hyperparameters()

        self.data_dir = Path(data_dir).resolve()  # Absolute path for EEGChallengeDataset
        self.num_workers = num_workers

        # Default: all releases except R5 (competition validation)
        if releases is None:
            releases = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"]
        self.releases = releases

        # Window parameters
        self.window_length = window_length
        self.stride = stride
        self.crop_length = crop_length
        self.val_split = val_split
        self.test_release = test_release
        self.n_chans_select = n_chans_select
        self.sfreq = sfreq
        self.mini = mini
        self.seed = seed

        # Tasks to include (default: all 7 HBN tasks)
        if all_tasks is None:
            all_tasks = [
                "contrastChangeDetection",
                "restingState",
                "DespicableMe",
                "ThePresent",
                "DiaryOfAWimpyKid",
                "FunwithFractals",
                "surroundSupp",
            ]
        self.all_tasks = all_tasks

        # Initialize UniversalCacheManager (Zarr-based, lazy loading)
        cache_root = self.data_dir / "cache"  # Unified cache root
        self.cache_mgr = UniversalCacheManager(
            cache_root=str(cache_root),
            preprocessing_params={
                "sfreq": self.sfreq,
                "bandpass": None,  # No bandpass filtering (already done by EEGChallengeDataset)
                "n_channels": self.n_chans_select,
                "standardize": False,
            }
        )

        # Placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def batch_size(self) -> int:
        """Batch size for DataLoaders."""
        return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        """Update batch size (modified by Lightning's batch size scaler)."""
        self.hparams.batch_size = value

    def prepare_data(self):
        """Download data if needed."""
        # EEGChallengeDataset handles downloading automatically
        pass

    def setup(self, stage: Optional[str] = None):
        """Load and prepare data using UniversalCacheManager (Zarr-based, lazy loading)."""
        logger.info(f"Loading data from releases: {self.releases}")
        logger.info(f"Tasks: {self.all_tasks}")
        logger.info(f"Mini mode: {self.mini}")
        logger.info(f"Test release: {self.test_release}")

        # Build/load Level 1 raw cache for ALL releases
        # This automatically handles parallel processing, caching, and fault tolerance
        self.cache_mgr.build_raw(
            dataset="hbn",
            releases=self.releases,
            tasks=self.all_tasks,
            mini=self.mini
        )

        # Step 1: Separate test release from train/val
        if self.test_release:
            if self.test_release not in self.releases:
                raise ValueError(f"test_release '{self.test_release}' not in releases {self.releases}")
            test_releases = [self.test_release]
            train_val_releases = [r for r in self.releases if r != self.test_release]
            logger.info(f"Using {self.test_release} as test set")
            logger.info(f"Train/val releases: {train_val_releases}")
        else:
            test_releases = []
            train_val_releases = self.releases
            logger.info("No test release specified - no test split")

        # Step 2: Query recordings from Level 1 cache
        train_val_recordings = self.cache_mgr.query_raw(
            dataset="hbn",
            releases=train_val_releases,
            tasks=self.all_tasks,
            mini=self.mini
        )

        if test_releases:
            test_recordings = self.cache_mgr.query_raw(
                dataset="hbn",
                releases=test_releases,
                tasks=self.all_tasks,
                mini=self.mini
            )
        else:
            test_recordings = pd.DataFrame()

        logger.info(f"Train/val recordings: {len(train_val_recordings)}")
        logger.info(f"Test recordings: {len(test_recordings)}")

        if len(train_val_recordings) == 0:
            raise ValueError("No train/val recordings found after filtering!")

        # Step 3: Subject-level split for train/val (DataFrame-based)
        train_val_subjects = train_val_recordings["subject"].unique()
        logger.info(f"Train/val unique subjects: {len(train_val_subjects)}")

        if self.val_split > 0 and len(train_val_subjects) > 1:
            train_subjects, val_subjects = train_test_split(
                train_val_subjects,
                test_size=self.val_split,
                random_state=self.seed
            )

            train_recordings = train_val_recordings[
                train_val_recordings["subject"].isin(train_subjects)
            ]
            val_recordings = train_val_recordings[
                train_val_recordings["subject"].isin(val_subjects)
            ]
        else:
            # No val split - use all for training, create dummy val for Lightning
            train_subjects = train_val_subjects
            val_subjects = np.array([])
            train_recordings = train_val_recordings
            val_recordings = train_val_recordings.head(min(10, len(train_val_recordings)))
            logger.info("No validation split - using all train/val subjects for training")

        if len(test_recordings) == 0:
            # No test data - create dummy test for Lightning
            test_recordings = train_recordings.head(min(10, len(train_recordings)))
            logger.info("No test split - using dummy test set")

        logger.info(
            f"\nSubject split:\n"
            f"  Train: {len(train_subjects)} subjects, {len(train_recordings)} recordings\n"
            f"  Val:   {len(val_subjects)} subjects, {len(val_recordings)} recordings\n"
            f"  Test:  {test_recordings['subject'].nunique()} subjects, {len(test_recordings)} recordings"
        )

        # Step 4: Build window cache once for ALL recordings, then split by filtering
        logger.info(f"\nCreating windowed datasets ({self.window_length}s windows, {self.stride}s stride)...")

        # Combine all recordings (train/val + test) for window cache building
        all_recordings = pd.concat([train_val_recordings, test_recordings], ignore_index=True)
        logger.info(f"Building window cache for {len(all_recordings)} recordings...")

        # Build/load window cache with ALL recordings
        # This creates a single Zarr array + metadata for all recordings
        full_dataset = self.cache_mgr.get_windowed_dataset(
            recordings=all_recordings,
            window_len_s=self.window_length,
            stride_s=self.stride,
            crop_len_s=None,  # No crop during cache building
            mode='train'
        )
        logger.info(f"Window cache ready: {len(full_dataset):,} total windows")

        # Split by filtering metadata (no data copying, just metadata filtering)
        # This allows flexible splitting by release and/or subject
        logger.info("Splitting dataset by subjects...")

        # Filter by subjects for train/val/test
        self.train_dataset = full_dataset.filter_by_recordings(train_recordings["recording_id"].tolist())
        self.train_dataset.crop_len_s = self.crop_length  # Apply temporal augmentation
        self.train_dataset.mode = 'train'

        self.val_dataset = full_dataset.filter_by_recordings(val_recordings["recording_id"].tolist())
        self.val_dataset.crop_len_s = self.crop_length
        self.val_dataset.mode = 'val'

        self.test_dataset = full_dataset.filter_by_recordings(test_recordings["recording_id"].tolist())
        self.test_dataset.crop_len_s = self.crop_length
        self.test_dataset.mode = 'val'

        logger.info(
            f"\nFinal window counts:\n"
            f"  Train: {len(self.train_dataset):,} windows\n"
            f"  Val:   {len(self.val_dataset):,} windows\n"
            f"  Test:  {len(self.test_dataset):,} windows"
        )

    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
