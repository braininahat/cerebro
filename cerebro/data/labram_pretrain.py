"""LaBraM Pretrain DataModule for masked EEG modeling.

This module prepares EEG data for LaBraM's masked token prediction pretraining.
It loads passive task recordings (movies, resting state, etc.) and prepares
windows in the format expected by MEMPretrainModule: [B, N, A, T]

Where:
- B = batch size
- N = number of EEG channels (129 for HBN)
- A = number of temporal patches per window
- T = samples per patch
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from eegdash import EEGChallengeDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from cerebro.data.unified_cache import UniversalCacheManager

logger = logging.getLogger(__name__)


class LaBraMWindowDataset(Dataset):
    """Wraps raw EEG data and creates windows in [N, A, T] format for LaBraM.

    Args:
        raw_list: List of MNE Raw objects
        window_len_samples: Window length in samples
        patch_size: Samples per patch (e.g., 100 for 1s patches @ 100Hz)
        stride_samples: Stride between windows in samples
    """

    def __init__(
        self,
        raw_list: list,
        window_len_samples: int,
        patch_size: int,
        stride_samples: int,
    ):
        self.raw_list = raw_list
        self.window_len_samples = window_len_samples
        self.patch_size = patch_size
        self.stride_samples = stride_samples

        # Create window index
        self.windows = []
        for raw_idx, raw in enumerate(raw_list):
            n_samples = raw.n_times
            n_windows = (n_samples - window_len_samples) // stride_samples + 1

            for win_idx in range(n_windows):
                start_sample = win_idx * stride_samples
                self.windows.append((raw_idx, start_sample))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """Return EEG window in shape [N, T_total] where N=channels, T_total=total_samples.

        Note: The tokenizer model handles patching internally via _chunk_to_patches().
        We return raw temporal data here.
        """
        raw_idx, start_sample = self.windows[idx]
        raw = self.raw_list[raw_idx]

        # Extract window
        end_sample = start_sample + self.window_len_samples
        data, _ = raw[:, start_sample:end_sample]  # [N, window_len_samples]

        # Force copy to avoid memory sharing issues with pickled Raw objects
        data = data.copy()

        # Convert to tensor - let tokenizer handle patching
        x = torch.from_numpy(np.array(data, dtype=np.float32))  # [N, T_total]

        return x


class LaBraMPretrainDataModule(L.LightningDataModule):
    """DataModule for LaBraM masked EEG modeling pretraining.

    Args:
        data_dir: Root directory containing EEG data
        releases: List of release names to use (e.g., ["R1", "R2", ...])
        passive_tasks: List of passive task names (movies, resting state, etc.)
        batch_size: Batch size for DataLoader
        num_workers: Number of parallel data loading workers
        window_len: Length of EEG windows in seconds
        patch_size: Samples per patch (e.g., 100 for 1s @ 100Hz)
        sfreq: Sampling frequency in Hz
        use_mini: If True, use mini dataset for prototyping
        cache_dir: Directory for caching preprocessed data
        val_frac: Fraction of data for validation
        test_frac: Fraction of data for testing
        seed: Random seed for reproducibility
        excluded_subjects: List of subject IDs to exclude
        n_channels: Expected number of EEG channels
    """

    def __init__(
        self,
        data_dir: str = "data",
        releases: Optional[List[str]] = None,
        passive_tasks: Optional[List[str]] = None,
        batch_size: int = 512,
        num_workers: int = 8,
        window_len: float = 2.0,
        patch_size: int = 100,
        sfreq: int = 100,
        use_mini: bool = False,
        cache_dir: Optional[str] = None,
        test_release: Optional[str] = None,  # Specific release for test (e.g., "R5")
        val_frac: float = 0.1,
        seed: int = 2025,
        excluded_subjects: Optional[List[str]] = None,
        n_channels: int = 129,
        **kwargs  # Accept and ignore other HBN parameters
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir).resolve()  # Absolute path for EEGChallengeDataset
        self.releases = releases or ["R1"]
        self.passive_tasks = passive_tasks or ["restingState"]
        self.num_workers = num_workers
        self.window_len = window_len
        self.patch_size = patch_size
        self.sfreq = sfreq
        self.use_mini = use_mini
        self.test_release = test_release
        self.val_frac = val_frac
        self.seed = seed
        self.excluded_subjects = excluded_subjects or []
        self.n_channels = n_channels

        # Cache directory
        if cache_dir:
            cache_root = Path(cache_dir)
        else:
            cache_root = self.data_dir / "cache"

        # Initialize universal cache manager (two-level caching)
        self.cache_mgr = UniversalCacheManager(
            cache_root=str(cache_root),
            preprocessing_params={
                "sfreq": self.sfreq,
                "bandpass": None,  # No bandpass filtering for LaBraM
                "n_channels": self.n_channels,
                "standardize": False,
            }
        )

        # Window parameters
        self.window_len_samples = int(window_len * sfreq)
        self.stride_samples = self.window_len_samples  # Non-overlapping windows

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

    def setup(self, stage: Optional[str] = None):
        """Load and prepare data using universal two-level cache with subject-level splitting."""
        logger.info(f"Loading data from releases: {self.releases}")
        logger.info(f"Tasks: {self.passive_tasks}")
        logger.info(f"Mini mode: {self.use_mini}")
        logger.info(f"Test release: {self.test_release}")

        # Build/load Level 1 raw cache for ALL releases
        self.cache_mgr.build_raw(
            dataset="hbn",
            releases=self.releases,
            tasks=self.passive_tasks,
            mini=self.use_mini
        )

        # Step 1: Pop test release from releases list
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

        # Step 2: Query recordings separately for train/val and test
        train_val_recordings = self.cache_mgr.query_raw(
            dataset="hbn",
            releases=train_val_releases,
            tasks=self.passive_tasks,
            mini=self.use_mini
        )

        if test_releases:
            test_recordings = self.cache_mgr.query_raw(
                dataset="hbn",
                releases=test_releases,
                tasks=self.passive_tasks,
                mini=self.use_mini
            )
        else:
            test_recordings = pd.DataFrame()

        logger.info(f"Train/val recordings: {len(train_val_recordings)}")
        logger.info(f"Test recordings: {len(test_recordings)}")

        # Filter excluded subjects
        if self.excluded_subjects:
            train_val_recordings = train_val_recordings[
                ~train_val_recordings["subject"].isin(self.excluded_subjects)
            ]
            if len(test_recordings) > 0:
                test_recordings = test_recordings[
                    ~test_recordings["subject"].isin(self.excluded_subjects)
                ]
            logger.info(f"After excluding subjects: {len(train_val_recordings)} train/val, {len(test_recordings)} test")

        if len(train_val_recordings) == 0:
            raise ValueError("No train/val recordings found after filtering!")

        # Step 3: Subject-level split for train/val
        train_val_subjects = train_val_recordings["subject"].unique()
        logger.info(f"Train/val unique subjects: {len(train_val_subjects)}")

        if self.val_frac > 0 and len(train_val_subjects) > 1:
            train_subjects, val_subjects = train_test_split(
                train_val_subjects,
                test_size=self.val_frac,
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

        # Get windowed datasets from Level 2 cache (builds if missing)
        window_len_s = self.window_len_samples / self.sfreq
        stride_s = self.stride_samples / self.sfreq

        logger.info(f"\nCreating windowed datasets ({window_len_s}s windows, {stride_s}s stride)...")

        self.train_dataset = self.cache_mgr.get_windowed_dataset(
            recordings=train_recordings,
            window_len_s=window_len_s,
            stride_s=stride_s,
            mode='train'
        )

        self.val_dataset = self.cache_mgr.get_windowed_dataset(
            recordings=val_recordings,
            window_len_s=window_len_s,
            stride_s=stride_s,
            mode='val'
        )

        self.test_dataset = self.cache_mgr.get_windowed_dataset(
            recordings=test_recordings,
            window_len_s=window_len_s,
            stride_s=stride_s,
            mode='val'
        )

        logger.info(
            f"\nFinal window counts:\n"
            f"  Train: {len(self.train_dataset)} windows\n"
            f"  Val:   {len(self.val_dataset)} windows\n"
            f"  Test:  {len(self.test_dataset)} windows"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
