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
import torch
from eegdash import EEGChallengeDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

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
        """Return EEG window in shape [N, A, T] where N=channels, A=patches, T=samples_per_patch."""
        raw_idx, start_sample = self.windows[idx]
        raw = self.raw_list[raw_idx]

        # Extract window
        end_sample = start_sample + self.window_len_samples
        data, _ = raw[:, start_sample:end_sample]  # [N, window_len_samples]

        # Force copy to avoid memory sharing issues with pickled Raw objects
        data = data.copy()

        N, total_T = data.shape
        A = total_T // self.patch_size  # Number of patches

        # Reshape to [N, A, T] where T = patch_size
        if total_T % self.patch_size != 0:
            # Trim to fit exact number of patches
            data = data[:, : A * self.patch_size]

        # Reshape and convert to tensor
        data = data.reshape(N, A, self.patch_size)  # [N, A, T]
        x = torch.from_numpy(np.array(data, dtype=np.float32))

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
        val_frac: float = 0.1,
        test_frac: float = 0.1,
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
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.excluded_subjects = excluded_subjects or []
        self.n_channels = n_channels

        # Cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.data_dir / "cache" / "labram_pretrain"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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

    def _create_cache_key(self) -> str:
        """Create cache key from data configuration."""
        releases_str = "_".join(self.releases)
        tasks_str = "_".join(sorted(self.passive_tasks))
        tasks_hash = hashlib.md5(tasks_str.encode()).hexdigest()[:8]

        cache_key = (
            f"labram_raw_{releases_str}_"
            f"sfreq{self.sfreq}_"
            f"tasks{tasks_hash}_"
            f"mini{self.use_mini}.pkl"
        )
        return cache_key

    def _get_cache_path(self) -> Path:
        """Get path to cache file."""
        return self.cache_dir / self._create_cache_key()

    def setup(self, stage: Optional[str] = None):
        """Load and prepare data."""
        logger.info(f"Loading data from releases: {self.releases}")
        logger.info(f"Tasks: {self.passive_tasks}")
        logger.info(f"Mini mode: {self.use_mini}")

        cache_path = self._get_cache_path()

        # Try loading from cache FIRST
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path.name}")
            with open(cache_path, "rb") as f:
                all_raws = pickle.load(f)
            logger.info(f"✓ Loaded {len(all_raws)} recordings from cache")
        else:
            # Load datasets from scratch - ONE call per release using query
            all_datasets_list = []

            for release in self.releases:
                logger.info(f"  Loading {release}...")
                try:
                    dataset = EEGChallengeDataset(
                        release=release,
                        cache_dir=str(self.data_dir),
                        mini=self.use_mini,
                        query={"task": self.passive_tasks},  # Load all tasks at once!
                    )
                    all_datasets_list.append(dataset)
                    logger.info(f"  ✓ {release}: {len(dataset.datasets)} recordings")
                except Exception as e:
                    logger.warning(f"  ✗ {release}: {type(e).__name__}: {str(e)[:60]}")
                    continue

            if not all_datasets_list:
                raise ValueError("No datasets loaded successfully!")

            # Flatten all datasets
            all_datasets = []
            for dataset_list in all_datasets_list:
                all_datasets.extend(dataset_list.datasets)

            logger.info(f"Total recordings: {len(all_datasets)}")

            # Filter excluded subjects and quality checks
            filtered_raws = []
            for ds in all_datasets:
                subj = ds.description.get("subject", "")
                if subj in self.excluded_subjects:
                    continue

                try:
                    raw = ds.raw
                    if len(raw.ch_names) != self.n_channels:
                        continue
                    filtered_raws.append(raw)
                except Exception:
                    continue

            logger.info(f"After filtering: {len(filtered_raws)} recordings")

            # Cache raw objects
            with open(cache_path, "wb") as f:
                pickle.dump(filtered_raws, f)
            logger.info(f"Cached raw objects to {cache_path}")

            all_raws = filtered_raws

        # Split at recording level (subject-level would be better but requires metadata)
        indices = list(range(len(all_raws)))

        # Train/val/test split
        test_size = self.test_frac
        val_size = self.val_frac / (1 - test_size)

        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=self.seed
        )
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size, random_state=self.seed
        )

        # Create datasets
        train_raws = [all_raws[i] for i in train_idx]
        val_raws = [all_raws[i] for i in val_idx]
        test_raws = [all_raws[i] for i in test_idx]

        self.train_dataset = LaBraMWindowDataset(
            train_raws, self.window_len_samples, self.patch_size, self.stride_samples
        )
        self.val_dataset = LaBraMWindowDataset(
            val_raws, self.window_len_samples, self.patch_size, self.stride_samples
        )
        self.test_dataset = LaBraMWindowDataset(
            test_raws, self.window_len_samples, self.patch_size, self.stride_samples
        )

        logger.info(
            f"Split: train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, "
            f"test={len(self.test_dataset)}"
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
