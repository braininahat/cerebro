"""JEPA Phase 1 pretraining data module.

Loads ALL tasks (CCD, movies, resting state, etc.) without labels.
No trial filtering - pure self-supervised learning on raw EEG.

From magnum_opus.md Phase 1:
- Use all available tasks
- 4s windows with random 2s crops (augmentation)
- Mix all tasks to learn general representations
"""

from pathlib import Path
from typing import Literal, Optional
import pickle
import os

import lightning as L
import mne
import numpy as np
import torch
from eegdash import EEGChallengeDataset
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from joblib import Parallel, delayed


class RawEEGDataset(Dataset):
    """Dataset of raw EEG windows without labels.

    Loads EEG recordings and creates fixed-length windows.
    No task-specific logic - just raw EEG for SSL.

    Args:
        raw_list: List of MNE Raw objects
        window_length: Window length in seconds (default: 4s)
        stride: Stride between windows in seconds (default: 2s)
        crop_length: Random crop length in seconds (default: 2s, None = no crop)
        sfreq: Target sampling frequency (default: 100 Hz)
        n_chans_select: Number of channels to select (default: 129)
            Note: HBN dataset has 129 channels including Cz reference (per startkit)
    """

    def __init__(
        self,
        raw_list: list[mne.io.BaseRaw],
        window_length: float = 4.0,
        stride: float = 2.0,
        crop_length: Optional[float] = 2.0,
        sfreq: int = 100,
        n_chans_select: int = 129,
    ):
        self.raw_list = raw_list
        self.window_length = window_length
        self.stride = stride
        self.crop_length = crop_length
        self.sfreq = sfreq
        self.n_chans_select = n_chans_select

        # Window parameters
        self.window_samples = int(window_length * sfreq)
        self.stride_samples = int(stride * sfreq)
        if crop_length is not None:
            self.crop_samples = int(crop_length * sfreq)
        else:
            self.crop_samples = None

        # Create window index
        self.windows = []
        # Only show progress bar for large datasets (>100 recordings)
        raw_iterator = tqdm(raw_list, desc="Indexing windows", unit="recording", disable=len(raw_list) < 100)
        for raw_idx, raw in enumerate(raw_iterator):
            n_samples = raw.n_times
            n_windows = (n_samples - self.window_samples) // self.stride_samples + 1

            for win_idx in range(n_windows):
                start_sample = win_idx * self.stride_samples
                self.windows.append((raw_idx, start_sample))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a window of EEG data.

        Returns:
            EEG tensor (n_chans, n_times)
        """
        raw_idx, start_sample = self.windows[idx]
        raw = self.raw_list[raw_idx]

        # Extract window
        end_sample = start_sample + self.window_samples
        data, _ = raw[:, start_sample:end_sample]  # (n_chans, window_samples)

        # CRITICAL: Force numpy copy to avoid memory sharing with pickled Raw objects
        # Without this, collation with persistent_workers fails with:
        # "RuntimeError: Trying to resize storage that is not resizable"
        data = data.copy()

        # Select subset of channels (e.g., first 128)
        # Note: Some recordings may have fewer channels than n_chans_select
        # Handle both cases: too many channels (slice) and too few channels (pad with zeros)
        if data.shape[0] > self.n_chans_select:
            # Too many channels: select first n_chans_select
            data = data[:self.n_chans_select, :]
        elif data.shape[0] < self.n_chans_select:
            # Too few channels: pad with zeros to match expected shape
            # This ensures all tensors in a batch have the same shape
            pad_width = ((0, self.n_chans_select - data.shape[0]), (0, 0))
            data = np.pad(data, pad_width, mode='constant', constant_values=0)

        # Random crop if specified (data augmentation)
        if self.crop_samples is not None and self.crop_samples < data.shape[1]:
            max_start = data.shape[1] - self.crop_samples
            crop_start = np.random.randint(0, max_start + 1)
            data = data[:, crop_start:crop_start + self.crop_samples]

        # Convert to tensor with explicit memory copy
        # CRITICAL: Use np.array() to force a final copy before torch conversion
        # Even though we called .copy() earlier, subsequent numpy slicing operations
        # (channel selection, cropping) create views. np.array() ensures the final
        # data is contiguous and independent before PyTorch conversion.
        eeg = torch.from_numpy(np.array(data, dtype=np.float32))

        return eeg


class JEPAPretrainDataModule(L.LightningDataModule):
    """Data module for JEPA Phase 1 self-supervised pretraining.

    Loads ALL tasks from HBN dataset without labels.
    Creates fixed-length windows from all available recordings.

    Args:
        data_dir: Root directory for EEG data
        releases: List of release IDs (default: all except R5)
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        window_length: Window length in seconds (default: 4s)
        stride: Stride between windows (default: 2s)
        crop_length: Random crop length (default: 2s for 200 samples @ 100Hz)
        val_split: Validation split fraction (default: 0.1)
        test_split: Test split fraction (default: 0.1)
        n_chans_select: Number of channels to use (default: 129)
        sfreq: Target sampling frequency (default: 100 Hz)
        mini: If True, use mini dataset for fast prototyping
        all_tasks: List of tasks to include (default: all available)
            Options: ["contrastChangeDetection", "restingState",
                     "DespicableMe", "ThePresent", "DiaryOfAWimpyKid",
                     "FunwithFractals", "surroundSupp"]

    Example:
        >>> dm = JEPAPretrainDataModule(
        ...     data_dir="./data",
        ...     releases=["R1", "R2", "R3", "R4"],
        ...     batch_size=128
        ... )
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        releases: Optional[list[str]] = None,
        batch_size: int = 128,
        num_workers: int = 8,
        window_length: float = 4.0,
        stride: float = 2.0,
        crop_length: float = 2.0,
        val_split: float = 0.1,
        test_split: float = 0.1,
        n_chans_select: int = 129,
        sfreq: int = 100,
        mini: bool = False,
        all_tasks: Optional[list[str]] = None,
    ):
        super().__init__()

        # Save all hyperparameters for Lightning compatibility
        # This enables batch_size property delegation and checkpoint saving
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.num_workers = num_workers

        # Default: all releases except R5 (competition validation)
        if releases is None:
            releases = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"]
        self.releases = releases

        # Validate R5 not in training releases
        if "R5" in self.releases:
            raise ValueError(
                "R5 is the competition validation set and should NEVER be in training releases. "
                f"Got releases={self.releases}"
            )

        # Window parameters
        self.window_length = window_length
        self.stride = stride
        self.crop_length = crop_length
        self.val_split = val_split
        self.test_split = test_split
        self.n_chans_select = n_chans_select
        self.sfreq = sfreq
        self.mini = mini

        # Tasks to include (default: all)
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

        # Cache directory
        self.cache_dir = self.data_dir / "cache" / "jepa_pretrain"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def batch_size(self) -> int:
        """Batch size for DataLoaders.

        Property accessor for Lightning's batch size scaler compatibility.
        The actual value is stored in self.hparams.batch_size.
        """
        return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        """Update batch size (modified by Lightning's batch size scaler)."""
        self.hparams.batch_size = value

    def _create_cache_key(self) -> str:
        """Create cache key from data configuration."""
        import hashlib

        releases_str = "_".join(self.releases)
        tasks_str = "_".join(sorted(self.all_tasks))
        tasks_hash = hashlib.md5(tasks_str.encode()).hexdigest()[:8]

        cache_key = f"jepa_raw_{releases_str}_sfreq{self.sfreq}_tasks{tasks_hash}_mini{self.mini}.pkl"
        return cache_key

    def _get_cache_path(self) -> Path:
        """Get path to cache file."""
        return self.cache_dir / self._create_cache_key()

    def prepare_data(self):
        """Download data if needed."""
        # EEGChallengeDataset handles downloading automatically
        pass

    def setup(self, stage: Optional[str] = None):
        """Load and split data."""
        print(f"Loading data from releases: {self.releases}")
        print(f"Tasks: {self.all_tasks}")
        print(f"Mini mode: {self.mini}")
        print()

        cache_path = self._get_cache_path()

        # Try loading from cache
        if cache_path.exists():
            print(f"Loading from cache: {cache_path.name}")
            with open(cache_path, "rb") as f:
                all_raws = pickle.load(f)
            print(f"✓ Loaded {len(all_raws)} recordings from cache\n")
        else:
            # Load datasets from scratch
            all_datasets_list = []

            for release in self.releases:
                print(f"  Loading {release}...", end="", flush=True)
                try:
                    dataset = EEGChallengeDataset(
                        release=release,
                        cache_dir=str(self.data_dir),
                        mini=self.mini,
                        query={'task': self.all_tasks}  # Load all tasks at once
                    )
                    all_datasets_list.append(dataset)
                    print(f" ✓ {len(dataset.datasets)} recordings")
                except Exception as e:
                    print(f" ✗ {type(e).__name__}: {str(e)[:60]}")
                    continue

            if len(all_datasets_list) == 0:
                raise ValueError("No datasets loaded! Check releases and tasks.")

            # Extract all Raw objects (parallelized)
            print(f"\nExtracting Raw objects from {len(all_datasets_list)} datasets...")

            # Flatten datasets into list of individual recordings
            all_dataset_items = []
            for dataset_obj in all_datasets_list:
                all_dataset_items.extend(dataset_obj.datasets)

            print(f"Total raw recordings to process: {len(all_dataset_items)}")

            # Define processing function
            def extract_and_resample_raw(ds, target_sfreq):
                """Extract raw and resample if needed."""
                raw = ds.raw
                if raw.info["sfreq"] != target_sfreq:
                    raw.resample(target_sfreq)
                return raw

            # Parallel processing with progress bar
            n_jobs = os.cpu_count()
            print(f"Using {n_jobs} CPU cores for parallel extraction...")
            all_raws = Parallel(n_jobs=n_jobs)(
                delayed(extract_and_resample_raw)(ds, self.sfreq)
                for ds in tqdm(all_dataset_items, desc="Extracting & resampling", unit="recording")
            )

            # Save to cache
            print(f"\n⚠ Caching to: {cache_path.name}")
            with open(cache_path, "wb") as f:
                pickle.dump(all_raws, f)
            print(f"⚠ Delete cache if config changes: rm {cache_path}")
            print(f"Total recordings: {len(all_raws)}\n")

        if len(all_raws) == 0:
            raise ValueError("No recordings found! Check releases and tasks.")

        # Split into train/val/test at SUBJECT level (not recording level)
        # This prevents data leakage - same subject cannot appear in multiple splits

        # Group recordings by subject
        subject_to_indices = {}
        for idx, raw in enumerate(all_raws):
            subject_id = raw.info['subject_info']['his_id']
            if subject_id not in subject_to_indices:
                subject_to_indices[subject_id] = []
            subject_to_indices[subject_id].append(idx)

        unique_subjects = list(subject_to_indices.keys())
        n_subjects = len(unique_subjects)

        print(f"Subject-level splitting:")
        print(f"  Total recordings: {len(all_raws)}")
        print(f"  Unique subjects: {n_subjects}")
        print(f"  Recordings per subject: {len(all_raws) / n_subjects:.1f} average")

        # Split subjects (not recordings) into train/val/test
        n_val_subj = int(n_subjects * self.val_split)
        n_test_subj = int(n_subjects * self.test_split)
        n_train_subj = n_subjects - n_val_subj - n_test_subj

        # Random shuffle subjects for splitting
        rng = np.random.RandomState(42)
        shuffled_subjects = rng.permutation(unique_subjects)

        train_subjects = shuffled_subjects[:n_train_subj]
        val_subjects = shuffled_subjects[n_train_subj:n_train_subj + n_val_subj]
        test_subjects = shuffled_subjects[n_train_subj + n_val_subj:]

        # Collect all recordings for each subject group
        train_indices = [idx for subj in train_subjects for idx in subject_to_indices[subj]]
        val_indices = [idx for subj in val_subjects for idx in subject_to_indices[subj]]
        test_indices = [idx for subj in test_subjects for idx in subject_to_indices[subj]]

        print(f"  Train: {n_train_subj} subjects → {len(train_indices)} recordings")
        print(f"  Val:   {n_val_subj} subjects → {len(val_indices)} recordings")
        print(f"  Test:  {n_test_subj} subjects → {len(test_indices)} recordings")

        # Create datasets
        train_raws = [all_raws[i] for i in train_indices]
        val_raws = [all_raws[i] for i in val_indices]
        test_raws = [all_raws[i] for i in test_indices]

        self.train_dataset = RawEEGDataset(
            train_raws,
            window_length=self.window_length,
            stride=self.stride,
            crop_length=self.crop_length,  # Random crop for augmentation
            sfreq=self.sfreq,
            n_chans_select=self.n_chans_select,
        )

        self.val_dataset = RawEEGDataset(
            val_raws,
            window_length=self.window_length,
            stride=self.stride,
            crop_length=self.crop_length,  # Fixed crop
            sfreq=self.sfreq,
            n_chans_select=self.n_chans_select,
        )

        self.test_dataset = RawEEGDataset(
            test_raws,
            window_length=self.window_length,
            stride=self.stride,
            crop_length=self.crop_length,
            sfreq=self.sfreq,
            n_chans_select=self.n_chans_select,
        )

        # Summary
        print(f"\n{'='*60}")
        print(f"Data loading complete:")
        print(f"  Train: {len(train_raws)} recordings → {len(self.train_dataset):,} windows")
        print(f"  Val:   {len(val_raws)} recordings → {len(self.val_dataset):,} windows")
        print(f"  Test:  {len(test_raws)} recordings → {len(self.test_dataset):,} windows")
        print(f"{'='*60}\n")

    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
