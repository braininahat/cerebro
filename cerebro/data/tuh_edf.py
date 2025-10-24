"""
TUH EEG Dataset with EDF backend and window caching.

This module provides a Lightning DataModule for loading TUH EEG data
directly from EDF files with preprocessing and window caching support.

Similar to HBNDataModule but adapted for TUH directory structure.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import lightning as L
import mne
import numpy as np
import pandas as pd
import torch
from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import Preprocessor, create_fixed_length_windows, preprocess
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Collate function that handles (x, y) tuples from CroppedWindowDataset.

    Args:
        batch: List of (x, y) tuples

    Returns:
        Tuple of (x_batch, y_batch) tensors
    """
    x_batch = torch.stack([item[0] for item in batch], dim=0)
    y_batch = torch.stack([item[1] for item in batch], dim=0)
    return x_batch, y_batch


class CroppedWindowDataset(Dataset):
    """Wraps a windows dataset with random or deterministic temporal cropping.

    Creates 2-second crops from 4-second windows, matching HBN preprocessing.
    Random crops during training, center crops for val/test.
    """

    def __init__(
        self,
        windows_dataset: BaseConcatDataset,
        crop_len: int,
        sfreq: int = 100,
        mode: str = "train",
    ):
        """
        Parameters
        ----------
        windows_dataset : BaseConcatDataset
            The input dataset of EEG windows (4-second windows).
        crop_len : int
            Crop length in samples (e.g., 2 * sfreq for 2-second crops).
        sfreq : int, optional
            Sampling frequency, default 100 Hz.
        mode : {"train", "val", "test"}
            Controls whether cropping is random or deterministic (center crop).
        """
        self.windows_dataset = windows_dataset
        self.crop_len = crop_len
        self.sfreq = sfreq
        self.mode = mode.lower()
        assert self.mode in {"train", "val", "test"}, f"Invalid mode: {self.mode}"

    def __len__(self):
        return len(self.windows_dataset.datasets)

    def __getitem__(self, idx):
        ds = self.windows_dataset.datasets[idx]
        x, y, _ = ds[0]  # (C, T)
        C, T = x.shape

        # Apply temporal cropping
        if T > self.crop_len:
            if self.mode == "train":
                # Random crop during training
                start = np.random.randint(0, T - self.crop_len + 1)
            else:
                # Deterministic crop (center) for val/test
                start = (T - self.crop_len) // 2
            x_crop = x[:, start:start + self.crop_len]
        else:
            # Pad if too short (shouldn't happen if we filter properly)
            x_crop = x[:, :self.crop_len]
            if x_crop.shape[1] < self.crop_len:
                pad = self.crop_len - x_crop.shape[1]
                x_crop = np.pad(x_crop, ((0, 0), (0, pad)), mode="constant")

        # Convert to tensors
        return torch.from_numpy(x_crop).float(), torch.tensor(y).float()


class TUHEDFDataset(BaseConcatDataset):
    """TUH EEG dataset loaded from EDF files.

    Scans TUH directory structure and loads EDF recordings into braindecode format.

    Directory structure:
        tuh_dir/
            edf/
                {numeric_group}/
                    {subject_id}/
                        {session_id}/
                            {montage}/
                                {recording_id}.edf

    Example: edf/000/aaaaaaaa/s001_2015/01_tcp_ar/aaaaaaaa_s001_t000.edf

    Args:
        tuh_dir: Root TUH directory containing edf/ subdirectory
        target_name: Target variable name (None for unsupervised)
        recording_ids: Specific recording paths to load (optional)
        subjects: Filter by subject IDs (optional)
        sessions: Filter by session IDs (optional)
        montages: Filter by montage type (optional)
        max_recordings: Maximum number of recordings to load (for testing)
    """

    def __init__(
        self,
        tuh_dir: str,
        target_name: Optional[str] = None,
        recording_ids: Optional[List[str]] = None,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        montages: Optional[List[str]] = None,
        max_recordings: Optional[int] = None,
    ):
        self.tuh_dir = Path(tuh_dir)
        self.target_name = target_name

        # Find all EDF files
        logger.info(f"Scanning TUH directory: {tuh_dir}")
        edf_files = self._find_edf_files(recording_ids, subjects, sessions, montages, max_recordings)
        logger.info(f"Found {len(edf_files)} EDF files")

        # Create BaseDataset objects
        datasets = []
        for edf_path, metadata in tqdm(edf_files, desc="Loading EDF files", unit="file"):
            try:
                ds = self._load_edf(edf_path, metadata)
                if ds is not None:
                    datasets.append(ds)
            except Exception as e:
                logger.warning(f"Failed to load {edf_path}: {e}")

        logger.info(f"Successfully loaded {len(datasets)} recordings")
        super().__init__(datasets)

    def _find_edf_files(
        self,
        recording_ids: Optional[List[str]],
        subjects: Optional[List[str]],
        sessions: Optional[List[str]],
        montages: Optional[List[str]],
        max_recordings: Optional[int],
    ) -> List[Tuple[Path, dict]]:
        """Find all EDF files matching filters."""
        edf_dir = self.tuh_dir / "edf"

        if not edf_dir.exists():
            raise FileNotFoundError(f"TUH EDF directory not found: {edf_dir}")

        edf_files = []

        # Find all .edf files recursively under edf/
        for edf_path in edf_dir.rglob("*.edf"):
            # Extract metadata from path
            # Path structure: .../edf/{numeric_group}/{subject_id}/{session_id}/{montage}/{recording}.edf
            # Example: edf/000/aaaaaaaa/s001_2015/01_tcp_ar/aaaaaaaa_s001_t000.edf
            try:
                parts = edf_path.relative_to(edf_dir).parts
                if len(parts) < 5:
                    continue

                # parts[0] = numeric grouping (e.g., '000')
                # parts[1] = subject_id (e.g., 'aaaaaaaa')
                # parts[2] = session_id (e.g., 's001_2015')
                # parts[3] = montage (e.g., '01_tcp_ar')
                # parts[4] = filename (e.g., 'aaaaaaaa_s001_t000.edf')
                subject_id = parts[1]
                session_id = parts[2]
                montage = parts[3]
                recording_name = edf_path.stem

                # Apply filters
                if recording_ids is not None and str(edf_path) not in recording_ids:
                    continue
                if subjects is not None and subject_id not in subjects:
                    continue
                if sessions is not None and session_id not in sessions:
                    continue
                if montages is not None and montage not in montages:
                    continue

                metadata = {
                    'subject_id': subject_id,
                    'session_id': session_id,
                    'recording_name': recording_name,
                    'montage': montage,
                    'file_path': str(edf_path),
                }

                edf_files.append((edf_path, metadata))

                if max_recordings is not None and len(edf_files) >= max_recordings:
                    return edf_files

            except Exception as e:
                logger.warning(f"Failed to parse path {edf_path}: {e}")
                continue

        return edf_files

    def _load_edf(self, edf_path: Path, metadata: dict) -> Optional[BaseDataset]:
        """Load single EDF file as BaseDataset."""
        try:
            # Load EDF with MNE
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)

            # Create description from metadata
            description = pd.Series(metadata)

            # Add dummy target if specified (for unsupervised learning)
            if self.target_name is not None:
                # Set dummy target value of 0 (no labels in TUH)
                description[self.target_name] = 0.0

            return BaseDataset(raw, description, target_name=self.target_name)

        except Exception as e:
            logger.warning(f"Failed to load {edf_path}: {e}")
            return None


class TUHEDFDataModule(L.LightningDataModule):
    """Lightning DataModule for TUH EEG dataset with EDF backend and caching.

    Loads EDF files directly, applies HBN-matching preprocessing, creates windows, and caches results.

    **Preprocessing Pipeline (Matching HBN)**:
    1. Load EDF files
    2. Filter out recordings < 4 seconds
    3. Bandpass filter: 0.5-50 Hz
    4. Resample to 100 Hz
    5. Create 4-second fixed-length windows with 2-second stride (2s overlap)
    6. Cache preprocessed windows
    7. Apply random 2-second crops from 4s windows in DataLoader (train: random, val/test: center)

    Args:
        tuh_dir: Root TUH directory containing edf/ subdirectory
        batch_size: Batch size for DataLoaders (default: 512)
        num_workers: Number of workers for DataLoaders (default: 8)
        target_name: Target variable name (None for unsupervised)
        train_ratio: Ratio of subjects for training (default: 0.8)
        val_ratio: Ratio of subjects for validation (default: 0.1)
        test_ratio: Ratio of subjects for test (default: 0.1)
        seed: Random seed for splits (default: 42)
        recording_ids: Specific recording paths to load (optional)
        subjects: Filter by subject IDs (optional)
        sessions: Filter by session IDs (optional)
        montages: Filter by montage type (optional, e.g., ['01_tcp_ar'])
        max_recordings: Maximum recordings to load (for testing, optional)
        window_size_s: Window size in seconds (default: 4.0, matching HBN)
        window_stride_s: Window stride in seconds (default: 2.0, matching HBN)
        crop_size_s: Final crop size in seconds (default: 2.0, matching HBN)
        sfreq: Target sampling frequency in Hz (default: 100.0, matching HBN)
        apply_bandpass: Apply bandpass filter (default: True, matching HBN)
        l_freq: Low frequency for bandpass (default: 0.5, matching HBN)
        h_freq: High frequency for bandpass (default: 50.0, matching HBN)
        min_recording_duration_s: Minimum recording duration (default: 4.0, matching HBN)
        cache_dir: Directory for caching windows (default: tuh_dir/cache)
        use_cache: Use cached windows if available (default: True)
        preprocessing_batch_size: Number of recordings to load at once during preprocessing (default: 100)
            Decrease to 50 or 20 if hitting OOM errors, increase to 200+ if memory allows for faster processing

    Example:
        >>> datamodule = TUHEDFDataModule(
        ...     tuh_dir='/path/to/tuh/tueg/v2.0.1',
        ...     montages=['01_tcp_ar'],
        ...     batch_size=256,
        ... )
        >>> datamodule.setup()
        >>> train_loader = datamodule.train_dataloader()
        >>> # Returns batches of shape (batch_size, 22, 200) - 2s @ 100Hz, 22 channels
    """

    def __init__(
        self,
        tuh_dir: str,
        batch_size: int = 512,
        num_workers: int = 8,
        target_name: Optional[str] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        recording_ids: Optional[List[str]] = None,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        montages: Optional[List[str]] = None,
        max_recordings: Optional[int] = None,
        window_size_s: float = 4.0,  # 4-second windows (matching HBN)
        window_stride_s: float = 2.0,  # 2-second stride = 2s overlap (matching HBN)
        crop_size_s: float = 2.0,  # Final crop size (matching HBN)
        sfreq: float = 100.0,  # Resample to 100Hz (matching HBN)
        apply_bandpass: bool = True,  # Apply bandpass by default (matching HBN)
        l_freq: float = 0.5,  # Low freq cutoff (matching HBN)
        h_freq: float = 50.0,  # High freq cutoff (matching HBN)
        min_recording_duration_s: float = 4.0,  # Filter out recordings < 4s
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        preprocessing_batch_size: int = 100,  # Number of recordings to process at once (OOM control)
    ):
        super().__init__()
        self.save_hyperparameters()

        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must sum to 1.0, "
                f"got {train_ratio + val_ratio + test_ratio}"
            )

        # Convert to Path
        self.tuh_dir = Path(tuh_dir)
        if not self.tuh_dir.exists():
            raise FileNotFoundError(f"TUH directory not found: {tuh_dir}")

        # Setup cache directory
        if cache_dir is None:
            self.cache_dir = self.tuh_dir / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated in setup()
        self.train_set = None
        self.val_set = None
        self.test_set = None

    @property
    def batch_size(self) -> int:
        """Batch size for DataLoaders (for Lightning compatibility)."""
        return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        """Update batch size (modified by Lightning's batch size scaler)."""
        self.hparams.batch_size = value

    def setup(self, stage: Optional[str] = None):
        """Setup datasets with preprocessing and caching.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict' (unused, loads all)
        """
        logger.info("\n" + "="*60)
        logger.info("[bold cyan]TUH EDF DATA SETUP[/bold cyan]")
        logger.info("="*60)

        # Create cache key from parameters
        cache_key = self._create_cache_key()
        cache_path = self.cache_dir / cache_key

        # Try loading from cache
        if self.hparams.use_cache and cache_path.exists():
            logger.info("[bold cyan]LOADING FROM CACHE[/bold cyan]")
            logger.info(f"[green]✓[/green] Loading cached windows from: {cache_key}")
            with open(cache_path, "rb") as f:
                windows = pickle.load(f)
            logger.info(f"[green]✓[/green] Loaded {len(windows)} windows from cache")
        else:
            # Cache miss - load and preprocess
            windows = self._load_and_preprocess()

            # Save to cache
            if self.hparams.use_cache:
                logger.info(f"[yellow]⚠[/yellow] Caching windows to: {cache_key}")
                with open(cache_path, "wb") as f:
                    pickle.dump(windows, f)
                logger.info(f"[green]✓[/green] Cached {len(windows)} windows")

        # Create subject-level splits
        self._create_splits(windows)

    def _create_cache_key(self) -> str:
        """Create cache key from preprocessing parameters."""
        params = [
            f"montage_{'-'.join(self.hparams.montages) if self.hparams.montages else 'all'}",
            f"win{int(self.hparams.window_size_s*10)}",
            f"stride{int(self.hparams.window_stride_s*10)}",
            f"crop{int(self.hparams.crop_size_s*10)}",
            f"sfreq{int(self.hparams.sfreq)}",
            f"bp{int(self.hparams.apply_bandpass)}",
            f"mindur{int(self.hparams.min_recording_duration_s)}",
            f"max{self.hparams.max_recordings or 'all'}",
        ]
        return "windows_" + "_".join(params) + ".pkl"

    def _load_and_preprocess(self) -> BaseConcatDataset:
        """Load EDF files in batches, preprocess, and create windows to avoid OOM."""
        logger.info("[bold cyan]LOADING EDF FILES (BATCH MODE)[/bold cyan]")

        # Load dataset metadata only (preload=False in TUHEDFDataset)
        dataset = TUHEDFDataset(
            tuh_dir=str(self.tuh_dir),
            target_name=self.hparams.target_name,
            recording_ids=self.hparams.recording_ids,
            subjects=self.hparams.subjects,
            sessions=self.hparams.sessions,
            montages=self.hparams.montages,
            max_recordings=self.hparams.max_recordings,
        )

        logger.info(f"[bold]Total recordings:[/bold] {len(dataset.datasets)}")

        # Filter by duration BEFORE loading (check times[-1] without preload)
        logger.info("\n[bold cyan]FILTERING BY DURATION (LAZY)[/bold cyan]")
        logger.info(f"Removing recordings < {self.hparams.min_recording_duration_s}s")

        filtered_datasets = []
        dropped_count = 0
        for ds in tqdm(dataset.datasets, desc="Filtering"):
            # This loads times only, not full data (cheap operation)
            duration = ds.raw.times[-1]
            if duration >= self.hparams.min_recording_duration_s:
                filtered_datasets.append(ds)
            else:
                dropped_count += 1

        logger.info(f"Dropped {dropped_count} recordings < {self.hparams.min_recording_duration_s}s")
        logger.info(f"Kept {len(filtered_datasets)} recordings")

        # Process in batches to avoid OOM
        batch_size = self.hparams.preprocessing_batch_size
        all_windows = []

        logger.info(f"\n[bold cyan]BATCH PROCESSING ({len(filtered_datasets)} recordings)[/bold cyan]")
        logger.info(f"Batch size: {batch_size} recordings (tune with preprocessing_batch_size parameter)")

        for batch_idx in range(0, len(filtered_datasets), batch_size):
            batch_end = min(batch_idx + batch_size, len(filtered_datasets))
            batch_datasets = filtered_datasets[batch_idx:batch_end]

            logger.info(f"\n[yellow]Batch {batch_idx // batch_size + 1}/{(len(filtered_datasets) + batch_size - 1) // batch_size}[/yellow] (recordings {batch_idx}-{batch_end-1})")

            # Create batch dataset
            batch_concat = BaseConcatDataset(batch_datasets)

            # Preload this batch only
            logger.info(f"  Loading {len(batch_datasets)} recordings into memory...")
            for ds in batch_datasets:
                ds.raw.load_data()

            # Apply preprocessing to batch
            logger.info(f"  Preprocessing...")
            preprocessors = []

            if self.hparams.apply_bandpass:
                preprocessors.append(
                    Preprocessor('filter', l_freq=self.hparams.l_freq, h_freq=self.hparams.h_freq)
                )

            preprocessors.append(
                Preprocessor('resample', sfreq=self.hparams.sfreq)
            )

            if preprocessors:
                preprocess(batch_concat, preprocessors, n_jobs=-1)

            # Create windows from batch
            logger.info(f"  Creating windows...")
            actual_sfreq = batch_concat.datasets[0].raw.info['sfreq']
            window_size_samples = int(self.hparams.window_size_s * actual_sfreq)
            window_stride_samples = int(self.hparams.window_stride_s * actual_sfreq)

            batch_windows = create_fixed_length_windows(
                batch_concat,
                window_size_samples=window_size_samples,
                window_stride_samples=window_stride_samples,
                drop_last_window=True,
                preload=True,
            )

            logger.info(f"  Created {len(batch_windows)} windows from batch")
            all_windows.extend(batch_windows.datasets)

            # Explicitly free memory
            del batch_concat
            for ds in batch_datasets:
                ds.raw._data = None  # Free loaded data
            import gc
            gc.collect()

        # Combine all windows
        windows = BaseConcatDataset(all_windows)

        logger.info(f"\n[bold green]✓ TOTAL: {len(windows)} 4-second windows[/bold green]")
        logger.info(f"[bold]Window size: {self.hparams.window_size_s}s[/bold]")
        logger.info(f"[bold]Window stride: {self.hparams.window_stride_s}s[/bold]")
        logger.info(f"[bold]Crop size: {self.hparams.crop_size_s}s (applied in DataLoader)[/bold]")
        return windows

    def _create_splits(self, windows: BaseConcatDataset):
        """Create subject-level train/val/test splits."""
        logger.info("\n[bold cyan]SPLITTING DATA[/bold cyan]")

        # Get metadata and unique subjects
        metadata = windows.get_metadata()
        subjects = metadata['subject_id'].unique()
        logger.info(f"[bold]Total subjects:[/bold] {len(subjects)}")

        # Set random state
        rng = check_random_state(self.hparams.seed)

        # Split: train / (val + test)
        train_subj, val_test_subj = train_test_split(
            subjects,
            test_size=(self.hparams.val_ratio + self.hparams.test_ratio),
            random_state=rng,
            shuffle=True
        )

        # Split: val / test
        val_subj, test_subj = train_test_split(
            val_test_subj,
            test_size=self.hparams.test_ratio / (self.hparams.val_ratio + self.hparams.test_ratio),
            random_state=check_random_state(self.hparams.seed + 1),
            shuffle=True
        )

        logger.info(f"Train subjects: {len(train_subj)}")
        logger.info(f"Val subjects: {len(val_subj)}")
        logger.info(f"Test subjects: {len(test_subj)}")

        # Create splits using subject filter
        subject_split = windows.split("subject_id")
        self.train_set = BaseConcatDataset(
            [subject_split[s] for s in train_subj if s in subject_split]
        )
        self.val_set = BaseConcatDataset(
            [subject_split[s] for s in val_subj if s in subject_split]
        )
        self.test_set = BaseConcatDataset(
            [subject_split[s] for s in test_subj if s in subject_split]
        )

        logger.info(f"\n[bold]Window counts:[/bold]")
        logger.info(f"  Train: {len(self.train_set)}")
        logger.info(f"  Val: {len(self.val_set)}")
        logger.info(f"  Test: {len(self.test_set)}")

    def train_dataloader(self):
        """Create training DataLoader with CroppedWindowDataset wrapper."""
        if self.train_set is None:
            raise RuntimeError("train_set is None. Did you call setup()?")

        # Wrap with CroppedWindowDataset for 2-second random crops
        crop_len = int(self.hparams.crop_size_s * self.hparams.sfreq)
        dataset = CroppedWindowDataset(
            self.train_set,
            crop_len=crop_len,
            sfreq=int(self.hparams.sfreq),
            mode="train",
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation DataLoader with CroppedWindowDataset wrapper."""
        if self.val_set is None:
            raise RuntimeError("val_set is None. Did you call setup()?")

        # Wrap with CroppedWindowDataset for 2-second center crops
        crop_len = int(self.hparams.crop_size_s * self.hparams.sfreq)
        dataset = CroppedWindowDataset(
            self.val_set,
            crop_len=crop_len,
            sfreq=int(self.hparams.sfreq),
            mode="val",
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test DataLoader with CroppedWindowDataset wrapper."""
        if self.test_set is None:
            raise RuntimeError("test_set is None. Did you call setup()?")

        # Wrap with CroppedWindowDataset for 2-second center crops
        crop_len = int(self.hparams.crop_size_s * self.hparams.sfreq)
        dataset = CroppedWindowDataset(
            self.test_set,
            crop_len=crop_len,
            sfreq=int(self.hparams.sfreq),
            mode="test",
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
