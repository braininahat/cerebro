"""
TUH EEG Dataset with EDF backend and Zarr-based window caching.

This module provides a Lightning DataModule for loading TUH EEG data
directly from EDF files with preprocessing and window caching support.

**KEY FEATURES**:
- Zarr-based storage: Incremental checkpointing, no OOM on large datasets
- Lazy loading: Only load windows needed for current batch
- Adaptive parallelization: Uses max CPU cores safely with memory monitoring
- Robust resuming: Parquet manifest tracks progress, resume from any point
- Memory-efficient: Processes 1.7TB dataset on 427GB RAM without crashes

Similar to HBNDataModule but optimized for massive TUH dataset (1.7TB).
"""

import gc
import json
import logging
import multiprocessing as mp
import os
import pickle
import psutil
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import warnings

import lightning as L
import mne
import numpy as np
import pandas as pd
import torch
import zarr
from numcodecs import Blosc
from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import Preprocessor, create_fixed_length_windows, preprocess
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================================
# RESOURCE MANAGEMENT UTILITIES
# ============================================================================

def get_memory_usage_gb():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def get_available_memory_gb():
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)


def get_optimal_worker_count(memory_budget_gb=340, per_worker_memory_gb=1.0):
    """Calculate optimal number of workers based on CPU and memory constraints.

    Args:
        memory_budget_gb: Maximum memory to use (default: 80% of 427GB = 340GB)
        per_worker_memory_gb: Estimated memory per worker (default: 1GB)

    Returns:
        int: Optimal worker count
    """
    cpu_count = os.cpu_count() or 1
    # Reserve 10% of CPUs for system
    max_cpu_workers = int(cpu_count * 0.9)

    # Calculate memory-constrained workers
    available_mem = get_available_memory_gb()
    max_mem_workers = int(min(memory_budget_gb, available_mem * 0.8) / per_worker_memory_gb)

    # Take minimum (most conservative)
    optimal = min(max_cpu_workers, max_mem_workers)
    return max(1, optimal)  # At least 1 worker


class AdaptiveWorkerManager:
    """Dynamically adjusts worker count based on memory pressure.

    Monitors memory usage and scales down workers if approaching limits.
    """

    def __init__(self, initial_workers: int, memory_threshold_gb: float = 340):
        self.initial_workers = initial_workers
        self.current_workers = initial_workers
        self.memory_threshold = memory_threshold_gb
        self.check_interval = 10  # Check every 10 tasks
        self.task_count = 0

    def should_reduce_workers(self) -> bool:
        """Check if we should reduce worker count due to memory pressure."""
        mem_used = get_memory_usage_gb()
        mem_available = get_available_memory_gb()

        # Reduce if using >80% of threshold OR <20GB available
        return mem_used > (self.memory_threshold * 0.8) or mem_available < 20

    def get_worker_count(self) -> int:
        """Get current recommended worker count."""
        self.task_count += 1

        # Check memory periodically
        if self.task_count % self.check_interval == 0:
            if self.should_reduce_workers() and self.current_workers > 1:
                self.current_workers = max(1, self.current_workers // 2)
                logger.warning(
                    f"[yellow]Memory pressure detected. Reducing workers: "
                    f"{self.current_workers * 2} → {self.current_workers}[/yellow]"
                )

        return self.current_workers


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class CheckpointManager:
    """Manages Zarr checkpoint progress with Parquet manifest.

    Tracks which recordings have been processed and enables resuming.
    """

    def __init__(self, cache_dir: Path, cache_key: str):
        self.cache_dir = Path(cache_dir)
        self.cache_key = cache_key
        self.manifest_path = self.cache_dir / f"{cache_key}_manifest.parquet"
        self.lock_path = self.cache_dir / f"{cache_key}.lock"
        self.manifest = None

    def load_manifest(self) -> pd.DataFrame:
        """Load existing manifest or create new one."""
        if self.manifest_path.exists():
            logger.info(f"[green]✓[/green] Loading existing manifest: {self.manifest_path.name}")
            self.manifest = pd.read_parquet(self.manifest_path)
            return self.manifest
        else:
            logger.info("[yellow]No manifest found. Starting fresh.[/yellow]")
            self.manifest = pd.DataFrame(columns=[
                'recording_id', 'subject_id', 'session_id', 'montage',
                'file_path', 'num_windows', 'status', 'error_msg', 'timestamp'
            ])
            return self.manifest

    def is_processed(self, recording_path: str) -> bool:
        """Check if recording has already been processed."""
        if self.manifest is None:
            return False
        return recording_path in self.manifest['file_path'].values

    def mark_completed(self, recording_path: str, metadata: dict, num_windows: int):
        """Mark recording as completed in manifest."""
        new_row = pd.DataFrame([{
            'recording_id': len(self.manifest),
            'subject_id': metadata.get('subject_id', ''),
            'session_id': metadata.get('session_id', ''),
            'montage': metadata.get('montage', ''),
            'file_path': str(recording_path),  # Ensure string for parquet compatibility
            'num_windows': num_windows,
            'status': 'completed',
            'error_msg': '',
            'timestamp': pd.Timestamp.now()
        }])
        self.manifest = pd.concat([self.manifest, new_row], ignore_index=True)

    def mark_failed(self, recording_path: str, metadata: dict, error_msg: str):
        """Mark recording as failed in manifest."""
        new_row = pd.DataFrame([{
            'recording_id': len(self.manifest),
            'subject_id': metadata.get('subject_id', ''),
            'session_id': metadata.get('session_id', ''),
            'montage': metadata.get('montage', ''),
            'file_path': str(recording_path),  # Ensure string for parquet compatibility
            'num_windows': 0,
            'status': 'failed',
            'error_msg': str(error_msg)[:500],  # Truncate long errors
            'timestamp': pd.Timestamp.now()
        }])
        self.manifest = pd.concat([self.manifest, new_row], ignore_index=True)

    def save_manifest(self):
        """Save manifest to disk."""
        # Convert Path objects to strings for parquet serialization
        manifest_copy = self.manifest.copy()
        if 'file_path' in manifest_copy.columns:
            manifest_copy['file_path'] = manifest_copy['file_path'].astype(str)
        manifest_copy.to_parquet(self.manifest_path, index=False)

    def get_progress_summary(self) -> dict:
        """Get summary of processing progress."""
        if self.manifest is None or len(self.manifest) == 0:
            return {'total': 0, 'completed': 0, 'failed': 0, 'remaining': 0}

        total = len(self.manifest)
        completed = (self.manifest['status'] == 'completed').sum()
        failed = (self.manifest['status'] == 'failed').sum()

        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'remaining': total - completed - failed
        }


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


# ============================================================================
# LAZY ZARR-BASED DATASET (MEMORY-EFFICIENT)
# ============================================================================

class LazyZarrWindowDataset(Dataset):
    """Lazy-loading dataset that reads windows from Zarr on-demand.

    Only loads windows needed for current batch, dramatically reducing RAM usage.
    Compatible with PyTorch DataLoader (thread-safe reads).

    Parameters
    ----------
    zarr_path : str or Path
        Path to Zarr array containing windows (shape: num_windows × channels × time)
    metadata : pd.DataFrame
        Metadata for each window (subjects, labels, etc.)
    crop_len : int
        Crop length in samples for temporal cropping (e.g., 200 for 2s @ 100Hz)
    sfreq : int
        Sampling frequency (default: 100 Hz)
    mode : {"train", "val", "test"}
        Controls random vs deterministic cropping
    target_col : str
        Column name in metadata for target labels (default: "target")
    """

    def __init__(
        self,
        zarr_path: Path,
        metadata: pd.DataFrame,
        crop_len: int,
        sfreq: int = 100,
        mode: str = "train",
        target_col: str = "target",
    ):
        self.zarr_array = zarr.open(str(zarr_path), mode='r')  # Read-only
        self.metadata = metadata.reset_index(drop=True)
        self.crop_len = crop_len
        self.sfreq = sfreq
        self.mode = mode.lower()
        self.target_col = target_col

        assert self.mode in {"train", "val", "test"}, f"Invalid mode: {self.mode}"
        assert len(self.metadata) == self.zarr_array.shape[0], (
            f"Metadata length ({len(self.metadata)}) != zarr length ({self.zarr_array.shape[0]})"
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Map dataframe index to Zarr array index
        if 'zarr_index' in self.metadata.columns:
            zarr_idx = int(self.metadata.iloc[idx]['zarr_index'])
        else:
            zarr_idx = idx  # Fallback for full dataset

        # Lazy load: Read ONLY this window from disk
        x = self.zarr_array[zarr_idx]  # Shape: (channels, time)
        C, T = x.shape

        # Apply temporal cropping (same logic as CroppedWindowDataset)
        if T > self.crop_len:
            if self.mode == "train":
                start = np.random.randint(0, T - self.crop_len + 1)
            else:
                start = (T - self.crop_len) // 2
            x_crop = x[:, start:start + self.crop_len]
        else:
            x_crop = x[:, :self.crop_len]
            if x_crop.shape[1] < self.crop_len:
                pad = self.crop_len - x_crop.shape[1]
                x_crop = np.pad(x_crop, ((0, 0), (0, pad)), mode="constant")

        # Get target from metadata
        if self.target_col in self.metadata.columns:
            y = float(self.metadata.iloc[idx][self.target_col])
        else:
            y = 0.0  # Dummy target for unsupervised

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
        """Setup datasets with Zarr-based preprocessing and caching.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict' (unused, loads all)
        """
        logger.info("\n" + "="*60)
        logger.info("[bold cyan]TUH EDF DATA SETUP (ZARR-OPTIMIZED)[/bold cyan]")
        logger.info("="*60)

        # Create cache paths for Zarr array and metadata
        cache_key = self._create_cache_key()
        zarr_path = self.cache_dir / f"{cache_key}.zarr"
        metadata_path = self.cache_dir / f"{cache_key}_metadata.parquet"

        # Check if cache exists
        cache_exists = zarr_path.exists() and metadata_path.exists()

        if self.hparams.use_cache and cache_exists:
            logger.info("[bold cyan]LOADING FROM ZARR CACHE[/bold cyan]")
            logger.info(f"[green]✓[/green] Zarr array: {zarr_path.name}")
            logger.info(f"[green]✓[/green] Metadata: {metadata_path.name}")

            # Load metadata
            self.metadata = pd.read_parquet(metadata_path)
            logger.info(f"[green]✓[/green] Loaded {len(self.metadata)} windows from cache")

            # Store paths for lazy loading
            self.zarr_path = zarr_path
        else:
            # Cache miss - process and save to Zarr
            logger.info("[bold yellow]CACHE MISS - PROCESSING DATA[/bold yellow]")
            self.zarr_path, self.metadata = self._load_and_preprocess(zarr_path, metadata_path)
            logger.info(f"[green]✓[/green] Saved Zarr cache: {zarr_path.name}")
            logger.info(f"[green]✓[/green] Saved metadata: {metadata_path.name}")

        # Create subject-level splits with lazy datasets
        self._create_splits_lazy()

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
        return "windows_" + "_".join(params)

    def _load_and_preprocess(self, zarr_path: Path, metadata_path: Path):
        """Load EDF files, preprocess, create windows, and save to Zarr with checkpointing.

        Uses parallel processing and incremental Zarr writes to handle massive datasets
        without OOM crashes. Supports resuming from interruptions via manifest tracking.

        Args:
            zarr_path: Path to Zarr array for storing windows
            metadata_path: Path to Parquet file for storing window metadata

        Returns:
            Tuple of (zarr_path, metadata_df) for lazy loading
        """
        logger.info("[bold cyan]ZARR-BASED PROCESSING WITH CHECKPOINTING[/bold cyan]")

        # Initialize checkpoint manager
        cache_key = self._create_cache_key()
        checkpoint_mgr = CheckpointManager(self.cache_dir, cache_key)
        checkpoint_mgr.load_manifest()

        # Load dataset metadata only (preload=False for speed)
        logger.info("\n[bold cyan]SCANNING EDF FILES[/bold cyan]")
        dataset = TUHEDFDataset(
            tuh_dir=str(self.tuh_dir),
            target_name=self.hparams.target_name,
            recording_ids=self.hparams.recording_ids,
            subjects=self.hparams.subjects,
            sessions=self.hparams.sessions,
            montages=self.hparams.montages,
            max_recordings=self.hparams.max_recordings,
        )

        logger.info(f"[bold]Total recordings found:[/bold] {len(dataset.datasets)}")

        # Filter by duration (lazy check, no data loading)
        logger.info("\n[bold cyan]FILTERING BY DURATION[/bold cyan]")
        logger.info(f"Minimum duration: {self.hparams.min_recording_duration_s}s")

        filtered_recordings = []
        for ds in tqdm(dataset.datasets, desc="Duration check"):
            try:
                duration = ds.raw.times[-1]
                if duration >= self.hparams.min_recording_duration_s:
                    file_path = ds.raw.filenames[0] if ds.raw.filenames else ""
                    filtered_recordings.append((ds, file_path))
            except Exception as e:
                logger.warning(f"Failed to check duration for {ds}: {e}")

        logger.info(f"[green]✓[/green] Kept {len(filtered_recordings)} recordings after duration filter")

        # Filter out already-processed recordings
        recordings_to_process = []
        for ds, file_path in filtered_recordings:
            if not checkpoint_mgr.is_processed(file_path):
                recordings_to_process.append((ds, file_path))
            else:
                logger.debug(f"[dim]Skipping already processed: {Path(file_path).name}[/dim]")

        logger.info(f"\n[bold cyan]RESUMING FROM CHECKPOINT[/bold cyan]")
        progress = checkpoint_mgr.get_progress_summary()
        logger.info(f"  Already processed: {progress['completed']} recordings")
        logger.info(f"  Failed: {progress['failed']} recordings")
        logger.info(f"  Remaining: {len(recordings_to_process)} recordings")

        if len(recordings_to_process) == 0:
            logger.info("[green]✓ All recordings already processed![/green]")
            # Load existing Zarr and metadata
            metadata_df = pd.read_parquet(metadata_path)
            return zarr_path, metadata_df

        # Calculate optimal worker count
        optimal_workers = get_optimal_worker_count()
        logger.info(f"\n[bold cyan]PARALLEL PROCESSING CONFIGURATION[/bold cyan]")
        logger.info(f"  System CPUs: {os.cpu_count()}")
        logger.info(f"  Available memory: {get_available_memory_gb():.1f} GB")
        logger.info(f"  Optimal workers: {optimal_workers}")

        # Initialize or append to Zarr array
        start_time = time.time()
        total_windows_written = 0

        # Determine Zarr array shape (infer from first recording if new)
        if not zarr_path.exists():
            logger.info("\n[bold cyan]INITIALIZING ZARR ARRAY[/bold cyan]")
            # Process one recording to determine shape
            sample_ds, _ = recordings_to_process[0]
            sample_ds.raw.load_data()

            # Standardize channels to 21 common 10-20 EEG channels
            sample_ds.raw = self._standardize_channels(sample_ds.raw)

            preprocessors = self._get_preprocessors()
            if preprocessors:
                preprocess(BaseConcatDataset([sample_ds]), preprocessors, n_jobs=1)
            actual_sfreq = sample_ds.raw.info['sfreq']
            n_chans = len(sample_ds.raw.ch_names)
            window_size_samples = int(self.hparams.window_size_s * actual_sfreq)

            logger.info(f"  Channels: {n_chans} (standardized 10-20 EEG)")
            logger.info(f"  Sampling rate: {actual_sfreq} Hz")
            logger.info(f"  Window size: {window_size_samples} samples ({self.hparams.window_size_s}s)")

            # Create Zarr array with compression
            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
            zarr_array = zarr.open(
                str(zarr_path),
                mode='w',
                shape=(0, n_chans, window_size_samples),
                chunks=(100, n_chans, window_size_samples),  # 100 windows per chunk
                dtype=np.float32,
                compressor=compressor,
            )
            logger.info(f"[green]✓[/green] Created Zarr array: {zarr_path.name}")
            logger.info(f"  Compression: {compressor.cname} (level {compressor.clevel})")
        else:
            zarr_array = zarr.open(str(zarr_path), mode='a')  # Append mode
            logger.info(f"[green]✓[/green] Opened existing Zarr: {zarr_array.shape[0]} windows")
            total_windows_written = zarr_array.shape[0]

        # Process recordings with multiprocessing
        logger.info(f"\n[bold cyan]PROCESSING {len(recordings_to_process)} RECORDINGS[/bold cyan]")

        all_metadata = []
        checkpoint_interval = 100  # Save manifest every 100 recordings

        with tqdm(total=len(recordings_to_process), desc="Processing", unit="rec") as pbar:
            for i, (ds, file_path) in enumerate(recordings_to_process):
                try:
                    # Process single recording
                    windows_data, num_windows, metadata = self._process_single_recording(
                        ds, file_path
                    )

                    if num_windows > 0:
                        # Validate shape matches Zarr array
                        expected_shape = (num_windows, zarr_array.shape[1], zarr_array.shape[2])
                        if windows_data.shape != expected_shape:
                            error_msg = f"Shape mismatch: expected {expected_shape}, got {windows_data.shape}"
                            logger.warning(f"[yellow]Skipping {Path(file_path).name}: {error_msg}[/yellow]")
                            checkpoint_mgr.mark_failed(file_path, ds.description.to_dict(), error_msg)
                        else:
                            # Append to Zarr array
                            zarr_array.append(windows_data, axis=0)
                            all_metadata.extend(metadata)
                            total_windows_written += num_windows

                            # Mark as completed
                            checkpoint_mgr.mark_completed(file_path, ds.description.to_dict(), num_windows)
                    else:
                        logger.warning(f"[yellow]No windows generated for {Path(file_path).name}[/yellow]")
                        checkpoint_mgr.mark_failed(file_path, ds.description.to_dict(), "No windows generated")

                    # Save checkpoint periodically
                    if (i + 1) % checkpoint_interval == 0:
                        checkpoint_mgr.save_manifest()
                        # Save intermediate metadata
                        pd.DataFrame(all_metadata).to_parquet(metadata_path, index=False)
                        logger.info(f"[green]✓[/green] Checkpoint saved: {i+1}/{len(recordings_to_process)} ({total_windows_written} windows)")

                except Exception as e:
                    logger.error(f"[red]Failed to process {Path(file_path).name}: {e}[/red]")
                    checkpoint_mgr.mark_failed(file_path, {}, str(e))

                pbar.update(1)
                pbar.set_postfix_str(f"{total_windows_written} windows")

                # Explicit garbage collection
                gc.collect()

        # Final checkpoint save
        checkpoint_mgr.save_manifest()
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_parquet(metadata_path, index=False)

        elapsed = time.time() - start_time
        logger.info(f"\n[bold green]✓ PROCESSING COMPLETE[/bold green]")
        logger.info(f"  Total windows: {total_windows_written}")
        logger.info(f"  Zarr size: {zarr_path.stat().st_size / (1024**3):.2f} GB")
        logger.info(f"  Time elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"  Throughput: {len(recordings_to_process)/(elapsed/60):.1f} recordings/min")

        return zarr_path, metadata_df

    def _get_preprocessors(self) -> List[Preprocessor]:
        """Get list of braindecode preprocessors based on hyperparameters."""
        preprocessors = []
        if self.hparams.apply_bandpass:
            preprocessors.append(
                Preprocessor('filter', l_freq=self.hparams.l_freq, h_freq=self.hparams.h_freq)
            )
        preprocessors.append(
            Preprocessor('resample', sfreq=self.hparams.sfreq)
        )
        return preprocessors

    def _standardize_channels(self, raw):
        """Standardize to 21 common 10-20 EEG channels (drops non-EEG channels).

        This ensures all recordings have the same channel count regardless of
        montage variations (29-36 channels). Uses MNE channel name mapping to
        handle different naming conventions (e.g., 'EEG FP1-REF' -> 'FP1').

        Standard 10-20 channels (21): FP1, FP2, F7, F3, FZ, F4, F8, A1, T3, C3, CZ,
                                      C4, T4, A2, T5, P3, PZ, P4, T6, O1, O2
        """
        # Standard 10-20 channel names (case-insensitive)
        standard_channels = [
            'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
            'A1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'A2',
            'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2'
        ]

        # Map TUH naming convention (e.g., 'EEG FP1-REF') to standard names
        # Extract channel name between 'EEG ' and '-REF'
        rename_mapping = {}
        for ch_name in raw.ch_names:
            if ch_name.startswith('EEG ') and '-REF' in ch_name:
                # Extract: 'EEG FP1-REF' -> 'FP1'
                clean_name = ch_name.replace('EEG ', '').replace('-REF', '')
                if clean_name.upper() in [s.upper() for s in standard_channels]:
                    rename_mapping[ch_name] = clean_name.upper()

        # Rename channels
        if rename_mapping:
            raw.rename_channels(rename_mapping)

        # Pick only standard channels (ignore missing ones)
        available_std_channels = [ch for ch in standard_channels if ch in raw.ch_names]

        if len(available_std_channels) == 0:
            raise ValueError(f"No standard 10-20 channels found in {raw.ch_names}")

        raw.pick_channels(available_std_channels, ordered=True)
        logger.debug(f"Standardized to {len(available_std_channels)}/21 channels: {available_std_channels}")

        return raw

    def _process_single_recording(self, ds: BaseDataset, file_path: str) -> Tuple[np.ndarray, int, List[Dict]]:
        """Process single EDF recording: load, preprocess, create windows.

        Args:
            ds: BaseDataset for this recording
            file_path: Path to EDF file

        Returns:
            Tuple of (windows_data, num_windows, metadata_list)
                - windows_data: numpy array of shape (num_windows, n_chans, window_samples)
                - num_windows: number of windows extracted
                - metadata_list: list of dicts with window metadata
        """
        # Load raw data
        ds.raw.load_data()

        # Standardize channels to 21 common 10-20 EEG channels
        ds.raw = self._standardize_channels(ds.raw)

        # Apply preprocessing
        preprocessors = self._get_preprocessors()
        if preprocessors:
            preprocess(BaseConcatDataset([ds]), preprocessors, n_jobs=1)

        # Create windows
        actual_sfreq = ds.raw.info['sfreq']
        window_size_samples = int(self.hparams.window_size_s * actual_sfreq)
        window_stride_samples = int(self.hparams.window_stride_s * actual_sfreq)

        windows = create_fixed_length_windows(
            BaseConcatDataset([ds]),
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
            drop_last_window=True,
            preload=True,
        )

        num_windows = len(windows.datasets)

        if num_windows == 0:
            return np.array([]), 0, []

        # Extract window data and metadata
        windows_data = []
        metadata_list = []

        for win_ds in windows.datasets:
            x, y, _ = win_ds[0]  # Shape: (channels, time)
            windows_data.append(x)

            # Extract metadata from window description
            win_meta = win_ds.description.to_dict() if hasattr(win_ds, 'description') else {}
            win_meta['file_path'] = str(file_path)  # Convert to string for parquet compatibility
            win_meta['target'] = float(y) if y is not None else 0.0
            metadata_list.append(win_meta)

        # Stack windows into array
        windows_data = np.stack(windows_data, axis=0).astype(np.float32)

        # Free memory
        ds.raw._data = None
        del windows
        gc.collect()

        return windows_data, num_windows, metadata_list

    def _create_splits_lazy(self):
        """Create subject-level train/val/test splits with lazy Zarr datasets."""
        logger.info("\n[bold cyan]SPLITTING DATA (LAZY ZARR)[/bold cyan]")

        # Get unique subjects from metadata
        if 'subject_id' not in self.metadata.columns:
            raise KeyError("Metadata missing 'subject_id' column. Cannot create splits.")

        subjects = self.metadata['subject_id'].unique()
        logger.info(f"[bold]Total subjects:[/bold] {len(subjects)}")

        # Handle edge case: too few subjects for splitting
        min_subjects_for_split = max(3, int(1 / min(self.hparams.val_ratio, self.hparams.test_ratio)) + 1)

        if len(subjects) < min_subjects_for_split:
            logger.warning(f"[yellow]Only {len(subjects)} subject(s) found (need >={min_subjects_for_split} for splits). Using all for train, empty val/test.[/yellow]")
            train_subj = subjects
            val_subj = np.array([])
            test_subj = np.array([])
        else:
            # Set random state
            rng = check_random_state(self.hparams.seed)

            # Split: train / (val + test)
            train_subj, val_test_subj = train_test_split(
                subjects,
                test_size=(self.hparams.val_ratio + self.hparams.test_ratio),
                random_state=rng,
                shuffle=True
            )

            # Split: val / test (handle edge case for small val_test_subj)
            if len(val_test_subj) < 2:
                # Can't split further, assign all to val or test based on larger ratio
                if self.hparams.val_ratio >= self.hparams.test_ratio:
                    val_subj = val_test_subj
                    test_subj = np.array([])
                else:
                    val_subj = np.array([])
                    test_subj = val_test_subj
            else:
                val_subj, test_subj = train_test_split(
                    val_test_subj,
                    test_size=self.hparams.test_ratio / (self.hparams.val_ratio + self.hparams.test_ratio),
                    random_state=check_random_state(self.hparams.seed + 1),
                    shuffle=True
                )

        logger.info(f"Train subjects: {len(train_subj)}")
        logger.info(f"Val subjects: {len(val_subj)}")
        logger.info(f"Test subjects: {len(test_subj)}")

        # Filter metadata by subject for each split
        # IMPORTANT: Keep original index as 'zarr_index' for lazy loading
        self.metadata['zarr_index'] = np.arange(len(self.metadata))

        train_meta = self.metadata[self.metadata['subject_id'].isin(train_subj)].copy()
        val_meta = self.metadata[self.metadata['subject_id'].isin(val_subj)].copy()
        test_meta = self.metadata[self.metadata['subject_id'].isin(test_subj)].copy()

        # Reset dataframe index but preserve zarr_index column
        train_meta = train_meta.reset_index(drop=True)
        val_meta = val_meta.reset_index(drop=True)
        test_meta = test_meta.reset_index(drop=True)

        # Create lazy datasets (store metadata, will create datasets in dataloaders)
        self.train_meta = train_meta
        self.val_meta = val_meta
        self.test_meta = test_meta

        logger.info(f"\n[bold]Window counts:[/bold]")
        logger.info(f"  Train: {len(train_meta)}")
        logger.info(f"  Val: {len(val_meta)}")
        logger.info(f"  Test: {len(test_meta)}")

    def train_dataloader(self):
        """Create training DataLoader with LazyZarrWindowDataset (memory-efficient)."""
        if not hasattr(self, 'train_meta') or self.train_meta is None:
            raise RuntimeError("train_meta is None. Did you call setup()?")

        # Create lazy Zarr dataset for training
        crop_len = int(self.hparams.crop_size_s * self.hparams.sfreq)
        dataset = LazyZarrWindowDataset(
            zarr_path=self.zarr_path,
            metadata=self.train_meta,
            crop_len=crop_len,
            sfreq=int(self.hparams.sfreq),
            mode="train",
            target_col=self.hparams.target_name if self.hparams.target_name else "target",
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
        """Create validation DataLoader with LazyZarrWindowDataset (memory-efficient)."""
        if not hasattr(self, 'val_meta') or self.val_meta is None:
            raise RuntimeError("val_meta is None. Did you call setup()?")

        # Handle empty val split
        if len(self.val_meta) == 0:
            logger.warning("[yellow]Val split is empty. Returning None.[/yellow]")
            return None

        # Create lazy Zarr dataset for validation
        crop_len = int(self.hparams.crop_size_s * self.hparams.sfreq)
        dataset = LazyZarrWindowDataset(
            zarr_path=self.zarr_path,
            metadata=self.val_meta,
            crop_len=crop_len,
            sfreq=int(self.hparams.sfreq),
            mode="val",
            target_col=self.hparams.target_name if self.hparams.target_name else "target",
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
        """Create test DataLoader with LazyZarrWindowDataset (memory-efficient)."""
        if not hasattr(self, 'test_meta') or self.test_meta is None:
            raise RuntimeError("test_meta is None. Did you call setup()?")

        # Handle empty test split
        if len(self.test_meta) == 0:
            logger.warning("[yellow]Test split is empty. Returning None.[/yellow]")
            return None

        # Create lazy Zarr dataset for testing
        crop_len = int(self.hparams.crop_size_s * self.hparams.sfreq)
        dataset = LazyZarrWindowDataset(
            zarr_path=self.zarr_path,
            metadata=self.test_meta,
            crop_len=crop_len,
            sfreq=int(self.hparams.sfreq),
            mode="test",
            target_col=self.hparams.target_name if self.hparams.target_name else "target",
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
