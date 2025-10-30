"""Lazy loading dataset for windowed data.

Provides memory-efficient on-demand loading from memory-mapped numpy arrays.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LazyZarrWindowDataset(Dataset):
    """Lazy-loading dataset that reads windows from Zarr on-demand.

    Args:
        zarr_path: Path to Zarr array
        metadata: DataFrame with window metadata (must include 'zarr_index' column)
        crop_len_s: Optional crop length for temporal augmentation
        sfreq: Sampling frequency in Hz
        mode: 'train' or 'val' (affects cropping strategy)
    """

    def __init__(
        self,
        zarr_path: Path,
        metadata: pd.DataFrame,
        crop_len_s: Optional[float] = None,
        sfreq: int = 100,
        mode: str = 'train'
    ):
        self.zarr_path = Path(zarr_path)
        self.zarr_array = zarr.open(str(zarr_path), mode='r')
        self.metadata = metadata.reset_index(drop=True)
        self._crop_len_s = crop_len_s
        self.sfreq = sfreq
        self.mode = mode

        # Validate zarr_index
        assert 'zarr_index' in self.metadata.columns, "Metadata must have 'zarr_index' column"
        if len(self.metadata) > 0:
            max_idx = self.metadata['zarr_index'].max()
            assert max_idx < self.zarr_array.shape[0], \
                f"Invalid zarr_index {max_idx} >= array size {self.zarr_array.shape[0]}"

        logger.info(f"LazyZarrWindowDataset created:")
        logger.info(f"  Windows: {len(self)} (from {self.zarr_array.shape[0]} total in Zarr)")
        logger.info(f"  Shape: {self.zarr_array.shape[1:]} (channels × samples)")
        logger.info(f"  Crop: {crop_len_s}s ({self.crop_samples} samples)" if crop_len_s else "  Crop: None")

    @property
    def crop_len_s(self) -> Optional[float]:
        """Get crop length in seconds."""
        return self._crop_len_s

    @crop_len_s.setter
    def crop_len_s(self, value: Optional[float]):
        """Set crop length in seconds and update crop_samples."""
        self._crop_len_s = value

    @property
    def crop_samples(self) -> Optional[int]:
        """Get crop length in samples (computed from crop_len_s)."""
        if self._crop_len_s is not None:
            return int(self._crop_len_s * self.sfreq)
        return None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a window with optional cropping.

        Args:
            idx: Index in filtered metadata

        Returns:
            EEG tensor (n_channels, n_samples)
        """
        # Map filtered index → Zarr index
        zarr_idx = int(self.metadata.iloc[idx]['zarr_index'])

        # Lazy load: Read ONLY this window from disk
        x = self.zarr_array[zarr_idx]  # Shape: (n_channels, window_samples)

        # Apply temporal cropping if specified
        if self.crop_samples is not None and self.crop_samples < x.shape[1]:
            x = self._apply_cropping(x)

        return torch.from_numpy(x).float()

    def _apply_cropping(self, x: np.ndarray) -> np.ndarray:
        """Apply temporal cropping.

        Args:
            x: Window array (n_channels, window_samples)

        Returns:
            Cropped array (n_channels, crop_samples)
        """
        max_start = x.shape[1] - self.crop_samples

        if self.mode == 'train':
            # Random crop for training
            crop_start = np.random.randint(0, max_start + 1)
        else:
            # Center crop for validation/test
            crop_start = max_start // 2

        return x[:, crop_start:crop_start + self.crop_samples]

    def get_metadata_row(self, idx: int) -> pd.Series:
        """Get metadata for a specific window.

        Args:
            idx: Index in filtered metadata

        Returns:
            Metadata as pandas Series
        """
        return self.metadata.iloc[idx]

    def get_subjects(self) -> list:
        """Get unique subject IDs in this dataset.

        Returns:
            List of unique subject IDs
        """
        if 'subject' in self.metadata.columns:
            return self.metadata['subject'].unique().tolist()
        return []

    def get_tasks(self) -> list:
        """Get unique task names in this dataset.

        Returns:
            List of unique task names
        """
        if 'task' in self.metadata.columns:
            return self.metadata['task'].unique().tolist()
        return []

    def get_stats(self) -> dict:
        """Get dataset statistics.

        Returns:
            Dict with statistics
        """
        stats = {
            "n_windows": len(self),
            "n_channels": self.zarr_array.shape[1],
            "window_samples": self.zarr_array.shape[2],
            "crop_samples": self.crop_samples,
        }

        if 'subject' in self.metadata.columns:
            stats["n_subjects"] = self.metadata['subject'].nunique()

        if 'task' in self.metadata.columns:
            stats["n_tasks"] = self.metadata['task'].nunique()

        if 'release' in self.metadata.columns:
            stats["releases"] = self.metadata['release'].unique().tolist()

        return stats

    def filter_by_subjects(self, subjects: list) -> "LazyZarrWindowDataset":
        """Create a new dataset filtered by subject IDs.

        Args:
            subjects: List of subject IDs to include

        Returns:
            New LazyZarrWindowDataset with filtered metadata
        """
        if 'subject' not in self.metadata.columns:
            raise ValueError("Metadata does not have 'subject' column")

        filtered_metadata = self.metadata[
            self.metadata['subject'].isin(subjects)
        ].reset_index(drop=True)

        return LazyZarrWindowDataset(
            zarr_path=self.zarr_path,
            metadata=filtered_metadata,
            crop_len_s=self.crop_len_s,
            sfreq=self.sfreq,
            mode=self.mode
        )

    def filter_by_releases(self, releases: list) -> "LazyZarrWindowDataset":
        """Create a new dataset filtered by release IDs.

        Args:
            releases: List of release IDs to include

        Returns:
            New LazyZarrWindowDataset with filtered metadata
        """
        if 'release' not in self.metadata.columns:
            raise ValueError("Metadata does not have 'release' column")

        filtered_metadata = self.metadata[
            self.metadata['release'].isin(releases)
        ].reset_index(drop=True)

        return LazyZarrWindowDataset(
            zarr_path=self.zarr_path,
            metadata=filtered_metadata,
            crop_len_s=self.crop_len_s,
            sfreq=self.sfreq,
            mode=self.mode
        )

    def filter_by_recordings(self, recording_ids: list) -> "LazyZarrWindowDataset":
        """Create a new dataset filtered by recording IDs.

        Args:
            recording_ids: List of recording IDs to include

        Returns:
            New LazyZarrWindowDataset with filtered metadata
        """
        if 'recording_id' not in self.metadata.columns:
            raise ValueError("Metadata does not have 'recording_id' column")

        filtered_metadata = self.metadata[
            self.metadata['recording_id'].isin(recording_ids)
        ].reset_index(drop=True)

        return LazyZarrWindowDataset(
            zarr_path=self.zarr_path,
            metadata=filtered_metadata,
            crop_len_s=self.crop_len_s,
            sfreq=self.sfreq,
            mode=self.mode
        )


class MemmapWindowDataset(Dataset):
    """Memory-mapped dataset for zero-copy window access.

    Uses numpy.memmap for direct memory mapping of uncompressed window arrays.
    Provides zero-copy access with automatic OS-level page cache management.

    Args:
        memmap_path: Path to .npy file containing windows
        metadata: DataFrame with window metadata (must include 'array_index' column)
        crop_len_s: Optional crop length for temporal augmentation
        sfreq: Sampling frequency in Hz
        mode: 'train' or 'val' (affects cropping strategy)
    """

    def __init__(
        self,
        memmap_path: Path,
        metadata: pd.DataFrame,
        crop_len_s: Optional[float] = None,
        sfreq: int = 100,
        mode: str = 'train'
    ):
        self.memmap_path = Path(memmap_path)
        self.metadata = metadata.reset_index(drop=True)
        self._crop_len_s = crop_len_s
        self.sfreq = sfreq
        self.mode = mode

        # Load array shape from .npy header without loading data
        with open(memmap_path, 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)

        # Open as memory-mapped array (zero-copy)
        self.data = np.load(str(memmap_path), mmap_mode='r')  # Read-only mmap

        # Validate array_index
        assert 'array_index' in self.metadata.columns, "Metadata must have 'array_index' column"
        if len(self.metadata) > 0:
            max_idx = self.metadata['array_index'].max()
            assert max_idx < self.data.shape[0], \
                f"Invalid array_index {max_idx} >= array size {self.data.shape[0]}"

        logger.info(f"MemmapWindowDataset created:")
        logger.info(f"  Windows: {len(self)} (from {self.data.shape[0]} total in array)")
        logger.info(f"  Shape: {self.data.shape[1:]} (channels × samples)")
        logger.info(f"  Size: {self.data.nbytes / 1e9:.2f} GB (memory-mapped)")
        logger.info(f"  Crop: {crop_len_s}s ({self.crop_samples} samples)" if crop_len_s else "  Crop: None")

    @property
    def crop_len_s(self) -> Optional[float]:
        """Get crop length in seconds."""
        return self._crop_len_s

    @crop_len_s.setter
    def crop_len_s(self, value: Optional[float]):
        """Set crop length in seconds and update crop_samples."""
        self._crop_len_s = value

    @property
    def crop_samples(self) -> Optional[int]:
        """Get crop length in samples (computed from crop_len_s)."""
        if self._crop_len_s is not None:
            return int(self._crop_len_s * self.sfreq)
        return None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a window with optional cropping (zero-copy from memmap).

        Args:
            idx: Index in filtered metadata

        Returns:
            EEG tensor (n_channels, n_samples)
        """
        # Map filtered index → array index
        array_idx = int(self.metadata.iloc[idx]['array_index'])

        # Zero-copy load: Memory-mapped array read (OS handles paging)
        x = self.data[array_idx]  # Shape: (n_channels, window_samples)

        # Apply temporal cropping if specified
        if self.crop_samples is not None and self.crop_samples < x.shape[1]:
            x = self._apply_cropping(x)

        # Convert to PyTorch tensor (shares underlying memory)
        return torch.from_numpy(x).float()

    def _apply_cropping(self, x: np.ndarray) -> np.ndarray:
        """Apply temporal cropping.

        Args:
            x: Window array (n_channels, window_samples)

        Returns:
            Cropped array (n_channels, crop_samples)
        """
        max_start = x.shape[1] - self.crop_samples

        if self.mode == 'train':
            # Random crop for training
            crop_start = np.random.randint(0, max_start + 1)
        else:
            # Center crop for validation/test
            crop_start = max_start // 2

        return x[:, crop_start:crop_start + self.crop_samples]

    def get_metadata_row(self, idx: int) -> pd.Series:
        """Get metadata for a specific window.

        Args:
            idx: Index in filtered metadata

        Returns:
            Metadata as pandas Series
        """
        return self.metadata.iloc[idx]

    def get_subjects(self) -> list:
        """Get unique subject IDs in this dataset.

        Returns:
            List of unique subject IDs
        """
        if 'subject' in self.metadata.columns:
            return self.metadata['subject'].unique().tolist()
        return []

    def get_tasks(self) -> list:
        """Get unique task names in this dataset.

        Returns:
            List of unique task names
        """
        if 'task' in self.metadata.columns:
            return self.metadata['task'].unique().tolist()
        return []

    def get_stats(self) -> dict:
        """Get dataset statistics.

        Returns:
            Dict with statistics
        """
        stats = {
            "n_windows": len(self),
            "n_channels": self.data.shape[1],
            "window_samples": self.data.shape[2],
            "crop_samples": self.crop_samples,
        }

        if 'subject' in self.metadata.columns:
            stats["n_subjects"] = self.metadata['subject'].nunique()

        if 'task' in self.metadata.columns:
            stats["n_tasks"] = self.metadata['task'].nunique()

        if 'release' in self.metadata.columns:
            stats["releases"] = self.metadata['release'].unique().tolist()

        return stats

    def filter_by_subjects(self, subjects: list) -> "MemmapWindowDataset":
        """Create a new dataset filtered by subject IDs.

        Args:
            subjects: List of subject IDs to include

        Returns:
            New MemmapWindowDataset with filtered metadata
        """
        if 'subject' not in self.metadata.columns:
            raise ValueError("Metadata does not have 'subject' column")

        filtered_metadata = self.metadata[
            self.metadata['subject'].isin(subjects)
        ].reset_index(drop=True)

        return MemmapWindowDataset(
            memmap_path=self.memmap_path,
            metadata=filtered_metadata,
            crop_len_s=self.crop_len_s,
            sfreq=self.sfreq,
            mode=self.mode
        )

    def filter_by_releases(self, releases: list) -> "MemmapWindowDataset":
        """Create a new dataset filtered by release IDs.

        Args:
            releases: List of release IDs to include

        Returns:
            New MemmapWindowDataset with filtered metadata
        """
        if 'release' not in self.metadata.columns:
            raise ValueError("Metadata does not have 'release' column")

        filtered_metadata = self.metadata[
            self.metadata['release'].isin(releases)
        ].reset_index(drop=True)

        return MemmapWindowDataset(
            memmap_path=self.memmap_path,
            metadata=filtered_metadata,
            crop_len_s=self.crop_len_s,
            sfreq=self.sfreq,
            mode=self.mode
        )

    def filter_by_recordings(self, recording_ids: list) -> "MemmapWindowDataset":
        """Create a new dataset filtered by recording IDs.

        Args:
            recording_ids: List of recording IDs to include

        Returns:
            New MemmapWindowDataset with filtered metadata
        """
        if 'recording_id' not in self.metadata.columns:
            raise ValueError("Metadata does not have 'recording_id' column")

        filtered_metadata = self.metadata[
            self.metadata['recording_id'].isin(recording_ids)
        ].reset_index(drop=True)

        return MemmapWindowDataset(
            memmap_path=self.memmap_path,
            metadata=filtered_metadata,
            crop_len_s=self.crop_len_s,
            sfreq=self.sfreq,
            mode=self.mode
        )
