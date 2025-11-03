"""
TUH EEG Dataset with HDF5 backend for efficient loading.

This module provides braindecode-compatible dataset classes for the
Temple University Hospital EEG Corpus stored in HDF5 format.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import lightning as L
import mne
import numpy as np
import pandas as pd
import torch
from braindecode.datasets import BaseDataset, BaseConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TUHRecording(BaseDataset):
    """Single TUH recording loaded from HDF5."""

    def __init__(self, hdf5_file: h5py.File, recording_id: int, target_name: Optional[str] = None):
        """
        Initialize TUH recording.

        Parameters
        ----------
        hdf5_file : h5py.File
            Open HDF5 file handle
        recording_id : int
            Recording index in HDF5 file
        target_name : str, optional
            Target variable name ('age', 'gender', etc.)
        """
        self.hdf5_file = hdf5_file
        self.recording_id = recording_id

        # Load metadata
        self.metadata = self._load_metadata()

        # Load data from HDF5 and create Raw object
        recording_key = f'recording_{self.recording_id:06d}'
        data = self.hdf5_file['data'][recording_key][:]  # (n_channels, n_samples)

        # Create Info
        info = mne.create_info(
            ch_names=json.loads(self.metadata['ch_names']),
            sfreq=self.metadata['sfreq'],
            ch_types='eeg',
        )

        # Create RawArray
        raw = mne.io.RawArray(data, info, verbose=False)

        # Set description
        description = pd.Series(self.metadata)

        super().__init__(raw=raw, description=description, target_name=target_name)

    def _load_metadata(self) -> dict:
        """Load metadata for this recording."""
        recording_key = f'recording_{self.recording_id:06d}'
        metadata = {}

        # Load from HDF5 attributes or datasets
        if 'metadata' in self.hdf5_file:
            meta_group = self.hdf5_file['metadata']

            # Try loading from datasets first (newer format)
            if 'recording_id' in meta_group:
                idx = np.where(meta_group['recording_id'][:] == self.recording_id)[0]
                if len(idx) > 0:
                    idx = idx[0]
                    for key in meta_group.keys():
                        value = meta_group[key][idx]
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        metadata[key] = value

            # Fall back to attributes (older format)
            else:
                for key in meta_group.attrs.keys():
                    if key.startswith(recording_key):
                        attr_key = key.replace(f'{recording_key}_', '')
                        metadata[attr_key] = meta_group.attrs[key]

        return metadata


class TUHDataset(BaseConcatDataset):
    """
    TUH EEG Dataset loaded from HDF5.

    This class provides a braindecode-compatible interface to TUH EEG data
    stored in HDF5 format for efficient loading.

    Parameters
    ----------
    hdf5_path : str
        Path to HDF5 file created by process_tuh_to_hdf5.py
    recording_ids : list of int, optional
        Specific recording IDs to load. If None, loads all recordings.
    target_name : str, optional
        Target variable name for prediction ('age', 'gender', etc.)
    subjects : list of str, optional
        Filter by subject IDs
    sessions : list of int, optional
        Filter by session numbers
    montages : list of str, optional
        Filter by montage type (e.g., '01_tcp_ar')

    Examples
    --------
    >>> # Load all recordings
    >>> dataset = TUHDataset('tuh_eeg_processed.h5')
    >>> print(len(dataset))  # Number of recordings
    >>>
    >>> # Load specific subjects
    >>> dataset = TUHDataset('tuh_eeg_processed.h5', subjects=['aaaaaaaa', 'aaaaaaab'])
    >>>
    >>> # Access individual recording
    >>> raw, target = dataset[0]
    >>> print(raw.info)
    >>>
    >>> # Use with braindecode preprocessing
    >>> from braindecode.preprocessing import preprocess, Preprocessor
    >>> preprocessors = [
    ...     Preprocessor('filter', l_freq=0.5, h_freq=40),
    ...     Preprocessor('resample', sfreq=100),
    ... ]
    >>> preprocess(dataset, preprocessors)
    """

    def __init__(
        self,
        hdf5_path: str,
        recording_ids: Optional[List[int]] = None,
        target_name: Optional[str] = None,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[int]] = None,
        montages: Optional[List[str]] = None,
    ):
        """Initialize TUH Dataset from HDF5."""
        self.hdf5_path = hdf5_path
        self.target_name = target_name

        # Open HDF5 file
        self.h5f = h5py.File(hdf5_path, 'r')

        # Load metadata table
        self.metadata_df = self._load_metadata_table()

        # Apply filters
        if recording_ids is not None:
            self.metadata_df = self.metadata_df[
                self.metadata_df['recording_id'].isin(recording_ids)
            ]

        if subjects is not None:
            self.metadata_df = self.metadata_df[
                self.metadata_df['subject_id'].isin(subjects)
            ]

        if sessions is not None:
            self.metadata_df = self.metadata_df[
                self.metadata_df['session'].isin(sessions)
            ]

        if montages is not None:
            self.metadata_df = self.metadata_df[
                self.metadata_df['montage'].isin(montages)
            ]

        # Create BaseDataset instances
        datasets = [
            TUHRecording(self.h5f, rec_id, target_name)
            for rec_id in self.metadata_df['recording_id'].values
        ]

        super().__init__(datasets)

    def _load_metadata_table(self) -> pd.DataFrame:
        """Load metadata table from HDF5."""
        if 'metadata' not in self.h5f:
            raise ValueError("HDF5 file does not contain metadata group")

        meta_group = self.h5f['metadata']

        # Check if metadata is stored as datasets (newer format)
        if 'recording_id' in meta_group:
            data = {}
            for key in meta_group.keys():
                values = meta_group[key][:]
                # Decode bytes to strings
                if values.dtype.kind in ['S', 'O']:
                    values = np.array([v.decode('utf-8') if isinstance(v, bytes) else v for v in values])
                data[key] = values

            df = pd.DataFrame(data)

        # Fall back to attributes (older format)
        else:
            records = []
            for key in meta_group.attrs.keys():
                if key.startswith('recording_'):
                    parts = key.split('_', 2)
                    if len(parts) >= 3:
                        rec_id = int(parts[1])
                        attr_name = parts[2]

                        # Find or create record
                        record = next((r for r in records if r['recording_id'] == rec_id), None)
                        if record is None:
                            record = {'recording_id': rec_id}
                            records.append(record)

                        record[attr_name] = meta_group.attrs[key]

            df = pd.DataFrame(records)

        return df

    def __del__(self):
        """Close HDF5 file on deletion."""
        if hasattr(self, 'h5f'):
            self.h5f.close()

    def get_metadata(self) -> pd.DataFrame:
        """Get metadata DataFrame for all recordings."""
        return self.metadata_df.copy()

    def split_by_subjects(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
    ) -> Tuple['TUHDataset', 'TUHDataset', 'TUHDataset']:
        """
        Split dataset by subjects (no data leakage).

        Parameters
        ----------
        train_ratio : float
            Ratio of subjects for training
        val_ratio : float
            Ratio of subjects for validation
        seed : int
            Random seed

        Returns
        -------
        train_dataset : TUHDataset
            Training dataset
        val_dataset : TUHDataset
            Validation dataset
        test_dataset : TUHDataset
            Test dataset
        """
        # Get unique subjects
        subjects = self.metadata_df['subject_id'].unique()
        np.random.seed(seed)
        np.random.shuffle(subjects)

        # Split
        n_train = int(len(subjects) * train_ratio)
        n_val = int(len(subjects) * val_ratio)

        train_subjects = subjects[:n_train]
        val_subjects = subjects[n_train:n_train + n_val]
        test_subjects = subjects[n_train + n_val:]

        # Create datasets
        train_ds = TUHDataset(
            self.hdf5_path,
            subjects=train_subjects.tolist(),
            target_name=self.target_name,
        )
        val_ds = TUHDataset(
            self.hdf5_path,
            subjects=val_subjects.tolist(),
            target_name=self.target_name,
        )
        test_ds = TUHDataset(
            self.hdf5_path,
            subjects=test_subjects.tolist(),
            target_name=self.target_name,
        )

        return train_ds, val_ds, test_ds


def load_tuh_dataset(
    hdf5_path: str,
    target_name: Optional[str] = None,
    **kwargs,
) -> TUHDataset:
    """
    Convenience function to load TUH dataset.

    Parameters
    ----------
    hdf5_path : str
        Path to HDF5 file
    target_name : str, optional
        Target variable name
    **kwargs
        Additional arguments passed to TUHDataset

    Returns
    -------
    dataset : TUHDataset
        Loaded dataset

    Examples
    --------
    >>> dataset = load_tuh_dataset('tuh_eeg_processed.h5', target_name='age')
    >>> train, val, test = dataset.split_by_subjects(0.7, 0.15)
    """
    return TUHDataset(hdf5_path, target_name=target_name, **kwargs)


def collate_fn(batch):
    """Collate function that handles (x, y, i) tuples from braindecode datasets.

    Args:
        batch: List of (x, y, i) tuples from BaseConcatDataset

    Returns:
        Tuple of (x_batch, y_batch) tensors
    """
    x_batch = torch.stack([torch.from_numpy(item[0]).float() for item in batch], dim=0)
    y_batch = torch.stack([torch.tensor(item[1]).float() for item in batch], dim=0)
    return x_batch, y_batch


class TUHDataModule(L.LightningDataModule):
    """Lightning DataModule for TUH EEG dataset.

    Provides train/val/test splits with DataLoaders for TUH EEG data
    stored in HDF5 format.

    Args:
        hdf5_path: Path to HDF5 file (or virtual HDF5 file)
        target_name: Target variable name (e.g., 'age', 'gender')
        batch_size: Batch size for DataLoaders (default: 512)
        num_workers: Number of workers for DataLoaders (default: 8)
        train_ratio: Ratio of subjects for training (default: 0.8)
        val_ratio: Ratio of subjects for validation (default: 0.1)
        test_ratio: Ratio of subjects for test (default: 0.1)
        seed: Random seed for splits (default: 42)
        recording_ids: Specific recording IDs to load (optional)
        subjects: Filter by subject IDs (optional)
        sessions: Filter by session numbers (optional)
        montages: Filter by montage type (optional)

    Example:
        >>> datamodule = TUHDataModule(
        ...     hdf5_path='data/tuh/tuh_eeg.h5',
        ...     target_name='age',
        ...     batch_size=256,
        ...     num_workers=4,
        ... )
        >>> datamodule.setup()
        >>> train_loader = datamodule.train_dataloader()
    """

    def __init__(
        self,
        hdf5_path: str,
        target_name: Optional[str] = None,
        batch_size: int = 512,
        num_workers: int = 8,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        recording_ids: Optional[List[int]] = None,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[int]] = None,
        montages: Optional[List[str]] = None,
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
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        # Will be populated in setup()
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.full_dataset = None

    @property
    def batch_size(self) -> int:
        """Batch size for DataLoaders (for Lightning compatibility)."""
        return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        """Update batch size (modified by Lightning's batch size scaler)."""
        self.hparams.batch_size = value

    def setup(self, stage: Optional[str] = None):
        """Setup datasets with subject-level splits.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict' (unused, loads all)
        """
        logger.info("\n" + "="*60)
        logger.info("[bold cyan]TUH EEG DATA SETUP[/bold cyan]")
        logger.info("="*60)

        # Load full dataset
        logger.info(f"Loading HDF5 file: {self.hdf5_path}")
        self.full_dataset = TUHDataset(
            hdf5_path=str(self.hdf5_path),
            target_name=self.hparams.target_name,
            recording_ids=self.hparams.recording_ids,
            subjects=self.hparams.subjects,
            sessions=self.hparams.sessions,
            montages=self.hparams.montages,
        )

        logger.info(f"[bold]Total recordings:[/bold] {len(self.full_dataset)}")

        # Get metadata
        metadata = self.full_dataset.get_metadata()
        logger.info(f"[bold]Metadata columns:[/bold] {list(metadata.columns)}")

        # Display sample metadata
        logger.info(f"\n[bold]Sample metadata:[/bold]")
        logger.info(metadata.head().to_string())

        # Display target statistics if available
        if self.hparams.target_name and self.hparams.target_name in metadata.columns:
            logger.info(f"\n[bold]Target '{self.hparams.target_name}' statistics:[/bold]")
            logger.info(f"  Mean: {metadata[self.hparams.target_name].mean():.4f}")
            logger.info(f"  Std: {metadata[self.hparams.target_name].std():.4f}")
            logger.info(f"  Min: {metadata[self.hparams.target_name].min():.4f}")
            logger.info(f"  Max: {metadata[self.hparams.target_name].max():.4f}")

        # Create subject-level splits
        self._create_splits(metadata)

    def _create_splits(self, metadata: pd.DataFrame):
        """Create subject-level train/val/test splits."""
        logger.info("\n[bold cyan]SPLITTING DATA[/bold cyan]")

        # Get unique subjects
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

        # Create datasets for each split
        self.train_set = TUHDataset(
            hdf5_path=str(self.hdf5_path),
            target_name=self.hparams.target_name,
            subjects=train_subj.tolist(),
        )

        self.val_set = TUHDataset(
            hdf5_path=str(self.hdf5_path),
            target_name=self.hparams.target_name,
            subjects=val_subj.tolist(),
        )

        self.test_set = TUHDataset(
            hdf5_path=str(self.hdf5_path),
            target_name=self.hparams.target_name,
            subjects=test_subj.tolist(),
        )

        logger.info(f"\n[bold]Recording counts:[/bold]")
        logger.info(f"  Train: {len(self.train_set)}")
        logger.info(f"  Val: {len(self.val_set)}")
        logger.info(f"  Test: {len(self.test_set)}")

    def train_dataloader(self):
        """Create training DataLoader with persistence."""
        if self.train_set is None:
            raise RuntimeError("train_set is None. Did you call setup()?")

        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation DataLoader with persistence."""
        if self.val_set is None:
            raise RuntimeError("val_set is None. Did you call setup()?")

        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test DataLoader."""
        if self.test_set is None:
            raise RuntimeError("test_set is None. Did you call setup()?")

        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
