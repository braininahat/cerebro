"""
TUH EEG Dataset with HDF5 backend for efficient loading.

This module provides braindecode-compatible dataset classes for the
Temple University Hospital EEG Corpus stored in HDF5 format.
"""

import json
from typing import List, Optional, Tuple

import h5py
import mne
import numpy as np
import pandas as pd
import torch
from braindecode.datasets import BaseDataset, BaseConcatDataset


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
