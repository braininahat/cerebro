#!/usr/bin/env python3
"""
Process TUH EEG corpus to HDF5 format for efficient loading.

This script converts 69k+ EDF files to a single HDF5 file with:
- Compression to reduce size (from 1.7TB to ~500GB estimated)
- Metadata storage (subject, session, date, montage, etc.)
- Progress tracking and resumability
- Parallel processing support

Usage:
    python scripts/process_tuh_to_hdf5.py --tuh-dir /path/to/tuh --output-hdf5 tuh_processed.h5 --n-jobs 8
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import h5py
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def extract_metadata_from_path(file_path: str) -> Dict:
    """Extract subject, session, segment from TUH file path."""
    # Path structure: .../edf/000/aaaaaaaa/s001_2015/01_tcp_ar/aaaaaaaa_s001_t000.edf
    parts = Path(file_path).parts

    filename = Path(file_path).stem  # aaaaaaaa_s001_t000
    tokens = filename.split('_')

    subject_id = tokens[0]
    session = tokens[1]  # s001
    segment = tokens[2]  # t000

    # Extract year from parent directory
    session_dir = parts[-3]  # s001_2015
    year = int(session_dir.split('_')[1]) if '_' in session_dir else None

    # Extract montage/reference from parent directory
    montage_dir = parts[-2]  # 01_tcp_ar

    # Check for date metadata file
    date_file = file_path.replace('.edf', '_date.txt')
    date_info = {}
    if os.path.exists(date_file):
        with open(date_file, 'r') as f:
            date_info = json.load(f)

    return {
        'subject_id': subject_id,
        'session': int(session[1:]),  # s001 -> 1
        'segment': int(segment[1:]),  # t000 -> 0
        'year': date_info.get('year', year),
        'month': date_info.get('month', None),
        'day': date_info.get('day', None),
        'montage': montage_dir,
        'path': file_path,
    }


def process_single_file(file_path: str, idx: int) -> Dict:
    """Process a single EDF file and return data + metadata."""
    try:
        # Read EDF file
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # Extract metadata
        metadata = extract_metadata_from_path(file_path)
        metadata.update({
            'recording_id': idx,
            'sfreq': raw.info['sfreq'],
            'n_channels': len(raw.ch_names),
            'duration': raw.times[-1],
            'ch_names': raw.ch_names,
            'meas_date': str(raw.info['meas_date']) if raw.info['meas_date'] else None,
        })

        # Get data
        data = raw.get_data()  # (n_channels, n_samples)

        return {
            'data': data,
            'metadata': metadata,
            'success': True,
            'error': None,
        }

    except Exception as e:
        return {
            'data': None,
            'metadata': {'path': file_path, 'recording_id': idx},
            'success': False,
            'error': str(e),
        }


def create_hdf5_dataset(
    tuh_dir: str,
    output_hdf5: str,
    n_jobs: int = 8,
    compression: str = 'gzip',
    compression_level: int = 4,
    resume: bool = True,
    max_files: int = None,
):
    """
    Create HDF5 dataset from TUH EEG corpus.

    Parameters
    ----------
    tuh_dir : str
        Path to TUH directory (containing edf/ subdirectory)
    output_hdf5 : str
        Output HDF5 file path
    n_jobs : int
        Number of parallel jobs for reading EDF files
    compression : str
        HDF5 compression algorithm ('gzip', 'lzf', None)
    compression_level : int
        Compression level (0-9, only for gzip)
    resume : bool
        If True, resume from last processed file
    max_files : int, optional
        Process only first N files (for testing/subset creation)
    """

    # Find all EDF files
    print("Scanning for EDF files...")
    edf_files = sorted(glob.glob(os.path.join(
        tuh_dir, "**/*.edf"), recursive=True))

    # Limit to max_files if specified
    if max_files is not None:
        edf_files = edf_files[:max_files]
        print(
            f"Found {len(edf_files)} EDF files (limited to first {max_files})")
    else:
        print(f"Found {len(edf_files)} EDF files")

    # Check if resuming
    start_idx = 0
    if resume and os.path.exists(output_hdf5):
        with h5py.File(output_hdf5, 'r') as f:
            if 'metadata' in f:
                start_idx = len(f['metadata'])
                print(f"Resuming from file {start_idx}/{len(edf_files)}")

    # Open HDF5 file
    mode = 'a' if (resume and os.path.exists(output_hdf5)) else 'w'
    h5f = h5py.File(output_hdf5, mode)

    # Create groups if they don't exist
    if 'data' not in h5f:
        h5f.create_group('data')
    if 'metadata' not in h5f:
        h5f.create_group('metadata')

    # Process files in batches
    batch_size = 100
    metadata_records = []
    errors = []

    for batch_start in range(start_idx, len(edf_files), batch_size):
        batch_end = min(batch_start + batch_size, len(edf_files))
        batch_files = edf_files[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))

        print(
            f"\nProcessing batch {batch_start}-{batch_end} / {len(edf_files)}")

        # Process batch in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_file)(file_path, idx)
            for file_path, idx in tqdm(
                zip(batch_files, batch_indices),
                total=len(batch_files),
                desc="Reading EDFs"
            )
        )

        # Write to HDF5
        print("Writing to HDF5...")
        for result in tqdm(results, desc="Writing"):
            if result['success']:
                idx = result['metadata']['recording_id']

                # Store data with compression
                h5f['data'].create_dataset(
                    f'recording_{idx:06d}',
                    data=result['data'],
                    compression=compression,
                    compression_opts=compression_level if compression == 'gzip' else None,
                    dtype='float32',  # Convert to float32 to save space
                )

                # Store metadata
                for key, value in result['metadata'].items():
                    if key == 'ch_names':
                        value = json.dumps(value)  # Serialize list
                    if value is not None:
                        h5f['metadata'].attrs[f'recording_{idx:06d}_{key}'] = value

                metadata_records.append(result['metadata'])
            else:
                errors.append(result)
                print(
                    f"Error processing {result['metadata']['path']}: {result['error']}")

        # Flush to disk
        h5f.flush()

    # Create metadata DataFrame and store as table
    if metadata_records:
        df = pd.DataFrame(metadata_records)
        df['ch_names'] = df['ch_names'].apply(json.dumps)  # Serialize

        # Store as dataset (for efficient querying)
        dt = h5py.string_dtype(encoding='utf-8')
        for col in df.columns:
            if df[col].dtype == object:
                data = df[col].fillna('').astype(str).values
                h5f['metadata'].create_dataset(col, data=data, dtype=dt)
            else:
                h5f['metadata'].create_dataset(col, data=df[col].values)

    # Store errors log
    if errors:
        error_df = pd.DataFrame([e['metadata'] for e in errors])
        error_df['error'] = [e['error'] for e in errors]
        error_df.to_csv(output_hdf5.replace('.h5', '_errors.csv'), index=False)
        print(
            f"\n{len(errors)} files failed. See {output_hdf5.replace('.h5', '_errors.csv')}")

    h5f.close()

    print(f"\n✓ Processing complete!")
    print(f"  Output: {output_hdf5}")
    print(f"  Total recordings: {len(metadata_records)}")
    print(f"  Failed: {len(errors)}")

    # Print compression stats
    file_size_gb = os.path.getsize(output_hdf5) / (1024**3)
    print(f"  File size: {file_size_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TUH EEG corpus to HDF5")
    parser.add_argument(
        '--tuh-dir',
        type=str,
        required=True,
        help='Path to TUH directory (containing edf/ subdirectory)'
    )
    parser.add_argument(
        '--output-hdf5',
        type=str,
        default='tuh_eeg_processed.h5',
        help='Output HDF5 file path'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=8,
        help='Number of parallel jobs'
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='gzip',
        choices=['gzip', 'lzf', 'none'],
        help='Compression algorithm'
    )
    parser.add_argument(
        '--compression-level',
        type=int,
        default=4,
        help='Compression level (0-9 for gzip)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start from scratch (ignore existing file)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Process only first N files (for testing/subset creation)'
    )

    args = parser.parse_args()

    create_hdf5_dataset(
        tuh_dir=args.tuh_dir,
        output_hdf5=args.output_hdf5,
        n_jobs=args.n_jobs,
        compression=args.compression if args.compression != 'none' else None,
        compression_level=args.compression_level,
        resume=not args.no_resume,
        max_files=args.max_files,
    )


if __name__ == '__main__':
    main()
