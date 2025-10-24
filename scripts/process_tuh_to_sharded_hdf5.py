#!/usr/bin/env python3
"""
Process TUH EEG corpus to sharded HDF5 files for fast parallel writing.

This script creates multiple HDF5 files (shards) that can be written in parallel,
then combined into a virtual HDF5 file for transparent single-file access.

Speed: ~1.5-2 hours for 69k files (vs 15 hours for single file)

Usage:
    python scripts/process_tuh_to_sharded_hdf5.py \
        --tuh-dir /path/to/tuh \
        --output-dir data/tuh/shards \
        --recordings-per-shard 1000 \
        --n-parallel-shards 12
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

# Workaround for LLVM JIT crash on ARM64
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Disable MNE config file locking (prevents warnings with parallel workers)
os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'
os.environ['MNE_USE_CUDA'] = 'false'

import h5py
import mne
# Disable MNE verbose output and config updates
mne.set_log_level('ERROR')
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def extract_metadata_from_path(file_path: str) -> Dict:
    """Extract subject, session, segment from TUH file path."""
    parts = Path(file_path).parts
    filename = Path(file_path).stem
    tokens = filename.split('_')

    subject_id = tokens[0]
    session = tokens[1]
    segment = tokens[2]

    session_dir = parts[-3]
    year = int(session_dir.split('_')[1]) if '_' in session_dir else None
    montage_dir = parts[-2]

    return {
        'subject_id': subject_id,
        'session': int(session[1:]),
        'segment': int(segment[1:]),
        'year': year,
        'montage': montage_dir,
        'path': file_path,
    }


def process_single_file(file_path: str, idx: int) -> Dict:
    """Process a single EDF file and return data + metadata."""
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        metadata = extract_metadata_from_path(file_path)
        metadata.update({
            'recording_id': idx,
            'sfreq': raw.info['sfreq'],
            'n_channels': len(raw.ch_names),
            'duration': raw.times[-1],
            'ch_names': raw.ch_names,
            'meas_date': str(raw.info['meas_date']) if raw.info['meas_date'] else None,
        })

        data = raw.get_data()

        del raw
        import gc
        gc.collect()

        return {
            'data': data,
            'metadata': metadata,
            'success': True,
            'error': None,
        }

    except Exception as e:
        import gc
        gc.collect()
        return {
            'data': None,
            'metadata': {'path': file_path, 'recording_id': idx},
            'success': False,
            'error': str(e),
        }


def process_shard(
    shard_num: int,
    shard_files: List[str],
    shard_indices: List[int],
    output_path: str,
    compression: str = None,
    shard_num_workers: int = 24,
) -> Dict:
    """Process one shard (batch of files) into a single HDF5 file."""

    print(f"\n[Shard {shard_num:03d}] Processing {len(shard_files)} files...")

    # Read all files in this shard in parallel
    # Workers calculated based on total cores and parallel shards
    # (set dynamically by create_sharded_hdf5 function)
    results = Parallel(
        n_jobs=shard_num_workers,  # Passed as parameter
        backend='loky',
        verbose=10,
        timeout=None,
    )(
        delayed(process_single_file)(file_path, idx)
        for file_path, idx in zip(shard_files, shard_indices)
    )

    # Write to HDF5
    print(f"[Shard {shard_num:03d}] Reading complete. Writing to {output_path}...")
    h5f = h5py.File(output_path, 'w')
    h5f.create_group('data')
    h5f.create_group('metadata')

    metadata_records = []
    errors = []

    write_interval = max(1, len(results) // 10)  # Print every 10%
    for i, result in enumerate(results):
        if result['success']:
            idx = result['metadata']['recording_id']
            dataset_name = f'recording_{idx:06d}'

            h5f['data'].create_dataset(
                dataset_name,
                data=result['data'],
                compression=compression,
                dtype='float32',
            )

            for key, value in result['metadata'].items():
                if key == 'ch_names':
                    value = json.dumps(value)
                if value is not None:
                    h5f['metadata'].attrs[f'{dataset_name}_{key}'] = value

            metadata_records.append(result['metadata'])
        else:
            errors.append(result)

        # Progress update every 10%
        if (i + 1) % write_interval == 0 or (i + 1) == len(results):
            progress_pct = ((i + 1) / len(results)) * 100
            print(f"[Shard {shard_num:03d}] Written {i + 1}/{len(results)} ({progress_pct:.0f}%)")

    h5f.close()
    print(f"[Shard {shard_num:03d}] ✓ Complete: {len(metadata_records)} success, {len(errors)} errors")

    return {
        'shard_num': shard_num,
        'output_path': output_path,
        'n_success': len(metadata_records),
        'n_errors': len(errors),
        'metadata': metadata_records,
        'errors': errors,
    }


def create_sharded_hdf5(
    tuh_dir: str,
    output_dir: str,
    recordings_per_shard: int = 1000,
    n_parallel_shards: int = 12,
    compression: str = None,
):
    """
    Create sharded HDF5 dataset from TUH EEG corpus.

    Parameters
    ----------
    tuh_dir : str
        Path to TUH directory
    output_dir : str
        Output directory for shard files
    recordings_per_shard : int
        Number of recordings per shard file
    n_parallel_shards : int
        Number of shards to process in parallel
    compression : str
        HDF5 compression ('gzip', 'lzf', or None)
    """

    os.makedirs(output_dir, exist_ok=True)

    # Auto-calculate optimal parallelization based on system resources
    import psutil
    total_cores = os.cpu_count() or 72
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Process 2 shards in parallel (empirically safe, 4 caused OOM)
    # Using same backend for both levels avoids nesting issues
    optimal_parallel_shards = 2
    workers_per_shard = 36  # 72 cores / 2 shards = 36 per shard

    # Memory estimation (for display only)
    memory_per_shard_gb = 200
    theoretical_max_shards = int(available_memory_gb / memory_per_shard_gb)

    # Override user parameter if it would cause OOM
    if n_parallel_shards > optimal_parallel_shards:
        print(f"⚠ Reducing n_parallel_shards from {n_parallel_shards} to {optimal_parallel_shards} (available memory: {available_memory_gb:.0f} GB)")
        n_parallel_shards = optimal_parallel_shards

    print(f"\n{'='*60}")
    print(f"Resource allocation:")
    print(f"  Total cores: {total_cores}")
    print(f"  Available memory: {available_memory_gb:.0f} GB")
    print(f"  Parallel shards: {n_parallel_shards}")
    print(f"  Workers per shard: {workers_per_shard}")
    print(f"  Total concurrent workers: {n_parallel_shards * workers_per_shard}")
    print(f"  Estimated peak memory: ~{n_parallel_shards * memory_per_shard_gb} GB")
    print(f"{'='*60}\n")

    # Find all EDF files
    print("Scanning for EDF files...")
    cache_file = os.path.join(output_dir, 'filelist.txt')

    if os.path.exists(cache_file):
        print(f"Loading cached file list from {cache_file}")
        with open(cache_file, 'r') as f:
            edf_files = [line.strip() for line in f]
    else:
        print("Using find command for fast filesystem scan...")
        result = subprocess.run(
            ['find', tuh_dir, '-name', '*.edf', '-type', 'f'],
            capture_output=True,
            text=True,
            check=True
        )
        edf_files = sorted(result.stdout.strip().split('\n'))

        with open(cache_file, 'w') as f:
            f.write('\n'.join(edf_files))
        print(f"Cached file list to {cache_file}")

    print(f"Found {len(edf_files)} EDF files")

    # Divide into shards
    n_shards = (len(edf_files) + recordings_per_shard - 1) // recordings_per_shard
    print(f"Creating {n_shards} shards ({recordings_per_shard} recordings each)")

    # Check for existing shards (resume functionality)
    existing_shards = set()
    for shard_num in range(n_shards):
        shard_path = os.path.join(output_dir, f'tuh_shard_{shard_num:03d}.h5')
        if os.path.exists(shard_path):
            existing_shards.add(shard_num)

    if existing_shards:
        print(f"Found {len(existing_shards)} existing shards - will skip these")
        print(f"Remaining shards to process: {n_shards - len(existing_shards)}")

    all_metadata = []
    all_errors = []

    # Load metadata from existing shards
    corrupted_shards = []
    if existing_shards:
        print("Loading metadata from existing shards...")
        existing_list = sorted(existing_shards)
        for i, shard_num in enumerate(existing_list):
            if (i + 1) % 10 == 0 or (i + 1) == len(existing_list):
                print(f"  Loaded {i + 1}/{len(existing_list)} shards...")

            shard_path = os.path.join(output_dir, f'tuh_shard_{shard_num:03d}.h5')

            try:
                with h5py.File(shard_path, 'r') as f:
                    for attr_name in f['metadata'].attrs:
                        if attr_name.endswith('_recording_id'):
                            # Reconstruct metadata for this recording
                            recording_name = attr_name.replace('_recording_id', '')
                            metadata = {}
                            for key in ['recording_id', 'subject_id', 'session', 'segment',
                                        'year', 'montage', 'path', 'sfreq', 'n_channels',
                                        'duration', 'ch_names', 'meas_date']:
                                full_key = f'{recording_name}_{key}'
                                if full_key in f['metadata'].attrs:
                                    metadata[key] = f['metadata'].attrs[full_key]
                            if metadata:
                                all_metadata.append(metadata)
            except (OSError, Exception) as e:
                # Corrupted shard - mark for deletion and regeneration
                print(f"  ⚠ Shard {shard_num:03d} is corrupted ({e.__class__.__name__}), will regenerate")
                corrupted_shards.append(shard_num)
                # Delete corrupted file
                try:
                    os.remove(shard_path)
                    print(f"  ✓ Deleted corrupted shard {shard_num:03d}")
                except:
                    pass

        # Remove corrupted shards from existing set
        for shard_num in corrupted_shards:
            existing_shards.discard(shard_num)

        if corrupted_shards:
            print(f"\n⚠ Found {len(corrupted_shards)} corrupted shard(s), will regenerate: {corrupted_shards}")
            print(f"Remaining valid shards: {len(existing_shards)}")

    # Process shards in batches (n_parallel_shards at a time)
    for batch_start in range(0, n_shards, n_parallel_shards):
        batch_end = min(batch_start + n_parallel_shards, n_shards)
        batch_shard_nums = list(range(batch_start, batch_end))

        print(f"\n{'='*60}")
        print(f"Processing shard batch {batch_start}-{batch_end-1} / {n_shards}")
        print(f"{'='*60}")

        # Prepare shard tasks (skip existing shards)
        shard_tasks = []
        for shard_num in batch_shard_nums:
            # Skip if already processed
            if shard_num in existing_shards:
                continue

            start_idx = shard_num * recordings_per_shard
            end_idx = min(start_idx + recordings_per_shard, len(edf_files))

            shard_files = edf_files[start_idx:end_idx]
            shard_indices = list(range(start_idx, end_idx))
            output_path = os.path.join(output_dir, f'tuh_shard_{shard_num:03d}.h5')

            shard_tasks.append((shard_num, shard_files, shard_indices, output_path, compression, workers_per_shard))

        # Skip this batch if all shards already exist
        if not shard_tasks:
            continue

        # Process shards in parallel using SAME backend as inner loop (loky)
        # Both outer and inner use loky - this is supported
        if len(shard_tasks) == 1:
            # Just run directly for single shard (no overhead)
            shard_results = []
            for shard_num, shard_files, shard_indices, output_path, compression, workers in shard_tasks:
                result = process_shard(shard_num, shard_files, shard_indices, output_path, compression, workers)
                shard_results.append(result)
        else:
            # Use loky for outer loop too (nested loky IS supported)
            shard_results = Parallel(n_jobs=len(shard_tasks), backend='loky', prefer='processes')(
                delayed(process_shard)(shard_num, shard_files, shard_indices, output_path, compression, workers)
                for shard_num, shard_files, shard_indices, output_path, compression, workers in shard_tasks
            )

        # Collect results
        for result in shard_results:
            all_metadata.extend(result['metadata'])
            all_errors.extend(result['errors'])
            print(f"\n[Shard {result['shard_num']:03d}] ✓ Complete: {result['n_success']} success, {result['n_errors']} errors")

    # Save master metadata CSV
    print("\nSaving master metadata CSV...")
    df = pd.DataFrame(all_metadata)
    df['ch_names'] = df['ch_names'].apply(json.dumps)
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    df.to_csv(metadata_path, index=False)
    print(f"Saved metadata to {metadata_path}")

    # Save errors
    if all_errors:
        error_df = pd.DataFrame([e['metadata'] for e in all_errors])
        error_df['error'] = [e['error'] for e in all_errors]
        error_path = os.path.join(output_dir, 'errors.csv')
        error_df.to_csv(error_path, index=False)
        print(f"Saved {len(all_errors)} errors to {error_path}")

    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total recordings: {len(all_metadata)}")
    print(f"  Total shards: {n_shards}")
    print(f"  Failed: {len(all_errors)}")
    print(f"\nNext step: Create virtual HDF5 file")
    print(f"  python scripts/create_virtual_hdf5.py --shard-dir {output_dir} --output {output_dir}/../tuh_complete.h5")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Convert TUH EEG to sharded HDF5")
    parser.add_argument('--tuh-dir', type=str, required=True,
                        help='Path to TUH directory')
    parser.add_argument('--output-dir', type=str, default='data/tuh/shards',
                        help='Output directory for shard files')
    parser.add_argument('--recordings-per-shard', type=int, default=1000,
                        help='Number of recordings per shard')
    parser.add_argument('--n-parallel-shards', type=int, default=2,
                        help='Number of shards to process in parallel')
    parser.add_argument('--compression', type=str, default='none',
                        choices=['gzip', 'lzf', 'none'],
                        help='HDF5 compression algorithm')

    args = parser.parse_args()

    create_sharded_hdf5(
        tuh_dir=args.tuh_dir,
        output_dir=args.output_dir,
        recordings_per_shard=args.recordings_per_shard,
        n_parallel_shards=args.n_parallel_shards,
        compression=args.compression if args.compression != 'none' else None,
    )


if __name__ == '__main__':
    main()
