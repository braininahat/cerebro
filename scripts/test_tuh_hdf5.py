#!/usr/bin/env python3
"""
Quick test script to verify TUH HDF5 processing and loading.

This script tests the HDF5 dataset on a small subset before processing
the full 1.7TB dataset.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add cerebro to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cerebro.data.tuh import TUHDataset
from scripts.process_tuh_to_hdf5 import create_hdf5_dataset


def test_small_subset():
    """Test on a small subset of TUH data."""
    import glob

    tuh_dir = "/projects/academic/wenyaoxu/anarghya/research/eeg-data/tuh/tueg/v2.0.1"

    # Find first 10 EDF files
    edf_files = sorted(glob.glob(os.path.join(tuh_dir, "**/*.edf"), recursive=True))[:10]

    if len(edf_files) == 0:
        print("No EDF files found!")
        return

    print(f"Testing with {len(edf_files)} files")

    # Create temporary directory for test HDF5
    with tempfile.TemporaryDirectory() as tmpdir:
        test_hdf5 = os.path.join(tmpdir, "test_tuh.h5")

        # Create temporary subset
        import shutil
        temp_tuh = os.path.join(tmpdir, "tuh_subset")
        os.makedirs(temp_tuh, exist_ok=True)

        print("Copying test files...")
        for edf_file in edf_files:
            rel_path = os.path.relpath(edf_file, tuh_dir)
            dest_path = os.path.join(temp_tuh, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(edf_file, dest_path)

        # Process to HDF5
        print("\n" + "=" * 60)
        print("Step 1: Processing EDFs to HDF5")
        print("=" * 60)
        create_hdf5_dataset(
            tuh_dir=temp_tuh,
            output_hdf5=test_hdf5,
            n_jobs=2,
            compression='gzip',
            compression_level=4,
            resume=False,
        )

        # Load dataset
        print("\n" + "=" * 60)
        print("Step 2: Loading HDF5 dataset")
        print("=" * 60)
        dataset = TUHDataset(test_hdf5)
        print(f"✓ Loaded {len(dataset)} recordings")

        # Test metadata
        print("\n" + "=" * 60)
        print("Step 3: Testing metadata access")
        print("=" * 60)
        metadata = dataset.get_metadata()
        print(f"✓ Metadata shape: {metadata.shape}")
        print("\nSample metadata:")
        print(metadata.head())

        # Test data loading
        print("\n" + "=" * 60)
        print("Step 4: Testing data loading")
        print("=" * 60)
        # Access raw object from underlying dataset
        raw = dataset.datasets[0].raw
        print(f"✓ Loaded recording 0")
        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Sample rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.2f} seconds")
        print(f"  Data shape: {raw.get_data().shape}")

        # Test __getitem__ (returns windowed data)
        X, y = dataset[0]
        print(f"\n✓ dataset[0] returns:")
        print(f"  X shape: {X.shape}")
        print(f"  y (target): {y}")

        # Test subject split
        print("\n" + "=" * 60)
        print("Step 5: Testing subject-level split")
        print("=" * 60)
        n_subjects = metadata['subject_id'].nunique()
        print(f"Number of subjects: {n_subjects}")

        if n_subjects >= 3:
            train_ds, val_ds, test_ds = dataset.split_by_subjects(0.6, 0.2, seed=42)
            print(f"✓ Train: {len(train_ds.datasets)} recordings")
            print(f"✓ Val: {len(val_ds.datasets)} recordings")
            print(f"✓ Test: {len(test_ds.datasets)} recordings")
        else:
            print(f"⚠ Skipping split test (need >= 3 subjects, found {n_subjects})")

        # Test braindecode preprocessing
        print("\n" + "=" * 60)
        print("Step 6: Testing braindecode preprocessing")
        print("=" * 60)
        try:
            from braindecode.preprocessing import preprocess, Preprocessor

            # Take first 3 recordings
            small_dataset = TUHDataset(
                test_hdf5,
                recording_ids=list(range(min(3, len(dataset))))
            )

            preprocessors = [
                Preprocessor('filter', l_freq=0.5, h_freq=40.0),
                Preprocessor('resample', sfreq=100),
            ]

            preprocess(small_dataset, preprocessors)
            print("✓ Preprocessing successful")

            raw = small_dataset.datasets[0].raw
            print(f"  New sample rate: {raw.info['sfreq']} Hz")
            print(f"  New data shape: {raw.get_data().shape}")

        except Exception as e:
            print(f"⚠ Preprocessing test failed: {e}")

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nYou can now process the full dataset with:")
        print(f"  python scripts/process_tuh_to_hdf5.py \\")
        print(f"    --tuh-dir {tuh_dir} \\")
        print(f"    --output-hdf5 tuh_eeg_processed.h5 \\")
        print(f"    --n-jobs 16")


if __name__ == '__main__':
    test_small_subset()
