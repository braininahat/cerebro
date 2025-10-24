#!/usr/bin/env python3
"""
Create a virtual HDF5 file that combines all shards into a single logical file.

This creates a small HDF5 file (few KB) that acts as a "view" over all shard files.
The virtual file can be used with braindecode and any HDF5 library as if it were
a single monolithic file.

Speed: ~1 minute for 70 shards

Usage:
    python scripts/create_virtual_hdf5.py \
        --shard-dir data/tuh/shards \
        --output data/tuh/tuh_complete.h5
"""

import argparse
import glob
import os
from pathlib import Path

import h5py
import pandas as pd


def create_virtual_hdf5(shard_dir: str, output_path: str):
    """
    Create virtual HDF5 file combining all shards.

    Parameters
    ----------
    shard_dir : str
        Directory containing tuh_shard_*.h5 files
    output_path : str
        Output path for virtual HDF5 file
    """

    print(f"Creating virtual HDF5 file: {output_path}")
    print(f"From shards in: {shard_dir}")

    # Find all shard files
    shard_pattern = os.path.join(shard_dir, 'tuh_shard_*.h5')
    shard_files = sorted(glob.glob(shard_pattern))

    if not shard_files:
        raise FileNotFoundError(f"No shard files found matching {shard_pattern}")

    print(f"Found {len(shard_files)} shard files")

    # Load metadata to get recording info
    metadata_path = os.path.join(shard_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    print(f"Total recordings: {len(metadata)}")

    # Peek at first shard to get data shape per recording
    with h5py.File(shard_files[0], 'r') as f:
        first_recording = list(f['data'].keys())[0]
        sample_shape = f['data'][first_recording].shape
        sample_dtype = f['data'][first_recording].dtype
        print(f"Recording shape: {sample_shape}, dtype: {sample_dtype}")

    # Create virtual HDF5 file
    print("\nCreating virtual datasets...")

    with h5py.File(output_path, 'w') as vf:
        # Create groups
        data_group = vf.create_group('data')
        metadata_group = vf.create_group('metadata')

        # Process each shard
        print(f"Linking {len(shard_files)} shards...")
        for i, shard_file in enumerate(shard_files):
            if (i + 1) % 10 == 0 or (i + 1) == len(shard_files):
                print(f"  Linked {i + 1}/{len(shard_files)} shards...")

            shard_path = os.path.abspath(shard_file)  # Must be absolute path

            with h5py.File(shard_file, 'r') as sf:
                # Link all recordings from this shard
                for recording_name in sf['data'].keys():
                    # Create virtual dataset that points to shard
                    source = h5py.ExternalLink(shard_path, f'/data/{recording_name}')
                    data_group[recording_name] = source

                # Copy metadata attributes
                for attr_name, attr_value in sf['metadata'].attrs.items():
                    metadata_group.attrs[attr_name] = attr_value

        # Store master metadata as datasets (for efficient queries)
        print("Storing metadata tables...")
        for col in metadata.columns:
            if metadata[col].dtype == object:
                # String columns
                dt = h5py.string_dtype(encoding='utf-8')
                data = metadata[col].fillna('').astype(str).values
                metadata_group.create_dataset(col, data=data, dtype=dt)
            else:
                # Numeric columns
                metadata_group.create_dataset(col, data=metadata[col].values)

    print(f"\n✓ Virtual HDF5 file created: {output_path}")

    # Verify
    print("\nVerifying virtual file...")
    with h5py.File(output_path, 'r') as vf:
        n_recordings = len(vf['data'].keys())
        print(f"  Accessible recordings: {n_recordings}")

        # Test reading one recording
        first_rec = list(vf['data'].keys())[0]
        test_data = vf['data'][first_rec][:]
        print(f"  Test read: {first_rec} shape={test_data.shape}")

    print(f"\n{'='*60}")
    print("Success! You can now use the virtual file with braindecode:")
    print(f"  import h5py")
    print(f"  f = h5py.File('{output_path}', 'r')")
    print(f"  # Access like single file")
    print(f"  data = f['data']['recording_000000'][:]")
    print(f"  metadata = pd.DataFrame({{")
    print(f"      col: f['metadata'][col][:]")
    print(f"      for col in f['metadata'].keys()")
    print(f"  }})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Create virtual HDF5 from sharded files"
    )
    parser.add_argument(
        '--shard-dir',
        type=str,
        required=True,
        help='Directory containing tuh_shard_*.h5 files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for virtual HDF5 file'
    )

    args = parser.parse_args()

    create_virtual_hdf5(
        shard_dir=args.shard_dir,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
