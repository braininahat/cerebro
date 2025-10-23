#!/usr/bin/env python3
"""Process a subset of TUH files for quick experimentation."""

import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.process_tuh_to_hdf5 import create_hdf5_dataset

# Configuration
TUH_DIR = "/projects/academic/wenyaoxu/anarghya/research/eeg-data/tuh/tueg/v2.0.1"
N_FILES = 1000  # First 1000 files (~2 minutes to process)

# Find first N files
print(f"Finding first {N_FILES} EDF files...")
all_files = sorted(glob.glob(os.path.join(TUH_DIR, "**/*.edf"), recursive=True))
subset_files = all_files[:N_FILES]

print(f"Found {len(subset_files)} files")
print(f"Estimated time: ~{N_FILES * 0.06 / 60:.1f} minutes")

# Create temporary subset directory
import tempfile
import shutil

with tempfile.TemporaryDirectory() as tmpdir:
    subset_dir = os.path.join(tmpdir, "tuh_subset")
    os.makedirs(subset_dir, exist_ok=True)

    print(f"\nCopying {N_FILES} files to temporary directory...")
    for i, edf_file in enumerate(subset_files):
        rel_path = os.path.relpath(edf_file, TUH_DIR)
        dest_path = os.path.join(subset_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(edf_file, dest_path)
        if (i + 1) % 100 == 0:
            print(f"  Copied {i+1}/{N_FILES} files")

    # Process subset
    output_hdf5 = f"tuh_subset_{N_FILES}_files.h5"
    print(f"\nProcessing to {output_hdf5}...")

    create_hdf5_dataset(
        tuh_dir=subset_dir,
        output_hdf5=output_hdf5,
        n_jobs=16,
        compression='lzf',  # Fast compression
        compression_level=4,
        resume=False,
    )

    print(f"\n✓ Done! Subset saved to {output_hdf5}")
    print(f"  You can now use this for testing while the full dataset processes")
