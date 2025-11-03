#!/usr/bin/env python3
"""Reconstruct metadata for completed .npy cache file.

The window data (.npy) was successfully saved, but metadata crashed.
This reconstructs the metadata by re-windowing WITHOUT loading the actual EEG data.
"""

import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
cache_root = Path("/media/varun/OS/Users/varun/DATASETS/HBN/cache")
window_config_id = "windows_082f5841_win20_stride10_full"
memmap_path = cache_root / "windowed" / f"{window_config_id}.npy"
metadata_path = cache_root / "windowed" / "windows" / f"{window_config_id}_meta.parquet"
complete_marker = cache_root / "windowed" / f"{window_config_id}.complete"

# Verify .npy exists
if not memmap_path.exists():
    raise FileNotFoundError(f".npy file not found: {memmap_path}")

logger.info(f"Found complete .npy file: {memmap_path}")
logger.info(f"  Size: {memmap_path.stat().st_size / 1e9:.2f} GB")

# Load .npy to get actual window count
logger.info("Loading .npy header to get window count...")
windows_array = np.load(memmap_path, mmap_mode='r')
total_windows = windows_array.shape[0]
logger.info(f"  Total windows in .npy: {total_windows:,}")

# Load raw cache manifest to get recordings
logger.info("Loading raw cache manifest...")
raw_manifest_path = cache_root / "raw" / "raw_manifest.parquet"
raw_manifest = pd.read_parquet(raw_manifest_path)

# Filter to full dataset (not mini)
recordings = raw_manifest[raw_manifest["mini"] == False].copy()
logger.info(f"  Found {len(recordings)} recordings in raw cache")

# Reconstruct metadata by computing windows for each recording
logger.info("Reconstructing metadata...")
window_len_s = 2.0
stride_s = 1.0
sfreq = 100
window_samples = int(window_len_s * sfreq)
stride_samples = int(stride_s * sfreq)

metadata_rows = []
total_computed = 0

for idx, rec in recordings.iterrows():
    # Skip if raw zarr doesn't exist (failed during cache build)
    raw_path = Path(rec["raw_zarr_path"])
    if not raw_path.exists():
        logger.warning(f"  Skipping missing recording: {rec['recording_id']}")
        continue

    try:
        # Load just the shape from raw zarr (fast - no data loaded)
        raw_zarr = zarr.open(str(rec["raw_zarr_path"]), mode='r')
        n_channels, n_samples = raw_zarr.shape

        # Compute number of windows
        n_windows = max(0, (n_samples - window_samples) // stride_samples + 1)

        # Generate metadata for this recording
        for i in range(n_windows):
            metadata_rows.append({
                "recording_id": rec["recording_id"],
                "dataset": rec["dataset"],
                "release": rec["release"],
                "subject": rec["subject"],
                "task": rec["task"],
                "mini": rec["mini"],
                "time_offset_s": i * stride_s,
                "window_idx": total_computed + i,
                "array_index": total_computed + i,
            })

        total_computed += n_windows

    except Exception as e:
        logger.warning(f"  Failed to process {rec['recording_id']}: {e}")
        continue

    if (idx + 1) % 100 == 0:
        logger.info(f"  Processed {idx + 1}/{len(recordings)} recordings, {total_computed:,} windows")

logger.info(f"Computed metadata for {total_computed:,} windows")

# Verify counts match
if total_computed != total_windows:
    logger.error(f"ERROR: Window count mismatch!")
    logger.error(f"  .npy file has: {total_windows:,}")
    logger.error(f"  Computed from recordings: {total_computed:,}")
    logger.error(f"  Difference: {abs(total_windows - total_computed):,}")

    # This might happen if some recordings failed during processing
    # Trim metadata to match actual .npy size
    if total_computed > total_windows:
        logger.warning(f"Trimming metadata to match .npy size ({total_windows:,})")
        metadata_rows = metadata_rows[:total_windows]
        total_computed = total_windows

# Create DataFrame
metadata_df = pd.DataFrame(metadata_rows)

# Save metadata
logger.info(f"Saving metadata to {metadata_path}...")
metadata_path.parent.mkdir(parents=True, exist_ok=True)
metadata_df.to_parquet(metadata_path, index=False)
logger.info(f"✓ Metadata saved ({len(metadata_df):,} rows)")

# Create completion marker
logger.info("Creating completion marker...")
complete_marker.write_text(f"completed at {datetime.now().isoformat()}")

logger.info("")
logger.info("=" * 70)
logger.info("✓ RECOVERY COMPLETE!")
logger.info("=" * 70)
logger.info(f"  Window data: {memmap_path}")
logger.info(f"  Metadata: {metadata_path}")
logger.info(f"  Total windows: {total_windows:,}")
logger.info("")
logger.info("You can now run your training command - cache is ready!")
