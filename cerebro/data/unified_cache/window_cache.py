"""Level 2: Window cache builder.

Builds windowed datasets from Level 1 raw cache.
Multiple window configurations can coexist.
"""

import gc
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import zarr
from joblib import Parallel, delayed
from numcodecs import Blosc
from tqdm import tqdm

from cerebro.data.unified_cache.lazy_dataset import MemmapWindowDataset
from cerebro.data.unified_cache.raw_cache import GracefulInterruptHandler

logger = logging.getLogger(__name__)


def _extract_windows_vectorized(
    data: np.ndarray,
    window_samples: int,
    stride_samples: int
) -> np.ndarray:
    """Extract sliding windows using vectorized numpy operations.

    Uses numpy.lib.stride_tricks.sliding_window_view for fast windowing.
    ~10-50x faster than loop-based approach.

    Args:
        data: Raw EEG data (n_channels, n_samples)
        window_samples: Window size in samples
        stride_samples: Stride in samples

    Returns:
        Windows array (n_windows, n_channels, window_samples)
    """
    n_channels, n_samples = data.shape
    n_windows = max(0, (n_samples - window_samples) // stride_samples + 1)

    if n_windows == 0:
        return np.zeros((0, n_channels, window_samples), dtype=np.float32)

    # Use sliding_window_view for vectorized windowing
    from numpy.lib.stride_tricks import sliding_window_view

    # Create sliding windows along time axis
    # Result: (n_channels, n_samples - window_samples + 1, window_samples)
    windowed = sliding_window_view(data, window_samples, axis=1)

    # Sample at stride intervals
    # Result: (n_channels, n_windows, window_samples)
    windowed = windowed[:, ::stride_samples, :]

    # Transpose to (n_windows, n_channels, window_samples)
    # Force memory copy to break stride view reference to original data
    windows = windowed.transpose(1, 0, 2).astype(np.float32).copy()

    return windows


def _process_single_recording(
    rec_tuple: Tuple[int, pd.Series],
    window_len_s: float,
    stride_s: float,
    sfreq: int
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Process one recording to extract windows.

    Joblib calls this function for each recording, pulling from queue automatically.
    Memory-efficient: processes one recording at a time, cleans up immediately.

    Args:
        rec_tuple: (index, recording_series) from DataFrame.iterrows()
        window_len_s: Window length in seconds
        stride_s: Stride in seconds
        sfreq: Sampling frequency

    Returns:
        Tuple of (windows, metadata_rows) for this recording
    """
    idx, rec = rec_tuple
    window_samples = int(window_len_s * sfreq)
    stride_samples = int(stride_s * sfreq)

    try:
        # Load raw from Level 1 cache
        raw_zarr = zarr.open(str(rec["raw_zarr_path"]), mode='r')
        data = np.array(raw_zarr)  # Shape: (n_channels, n_samples)

        # Extract windows using vectorized approach
        windows = _extract_windows_vectorized(data, window_samples, stride_samples)

        # Explicitly close zarr file handle to prevent accumulation
        if hasattr(raw_zarr.store, 'close'):
            raw_zarr.store.close()

        n_windows = windows.shape[0]
        if n_windows == 0:
            # Return empty arrays for short recordings
            return np.zeros((0, data.shape[0], window_samples), dtype=np.float32), []

        # Build metadata for each window
        metadata_rows = []
        for i in range(n_windows):
            metadata_rows.append({
                "recording_id": rec["recording_id"],
                "dataset": rec["dataset"],
                "release": rec["release"],
                "subject": rec["subject"],
                "task": rec["task"],
                "mini": rec["mini"],
                "time_offset_s": i * stride_s,
            })

        # Explicit cleanup before return
        del data, raw_zarr
        gc.collect()

        return windows, metadata_rows

    except Exception as e:
        logger.error(f"Failed to window {rec['recording_id']}: {e}")
        # Return empty arrays on failure
        return np.zeros((0, 129, window_samples), dtype=np.float32), []


def partition_list(items: List, n_partitions: int) -> List[List]:
    """Divide list into n roughly equal chunks.

    Args:
        items: List to partition
        n_partitions: Number of partitions

    Returns:
        List of chunks
    """
    chunk_size = len(items) // n_partitions
    remainder = len(items) % n_partitions
    chunks = []
    start = 0
    for i in range(n_partitions):
        # Distribute remainder across first chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


class WindowCacheBuilder:
    """Builds Level 2 window cache from Level 1 raw cache.

    Args:
        cache_manager: Parent UniversalCacheManager instance
    """

    def __init__(self, cache_manager):
        self.cache_mgr = cache_manager
        self.window_manifest = self.cache_mgr.window_manifest

    def get_or_build(
        self,
        recordings: pd.DataFrame,
        window_len_s: float,
        stride_s: float,
        crop_len_s: Optional[float] = None,
        mode: str = 'train'
    ) -> MemmapWindowDataset:
        """Get windowed dataset (builds if missing).

        Args:
            recordings: DataFrame from query_raw()
            window_len_s: Window length in seconds
            stride_s: Stride in seconds
            crop_len_s: Optional crop length for augmentation

        Returns:
            MemmapWindowDataset instance
        """
        # Infer mini flag from recordings (all should be same)
        if len(recordings) == 0:
            raise ValueError("Cannot create window dataset from empty recordings DataFrame")
        mini = bool(recordings["mini"].iloc[0])

        # Validate channel counts - filter out non-standard recordings
        # Use mode (most common value) as the expected channel count
        if "n_channels" in recordings.columns:
            expected_channels = int(recordings["n_channels"].mode()[0])
            invalid_mask = recordings["n_channels"] != expected_channels

            if invalid_mask.any():
                invalid_recs = recordings[invalid_mask]
                logger.warning(
                    f"⚠️  Filtering {len(invalid_recs)} recording(s) with non-standard channel counts "
                    f"(expected {expected_channels} channels):"
                )
                for _, rec in invalid_recs.iterrows():
                    logger.warning(
                        f"    - {rec['recording_id']}: {rec['n_channels']} channels "
                        f"(subject={rec['subject']}, task={rec['task']}, release={rec['release']})"
                    )

                # Filter to only standard recordings
                recordings = recordings[~invalid_mask].reset_index(drop=True)
                logger.info(f"  Continuing with {len(recordings)} recordings after filtering")

                # Check if we have any recordings left
                if len(recordings) == 0:
                    raise ValueError(
                        f"All recordings filtered out due to non-standard channel counts. "
                        f"Expected {expected_channels} channels."
                    )

        # Get window config ID (includes mini to prevent mini/full collision)
        window_config_id = self.cache_mgr._get_window_config_id(window_len_s, stride_s, mini)
        memmap_path = self.cache_mgr.window_cache_dir / f"{window_config_id}.npy"
        metadata_path = self.cache_mgr._get_window_metadata_path(window_config_id)
        complete_marker_path = memmap_path.parent / f"{window_config_id}.complete"

        # Check if window cache exists and is complete
        if memmap_path.exists() and metadata_path.exists() and complete_marker_path.exists():
            logger.info(f"Loading windowed dataset from cache (memmap): {window_config_id}")
            metadata = pd.read_parquet(metadata_path)

            # Filter metadata to match requested recordings
            metadata_filtered = metadata[
                metadata["recording_id"].isin(recordings["recording_id"].values)
            ].reset_index(drop=True)

            # Debug: Check for NaN after filtering
            if metadata_filtered["array_index"].isna().any():
                logger.error(f"NaN array_index after filtering!")
                logger.error(f"  Requested recordings: {len(recordings)}")
                logger.error(f"  Filtered metadata: {len(metadata_filtered)}")
                logger.error(f"  NaN count: {metadata_filtered['array_index'].isna().sum()}")
                # Drop NaN rows as defensive fix
                logger.warning(f"Dropping {metadata_filtered['array_index'].isna().sum()} rows with NaN array_index")
                metadata_filtered = metadata_filtered.dropna(subset=["array_index"]).reset_index(drop=True)

            return MemmapWindowDataset(
                memmap_path=memmap_path,
                metadata=metadata_filtered,
                crop_len_s=crop_len_s,
                sfreq=self.cache_mgr.preprocessing_params["sfreq"],
                mode=mode
            )

        # Build window cache (no incremental builds for memmap)
        logger.info(f"Building windowed dataset (memmap): {window_config_id}")
        return self._build_window_cache(
            recordings, window_len_s, stride_s, crop_len_s,
            window_config_id, memmap_path, metadata_path, complete_marker_path, mode
        )

    def _build_window_cache(
        self,
        recordings: pd.DataFrame,
        window_len_s: float,
        stride_s: float,
        crop_len_s: Optional[float],
        window_config_id: str,
        memmap_path: Path,
        metadata_path: Path,
        complete_marker_path: Path,
        mode: str = 'train'
    ) -> MemmapWindowDataset:
        """Build window cache from raw cache using parallel workers.

        Note: Memmap does not support incremental builds. Always rebuilds from scratch.

        Args:
            recordings: DataFrame of recordings to window
            window_len_s: Window length in seconds
            stride_s: Stride in seconds
            crop_len_s: Optional crop length
            window_config_id: Config ID for this window setup
            memmap_path: Path to save numpy array (.npy file)
            metadata_path: Path to save metadata
            complete_marker_path: Path to completion marker file

        Returns:
            MemmapWindowDataset
        """
        sfreq = self.cache_mgr.preprocessing_params["sfreq"]
        window_samples = int(window_len_s * sfreq)

        # Ensure output directory exists
        memmap_path.parent.mkdir(parents=True, exist_ok=True)

        # Estimate total windows needed (for pre-allocation)
        # Use average of 500 windows per recording as conservative estimate
        estimated_windows = len(recordings) * 500

        # Get channel count from first recording
        if len(recordings) == 0:
            raise ValueError("Cannot create window dataset from empty recordings DataFrame")
        first_raw_path = recordings.iloc[0]["raw_zarr_path"]
        first_raw = zarr.open(str(first_raw_path), mode='r')
        n_channels = first_raw.shape[0]

        # Pre-allocate memmap file with estimated size
        logger.info(f"Pre-allocating memmap file: estimated {estimated_windows:,} windows...")
        memmap_array = np.lib.format.open_memmap(
            str(memmap_path),
            mode='w+',
            dtype=np.float32,
            shape=(estimated_windows, n_channels, window_samples)
        )

        # Track progress
        all_metadata = []
        total_windows = 0

        # Process all recordings in chunks
        max_workers = 32
        chunk_size = 10

        recordings_list = list(recordings.iterrows())
        n_recordings = len(recordings_list)
        n_chunks = (n_recordings + chunk_size - 1) // chunk_size

        logger.info(f"Windowing with {max_workers} parallel workers (chunked processing)")
        logger.info(f"Processing {n_recordings} recordings in {n_chunks} chunks of {chunk_size}")

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_recordings)
            chunk = recordings_list[chunk_start:chunk_end]

            logger.info(f"Processing chunk {chunk_idx + 1}/{n_chunks} ({len(chunk)} recordings)...")

            # Process chunk in parallel with joblib
            chunk_results = Parallel(
                n_jobs=max_workers,
                backend='loky',
                batch_size=1,
                verbose=0,
                max_nbytes='100M',
                pre_dispatch='2*n_jobs'
            )(
                delayed(_process_single_recording)(
                    rec, window_len_s, stride_s, sfreq
                )
                for rec in tqdm(chunk, desc=f"Chunk {chunk_idx + 1}/{n_chunks}", unit="rec")
            )

            # Write windows directly to memmap file (incremental, no RAM accumulation)
            for windows, rec_metadata in chunk_results:
                if windows.shape[0] > 0:
                    n_windows = windows.shape[0]

                    # Check if we need to expand the array (shouldn't happen with good estimate)
                    if total_windows + n_windows > memmap_array.shape[0]:
                        logger.warning(f"Expanding memmap array (estimate was too small)")
                        # Flush current memmap
                        del memmap_array
                        # Resize file
                        new_size = (total_windows + n_windows) * 2  # Double to avoid frequent resizes
                        logger.info(f"  Resizing from {estimated_windows:,} to {new_size:,} windows...")
                        # Reload with new size (numpy doesn't support in-place resize, so we copy)
                        temp_path = memmap_path.parent / f"{memmap_path.stem}_temp.npy"
                        old_data = np.load(memmap_path, mmap_mode='r')[:total_windows]
                        new_array = np.lib.format.open_memmap(
                            str(temp_path),
                            mode='w+',
                            dtype=np.float32,
                            shape=(new_size, n_channels, window_samples)
                        )
                        new_array[:total_windows] = old_data
                        del old_data
                        temp_path.rename(memmap_path)
                        memmap_array = new_array
                        estimated_windows = new_size

                    # Write windows directly to memmap (writes to disk immediately)
                    memmap_array[total_windows:total_windows + n_windows] = windows

                    # Update metadata with array indices
                    for i, meta in enumerate(rec_metadata):
                        meta["window_idx"] = total_windows + i
                        meta["array_index"] = total_windows + i
                        all_metadata.append(meta)

                    total_windows += n_windows
                    del windows
                    del rec_metadata

            # Clear memory after each chunk
            del chunk_results
            gc.collect()

            # Flush memmap to disk periodically
            if chunk_idx % 10 == 0:  # Flush every 10 chunks
                memmap_array.flush()

            logger.info(f"  ✓ Chunk {chunk_idx + 1}/{n_chunks} complete: {total_windows:,} total windows written to disk")

        # Trim memmap file to actual size (if estimated size was larger than actual)
        if total_windows < estimated_windows:
            logger.info(f"Trimming memmap file from {estimated_windows:,} to {total_windows:,} windows...")
            # Flush current data
            memmap_array.flush()
            del memmap_array

            # Load actual data
            actual_data = np.load(memmap_path, mmap_mode='r')[:total_windows]

            # Overwrite file with trimmed data
            trimmed_array = np.lib.format.open_memmap(
                str(memmap_path),
                mode='w+',
                dtype=np.float32,
                shape=(total_windows, n_channels, window_samples)
            )
            trimmed_array[:] = actual_data
            trimmed_array.flush()
            del actual_data, trimmed_array
            gc.collect()
        else:
            # Just flush if size was perfect
            memmap_array.flush()
            del memmap_array
            gc.collect()

        logger.info(f"✓ Saved {total_windows:,} windows to {memmap_path.name} ({total_windows * n_channels * window_samples * 4 / 1e9:.2f} GB)")

        # Save metadata
        logger.info(f"Saving metadata ({len(all_metadata)} rows)...")
        metadata_df = pd.DataFrame(all_metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        metadata_df.to_parquet(metadata_path, index=False)
        del all_metadata
        gc.collect()

        # Create completion marker
        logger.info("Creating completion marker...")
        complete_marker_path.write_text(f"completed at {datetime.now().isoformat()}")

        # Update window manifest
        self.window_manifest.mark_complete({
            "window_config_id": window_config_id,
            "raw_config_hash": self.cache_mgr.preprocessing_hash,
            "window_len_s": window_len_s,
            "stride_s": stride_s,
            "n_windows": total_windows,
            "window_zarr_path": str(memmap_path),  # Keep same field name for compatibility
            "metadata_path": str(metadata_path),
            "error_msg": ""
        })
        self.window_manifest.save()

        logger.info(f"✓ Created {total_windows:,} windows using {max_workers} workers")

        # Return dataset
        return MemmapWindowDataset(
            memmap_path=memmap_path,
            metadata=metadata_df,
            crop_len_s=crop_len_s,
            sfreq=sfreq,
            mode=mode
        )
