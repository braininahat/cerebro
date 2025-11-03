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
from joblib import Parallel, delayed
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


def _extract_windows_at_times(
    data: np.ndarray,
    start_times_samples: List[int],
    window_samples: int
) -> np.ndarray:
    """Extract windows at arbitrary start times (unified for sliding + event-locked).

    This function unifies two windowing paradigms:
    - Sliding window: start_times = [0, stride, 2*stride, ...]
    - Event-locked: start_times = [event1_onset, event2_onset, ...]

    Args:
        data: Raw EEG data (n_channels, n_samples)
        start_times_samples: List of window start times in samples
        window_samples: Window size in samples

    Returns:
        Windows array (n_windows, n_channels, window_samples)
    """
    n_channels, n_samples = data.shape
    n_windows = len(start_times_samples)

    if n_windows == 0:
        return np.zeros((0, n_channels, window_samples), dtype=np.float32)

    # Pre-allocate output array
    windows = np.zeros((n_windows, n_channels, window_samples), dtype=np.float32)

    # Extract each window
    for i, start_sample in enumerate(start_times_samples):
        end_sample = start_sample + window_samples

        # Boundary check: skip windows that extend past recording end
        if end_sample > n_samples or start_sample < 0:
            # Fill with zeros (will be filtered out later via metadata)
            continue

        # Extract window
        windows[i] = data[:, start_sample:end_sample]

    return windows


def _process_single_recording(
    rec_tuple: Tuple[int, pd.Series],
    window_len_s: float,
    stride_s: Optional[float],
    event_config: Optional[Dict[str, Any]],
    sfreq: int
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Process one recording to extract windows (sliding or event-locked).

    Joblib calls this function for each recording, pulling from queue automatically.
    Memory-efficient: processes one recording at a time, cleans up immediately.

    Args:
        rec_tuple: (index, recording_series) from DataFrame.iterrows()
        window_len_s: Window length in seconds
        stride_s: Stride in seconds (None for event-locked)
        event_config: Event configuration dict (None for sliding window)
        sfreq: Sampling frequency

    Returns:
        Tuple of (windows, metadata_rows) for this recording
    """
    idx, rec = rec_tuple
    window_samples = int(window_len_s * sfreq)

    try:
        # Load raw from Level 1 cache (memory-mapped .npy)
        raw_path = rec["raw_zarr_path"]  # Field name kept for compatibility
        data = np.load(str(raw_path), mmap_mode='r')  # Shape: (n_channels, n_samples)

        if event_config is None:
            # ===== SLIDING WINDOW MODE (existing logic) =====
            stride_samples = int(stride_s * sfreq)
            windows = _extract_windows_vectorized(data, window_samples, stride_samples)

            n_windows = windows.shape[0]
            if n_windows == 0:
                return np.zeros((0, data.shape[0], window_samples), dtype=np.float32), []

            # Build metadata for sliding windows
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

        else:
            # ===== EVENT-LOCKED MODE (new logic) =====
            # Load annotations from JSON
            from pathlib import Path
            import json

            annotations_path = Path(raw_path).with_suffix('.annotations.json')

            if not annotations_path.exists():
                # No annotations = no events = no windows
                return np.zeros((0, data.shape[0], window_samples), dtype=np.float32), []

            with open(annotations_path, 'r') as f:
                annotations_data = json.load(f)

            # Filter events by description (e.g., 'stimulus_anchor')
            event_type = event_config.get('event_type', 'stimulus_anchor')
            shift_after_event = event_config.get('shift_after_event', 0.5)
            target_field = event_config.get('target_field', None)

            # Find matching events
            event_indices = [
                i for i, desc in enumerate(annotations_data['descriptions'])
                if desc == event_type
            ]

            if len(event_indices) == 0:
                # No matching events
                return np.zeros((0, data.shape[0], window_samples), dtype=np.float32), []

            # Compute window start times (event onset + shift)
            start_times_s = [
                annotations_data['onsets'][i] + shift_after_event
                for i in event_indices
            ]
            start_times_samples = [int(t * sfreq) for t in start_times_s]

            # Extract windows at event times
            windows = _extract_windows_at_times(data, start_times_samples, window_samples)

            # Build metadata with event information and targets
            metadata_rows = []
            extras_list = annotations_data.get('extras', [None] * len(annotations_data['onsets']))

            for i, event_idx in enumerate(event_indices):
                row = {
                    "recording_id": rec["recording_id"],
                    "dataset": rec["dataset"],
                    "release": rec["release"],
                    "subject": rec["subject"],
                    "task": rec["task"],
                    "mini": rec["mini"],
                    "event_type": event_type,
                    "event_onset_s": annotations_data['onsets'][event_idx],
                }

                # Add target if requested (e.g., RT from stimulus)
                if target_field and extras_list[event_idx]:
                    extras = extras_list[event_idx]
                    if target_field in extras:
                        row["target"] = extras[target_field]

                        # Also add other useful event metadata
                        for key in ['correct', 'response_type', 'rt_from_trialstart']:
                            if key in extras:
                                row[key] = extras[key]

                metadata_rows.append(row)

        # Explicit cleanup before return
        del data
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
        stride_s: Optional[float],
        event_config: Optional[Dict[str, Any]],
        crop_len_s: Optional[float] = None,
        mode: str = 'train'
    ) -> MemmapWindowDataset:
        """Get windowed dataset (builds if missing). Supports sliding and event-locked modes.

        Args:
            recordings: DataFrame from query_raw()
            window_len_s: Window length in seconds
            stride_s: Stride in seconds (None for event-locked)
            event_config: Event configuration dict (None for sliding window)
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

        # Get window config ID (includes event_config hash for event-locked mode)
        window_config_id = self.cache_mgr._get_window_config_id(window_len_s, stride_s, event_config, mini)
        memmap_path = self.cache_mgr.window_cache_dir / f"{window_config_id}.npy"
        metadata_path = self.cache_mgr._get_window_metadata_path(window_config_id)
        progress_path = self.cache_mgr.window_cache_dir / f"{window_config_id}_completed.txt"

        # Check for incomplete build first (before loading cache)
        if progress_path.exists():
            # Progress file exists = incomplete build, must resume
            logger.info(f"Detected incomplete cache build (progress file exists), attempting resume...")
            return self._resume_window_cache(
                recordings, window_len_s, stride_s, event_config, crop_len_s,
                window_config_id, memmap_path, metadata_path, progress_path, mode
            )

        # Check if window cache exists and is complete (no progress file)
        if memmap_path.exists() and metadata_path.exists():
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

        # Check for corrupted cache (memmap exists but metadata missing, no progress file)
        if memmap_path.exists():
            logger.warning(f"Found memmap without metadata (and no progress file) - corrupted cache")
            logger.info(f"Deleting corrupted memmap and rebuilding from scratch...")
            memmap_path.unlink()

        # Build window cache from scratch
        logger.info(f"Building windowed dataset (memmap): {window_config_id}")
        return self._build_window_cache(
            recordings, window_len_s, stride_s, event_config, crop_len_s,
            window_config_id, memmap_path, metadata_path, mode
        )

    def _build_window_cache(
        self,
        recordings: pd.DataFrame,
        window_len_s: float,
        stride_s: Optional[float],
        event_config: Optional[Dict[str, Any]],
        crop_len_s: Optional[float],
        window_config_id: str,
        memmap_path: Path,
        metadata_path: Path,
        mode: str = 'train'
    ) -> MemmapWindowDataset:
        """Build window cache from raw cache using parallel workers.

        Note: Memmap does not support incremental builds. Always rebuilds from scratch.

        Args:
            recordings: DataFrame of recordings to window
            window_len_s: Window length in seconds
            stride_s: Stride in seconds (None for event-locked)
            event_config: Event configuration dict (None for sliding window)
            crop_len_s: Optional crop length
            window_config_id: Config ID for this window setup
            memmap_path: Path to save numpy array (.npy file)
            metadata_path: Path to save metadata

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
        first_raw_path = recordings.iloc[0]["raw_zarr_path"]  # Field name kept for compatibility
        first_raw = np.load(str(first_raw_path), mmap_mode='r')
        n_channels = first_raw.shape[0]
        del first_raw  # Release memmap handle

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

        # Initialize simple progress tracking (just recording IDs)
        completed_path = memmap_path.parent / f"{window_config_id}_completed.txt"
        completed_ids = set()

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
                    rec, window_len_s, stride_s, event_config, sfreq
                )
                for rec in tqdm(chunk, desc=f"Chunk {chunk_idx + 1}/{n_chunks}", unit="rec")
            )

            # Write windows directly to memmap file (incremental, no RAM accumulation)
            for windows, rec_metadata in chunk_results:
                if windows.shape[0] > 0:
                    n_windows = windows.shape[0]
                    recording_id = rec_metadata[0]["recording_id"]  # All windows from same recording
                    array_start_idx = total_windows

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

                    # Mark this recording as complete
                    completed_ids.add(recording_id)

                    del windows
                    del rec_metadata

            # Clear memory after each chunk
            del chunk_results
            gc.collect()

            # Flush memmap to disk periodically
            if chunk_idx % 10 == 0:  # Flush every 10 chunks
                memmap_array.flush()

            # Save metadata incrementally (append mode for resilience)
            if len(all_metadata) > 0:
                metadata_df = pd.DataFrame(all_metadata)
                if metadata_path.exists():
                    # Append to existing metadata
                    existing_metadata = pd.read_parquet(metadata_path)
                    metadata_df = pd.concat([existing_metadata, metadata_df], ignore_index=True)
                metadata_df.to_parquet(metadata_path, index=False)
                logger.debug(f"  Saved metadata incrementally ({len(metadata_df)} total rows)")

                # Clear metadata buffer to avoid memory accumulation
                all_metadata.clear()
                del metadata_df
                gc.collect()

            # Save progress (survives interruptions)
            completed_path.write_text('\n'.join(sorted(completed_ids)) + '\n')

            logger.info(f"  ✓ Chunk {chunk_idx + 1}/{n_chunks} complete: {total_windows:,} total windows written to disk")

        # Final flush of metadata buffer (handles any remaining unflushed metadata)
        if len(all_metadata) > 0:
            metadata_df = pd.DataFrame(all_metadata)
            if metadata_path.exists():
                # Append to existing metadata
                existing_metadata = pd.read_parquet(metadata_path)
                metadata_df = pd.concat([existing_metadata, metadata_df], ignore_index=True)
            metadata_df.to_parquet(metadata_path, index=False)
            logger.debug(f"  Saved final metadata batch ({len(metadata_df)} total rows)")
            all_metadata.clear()
            del metadata_df
            gc.collect()

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

        # Metadata already saved incrementally during chunk processing
        # Verify it exists and load final count
        if not metadata_path.exists():
            # Provide diagnostic info for zero windows case
            event_type_msg = f"'{event_config.get('event_type')}'" if event_config else "N/A"
            raise ValueError(
                f"No windows were created from {len(recordings)} recordings.\n"
                f"Possible causes:\n"
                f"  - Event type {event_type_msg} not found in annotations\n"
                f"  - Annotation files missing (check: {memmap_path.parent})\n"
                f"  - Recordings too short for window_len={window_len_s}s\n"
                f"  - Total windows created: {total_windows}"
            )

        metadata_df = pd.read_parquet(metadata_path)
        logger.info(f"✓ Metadata complete: {len(metadata_df)} rows")

        # Clean up progress file (no longer needed after successful completion)
        if completed_path.exists():
            completed_path.unlink()
            logger.debug(f"Removed progress file: {completed_path.name}")

        del all_metadata
        gc.collect()

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

    def _resume_window_cache(
        self,
        recordings: pd.DataFrame,
        window_len_s: float,
        stride_s: Optional[float],
        event_config: Optional[Dict[str, Any]],
        crop_len_s: Optional[float],
        window_config_id: str,
        memmap_path: Path,
        metadata_path: Path,
        progress_path: Path,
        mode: str = 'train'
    ) -> MemmapWindowDataset:
        """Resume incomplete window cache build.

        Loads progress tracker to identify which recordings are complete,
        then processes only the missing recordings. Preserves all existing data.

        Args:
            recordings: DataFrame of all recordings needed
            window_len_s: Window length in seconds
            stride_s: Stride in seconds (None for event-locked)
            event_config: Event configuration dict (None for sliding window)
            crop_len_s: Optional crop length
            window_config_id: Config ID for this window setup
            memmap_path: Path to memmap array
            metadata_path: Path to metadata parquet
            progress_path: Path to progress tracker
            mode: 'train' or 'val'

        Returns:
            MemmapWindowDataset
        """
        sfreq = self.cache_mgr.preprocessing_params["sfreq"]
        window_samples = int(window_len_s * sfreq)

        # Load completed recordings from simple text file
        completed_ids = set()
        if progress_path.exists():
            completed_ids = set(line.strip() for line in progress_path.read_text().strip().split('\n') if line.strip())

        logger.info(f"Found progress file: {len(completed_ids)} recordings already complete")

        # Identify missing recordings
        missing_recordings = recordings[
            ~recordings["recording_id"].isin(completed_ids)
        ].reset_index(drop=True)

        logger.info(f"Missing recordings: {len(missing_recordings)} / {len(recordings)}")

        # Case 1: All requested recordings are complete
        if len(missing_recordings) == 0:
            logger.info("✓ All requested recordings already in cache!")

            # Load metadata and return dataset
            if metadata_path.exists():
                metadata = pd.read_parquet(metadata_path)
                metadata_filtered = metadata[
                    metadata["recording_id"].isin(recordings["recording_id"].values)
                ].reset_index(drop=True)

                return MemmapWindowDataset(
                    memmap_path=memmap_path,
                    metadata=metadata_filtered,
                    crop_len_s=crop_len_s,
                    sfreq=sfreq,
                    mode=mode
                )
            else:
                logger.error("Metadata file missing despite complete progress tracker!")
                logger.info("Falling back to full rebuild...")
                return self._build_window_cache(
                    recordings, window_len_s, stride_s, event_config, crop_len_s,
                    window_config_id, memmap_path, metadata_path, mode
                )

        # Case 2: Some recordings are missing - resume build
        logger.info(f"Resuming build for {len(missing_recordings)} missing recordings...")

        # Validate existing memmap
        try:
            existing_array = np.load(str(memmap_path), mmap_mode='r')
            logger.info(f"Validated existing memmap: shape={existing_array.shape}")
            n_channels = existing_array.shape[1]
            del existing_array  # Release handle
        except Exception as e:
            logger.error(f"Failed to load existing memmap: {e}")
            logger.info("Memmap corrupted, starting fresh build...")
            # Delete corrupted files
            if memmap_path.exists():
                memmap_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            if progress_path.exists():
                progress_path.unlink()
            return self._build_window_cache(
                recordings, window_len_s, stride_s, event_config, crop_len_s,
                window_config_id, memmap_path, metadata_path, mode
            )

        # Load existing memmap in append mode
        # Get last array index from metadata if it exists
        total_windows = 0
        if metadata_path.exists():
            existing_metadata = pd.read_parquet(metadata_path)
            if len(existing_metadata) > 0:
                total_windows = existing_metadata["array_index"].max() + 1
        logger.info(f"Resuming from array index: {total_windows}")

        # Estimate additional windows needed
        estimated_additional = len(missing_recordings) * 500
        current_size = np.load(str(memmap_path), mmap_mode='r').shape[0]

        # Expand memmap if needed
        if total_windows + estimated_additional > current_size:
            new_size = total_windows + estimated_additional
            logger.info(f"Expanding memmap from {current_size:,} to {new_size:,} windows...")

            # Load existing data
            old_data = np.load(str(memmap_path), mmap_mode='r')[:total_windows]

            # Create larger array
            temp_path = memmap_path.parent / f"{memmap_path.stem}_temp.npy"
            new_array = np.lib.format.open_memmap(
                str(temp_path),
                mode='w+',
                dtype=np.float32,
                shape=(new_size, n_channels, window_samples)
            )
            new_array[:total_windows] = old_data
            new_array.flush()
            del old_data, new_array
            gc.collect()

            # Replace original
            temp_path.rename(memmap_path)
            logger.info(f"✓ Memmap expanded successfully")

        # Open memmap in append mode
        memmap_array = np.lib.format.open_memmap(
            str(memmap_path),
            mode='r+',
            dtype=np.float32,
        )

        # Process missing recordings (reuse same chunking logic)
        all_metadata = []
        max_workers = 32
        chunk_size = 10

        recordings_list = list(missing_recordings.iterrows())
        n_recordings = len(recordings_list)
        n_chunks = (n_recordings + chunk_size - 1) // chunk_size

        logger.info(f"Processing {n_recordings} missing recordings in {n_chunks} chunks")

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_recordings)
            chunk = recordings_list[chunk_start:chunk_end]

            logger.info(f"Processing chunk {chunk_idx + 1}/{n_chunks} ({len(chunk)} recordings)...")

            # Process chunk in parallel
            chunk_results = Parallel(
                n_jobs=max_workers,
                backend='loky',
                batch_size=1,
                verbose=0,
                max_nbytes='100M',
                pre_dispatch='2*n_jobs'
            )(
                delayed(_process_single_recording)(
                    rec, window_len_s, stride_s, event_config, sfreq
                )
                for rec in tqdm(chunk, desc=f"Resume chunk {chunk_idx + 1}/{n_chunks}", unit="rec")
            )

            # Append windows to memmap
            for windows, rec_metadata in chunk_results:
                if windows.shape[0] > 0:
                    n_windows = windows.shape[0]
                    recording_id = rec_metadata[0]["recording_id"]
                    array_start_idx = total_windows

                    # Check if expansion needed
                    if total_windows + n_windows > memmap_array.shape[0]:
                        logger.warning(f"Need to expand array during resume")
                        memmap_array.flush()
                        del memmap_array

                        new_size = (total_windows + n_windows) * 2
                        old_data = np.load(str(memmap_path), mmap_mode='r')[:total_windows]
                        temp_path = memmap_path.parent / f"{memmap_path.stem}_temp.npy"
                        new_array = np.lib.format.open_memmap(
                            str(temp_path),
                            mode='w+',
                            dtype=np.float32,
                            shape=(new_size, n_channels, window_samples)
                        )
                        new_array[:total_windows] = old_data
                        new_array.flush()
                        del old_data
                        temp_path.rename(memmap_path)
                        memmap_array = new_array

                    # Write windows
                    memmap_array[total_windows:total_windows + n_windows] = windows

                    # Update metadata
                    for i, meta in enumerate(rec_metadata):
                        meta["window_idx"] = total_windows + i
                        meta["array_index"] = total_windows + i
                        all_metadata.append(meta)

                    total_windows += n_windows

                    # Mark this recording as complete
                    completed_ids.add(recording_id)

                    del windows
                    del rec_metadata

            del chunk_results
            gc.collect()

            # Save incrementally
            if len(all_metadata) > 0:
                metadata_df = pd.DataFrame(all_metadata)
                if metadata_path.exists():
                    existing_metadata = pd.read_parquet(metadata_path)
                    metadata_df = pd.concat([existing_metadata, metadata_df], ignore_index=True)
                metadata_df.to_parquet(metadata_path, index=False)
                logger.debug(f"  Saved metadata incrementally ({len(metadata_df)} total rows)")
                all_metadata.clear()
                del metadata_df
                gc.collect()

            # Save progress after each chunk
            progress_path.write_text('\n'.join(sorted(completed_ids)) + '\n')
            memmap_array.flush()

            logger.info(f"  ✓ Resume chunk {chunk_idx + 1}/{n_chunks} complete: {total_windows:,} total windows")

        # Flush and cleanup
        memmap_array.flush()
        del memmap_array
        gc.collect()

        logger.info(f"✓ Resume complete: {total_windows:,} total windows")

        # Load final metadata
        metadata = pd.read_parquet(metadata_path)
        metadata_filtered = metadata[
            metadata["recording_id"].isin(recordings["recording_id"].values)
        ].reset_index(drop=True)

        # Clean up progress file
        if progress_path.exists():
            progress_path.unlink()
            logger.debug(f"Removed progress file: {progress_path.name}")

        # Update manifest
        self.window_manifest.mark_complete({
            "window_config_id": window_config_id,
            "raw_config_hash": self.cache_mgr.preprocessing_hash,
            "window_len_s": window_len_s,
            "stride_s": stride_s,
            "n_windows": total_windows,
            "window_zarr_path": str(memmap_path),
            "metadata_path": str(metadata_path),
            "error_msg": ""
        })
        self.window_manifest.save()

        return MemmapWindowDataset(
            memmap_path=memmap_path,
            metadata=metadata_filtered,
            crop_len_s=crop_len_s,
            sfreq=sfreq,
            mode=mode
        )
