"""Level 1: Raw preprocessed cache builder.

Builds per-recording Zarr cache with preprocessing applied.
Stores one Zarr array per (subject × task × release) recording.
"""

import logging
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import zarr
from eegdash import EEGChallengeDataset
from numcodecs import Blosc
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _process_recording_worker(
    recording_id: str,
    ds: Any,
    dataset: str,
    mini: bool,
    zarr_path: Path,
    preprocessing_params: Dict[str, Any],
    preprocessing_hash: str
) -> Dict[str, Any]:
    """Worker function for parallel recording processing.

    This is a module-level function (not a method) so it can be pickled
    for multiprocessing.

    Args:
        recording_id: Unique recording ID
        ds: Dataset object from EEGChallengeDataset
        dataset: Dataset name
        mini: Mini flag
        zarr_path: Path to save Zarr array
        preprocessing_params: Preprocessing parameters dict
        preprocessing_hash: Hash of preprocessing params

    Returns:
        Metadata dict for manifest, or None if failed
    """
    try:
        # Extract raw
        raw = ds.raw

        # Apply preprocessing
        target_sfreq = preprocessing_params["sfreq"]
        if raw.info["sfreq"] != target_sfreq:
            raw = raw.resample(target_sfreq)

        bandpass = preprocessing_params.get("bandpass")
        if bandpass:
            raw = raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])

        if preprocessing_params.get("standardize", False):
            data = raw.get_data()
            data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
            raw._data = data

        # Save to Zarr
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        data = raw.get_data()

        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        z = zarr.open(
            str(zarr_path),
            mode='w',
            shape=data.shape,
            chunks=(data.shape[0], min(10000, data.shape[1])),
            dtype=np.float32,
            compressor=compressor,
            zarr_format=2,
        )
        z[:] = data.astype(np.float32)

        # Return metadata
        return {
            "dataset": dataset,
            "release": ds.description.get("release_number", "unknown"),
            "subject": ds.description.get("subject", "unknown"),
            "task": ds.description.get("task", "unknown"),
            "mini": mini,
            "recording_id": recording_id,
            "sfreq": int(raw.info["sfreq"]),
            "n_channels": len(raw.ch_names),
            "n_samples": raw.n_times,
            "duration_s": raw.times[-1],
            "preprocessing_hash": preprocessing_hash,
            "raw_zarr_path": str(zarr_path),
            "error_msg": "",
            "success": True,
        }
    except Exception as e:
        logger.error(f"Worker failed on {recording_id}: {e}")
        return {
            "recording_id": recording_id,
            "error_msg": str(e),
            "success": False,
        }


class GracefulInterruptHandler:
    """Context manager for graceful SIGINT/SIGTERM handling."""

    def __init__(self):
        self.interrupted = False
        self.original_handler = None

    def __enter__(self):
        self.original_handler = signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_handler)

    def _handler(self, signum, frame):
        logger.warning("⚠ Interrupt received - finishing current recording...")
        self.interrupted = True


class RawCacheBuilder:
    """Builds Level 1 raw cache from EEGChallengeDataset.

    Args:
        cache_manager: Parent UniversalCacheManager instance
    """

    def __init__(self, cache_manager):
        self.cache_mgr = cache_manager
        self.raw_manifest = self.cache_mgr.raw_manifest

    def build(
        self,
        dataset: str,
        releases: List[str],
        tasks: List[str],
        mini: bool = False,
        **kwargs
    ):
        """Build raw cache for specified data.

        Args:
            dataset: Dataset name ("hbn", "tuh")
            releases: List of release IDs (e.g., ["R1", "R2"])
            tasks: List of task names or "all"
            mini: Use mini dataset
            **kwargs: Dataset-specific parameters
        """
        if dataset != "hbn":
            raise NotImplementedError(f"Dataset '{dataset}' not yet supported. Currently supports: 'hbn'")

        logger.info(f"Building raw cache for {dataset}")
        logger.info(f"  Releases: {releases}")
        logger.info(f"  Tasks: {tasks}")
        logger.info(f"  Mini: {mini}")

        # Load all recordings via EEGChallengeDataset
        all_recordings = []
        for release in releases:
            logger.info(f"  Loading {release}...")
            try:
                eeg_dataset = EEGChallengeDataset(
                    release=release,
                    cache_dir=str(self.cache_mgr.cache_root.parent),
                    mini=mini,
                    query={"task": tasks} if tasks != "all" else None
                )
                all_recordings.extend(eeg_dataset.datasets)
                logger.info(f"    ✓ Loaded {len(eeg_dataset.datasets)} recordings")
            except Exception as e:
                logger.warning(f"    ✗ Failed to load {release}: {e}")
                continue

        if not all_recordings:
            raise ValueError("No recordings loaded!")

        logger.info(f"\nTotal recordings to process: {len(all_recordings)}")

        # Identify missing recordings
        recording_ids = [
            self.cache_mgr._get_recording_id(
                dataset=dataset,
                release=ds.description.get("release", "unknown"),
                subject=ds.description.get("subject", "unknown"),
                task=ds.description.get("task", "unknown"),
                mini=mini
            )
            for ds in all_recordings
        ]

        missing_ids = self.raw_manifest.get_missing_recordings(recording_ids)
        missing_recordings = [
            (rid, rec) for rid, rec in zip(recording_ids, all_recordings)
            if rid in missing_ids
        ]

        logger.info(f"Already cached: {len(recording_ids) - len(missing_ids)} recordings")
        logger.info(f"To process: {len(missing_recordings)} recordings\n")

        if not missing_recordings:
            logger.info("✓ All recordings already cached!")
            return

        # Process missing recordings in parallel
        import os
        max_workers = min(32, os.cpu_count() or 1)  # Use up to 32 cores
        logger.info(f"Processing with {max_workers} parallel workers")

        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            for recording_id, ds in missing_recordings:
                zarr_path = self.cache_mgr._get_raw_zarr_path(recording_id)
                future = executor.submit(
                    _process_recording_worker,
                    recording_id,
                    ds,
                    dataset,
                    mini,
                    zarr_path,
                    self.cache_mgr.preprocessing_params,
                    self.cache_mgr.preprocessing_hash
                )
                futures.append((recording_id, future))

            # Collect results with progress bar
            completed = 0
            failed = 0
            with tqdm(total=len(futures), desc="Processing", unit="rec") as pbar:
                for recording_id, future in futures:
                    try:
                        result = future.result()
                        if result["success"]:
                            self.raw_manifest.mark_complete(result)
                            completed += 1
                        else:
                            self.raw_manifest.mark_failed(
                                {"recording_id": recording_id},
                                error=result["error_msg"]
                            )
                            failed += 1
                    except Exception as e:
                        logger.error(f"Failed to process {recording_id}: {e}")
                        self.raw_manifest.mark_failed(
                            {"recording_id": recording_id},
                            error=str(e)
                        )
                        failed += 1
                    finally:
                        pbar.update(1)

            # Save manifest
            self.raw_manifest.save()
            logger.info(f"\n✓ Raw cache build complete!")
            logger.info(f"  Completed: {completed}, Failed: {failed}")
            self.raw_manifest.print_status("Raw Cache Status")

    def _process_recording(
        self,
        recording_id: str,
        ds: Any,
        dataset: str,
        mini: bool
    ):
        """Process single recording and save to Zarr.

        Args:
            recording_id: Unique recording ID
            ds: Dataset object from EEGChallengeDataset
            dataset: Dataset name
            mini: Mini flag
        """
        # Extract raw
        raw = ds.raw

        # Apply preprocessing
        raw = self._preprocess_raw(raw)

        # Save to Zarr
        zarr_path = self.cache_mgr._get_raw_zarr_path(recording_id)
        self._save_to_zarr(raw, zarr_path)

        # Update manifest
        self.raw_manifest.mark_complete({
            "dataset": dataset,
            "release": ds.description.get("release_number", "unknown"),
            "subject": ds.description.get("subject", "unknown"),
            "task": ds.description.get("task", "unknown"),
            "mini": mini,
            "recording_id": recording_id,
            "sfreq": int(raw.info["sfreq"]),
            "n_channels": len(raw.ch_names),
            "n_samples": raw.n_times,
            "duration_s": raw.times[-1],
            "preprocessing_hash": self.cache_mgr.preprocessing_hash,
            "raw_zarr_path": str(zarr_path),
            "error_msg": "",
        })

    def _preprocess_raw(self, raw):
        """Apply preprocessing to raw recording.

        Args:
            raw: MNE Raw object

        Returns:
            Preprocessed MNE Raw object
        """
        # Resample if needed
        target_sfreq = self.cache_mgr.preprocessing_params["sfreq"]
        if raw.info["sfreq"] != target_sfreq:
            raw = raw.resample(target_sfreq)

        # Apply bandpass filter if specified
        bandpass = self.cache_mgr.preprocessing_params.get("bandpass")
        if bandpass:
            raw = raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])

        # Standardize if specified
        if self.cache_mgr.preprocessing_params.get("standardize", False):
            data = raw.get_data()
            data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
            raw._data = data

        return raw

    def _save_to_zarr(self, raw, zarr_path: Path):
        """Save MNE Raw to Zarr format.

        Args:
            raw: MNE Raw object
            zarr_path: Path to save Zarr array
        """
        zarr_path.parent.mkdir(parents=True, exist_ok=True)

        # Get data
        data = raw.get_data()  # Shape: (n_channels, n_samples)

        # Create Zarr array with compression (use v2 format for compatibility)
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        z = zarr.open(
            str(zarr_path),
            mode='w',
            shape=data.shape,
            chunks=(data.shape[0], min(10000, data.shape[1])),  # Chunk along time
            dtype=np.float32,
            compressor=compressor,
            zarr_format=2,  # Use v2 format for compressor API compatibility
        )

        # Write data
        z[:] = data.astype(np.float32)
