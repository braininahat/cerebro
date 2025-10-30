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
        # Check if already processed (defensive programming against race conditions)
        # Multiple workers might receive the same task from ProcessPoolExecutor
        if zarr_path.exists():
            # Already processed by another worker - skip
            return {
                "dataset": dataset,
                "release": "unknown",
                "subject": "unknown",
                "task": "unknown",
                "mini": mini,
                "recording_id": recording_id,
                "sfreq": 0,
                "n_channels": 0,
                "n_samples": 0,
                "duration_s": 0.0,
                "preprocessing_hash": preprocessing_hash,
                "raw_zarr_path": str(zarr_path),
                "error_msg": "Already processed by another worker",
                "success": True,
            }

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

        # Save to Zarr (use 'w-' mode to fail if exists - atomic check)
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        data = raw.get_data()

        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        try:
            # Use 'w-' mode (exclusive create) to atomically fail if exists
            z = zarr.open(
                str(zarr_path),
                mode='w-',  # Create only, fail if exists (prevents race conditions)
                shape=data.shape,
                chunks=(data.shape[0], min(10000, data.shape[1])),
                dtype=np.float32,
                compressor=compressor,
                zarr_format=2,
            )
            z[:] = data.astype(np.float32)
        except (FileExistsError, ValueError) as e:
            # Another worker created this zarr - that's ok, skip
            return {
                "dataset": dataset,
                "release": ds.description.get("release_number", "unknown"),
                "subject": ds.description.get("subject", "unknown"),
                "task": ds.description.get("task", "unknown"),
                "mini": mini,
                "recording_id": recording_id,
                "sfreq": 0,
                "n_channels": 0,
                "n_samples": 0,
                "duration_s": 0.0,
                "preprocessing_hash": preprocessing_hash,
                "raw_zarr_path": str(zarr_path),
                "error_msg": "Already created by another worker (race condition detected)",
                "success": True,
            }

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


def _process_recording_chunk(
    chunk: List[tuple],
    dataset: str,
    mini: bool,
    recordings_dir: Path,
    preprocessing_params: Dict[str, Any],
    preprocessing_hash: str,
    worker_id: int
) -> List[Dict[str, Any]]:
    """Process a chunk of recordings in a single worker process.

    With pre-partitioning, each worker gets an exclusive chunk of recordings,
    eliminating any possibility of race conditions.

    Args:
        chunk: List of (recording_id, ds) tuples to process
        dataset: Dataset name (e.g., "hbn")
        mini: Mini flag
        recordings_dir: Base directory for zarr files
        preprocessing_params: Preprocessing parameters dict
        preprocessing_hash: Hash of preprocessing params
        worker_id: Worker identifier for logging

    Returns:
        List of metadata dicts for each recording in chunk
    """
    results = []

    for recording_id, ds in chunk:
        try:
            # Build zarr path
            zarr_path = recordings_dir / f"{recording_id}_{preprocessing_hash}.zarr"

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
                mode='w',  # Safe to use 'w' - no collisions with pre-partitioning
                shape=data.shape,
                chunks=(data.shape[0], min(10000, data.shape[1])),
                dtype=np.float32,
                compressor=compressor,
                zarr_format=2,
            )
            z[:] = data.astype(np.float32)

            # Return metadata
            results.append({
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
            })
        except Exception as e:
            logger.error(f"Worker {worker_id} failed on {recording_id}: {e}")
            results.append({
                "recording_id": recording_id,
                "error_msg": str(e),
                "success": False,
            })

    return results


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

        # Process missing recordings in parallel with pre-partitioning
        # Pre-partition recordings into chunks to guarantee no worker collisions
        import os
        max_workers = min(32, os.cpu_count())

        # Partition recordings into chunks - each worker gets exclusive chunk
        def partition_list(items, n_partitions):
            """Divide list into n roughly equal chunks."""
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

        # Create more chunks for better load balancing (target ~50 recordings per chunk)
        target_chunk_size = 50
        n_chunks = max(max_workers, len(missing_recordings) // target_chunk_size)
        recording_chunks = partition_list(missing_recordings, n_chunks)

        logger.info(f"Processing with {max_workers} parallel workers")
        logger.info(f"Partitioned {len(missing_recordings)} recordings into {len(recording_chunks)} chunks (~{target_chunk_size} recordings each)")
        logger.info(f"  Chunk size range: {min(len(c) for c in recording_chunks if c)}-{max(len(c) for c in recording_chunks if c)} recordings")

        futures = []
        future_to_metadata = {}  # Map futures to their worker metadata

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks to executor
            for worker_id, chunk in enumerate(recording_chunks):
                if not chunk:  # Skip empty chunks
                    continue
                future = executor.submit(
                    _process_recording_chunk,
                    chunk,
                    dataset,
                    mini,
                    self.cache_mgr.raw_cache_dir / "recordings",
                    self.cache_mgr.preprocessing_params,
                    self.cache_mgr.preprocessing_hash,
                    worker_id
                )
                future_to_metadata[future] = (worker_id, chunk)
                futures.append(future)

            # Collect results using as_completed() for real-time progress
            from concurrent.futures import as_completed
            completed = 0
            failed = 0

            with GracefulInterruptHandler() as handler:
                with tqdm(total=len(missing_recordings), desc="Processing", unit="rec") as pbar:
                    for future in as_completed(futures):
                        if handler.interrupted:
                            logger.warning("⚠ Interrupt received - saving progress and stopping...")
                            break

                        worker_id, chunk = future_to_metadata[future]

                        try:
                            results = future.result()
                            for result in results:
                                if result["success"]:
                                    self.raw_manifest.mark_complete(result)
                                    completed += 1
                                else:
                                    self.raw_manifest.mark_failed(
                                        {"recording_id": result["recording_id"]},
                                        error=result["error_msg"]
                                    )
                                    failed += 1
                                pbar.update(1)

                            # Save after each chunk completes (incremental save)
                            self.raw_manifest.save()

                        except Exception as e:
                            logger.error(f"Worker {worker_id} failed: {e}")
                            # Mark all recordings in chunk as failed
                            for recording_id, _ in chunk:
                                self.raw_manifest.mark_failed(
                                    {"recording_id": recording_id},
                                    error=f"Worker crashed: {str(e)}"
                                )
                                failed += 1
                                pbar.update(1)

                            # Save even after failures
                            self.raw_manifest.save()

            # Final save to ensure everything is persisted
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
