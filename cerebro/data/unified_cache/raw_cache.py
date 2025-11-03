"""Level 1: Raw preprocessed cache builder.

Builds per-recording memory-mapped numpy cache with preprocessing applied.
Stores one .npy memory-mapped array per (subject × task × release) recording.
"""

import logging
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from eegdash import EEGChallengeDataset
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
        zarr_path: Path to save numpy array (parameter name kept for backward compatibility,
                   but now saves .npy memory-mapped files instead of Zarr)
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
        # CRITICAL: Apply bandpass BEFORE resampling for proper anti-aliasing
        # This matches HBN preprocessing (0.5-50Hz filter, then 500Hz→100Hz resample)
        target_sfreq = preprocessing_params["sfreq"]
        bandpass = preprocessing_params.get("bandpass")

        # Step 1: Apply bandpass filter at original sampling rate
        # Serves as anti-aliasing filter before downsampling
        if bandpass:
            raw = raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])

        # Step 2: Resample to target frequency
        # Filter has already removed frequencies > Nyquist of target rate
        if raw.info["sfreq"] != target_sfreq:
            raw = raw.resample(target_sfreq)

        if preprocessing_params.get("standardize", False):
            data = raw.get_data()
            data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
            raw._data = data

        # Save to memory-mapped numpy array (no compression for zero-copy access)
        npy_path = zarr_path.with_suffix('.npy')
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        data = raw.get_data()

        # Check if file exists (atomic check)
        if npy_path.exists():
            # Another worker created this file - that's ok, skip
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
                "raw_zarr_path": str(npy_path),  # Keep field name for compatibility
                "error_msg": "Already created by another worker (race condition detected)",
                "success": True,
            }

        # Write using memory-mapped array
        try:
            arr = np.lib.format.open_memmap(
                str(npy_path),
                mode='w+',
                dtype=np.float32,
                shape=data.shape
            )
            arr[:] = data.astype(np.float32)
            arr.flush()
            del arr
        except Exception as e:
            # Handle any write errors
            if npy_path.exists():
                npy_path.unlink()
            raise e

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
            "raw_zarr_path": str(npy_path),  # Keep field name for compatibility
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
            # Build npy path
            npy_path = recordings_dir / f"{recording_id}_{preprocessing_hash}.npy"

            # Extract raw
            raw = ds.raw

            # Apply preprocessing FIRST (before annotations to preserve extras)
            # CRITICAL: Apply bandpass BEFORE resampling for proper anti-aliasing
            target_sfreq = preprocessing_params["sfreq"]
            bandpass = preprocessing_params.get("bandpass")

            # Step 1: Apply bandpass filter at original sampling rate
            if bandpass:
                raw = raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])

            # Step 2: Resample to target frequency
            if raw.info["sfreq"] != target_sfreq:
                raw = raw.resample(target_sfreq)

            if preprocessing_params.get("standardize", False):
                data = raw.get_data()
                data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
                raw._data = data

            # Apply task-specific annotations AFTER preprocessing
            # (must happen after resampling to preserve annotations.extras)
            # Note: Annotations use time in seconds, so they remain valid after resampling
            task = ds.description.get("task", "unknown")
            if task == "contrastChangeDetection":
                from eegdash.hbn.windows import annotate_trials_with_target, add_aux_anchors

                # Step 1: Pair Stim+Response events and compute RT
                # This reads from BIDS _events.tsv via raw.filenames
                annotate_trials_with_target(
                    raw,
                    target_field='rt_from_stimulus',
                    epoch_length=2.0,  # Not used for windowing, just metadata
                    require_stimulus=True,
                    require_response=True
                )

                # Step 2: Add "stimulus_anchor" description for event-locked windowing
                add_aux_anchors(raw)
                # Let any errors propagate - don't fail silently!

            # Save to memory-mapped numpy array (no compression)
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            data = raw.get_data()

            # Write using memory-mapped array
            arr = np.lib.format.open_memmap(
                str(npy_path),
                mode='w+',
                dtype=np.float32,
                shape=data.shape
            )
            arr[:] = data.astype(np.float32)
            arr.flush()
            del arr

            # Save annotations alongside (for event-locked windowing)
            if raw.annotations is not None and len(raw.annotations) > 0:
                import json
                annotations_path = npy_path.with_suffix('.annotations.json')

                annotations_data = {
                    'onsets': raw.annotations.onset.tolist(),
                    'durations': raw.annotations.duration.tolist(),
                    'descriptions': raw.annotations.description.tolist(),
                }

                # Add extras (RT and other metadata)
                if hasattr(raw.annotations, 'extras') and raw.annotations.extras is not None:
                    extras_serializable = []
                    for extra in raw.annotations.extras:
                        # Use duck typing to handle MNE's custom dict-like objects
                        if extra is not None and hasattr(extra, 'items'):
                            extra_dict = {}
                            for k, v in extra.items():
                                if isinstance(v, (np.integer, np.floating)):
                                    extra_dict[k] = float(v)
                                elif isinstance(v, np.ndarray):
                                    extra_dict[k] = v.tolist()
                                else:
                                    extra_dict[k] = v
                            extras_serializable.append(extra_dict)
                        else:
                            extras_serializable.append(None)
                    annotations_data['extras'] = extras_serializable

                with open(annotations_path, 'w') as f:
                    json.dump(annotations_data, f, indent=2)

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
                "raw_zarr_path": str(npy_path),  # Keep field name for compatibility
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
        data_dir: Directory for raw dataset downloads (e.g., HBN_ROOT)
    """

    def __init__(self, cache_manager, data_dir: Optional[Path] = None):
        self.cache_mgr = cache_manager
        self.raw_manifest = self.cache_mgr.raw_manifest
        self.data_dir = data_dir

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
            releases: List of release IDs (e.g., ["R1", "R2"]) or TUH subset names
            tasks: List of task names or "all"
            mini: Use mini dataset
            **kwargs: Dataset-specific parameters (e.g., tuh_path for TUH)
        """
        if dataset == "hbn":
            self._build_hbn(releases, tasks, mini, **kwargs)
        elif dataset == "tuh":
            self._build_tuh(releases, tasks, mini, **kwargs)
        else:
            raise NotImplementedError(f"Dataset '{dataset}' not supported. Use 'hbn' or 'tuh'")

    def _build_hbn(
        self,
        releases: List[str],
        tasks: List[str],
        mini: bool = False,
        **kwargs
    ):
        """Build HBN cache from EEGChallengeDataset.

        Args:
            releases: List of release IDs (e.g., ["R1", "R2"])
            tasks: List of task names or "all"
            mini: Use mini dataset
            **kwargs: Additional parameters
        """
        dataset = "hbn"

        logger.info(f"Building raw cache for {dataset}")
        logger.info(f"  Releases: {releases}")
        logger.info(f"  Tasks: {tasks}")
        logger.info(f"  Mini: {mini}")

        # Load all recordings via EEGChallengeDataset
        all_recordings = []
        for release in releases:
            logger.info(f"  Loading {release}...")
            try:
                # Use data_dir for downloads (falls back to cache_root.parent if not set)
                download_dir = str(self.data_dir) if self.data_dir else str(self.cache_mgr.cache_root.parent)

                # Suppress EEGChallengeDataset competition notice warning
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*EEGChallengeDataset.*Competition Data Notice.*")
                    eeg_dataset = EEGChallengeDataset(
                        release=release,
                        cache_dir=download_dir,
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

        self._process_recordings_parallel(all_recordings, dataset, mini)

    def _build_tuh(
        self,
        releases: List[str],
        tasks: List[str],
        mini: bool = False,
        **kwargs
    ):
        """Build TUH cache from braindecode TUH loader.

        Args:
            releases: List of subset names (e.g., ["tuh_eeg_subset"])
            tasks: List of recording types (e.g., ["01_tcp_ar", "02_tcp_le"]) or "all"
            mini: Not used for TUH (kept for interface compatibility)
            **kwargs: Must include 'tuh_path' (path to TUH dataset root)
        """
        dataset = "tuh"
        tuh_path = kwargs.get("tuh_path")
        if not tuh_path:
            raise ValueError("tuh_path must be provided in kwargs for TUH dataset")

        tuh_path = Path(tuh_path)
        if not tuh_path.exists():
            raise ValueError(f"TUH path does not exist: {tuh_path}")

        logger.info(f"Building raw cache for {dataset}")
        logger.info(f"  TUH path: {tuh_path}")
        logger.info(f"  Recording types filter: {tasks if tasks != 'all' else 'all'}")

        # Import here to avoid requiring braindecode for HBN-only usage
        try:
            from braindecode.datasets import TUH as TUHDataset
        except ImportError:
            raise ImportError("braindecode is required for TUH dataset. Install with: uv pip install braindecode")

        # Load TUH dataset
        logger.info(f"  Loading TUH recordings from {tuh_path}...")
        try:
            tuh_ds = TUHDataset(
                path=str(tuh_path),
                recording_ids=None,  # Load all
                target_name=None,    # No classification target
                preload=False,       # Don't preload to memory
                add_physician_reports=False,
                n_jobs=1  # Single process for loading (parallel processing happens later)
            )
            logger.info(f"    ✓ Loaded {len(tuh_ds.datasets)} recordings")
        except Exception as e:
            raise RuntimeError(f"Failed to load TUH dataset: {e}")

        if not tuh_ds.datasets:
            raise ValueError("No recordings loaded from TUH dataset!")

        # Filter by recording types if specified
        all_recordings = tuh_ds.datasets
        if tasks != "all":
            # TUH description includes session info with recording type
            filtered_recordings = []
            for ds in all_recordings:
                # Recording type is in the path (e.g., "01_tcp_ar", "02_tcp_le")
                rec_type = ds.description.get("path", "").split("/")[-2] if "/" in ds.description.get("path", "") else ""
                if any(task in rec_type for task in tasks):
                    filtered_recordings.append(ds)
            logger.info(f"  Filtered to {len(filtered_recordings)} recordings matching {tasks}")
            all_recordings = filtered_recordings

        if not all_recordings:
            raise ValueError(f"No recordings found matching recording types: {tasks}")

        self._process_recordings_parallel(all_recordings, dataset, mini=False)

    def _process_recordings_parallel(
        self,
        all_recordings: List,
        dataset: str,
        mini: bool
    ):
        """Process recordings in parallel with chunked execution.

        Args:
            all_recordings: List of dataset objects (from EEGChallengeDataset or TUH)
            dataset: Dataset name ("hbn" or "tuh")
            mini: Mini dataset flag
        """

        logger.info(f"\nTotal recordings to process: {len(all_recordings)}")

        # Identify missing recordings
        recording_ids = []
        for ds in all_recordings:
            # Extract metadata based on dataset type
            if dataset == "hbn":
                release = ds.description.get("release", "unknown")
                subject = ds.description.get("subject", "unknown")
                task = ds.description.get("task", "unknown")
            elif dataset == "tuh":
                # TUH has different metadata structure
                release = "tuh_eeg"  # Use dataset name as "release"
                subject = ds.description.get("subject", "unknown")
                # Task is the session/recording type (e.g., "s001_2004/02_tcp_le")
                path_parts = ds.description.get("path", "").split("/")
                session = path_parts[-3] if len(path_parts) >= 3 else "unknown"
                rec_type = path_parts[-2] if len(path_parts) >= 2 else "unknown"
                task = f"{session}_{rec_type}"
            else:
                raise ValueError(f"Unknown dataset type: {dataset}")

            recording_id = self.cache_mgr._get_recording_id(
                dataset=dataset,
                release=release,
                subject=subject,
                task=task,
                mini=mini
            )
            recording_ids.append(recording_id)

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

        # Save to memory-mapped numpy
        npy_path = self.cache_mgr._get_raw_zarr_path(recording_id).with_suffix('.npy')
        self._save_to_npy(raw, npy_path)

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
            "raw_zarr_path": str(npy_path),  # Keep field name for compatibility
            "error_msg": "",
        })

    def _preprocess_raw(self, raw):
        """Apply preprocessing to raw recording.

        CRITICAL: Filter BEFORE resample for proper anti-aliasing.

        Args:
            raw: MNE Raw object

        Returns:
            Preprocessed MNE Raw object
        """
        target_sfreq = self.cache_mgr.preprocessing_params["sfreq"]
        bandpass = self.cache_mgr.preprocessing_params.get("bandpass")

        # Step 1: Bandpass filter at original sampling rate (anti-aliasing)
        if bandpass:
            raw = raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])

        # Step 2: Resample to target frequency
        if raw.info["sfreq"] != target_sfreq:
            raw = raw.resample(target_sfreq)

        # Step 3: Standardize if specified
        if self.cache_mgr.preprocessing_params.get("standardize", False):
            data = raw.get_data()
            data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
            raw._data = data

        return raw

    def _save_to_npy(self, raw, npy_path: Path):
        """Save MNE Raw to memory-mapped numpy format + annotations as JSON.

        Args:
            raw: MNE Raw object
            npy_path: Path to save .npy array
        """
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        # Get data
        data = raw.get_data()  # Shape: (n_channels, n_samples)

        # Create memory-mapped array (no compression for zero-copy access)
        arr = np.lib.format.open_memmap(
            str(npy_path),
            mode='w+',
            dtype=np.float32,
            shape=data.shape
        )

        # Write data
        arr[:] = data.astype(np.float32)
        arr.flush()
        del arr

        # Save annotations alongside (for event-locked windowing)
        if raw.annotations is not None and len(raw.annotations) > 0:
            self._save_annotations(raw.annotations, npy_path)

    def _save_annotations(self, annotations, npy_path: Path):
        """Save MNE annotations to JSON for event-locked windowing.

        Args:
            annotations: MNE Annotations object
            npy_path: Path to corresponding .npy file (will save as .annotations.json)
        """
        import json

        annotations_path = npy_path.with_suffix('.annotations.json')

        # Extract annotation data
        annotations_data = {
            'onsets': annotations.onset.tolist(),
            'durations': annotations.duration.tolist(),
            'descriptions': annotations.description.tolist(),
        }

        # Add extras (contains RT and other event metadata)
        if hasattr(annotations, 'extras') and annotations.extras is not None:
            # Convert extras to JSON-serializable format
            extras_serializable = []
            for extra in annotations.extras:
                if extra is not None and isinstance(extra, dict):
                    # Convert numpy types to Python types
                    extra_dict = {}
                    for k, v in extra.items():
                        if isinstance(v, (np.integer, np.floating)):
                            extra_dict[k] = float(v)
                        elif isinstance(v, np.ndarray):
                            extra_dict[k] = v.tolist()
                        else:
                            extra_dict[k] = v
                    extras_serializable.append(extra_dict)
                else:
                    extras_serializable.append(None)
            annotations_data['extras'] = extras_serializable

        # Save to JSON
        with open(annotations_path, 'w') as f:
            json.dump(annotations_data, f, indent=2)

    def _load_annotations(self, npy_path: Path) -> Optional[Dict]:
        """Load annotations from JSON file.

        Args:
            npy_path: Path to .npy file (will load .annotations.json)

        Returns:
            Annotations dict or None if file doesn't exist
        """
        import json

        annotations_path = npy_path.with_suffix('.annotations.json')

        if not annotations_path.exists():
            return None

        with open(annotations_path, 'r') as f:
            return json.load(f)
