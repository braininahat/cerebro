"""JEPA Phase 1 pretraining data module.

Loads ALL tasks (CCD, movies, resting state, etc.) without labels.
No trial filtering - pure self-supervised learning on raw EEG.

From magnum_opus.md Phase 1:
- Use all available tasks
- Windows with optional random temporal crops (augmentation)
- Mix all tasks to learn general representations

MIGRATED TO UniversalCacheManager (memory-mapped numpy-based lazy loading)
- Replaces old GranularCacheManager (pickle-based)
- Better memory efficiency via lazy loading with .npy memmap files
- Parallel cache building
- Fault tolerance with parquet manifests
"""

from pathlib import Path
from typing import Any, Optional
import logging
import os

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from cerebro.data.unified_cache import UniversalCacheManager

logger = logging.getLogger(__name__)


def _collate_braindecode_windows(batch):
    """Collate function for braindecode WindowsDataset.

    Braindecode returns (X, y, crop_inds) or (X, y, crop_inds, metadata)
    but trainers expect (X, y) batches.

    Args:
        batch: List of tuples from braindecode WindowsDataset

    Returns:
        Tuple of (x_batch, y_batch) as torch tensors
    """
    x_list = []
    y_list = []

    for item in batch:
        # Braindecode returns (X, y, crop_inds) or more
        # We only need X and y
        x, y = item[0], item[1]

        # Convert to tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        x_list.append(x)
        y_list.append(y)

    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)

    # Squeeze targets to 1D if they're [batch_size, 1]
    # MSE loss expects [batch_size] not [batch_size, 1]
    if y_batch.ndim == 2 and y_batch.shape[1] == 1:
        y_batch = y_batch.squeeze(1)

    return x_batch, y_batch


def _collate_multi_dataset(batch):
    """Collate function for MultiDatasetWrapper.

    MultiDatasetWrapper returns (x, unmapped_mask) tuples where:
    - HBN samples: (x, None) - 129 channels, no unmapped positions
    - TUH samples: (x, unmapped_mask) - 129 channels (projected), boolean mask for unmapped positions

    Args:
        batch: List of (x, unmapped_mask) tuples

    Returns:
        Tuple of (x_batch, unmapped_mask_batch) as torch tensors
        - x_batch: (batch_size, n_channels, n_samples)
        - unmapped_mask_batch: (batch_size, n_channels) boolean tensor (False for HBN, True for TUH unmapped)
    """
    x_list = []
    unmapped_mask_list = []

    for x, unmapped_mask in batch:
        # Convert to tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x_list.append(x)

        # Handle None masks (HBN samples - no unmapped channels)
        if unmapped_mask is None:
            # Create all-False mask (all channels are valid)
            n_channels = x.shape[0]
            unmapped_mask = torch.zeros(n_channels, dtype=torch.bool)

        if isinstance(unmapped_mask, np.ndarray):
            unmapped_mask = torch.from_numpy(unmapped_mask).bool()

        unmapped_mask_list.append(unmapped_mask)

    x_batch = torch.stack(x_list, dim=0)
    unmapped_mask_batch = torch.stack(unmapped_mask_list, dim=0)

    return x_batch, unmapped_mask_batch


class JEPAPretrainDataModule(L.LightningDataModule):
    """Data module for JEPA Phase 1 self-supervised pretraining.

    Loads ALL tasks from HBN dataset without labels.
    Creates fixed-length windows from all available recordings.

    Uses UniversalCacheManager for efficient memory-mapped numpy caching with lazy loading.

    Args:
        data_dir: Root directory for EEG data
        releases: List of release IDs (default: all except R5)
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        prefetch_factor: Number of batches each worker prefetches ahead (default: 2)
            Higher values = more memory but smoother GPU feeding. Try 4-8 for large num_workers.
        persistent_workers: Keep workers alive between epochs (default: False)
            Reduces epoch transition time but uses more memory.
        window_length: Window length in seconds (default: 2.0 for S-JEPA)
        stride: Stride between windows (default: 1.0)
        crop_length: Random crop length in seconds (default: None, set to enable temporal augmentation)
        val_split: Validation split fraction (default: 0.2 subject-level)
        test_release: Specific release to use as test set (e.g., "R5"), or None for no test
        n_chans_select: Number of channels to use (default: 129)
        sfreq: Target sampling frequency (default: 100 Hz)
        mini: If True, use mini dataset for fast prototyping
        all_tasks: List of tasks to include (default: all available)
            Options: ["contrastChangeDetection", "restingState",
                     "DespicableMe", "ThePresent", "DiaryOfAWimpyKid",
                     "FunwithFractals", "surroundSupp"]
        seed: Random seed for subject-level splits (default: 42)

    Example:
        >>> # Development mode with R5 test set
        >>> dm = JEPAPretrainDataModule(
        ...     data_dir="./data",
        ...     releases=["R1", "R2", "R3", "R4", "R5"],
        ...     test_release="R5",
        ...     batch_size=256
        ... )
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        releases: Optional[list[str]] = None,
        batch_size: int = 256,
        num_workers: int = 8,
        prefetch_factor: int = 2,  # Number of batches each worker prefetches
        persistent_workers: bool = False,  # Keep workers alive between epochs
        window_length: float = 2.0,  # 2s windows for S-JEPA
        stride: float = 1.0,
        crop_length: Optional[float] = None,  # Optional temporal augmentation
        val_split: float = 0.2,  # Subject-level split
        test_release: Optional[str] = None,  # Specific release for test (e.g., "R5")
        n_chans_select: int = 129,
        sfreq: int = 100,
        mini: bool = False,
        all_tasks: Optional[list[str]] = None,
        seed: int = 42,  # Random seed for subject-level splits
        task_windowing: Optional[Any] = None,  # Challenge 1: Event-based windowing
        subject_target_field: Optional[str] = None,  # Challenge 2: Subject-level targets
        datasets: Optional[list[dict]] = None,  # Multi-dataset configs (HBN, TUH, etc.)
        channel_projection: bool = True,  # Enable TUH→HBN projection
    ):
        super().__init__()

        # Save all hyperparameters for Lightning compatibility
        self.save_hyperparameters(ignore=['task_windowing'])  # Task instance not serializable

        self.data_dir = Path(data_dir).resolve()  # Absolute path for EEGChallengeDataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        # Default: all releases except R5 (competition validation)
        if releases is None:
            releases = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"]
        self.releases = releases

        # Window parameters
        self.window_length = window_length
        self.stride = stride
        self.crop_length = crop_length
        self.val_split = val_split
        self.test_release = test_release
        self.n_chans_select = n_chans_select
        self.sfreq = sfreq
        self.mini = mini
        self.seed = seed

        # Tasks to include (default: all 7 HBN tasks)
        if all_tasks is None:
            all_tasks = [
                "contrastChangeDetection",
                "restingState",
                "DespicableMe",
                "ThePresent",
                "DiaryOfAWimpyKid",
                "FunwithFractals",
                "surroundSupp",
            ]
        self.all_tasks = all_tasks

        # Finetuning modes
        self.task_windowing = task_windowing  # Challenge 1: Event-based windowing with trial targets
        self.subject_target_field = subject_target_field  # Challenge 2: Subject-level targets

        # Multi-dataset support
        self.datasets = datasets  # List of dataset configs (e.g., [{"name": "hbn", ...}, {"name": "tuh", ...}])
        self.channel_projection = channel_projection  # Enable TUH→HBN projection

        # Initialize UniversalCacheManager (memory-mapped numpy-based, lazy loading)
        # Use CACHE_PATH for processed caches (fail if not set)
        cache_path = os.getenv("CACHE_PATH")
        if not cache_path:
            raise ValueError(
                "CACHE_PATH environment variable not set. "
                "Set it in your .env file or export it before running."
            )
        self.cache_root = Path(cache_path)

        self.cache_mgr = UniversalCacheManager(
            cache_root=str(self.cache_root),
            preprocessing_params={
                "sfreq": self.sfreq,
                "bandpass": None,  # No bandpass filtering (already done by EEGChallengeDataset)
                "n_channels": self.n_chans_select,
                "standardize": False,
            },
            data_dir=str(self.data_dir)  # For EEGChallengeDataset downloads
        )

        # Placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def batch_size(self) -> int:
        """Batch size for DataLoaders."""
        return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        """Update batch size (modified by Lightning's batch size scaler)."""
        self.hparams.batch_size = value

    def _get_split_file_path(self) -> Path:
        """Generate path for split file based on releases, seed, val_split.

        Returns:
            Path to split JSON file
        """
        import hashlib
        releases_str = "_".join(sorted(self.releases))
        releases_hash = hashlib.md5(releases_str.encode()).hexdigest()[:8]
        val_str = f"{int(self.val_split*100)}"
        return self.data_dir / "splits" / f"splits_{releases_hash}_seed{self.seed}_val{val_str}.json"

    def _save_splits(self, train_subjects: np.ndarray, val_subjects: np.ndarray, test_subjects: Optional[np.ndarray] = None):
        """Save subject splits to disk for reproducibility.

        Args:
            train_subjects: Training subject IDs
            val_subjects: Validation subject IDs
            test_subjects: Test subject IDs (optional)
        """
        import json
        split_path = self._get_split_file_path()
        split_path.parent.mkdir(parents=True, exist_ok=True)

        splits = {
            "train": train_subjects.tolist(),
            "val": val_subjects.tolist(),
            "test": test_subjects.tolist() if test_subjects is not None else []
        }

        with open(split_path, 'w') as f:
            json.dump(splits, f, indent=2)

        logger.info(f"Saved splits to: {split_path}")

    def _load_splits(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load subject splits from disk if they exist.

        Returns:
            Tuple of (train_subjects, val_subjects, test_subjects) or (None, None, None)
        """
        import json
        split_path = self._get_split_file_path()

        if split_path.exists():
            with open(split_path, 'r') as f:
                splits = json.load(f)
            logger.info(f"Loaded existing splits from: {split_path}")
            return (
                np.array(splits["train"]),
                np.array(splits["val"]),
                np.array(splits.get("test", []))
            )

        return None, None, None

    def prepare_data(self):
        """Download data if needed."""
        # EEGChallengeDataset handles downloading automatically
        pass

    def _setup_event_based_windowing(self, train_recordings, val_recordings, test_recordings):
        """Setup Challenge 1: Event-based windowing with trial-level targets.

        Args:
            train_recordings: Training recording IDs from splits
            val_recordings: Validation recording IDs from splits
            test_recordings: Test recording IDs from splits
        """
        from braindecode.datasets import BaseConcatDataset
        from eegdash import EEGChallengeDataset

        logger.info("\n[bold cyan]MODE: Challenge 1 (Event-Based Windowing)[/bold cyan]")
        logger.info("Loading recordings for event-based windowing...")

        # Load recordings via EEGChallengeDataset (uses download cache)
        # We need braindecode datasets for Challenge1Task.create_windows()
        all_datasets = []
        for release in self.releases:
            try:
                dataset = EEGChallengeDataset(
                    release=release,
                    task=self.all_tasks[0] if len(self.all_tasks) == 1 else "contrastChangeDetection",
                    cache_dir=str(self.data_dir),
                    mini=self.mini
                )
                all_datasets.extend(dataset.datasets)
            except Exception as e:
                logger.warning(f"Failed to load {release}: {e}")

        logger.info(f"Loaded {len(all_datasets)} recordings")

        # Generate cache key for Challenge 1 windows
        import hashlib
        cache_params = {
            'releases': sorted(self.releases),
            'tasks': sorted(self.all_tasks),
            'mini': self.mini,
            'window_len': self.task_windowing.window_len,
            'shift_after_stim': self.task_windowing.shift_after_stim,
            'sfreq': self.task_windowing.sfreq,
            'epoch_len_s': self.task_windowing.epoch_len_s,
            'anchor': self.task_windowing.anchor,
        }
        cache_key = hashlib.md5(str(sorted(cache_params.items())).encode()).hexdigest()[:12]
        cache_dir = self.cache_root / "challenge1"
        cache_file_npy = cache_dir / f"windows_{cache_key}.npy"
        cache_file_meta = cache_dir / f"windows_{cache_key}_meta.parquet"

        # Try to load from cache (memmap + metadata)
        if cache_file_npy.exists() and cache_file_meta.exists():
            logger.info(f"[bold green]Loading cached Challenge 1 windows from: {cache_file_npy.name}[/bold green]")
            try:
                # Load metadata
                metadata = pd.read_parquet(cache_file_meta)

                # Load memmap data
                windows_array = np.load(str(cache_file_npy), mmap_mode='r')

                logger.info(f"[bold green]✓ Loaded {len(metadata)} cached windows[/bold green]")

                # Create dataset from memmap (will be handled below)
                windowed_dataset = None  # Signal that we have cache
                cached_metadata = metadata
                cached_array = windows_array
            except Exception as e:
                logger.warning(f"Failed to load cache ({e}), rebuilding...")
                cache_file_npy.unlink(missing_ok=True)
                cache_file_meta.unlink(missing_ok=True)
                windowed_dataset = None
                cached_metadata = None
                cached_array = None
        else:
            windowed_dataset = None
            cached_metadata = None
            cached_array = None

        # Build windows if not cached
        if cached_metadata is None:
            # Wrap in BaseConcatDataset
            concat_dataset = BaseConcatDataset(all_datasets)

            # Suppress braindecode's verbose "Used Annotations" logging
            import logging as stdlib_logging
            braindecode_logger = stdlib_logging.getLogger('braindecode')
            old_level = braindecode_logger.level
            braindecode_logger.setLevel(stdlib_logging.WARNING)

            # Use task windowing to create stimulus-locked windows
            logger.info("Applying task-specific windowing (will be cached)...")
            windowed_dataset = self.task_windowing.create_windows(concat_dataset)

            # Restore logging level
            braindecode_logger.setLevel(old_level)

            # Extract windows to memmap array and save metadata
            logger.info(f"Extracting windows to cache: {cache_file_npy.name}")
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Get metadata
            metadata = windowed_dataset.get_metadata()

            # Extract all windows to numpy array
            n_windows = len(windowed_dataset)
            if n_windows > 0:
                # Get shape from first window
                first_window = windowed_dataset[0][0]  # (X, y, crop_inds) -> X
                n_channels, n_samples = first_window.shape

                # Create memmap array
                logger.info(f"Creating memmap array: {n_windows} windows × {n_channels} channels × {n_samples} samples")
                windows_array = np.lib.format.open_memmap(
                    str(cache_file_npy),
                    mode='w+',
                    dtype=np.float32,
                    shape=(n_windows, n_channels, n_samples)
                )

                # Extract all windows
                logger.info("Extracting windows...")
                for i in range(n_windows):
                    X = windowed_dataset[i][0]  # Get data (ignore y and crop_inds)
                    windows_array[i] = X

                windows_array.flush()
                del windows_array

                # Save metadata
                metadata.to_parquet(cache_file_meta, index=False)

                logger.info(f"[bold green]✓ Cached {n_windows} windows to memmap + metadata[/bold green]")

                # Load back for use
                cached_metadata = metadata
                cached_array = np.load(str(cache_file_npy), mmap_mode='r')
            else:
                logger.warning("No windows to cache!")
                cached_metadata = None
                cached_array = None

        # Use cached metadata if available, otherwise get from windowed_dataset
        if cached_metadata is not None:
            # Using cached data - create custom dataset wrapper
            from cerebro.data.unified_cache.lazy_dataset import MemmapWindowDataset

            # Filter metadata by subjects
            train_mask = cached_metadata['subject'].isin(train_recordings)
            val_mask = cached_metadata['subject'].isin(val_recordings)
            test_mask = cached_metadata['subject'].isin(test_recordings) if len(test_recordings) > 0 else pd.Series([False] * len(cached_metadata))

            train_meta = cached_metadata[train_mask].reset_index(drop=True)
            val_meta = cached_metadata[val_mask].reset_index(drop=True)
            test_meta = cached_metadata[test_mask].reset_index(drop=True) if len(test_recordings) > 0 else cached_metadata.head(min(10, len(cached_metadata)))

            # Add array_index column for MemmapWindowDataset
            train_meta['array_index'] = train_meta.index
            val_meta['array_index'] = val_meta.index
            test_meta['array_index'] = test_meta.index

            # Create memmap datasets
            self.train_dataset = MemmapWindowDataset(
                memmap_path=cache_file_npy,
                metadata=train_meta,
                sfreq=self.sfreq,
                mode='train'
            )
            self.val_dataset = MemmapWindowDataset(
                memmap_path=cache_file_npy,
                metadata=val_meta,
                sfreq=self.sfreq,
                mode='val'
            )
            self.test_dataset = MemmapWindowDataset(
                memmap_path=cache_file_npy,
                metadata=test_meta,
                sfreq=self.sfreq,
                mode='val'
            )
        else:
            # Using freshly built windowed_dataset (braindecode format)
            metadata = windowed_dataset.get_metadata()

            # Split by subjects
            train_subjects = train_recordings
            val_subjects = val_recordings
            test_subjects = test_recordings if len(test_recordings) > 0 else []

            train_mask = metadata['subject'].isin(train_subjects)
            val_mask = metadata['subject'].isin(val_subjects)
            test_mask = metadata['subject'].isin(test_subjects) if len(test_subjects) > 0 else pd.Series([False] * len(metadata))

            # Create split datasets (braindecode WindowsDataset)
            # Filter by indices
            train_indices = metadata[train_mask].index.tolist()
            val_indices = metadata[val_mask].index.tolist()
            test_indices = metadata[test_mask].index.tolist() if len(test_subjects) > 0 else list(range(min(10, len(windowed_dataset))))

            from braindecode.datasets import BaseConcatDataset as BDConcatDataset
            self.train_dataset = BDConcatDataset([windowed_dataset.datasets[i] for i in train_indices])
            self.val_dataset = BDConcatDataset([windowed_dataset.datasets[i] for i in val_indices])
            self.test_dataset = BDConcatDataset([windowed_dataset.datasets[i] for i in test_indices])

        logger.info(f"Train windows: {len(self.train_dataset)}")
        logger.info(f"Val windows: {len(self.val_dataset)}")
        logger.info(f"Test windows: {len(self.test_dataset)}")

    def _setup_subject_target_windowing(self, train_recordings, val_recordings, test_recordings, all_recordings):
        """Setup Challenge 2: Fixed windowing + subject-level targets.

        Args:
            train_recordings: Training recordings DataFrame
            val_recordings: Validation recordings DataFrame
            test_recordings: Test recordings DataFrame
            all_recordings: Combined recordings DataFrame
        """
        logger.info("\n[bold cyan]MODE: Challenge 2 (Subject-Level Targets)[/bold cyan]")
        logger.info(f"Target field: {self.subject_target_field}")
        logger.info(f"Creating {self.window_length}s windows (reusing pretraining cache)...")

        # Use existing fixed windowing (reuses pretraining cache!)
        full_dataset = self.cache_mgr.get_windowed_dataset(
            recordings=all_recordings,
            window_len_s=self.window_length,
            stride_s=self.stride,
            crop_len_s=None,
            mode='train'
        )
        logger.info(f"Window cache ready: {len(full_dataset):,} total windows")

        # Load participant data with target scores
        logger.info(f"Loading participant data for target field: {self.subject_target_field}")
        participant_dfs = []

        # Map release IDs to dataset directory names
        release_to_ds = {
            'R1': 'ds005505-bdf', 'R2': 'ds005506-bdf', 'R3': 'ds005507-bdf',
            'R4': 'ds005508-bdf', 'R6': 'ds005510-bdf', 'R7': 'ds005511-bdf',
            'R8': 'ds005512-bdf', 'R9': 'ds005514-bdf', 'R10': 'ds005515-bdf',
            'R11': 'ds005516-bdf'
        }

        for release in self.releases:
            if release in release_to_ds:
                participants_file = self.data_dir / release_to_ds[release] / "participants.tsv"
                if participants_file.exists():
                    df = pd.read_csv(participants_file, sep='\t')
                    participant_dfs.append(df)
                    logger.info(f"  Loaded {len(df)} subjects from {release}")
                else:
                    logger.warning(f"  No participants.tsv found for {release}")

        if not participant_dfs:
            raise ValueError(f"No participants.tsv files found for releases {self.releases}")

        participants_df = pd.concat(participant_dfs, ignore_index=True)

        # Create subject->target mapping
        subject_targets = {}
        nan_count = 0
        for _, row in participants_df.iterrows():
            subject_id = row['participant_id'].replace('sub-', '')
            if self.subject_target_field in row:
                target_value = row[self.subject_target_field]
                # Handle NaN and n/a values
                if pd.notna(target_value) and str(target_value).lower() != 'n/a':
                    try:
                        subject_targets[subject_id] = float(target_value)
                    except (ValueError, TypeError):
                        nan_count += 1
                else:
                    nan_count += 1
            else:
                logger.warning(f"Target field '{self.subject_target_field}' not found in participants.tsv")
                break

        logger.info(f"Loaded targets for {len(subject_targets)} subjects ({nan_count} NaN/invalid)")

        # Attach targets to dataset metadata
        def attach_targets_to_metadata(dataset, subject_targets):
            """Attach subject-level targets to window dataset metadata."""
            # Add target column to metadata based on subject
            metadata = dataset.metadata.copy()
            metadata['target'] = metadata['subject'].map(subject_targets)

            # Filter out windows without valid targets
            valid_mask = metadata['target'].notna()
            metadata_with_targets = metadata[valid_mask].reset_index(drop=True)

            # Create new dataset with updated metadata
            # NOTE: Backward compatibility support for legacy Zarr-based datasets
            # Current implementation uses memmap_path with .npy files
            if hasattr(dataset, 'zarr_path'):  # LazyZarrWindowDataset (legacy)
                from cerebro.data.unified_cache.lazy_dataset import LazyZarrWindowDataset
                return LazyZarrWindowDataset(
                    zarr_path=dataset.zarr_path,
                    metadata=metadata_with_targets,
                    crop_len_s=dataset.crop_len_s,
                    sfreq=dataset.sfreq,
                    mode=dataset.mode
                )
            elif hasattr(dataset, 'memmap_path'):  # MemmapWindowDataset (current)
                from cerebro.data.unified_cache.lazy_dataset import MemmapWindowDataset
                return MemmapWindowDataset(
                    memmap_path=dataset.memmap_path,
                    metadata=metadata_with_targets,
                    crop_len_s=dataset.crop_len_s,
                    sfreq=dataset.sfreq,
                    mode=dataset.mode
                )
            else:
                raise ValueError(f"Unknown dataset type: {type(dataset)}")

        # Filter by recordings and attach targets
        train_filtered = full_dataset.filter_by_recordings(train_recordings["recording_id"].tolist())
        self.train_dataset = attach_targets_to_metadata(train_filtered, subject_targets)
        self.train_dataset.crop_len_s = self.crop_length
        self.train_dataset.mode = 'train'

        val_filtered = full_dataset.filter_by_recordings(val_recordings["recording_id"].tolist())
        self.val_dataset = attach_targets_to_metadata(val_filtered, subject_targets)
        self.val_dataset.crop_len_s = self.crop_length
        self.val_dataset.mode = 'val'

        test_filtered = full_dataset.filter_by_recordings(test_recordings["recording_id"].tolist())
        self.test_dataset = attach_targets_to_metadata(test_filtered, subject_targets)
        self.test_dataset.crop_len_s = self.crop_length
        self.test_dataset.mode = 'val'

        logger.info(f"Train windows: {len(self.train_dataset):,} (with targets)")
        logger.info(f"Val windows: {len(self.val_dataset):,} (with targets)")
        logger.info(f"Test windows: {len(self.test_dataset):,} (with targets)")

    def _setup_pretraining_windowing(self, train_recordings, val_recordings, test_recordings, all_recordings):
        """Setup Pretraining: Fixed windowing without targets.

        Args:
            train_recordings: Training recordings DataFrame
            val_recordings: Validation recordings DataFrame
            test_recordings: Test recordings DataFrame
            all_recordings: Combined recordings DataFrame
        """
        logger.info("\n[bold cyan]MODE: Pretraining (Fixed Windowing)[/bold cyan]")
        logger.info(f"Creating {self.window_length}s windows, {self.stride}s stride...")

        # Multi-dataset support
        if self.datasets is not None:
            logger.info(f"[cyan]Building windowed datasets for {len(self.datasets)} datasets[/cyan]")

            # Build windowed datasets for each dataset separately
            train_dataset_list = []
            val_dataset_list = []
            test_dataset_list = []

            for ds_config in self.datasets:
                ds_name = ds_config["name"]
                logger.info(f"\n[cyan]Processing {ds_name} windows:[/cyan]")

                # Filter recordings for this dataset
                ds_all_recordings = all_recordings[all_recordings["dataset"] == ds_name]
                ds_train_recordings = train_recordings[train_recordings["dataset"] == ds_name]
                ds_val_recordings = val_recordings[val_recordings["dataset"] == ds_name]
                ds_test_recordings = test_recordings[test_recordings["dataset"] == ds_name]

                logger.info(f"  Total recordings: {len(ds_all_recordings)}")
                logger.info(f"  Train recordings: {len(ds_train_recordings)}")
                logger.info(f"  Val recordings: {len(ds_val_recordings)}")
                logger.info(f"  Test recordings: {len(ds_test_recordings)}")

                # Build window cache for this dataset
                ds_full_dataset = self.cache_mgr.get_windowed_dataset(
                    recordings=ds_all_recordings,
                    window_len_s=self.window_length,
                    stride_s=self.stride,
                    crop_len_s=None,
                    mode='train'
                )
                logger.info(f"  Window cache: {len(ds_full_dataset):,} total windows")

                # Filter by recording IDs
                ds_train = ds_full_dataset.filter_by_recordings(ds_train_recordings["recording_id"].tolist())
                ds_train.crop_len_s = self.crop_length
                ds_train.mode = 'train'
                train_dataset_list.append((ds_name, ds_train))
                logger.info(f"  Train windows: {len(ds_train):,}")

                ds_val = ds_full_dataset.filter_by_recordings(ds_val_recordings["recording_id"].tolist())
                ds_val.crop_len_s = self.crop_length
                ds_val.mode = 'val'
                val_dataset_list.append((ds_name, ds_val))
                logger.info(f"  Val windows: {len(ds_val):,}")

                ds_test = ds_full_dataset.filter_by_recordings(ds_test_recordings["recording_id"].tolist())
                ds_test.crop_len_s = self.crop_length
                ds_test.mode = 'val'
                test_dataset_list.append((ds_name, ds_test))
                logger.info(f"  Test windows: {len(ds_test):,}")

            # Wrap in MultiDatasetWrapper
            from cerebro.data.multi_dataset import MultiDatasetWrapper

            self.train_dataset = MultiDatasetWrapper(
                dataset_list=train_dataset_list,
                projection_enabled=self.channel_projection
            )
            self.val_dataset = MultiDatasetWrapper(
                dataset_list=val_dataset_list,
                projection_enabled=self.channel_projection
            )
            self.test_dataset = MultiDatasetWrapper(
                dataset_list=test_dataset_list,
                projection_enabled=self.channel_projection
            )

            logger.info(f"\n[bold green]Multi-dataset wrapped:[/bold green]")
            logger.info(f"  Train: {len(self.train_dataset):,} total windows")
            logger.info(f"  Val: {len(self.val_dataset):,} total windows")
            logger.info(f"  Test: {len(self.test_dataset):,} total windows")

        else:
            # Single-dataset legacy behavior
            # Build/load window cache with ALL recordings
            full_dataset = self.cache_mgr.get_windowed_dataset(
                recordings=all_recordings,
                window_len_s=self.window_length,
                stride_s=self.stride,
                crop_len_s=None,
                mode='train'
            )
            logger.info(f"Window cache ready: {len(full_dataset):,} total windows")

            # Filter by subjects
            self.train_dataset = full_dataset.filter_by_recordings(train_recordings["recording_id"].tolist())
            self.train_dataset.crop_len_s = self.crop_length
            self.train_dataset.mode = 'train'

            self.val_dataset = full_dataset.filter_by_recordings(val_recordings["recording_id"].tolist())
            self.val_dataset.crop_len_s = self.crop_length
            self.val_dataset.mode = 'val'

            self.test_dataset = full_dataset.filter_by_recordings(test_recordings["recording_id"].tolist())
            self.test_dataset.crop_len_s = self.crop_length
            self.test_dataset.mode = 'val'

            logger.info(f"Train windows: {len(self.train_dataset):,}")
            logger.info(f"Val windows: {len(self.val_dataset):,}")
            logger.info(f"Test windows: {len(self.test_dataset):,}")

    def setup(self, stage: Optional[str] = None):
        """Load and prepare data using UniversalCacheManager (memory-mapped numpy-based, lazy loading)."""

        # Multi-dataset support
        if self.datasets is not None:
            logger.info(f"\n[bold cyan]Multi-Dataset Mode: {len(self.datasets)} datasets[/bold cyan]")

            # Build raw cache for each dataset
            for ds_config in self.datasets:
                ds_name = ds_config["name"]
                ds_releases = ds_config.get("releases", self.releases)
                ds_tasks = ds_config.get("tasks", self.all_tasks)
                ds_mini = ds_config.get("mini", self.mini)

                logger.info(f"\n[cyan]Building raw cache for {ds_name}:[/cyan]")
                logger.info(f"  Releases: {ds_releases}")
                logger.info(f"  Tasks: {ds_tasks}")
                logger.info(f"  Mini: {ds_mini}")

                if ds_name == "hbn":
                    self.cache_mgr.build_raw(
                        dataset="hbn",
                        releases=ds_releases,
                        tasks=ds_tasks,
                        mini=ds_mini
                    )
                elif ds_name == "tuh":
                    # TUH requires tuh_path parameter
                    tuh_path = ds_config.get("tuh_path")
                    if not tuh_path:
                        raise ValueError(f"TUH dataset requires 'tuh_path' in config: {ds_config}")

                    self.cache_mgr.build_raw(
                        dataset="tuh",
                        releases=ds_releases,
                        tasks=ds_tasks,
                        tuh_path=tuh_path,
                        mini=ds_mini
                    )
                else:
                    raise ValueError(f"Unknown dataset: {ds_name}. Supported: hbn, tuh")
        else:
            # Single-dataset legacy behavior
            logger.info(f"Loading data from releases: {self.releases}")
            logger.info(f"Tasks: {self.all_tasks}")
            logger.info(f"Mini mode: {self.mini}")
            logger.info(f"Test release: {self.test_release}")

            # Build/load Level 1 raw cache for ALL releases
            # This automatically handles parallel processing, caching, and fault tolerance
            self.cache_mgr.build_raw(
                dataset="hbn",
                releases=self.releases,
                tasks=self.all_tasks,
                mini=self.mini
            )

        # Step 1: Query recordings from Level 1 cache
        if self.datasets is not None:
            # Multi-dataset: Query each dataset and combine
            train_val_recordings_list = []
            test_recordings_list = []

            for ds_config in self.datasets:
                ds_name = ds_config["name"]
                ds_releases = ds_config.get("releases", self.releases)
                ds_tasks = ds_config.get("tasks", self.all_tasks)
                ds_mini = ds_config.get("mini", self.mini)

                # Separate test release for this dataset
                if self.test_release:
                    if self.test_release in ds_releases:
                        raise ValueError(
                            f"test_release '{self.test_release}' found in {ds_name} releases {ds_releases}. "
                            "This would cause data leakage! Remove it from the dataset's releases list."
                        )
                    # Test release is separate from this dataset's releases
                    ds_test_releases = [self.test_release]
                    ds_train_val_releases = ds_releases
                else:
                    ds_test_releases = []
                    ds_train_val_releases = ds_releases

                # Query train/val recordings
                if ds_train_val_releases:
                    ds_train_val_rec = self.cache_mgr.query_raw(
                        dataset=ds_name,
                        releases=ds_train_val_releases,
                        tasks=ds_tasks,
                        mini=ds_mini
                    )
                    ds_train_val_rec["dataset"] = ds_name  # Add dataset identifier
                    train_val_recordings_list.append(ds_train_val_rec)
                    logger.info(f"  {ds_name} train/val: {len(ds_train_val_rec)} recordings")

                # Query test recordings
                if ds_test_releases:
                    ds_test_rec = self.cache_mgr.query_raw(
                        dataset=ds_name,
                        releases=ds_test_releases,
                        tasks=ds_tasks,
                        mini=ds_mini
                    )
                    ds_test_rec["dataset"] = ds_name
                    test_recordings_list.append(ds_test_rec)
                    logger.info(f"  {ds_name} test: {len(ds_test_rec)} recordings")

            # Combine recordings from all datasets
            train_val_recordings = pd.concat(train_val_recordings_list, ignore_index=True) if train_val_recordings_list else pd.DataFrame()
            test_recordings = pd.concat(test_recordings_list, ignore_index=True) if test_recordings_list else pd.DataFrame()

        else:
            # Single-dataset legacy behavior
            # Separate test release from train/val
            if self.test_release:
                if self.test_release in self.releases:
                    raise ValueError(
                        f"test_release '{self.test_release}' found in training releases {self.releases}. "
                        "This would cause data leakage! Remove it from releases list."
                    )
                test_releases = [self.test_release]
                train_val_releases = self.releases  # No filtering needed - test_release not in list
                logger.info(f"Using {self.test_release} as test set")
                logger.info(f"Train/val releases: {train_val_releases}")
            else:
                test_releases = []
                train_val_releases = self.releases
                logger.info("No test release specified - no test split")

            # Query recordings from Level 1 cache
            train_val_recordings = self.cache_mgr.query_raw(
                dataset="hbn",
                releases=train_val_releases,
                tasks=self.all_tasks,
                mini=self.mini
            )

            if test_releases:
                test_recordings = self.cache_mgr.query_raw(
                    dataset="hbn",
                    releases=test_releases,
                    tasks=self.all_tasks,
                    mini=self.mini
                )
            else:
                test_recordings = pd.DataFrame()

        logger.info(f"Train/val recordings: {len(train_val_recordings)}")
        logger.info(f"Test recordings: {len(test_recordings)}")

        if len(train_val_recordings) == 0:
            raise ValueError("No train/val recordings found after filtering!")

        # Step 3: Subject-level split for train/val (DataFrame-based)
        train_val_subjects = train_val_recordings["subject"].unique()
        logger.info(f"Train/val unique subjects: {len(train_val_subjects)}")

        # Try to load existing splits for reproducibility
        train_subjects_saved, val_subjects_saved, _ = self._load_splits()

        if train_subjects_saved is not None:
            # Use saved splits
            train_subjects = train_subjects_saved
            val_subjects = val_subjects_saved
            logger.info("Using previously saved splits for reproducibility")

            train_recordings = train_val_recordings[
                train_val_recordings["subject"].isin(train_subjects)
            ]
            val_recordings = train_val_recordings[
                train_val_recordings["subject"].isin(val_subjects)
            ]
        elif self.val_split > 0 and len(train_val_subjects) > 1:
            # Create new splits
            train_subjects, val_subjects = train_test_split(
                train_val_subjects,
                test_size=self.val_split,
                random_state=self.seed
            )

            train_recordings = train_val_recordings[
                train_val_recordings["subject"].isin(train_subjects)
            ]
            val_recordings = train_val_recordings[
                train_val_recordings["subject"].isin(val_subjects)
            ]

            # Save splits for future runs
            self._save_splits(train_subjects, val_subjects)
        else:
            # No val split - use all for training, create dummy val for Lightning
            train_subjects = train_val_subjects
            val_subjects = np.array([])
            train_recordings = train_val_recordings
            val_recordings = train_val_recordings.head(min(10, len(train_val_recordings)))
            logger.info("No validation split - using all train/val subjects for training")

        if len(test_recordings) == 0:
            # No test data - create dummy test for Lightning
            test_recordings = train_recordings.head(min(10, len(train_recordings)))
            logger.info("No test split - using dummy test set")

        logger.info(
            f"\nSubject split:\n"
            f"  Train: {len(train_subjects)} subjects, {len(train_recordings)} recordings\n"
            f"  Val:   {len(val_subjects)} subjects, {len(val_recordings)} recordings\n"
            f"  Test:  {test_recordings['subject'].nunique()} subjects, {len(test_recordings)} recordings"
        )

        # Step 4: Windowing (mode-specific)
        # Combine all recordings (train/val + test) for window cache building
        all_recordings = pd.concat([train_val_recordings, test_recordings], ignore_index=True)

        # Choose windowing strategy based on configuration
        if self.task_windowing is not None:
            # Challenge 1: Event-based windowing with trial-level targets
            self._setup_event_based_windowing(
                train_subjects.tolist(),
                val_subjects.tolist(),
                test_recordings['subject'].unique().tolist() if len(test_recordings) > 0 else []
            )
        elif self.subject_target_field is not None:
            # Challenge 2: Fixed windowing + subject-level targets
            self._setup_subject_target_windowing(
                train_recordings,
                val_recordings,
                test_recordings,
                all_recordings
            )
        else:
            # Pretraining: Fixed windowing without targets
            self._setup_pretraining_windowing(
                train_recordings,
                val_recordings,
                test_recordings,
                all_recordings
            )

    def train_dataloader(self) -> DataLoader:
        """Training dataloader with prefetching support.

        NOTE: Lightning may call this method multiple times during training.
        This is normal behavior and doesn't rebuild the dataset (just creates new DataLoader instance).
        The actual dataset (self.train_dataset) is cached and reused.
        """
        # Debug: Log when this is called to track dataloader recreation
        import inspect
        caller = inspect.stack()[1]
        logger.debug(f"[DATALOADER] train_dataloader() called from {caller.function} at {caller.filename}:{caller.lineno}")

        # Select collate function based on mode:
        # - Multi-dataset: _collate_multi_dataset (handles channel projection)
        # - Challenge 1/2: _collate_braindecode_windows (handles braindecode format)
        # - Pretraining: None (default PyTorch collate)
        if self.datasets is not None:
            collate_fn = _collate_multi_dataset
        elif self.task_windowing is not None or self.subject_target_field is not None:
            collate_fn = _collate_braindecode_windows
        else:
            collate_fn = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=getattr(self.hparams, 'pin_memory', True),
            collate_fn=collate_fn,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader with prefetching support."""
        # Select collate function based on mode
        if self.datasets is not None:
            collate_fn = _collate_multi_dataset
        elif self.task_windowing is not None or self.subject_target_field is not None:
            collate_fn = _collate_braindecode_windows
        else:
            collate_fn = None

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=getattr(self.hparams, 'pin_memory', True),
            collate_fn=collate_fn,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader with prefetching support."""
        # Select collate function based on mode
        if self.datasets is not None:
            collate_fn = _collate_multi_dataset
        elif self.task_windowing is not None or self.subject_target_field is not None:
            collate_fn = _collate_braindecode_windows
        else:
            collate_fn = None

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=getattr(self.hparams, 'pin_memory', True),
            collate_fn=collate_fn,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
