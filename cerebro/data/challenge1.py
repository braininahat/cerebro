"""Challenge 1 DataModule: Response Time Prediction from CCD Task.

Extracted from notebook 04_train_challenge1.py. Implements the complete
preprocessing pipeline with caching support.

Pipeline:
1. Load CCD task data from EEGChallengeDataset
2. Preprocess: annotate trials with target, add anchors
3. Create stimulus-locked windows [stim+0.5s, stim+2.5s] → (129, 200)
4. Split at subject level (train/val/test)
5. Create DataLoaders with persistence

Key differences from Challenge 2:
- Windowing: Stimulus-locked (event-based) vs fixed-length
- Label: rt_from_stimulus (per-trial) vs externalizing (per-subject)
- Task: CCD only vs multi-task
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional

import lightning as L
import numpy as np
import torch
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import (Preprocessor, create_fixed_length_windows,
                                       create_windows_from_events, preprocess)
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (add_aux_anchors, add_extras_columns,
                                 annotate_trials_with_target,
                                 keep_only_recordings_with)
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def filter_recordings_with_valid_windows(
    concat_ds, anchor, shift_after_stim, window_len, sfreq
):
    """Filter out recordings where stimulus anchors violate window boundaries.

    Ensures all stimulus anchors have enough post-stimulus data for window extraction.
    This prevents create_windows_from_events from failing due to boundary violations.

    Args:
        concat_ds: BaseConcatDataset to filter
        anchor: Anchor event name (e.g., 'stimulus_anchor')
        shift_after_stim: Window start offset after stimulus (seconds)
        window_len: Window length (seconds)
        sfreq: Sampling frequency (Hz)

    Returns:
        BaseConcatDataset with only recordings that have valid windows
    """
    required_post_stimulus = shift_after_stim + window_len
    filtered = []
    dropped = []

    for ds in concat_ds.datasets:
        ann = ds.raw.annotations
        stimulus_mask = ann.description == anchor

        if not np.any(stimulus_mask):
            # No stimulus anchors (already filtered by keep_only_recordings_with)
            continue

        stimulus_times = ann.onset[stimulus_mask]
        recording_duration = ds.raw.times[-1]

        # Check if all stimuli have enough post-stimulus data
        latest_stimulus = stimulus_times.max()
        required_end_time = latest_stimulus + required_post_stimulus

        if required_end_time <= recording_duration:
            filtered.append(ds)
        else:
            shortfall = required_end_time - recording_duration
            dropped.append({
                'file': ds.raw.filenames[0],
                'latest_stimulus': latest_stimulus,
                'recording_duration': recording_duration,
                'required_end': required_end_time,
                'shortfall': shortfall
            })

    # Log dropped recordings
    if dropped:
        logger.warning(
            f"[yellow]Dropped {len(dropped)} recording(s) due to insufficient post-stimulus data:[/yellow]"
        )
        for info in dropped:
            logger.warning(
                f"  • {info['file']}\n"
                f"    Latest stimulus: {info['latest_stimulus']:.2f}s, "
                f"Recording ends: {info['recording_duration']:.2f}s, "
                f"Window needs: {info['required_end']:.2f}s "
                f"(shortfall: {info['shortfall']:.2f}s)"
            )

    return BaseConcatDataset(filtered)


class CroppedWindowDataset(Dataset):
    """Wraps a windows dataset with random or deterministic temporal cropping.

    Used for Challenge 2 and pre-training to apply random 2s crops from
    longer fixed-length windows during training.
    """

    def __init__(self, windows_dataset: BaseConcatDataset, crop_len: int,
                 sfreq: int = 100, mode: str = "train", data_req: str = "challenge2"):
        """
        Parameters
        ----------
        windows_dataset : BaseConcatDataset
            The input dataset of EEG windows.
        crop_len : int
            Crop length in samples (e.g., 2 * sfreq for 2-second crops).
        sfreq : int, optional
            Sampling frequency, default 100 Hz.
        mode : {"train", "val", "test"}
            Controls whether cropping is random or deterministic (center crop).
        data_req : {"challenge2", "pretrain"}
            Type of data request to determine label handling.
        """
        self.windows_dataset = windows_dataset
        self.crop_len = crop_len
        self.sfreq = sfreq
        self.mode = mode.lower()
        self.data_req = data_req
        assert self.mode in {"train", "val",
                             "test"}, f"Invalid mode: {self.mode}"

        # Get metadata for Challenge 2 (needs externalizing labels)
        if self.data_req == "challenge2":
            self.meta = self.windows_dataset.get_metadata().reset_index(drop=True)
            if "externalizing" not in self.meta.columns:
                raise KeyError(
                    f"Target column 'externalizing' not found in windows metadata. "
                    f"Available columns: {list(self.meta.columns)}"
                )

    def __len__(self):
        return len(self.windows_dataset.datasets)

    def __getitem__(self, idx):
        ds = self.windows_dataset.datasets[idx]
        x, y, _ = ds[0]  # (C, T)
        C, T = x.shape

        # Apply temporal cropping
        if T > self.crop_len:
            if self.mode == "train":
                # Random crop during training
                start = np.random.randint(0, T - self.crop_len + 1)
            else:
                # Deterministic crop (center) for val/test
                start = (T - self.crop_len) // 2
            x_crop = x[:, start:start + self.crop_len]
        else:
            # Pad if too short
            x_crop = x[:, :self.crop_len]
            if x_crop.shape[1] < self.crop_len:
                pad = self.crop_len - x_crop.shape[1]
                x_crop = np.pad(x_crop, ((0, 0), (0, pad)), mode="constant")

        # Handle labels based on data_req
        if self.data_req == "challenge2":
            # Pull externalizing from metadata (per-subject label)
            y_val = self.meta.iloc[idx]["externalizing"]
            y = float(np.asarray(y_val))
        elif self.data_req == "pretrain":
            # Pre-training doesn't need labels (self-supervised)
            y = 0.0  # Dummy label

        return torch.from_numpy(x_crop).float(), torch.tensor(y).float()


class Challenge1DataModule(L.LightningDataModule):
    """DataModule for Challenge 1, Challenge 2, and pre-training.

    Supports three data requirements (data_req):
    - "challenge1": CCD task, stimulus-locked windows, RT prediction
    - "challenge2": Multi-task, fixed windows + random crops, externalizing prediction
    - "pretrain": Passive tasks (movies), fixed windows + random crops, self-supervised

    Supports two modes:
    - "dev": Development mode with train/val/test splits from training releases
    - "submission": Final submission mode (train on all subjects, test on R5)

    Args:
        data_dir: Root directory containing HBN-EEG releases
        releases: List of releases to use (e.g., ["R1", "R2", ...])
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for parallel data loading
        excluded_subjects: List of subject IDs to exclude
        shift_after_stim: Start window X seconds after stimulus (Challenge 1 only, default: 0.5)
        window_len: Window length in seconds (Challenge 1 only, default: 2.0)
        use_mini: Use mini dataset for fast prototyping (default: False)
        cache_dir: Directory for caching preprocessed windows (default: None = data_dir/../cache)
        val_frac: Fraction of subjects for validation (default: 0.1)
        test_frac: Fraction of subjects for test (default: 0.1)
        seed: Random seed for splits (default: 2025)
        sfreq: Sampling frequency in Hz (default: 100)
        epoch_len_s: Epoch length for preprocessing in seconds (default: 2.0)
        anchor: Anchor name for windowing (Challenge 1 only, default: "stimulus_anchor")
        mode: "dev" (split training releases) or "submission" (train on all, test on R5)
        test_on_r5: Use R5 as test set (matches local_scoring.py behavior)
        r5_data_dir: Directory for R5 data (default: same as data_dir)
        data_req: "challenge1", "challenge2", or "pretrain"
        passive_tasks: List of passive tasks for pre-training (default: None)
        active_task: Active task name for Challenge 1 (default: "contrastChangeDetection")
        window_size_s: Fixed window size for Challenge 2/pretrain (default: 4.0)
        window_stride_s: Window stride for Challenge 2/pretrain (default: 2.0)

    Example (dev mode with local test):
        >>> datamodule = Challenge1DataModule(
        ...     data_dir="data/full",
        ...     releases=["R1", "R2", "R3"],
        ...     mode="dev",
        ...     test_on_r5=False,
        ... )

    Example (dev mode with R5 test):
        >>> datamodule = Challenge1DataModule(
        ...     data_dir="data/full",
        ...     releases=["R1", "R2", "R3"],
        ...     mode="dev",
        ...     test_on_r5=True,
        ... )

    Example (submission mode):
        >>> datamodule = Challenge1DataModule(
        ...     data_dir="data/full",
        ...     releases=["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"],
        ...     mode="submission",
        ...     test_on_r5=True,  # Required for submission mode
        ... )
    """

    def __init__(
        self,
        data_dir: str,
        releases: List[str],
        batch_size: int = 512,
        num_workers: int = 8,
        excluded_subjects: Optional[List[str]] = None,
        shift_after_stim: float = 0.5,
        window_len: float = 2.0,
        use_mini: bool = False,
        cache_dir: Optional[str] = None,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 2025,
        sfreq: int = 100,
        epoch_len_s: float = 2.0,
        anchor: str = "stimulus_anchor",
        mode: str = "dev",
        test_on_r5: bool = False,
        r5_data_dir: Optional[str] = None,
        data_req: str = "challenge1",
        passive_tasks: Optional[List[str]] = None,
        active_task: str = "contrastChangeDetection",
        window_size_s: float = 4.0,
        window_stride_s: float = 2.0,
        n_channels: int = 129,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_req = data_req

        # Validate data_req
        if data_req not in ["challenge1", "challenge2", "pretrain"]:
            raise ValueError(
                f"data_req must be 'challenge1', 'challenge2', or 'pretrain', got '{data_req}'"
            )

        # Validate mode
        if mode not in ["dev", "submission"]:
            raise ValueError(
                f"mode must be 'dev' or 'submission', got '{mode}'"
            )

        # Validate passive_tasks for pretrain
        if data_req == "pretrain" and (passive_tasks is None or len(passive_tasks) == 0):
            raise ValueError(
                "data_req='pretrain' requires passive_tasks to be specified"
            )

        # R5 guard: Prevent training on competition validation set
        if "R5" in releases:
            raise ValueError(
                "R5 is the COMPETITION VALIDATION SET and must NEVER be used for training! "
                f"Got releases={releases}. Use releases from [R1, R2, R3, R4, R6, R7, R8, R9, R10, R11] only."
            )

        # Submission mode validation
        if mode == "submission" and not test_on_r5:
            raise ValueError(
                "submission mode requires test_on_r5=True (final submission must be evaluated on R5)"
            )

        # Convert to Path objects
        self.data_dir = Path(data_dir)
        self.r5_data_dir = Path(
            r5_data_dir) if r5_data_dir is not None else self.data_dir

        # Setup cache directory based on data_req
        if cache_dir is None:
            cache_subdir = {
                "challenge1": "challenge1",
                "challenge2": "challenge2",
                "pretrain": "pretrain"
            }[data_req]
            self.cache_dir = self.data_dir.parent / "cache" / cache_subdir
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # R5 cache directory (separate to avoid accidental mixing)
        if test_on_r5:
            self.r5_cache_dir = self.cache_dir / "r5"
            self.r5_cache_dir.mkdir(parents=True, exist_ok=True)

        # Default excluded subjects from startkit
        if excluded_subjects is None:
            self.hparams.excluded_subjects = [
                "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
                "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
                "NDARBA381JGH"
            ]

        # Will be populated in setup()
        self.train_set = None
        self.val_set = None
        self.test_set = None

    @property
    def batch_size(self) -> int:
        """Batch size for DataLoaders.

        Property accessor for Lightning's batch size scaler compatibility.
        The actual value is stored in self.hparams.batch_size.
        """
        return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        """Update batch size (modified by Lightning's batch size scaler)."""
        self.hparams.batch_size = value

    def setup(self, stage: Optional[str] = None):
        """Setup datasets with caching support.

        Loads data, preprocesses, creates windows, and splits at subject level.
        Uses pickle cache to skip preprocessing on repeated runs.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict' (unused, loads all)
        """
        logger.info("\n" + "="*60)
        logger.info(
            f"[bold cyan]CHALLENGE 1 DATA SETUP (mode={self.hparams.mode}, test_on_r5={self.hparams.test_on_r5})[/bold cyan]")
        logger.info("="*60)

        # Create cache key from windowing parameters AND data_req
        releases_str = "_".join(self.hparams.releases)
        windowing_params = f"shift{int(self.hparams.shift_after_stim*10)}_len{int(self.hparams.window_len*10)}"
        cache_key = f"windows_{self.data_req}_{releases_str}_{windowing_params}_mini{self.hparams.use_mini}.pkl"
        cache_path = self.cache_dir / cache_key

        # Try loading from cache
        if cache_path.exists():
            logger.info("[bold cyan]LOADING FROM CACHE[/bold cyan]")
            logger.info(
                f"[green]✓[/green] Loading cached windows from: {cache_key}")
            with open(cache_path, "rb") as f:
                single_windows = pickle.load(f)
            logger.info(
                f"[green]✓[/green] Loaded {len(single_windows)} windows from cache")
        else:
            # Cache miss - run full pipeline
            single_windows = self._load_and_preprocess(cache_path)

        # Inspect metadata
        metadata = single_windows.get_metadata()
        logger.info(
            f"\n[bold]Metadata columns:[/bold] {len(list(metadata.columns))} columns")
        logger.info(f"\n[bold]Sample metadata:[/bold]")
        logger.info(metadata.head().to_string())

        # RT distribution statistics
        logger.info(
            f"\n[bold]Response Time Statistics (training releases):[/bold]")
        logger.info(f"  Mean: {metadata['target'].mean():.4f}s")
        logger.info(f"  Std: {metadata['target'].std():.4f}s")
        logger.info(f"  Min: {metadata['target'].min():.4f}s")
        logger.info(f"  Max: {metadata['target'].max():.4f}s")

        # Split data at subject level
        self._create_splits(single_windows, metadata)

        # Load R5 test set if requested
        if self.hparams.test_on_r5:
            logger.info("\n" + "="*60)
            logger.info("[bold yellow]LOADING R5 TEST SET[/bold yellow]")
            logger.info("="*60)
            self.test_set = self._load_r5_test()

    def _load_and_preprocess(self, cache_path: Path):
        """Load raw data, preprocess, create windows, and cache."""
        logger.info("[bold cyan]LOADING DATA[/bold cyan]")

        # Determine tasks to load based on data_req
        if self.data_req == "pretrain":
            tasks = self.hparams.passive_tasks
            description_fields = ["subject",
                                  "session", "run", "task", "age", "sex"]
        elif self.data_req == "challenge1":
            tasks = [self.hparams.active_task]
            description_fields = ["subject", "session",
                                  "run", "task", "age", "sex", "p_factor"]
        elif self.data_req == "challenge2":
            # Challenge 2 can use any task, but we need externalizing labels
            tasks = [self.hparams.active_task]
            description_fields = ["subject", "session", "run",
                                  "task", "age", "sex", "p_factor", "externalizing"]
        else:
            raise ValueError(f"Unknown data_req: {self.data_req}")

        # Load all releases
        all_datasets_list = []
        total_combinations = len(self.hparams.releases) * len(tasks)

        with tqdm(total=total_combinations, desc="Loading datasets", unit="dataset") as pbar:
            for release in self.hparams.releases:
                for task in tasks:
                    try:
                        pbar.set_postfix_str(f"{release}/{task}")
                        dataset = EEGChallengeDataset(
                            task=task,
                            release=release,
                            cache_dir=str(self.data_dir),
                            mini=self.hparams.use_mini,
                            description_fields=description_fields,
                        )
                        all_datasets_list.append(dataset)
                        pbar.set_postfix_str(
                            f"{release}/{task} ✓ {len(dataset.datasets)} recordings")
                    except Exception as e:
                        pbar.set_postfix_str(f"{release}/{task} ✗ Failed: {e}")
                    finally:
                        pbar.update(1)

        # Combine datasets
        all_datasets = BaseConcatDataset(all_datasets_list)
        logger.info(
            f"[bold]Total recordings:[/bold] {len(all_datasets.datasets)}")

        # Preload raws in parallel
        logger.info("Preloading raw data...")
        raws = Parallel(n_jobs=os.cpu_count())(
            delayed(lambda d: d.raw)(d) for d in all_datasets.datasets
        )
        logger.info("[green]Done loading raw data[/green]")

        # Apply quality filters
        logger.info("\n[bold cyan]FILTERING DATA[/bold cyan]")
        filtered_datasets = self._filter_datasets(all_datasets)

        # Preprocess and create windows based on data_req
        logger.info("\n[bold cyan]PREPROCESSING AND WINDOWING[/bold cyan]")
        if self.data_req == "challenge1":
            single_windows = self._create_challenge1_windows(filtered_datasets)
        elif self.data_req == "challenge2":
            single_windows = self._create_challenge2_windows(filtered_datasets)
        elif self.data_req == "pretrain":
            single_windows = self._create_pretrain_windows(filtered_datasets)
        else:
            raise ValueError(f"Unknown data_req: {self.data_req}")

        # Save to cache
        logger.info(
            f"[yellow]⚠[/yellow] Caching windows to: {cache_path.name}")
        with open(cache_path, "wb") as f:
            pickle.dump(single_windows, f)
        logger.info(
            f"[yellow]⚠[/yellow] Delete cache if windowing params change: rm {cache_path}")

        return single_windows

    def _filter_datasets(self, datasets: BaseConcatDataset) -> BaseConcatDataset:
        """Apply quality filters to datasets with detailed logging."""
        logger.info(f"Filtering {len(datasets.datasets)} datasets...")

        dropped = {
            'excluded_subject': [],
            'nan_externalizing': [],
            'duration_too_short': [],
            'wrong_channel_count': [],
            'raw_load_failed': [],
            'other_error': []
        }

        def check_dataset(ds):
            try:
                # Get file identifier for logging
                file_id = ds.raw.filenames[0] if hasattr(
                    ds.raw, 'filenames') else ds.description.get('subject', 'unknown')

                # Subject exclusion
                subj = ds.description.get('subject', '')
                if subj in self.hparams.excluded_subjects:
                    dropped['excluded_subject'].append({
                        'file': file_id,
                        'subject': subj
                    })
                    return None

                # For Challenge 2, filter out subjects with NaN externalizing
                if self.data_req == "challenge2":
                    import math
                    ext_val = ds.description.get("externalizing", float('nan'))
                    if math.isnan(ext_val):
                        dropped['nan_externalizing'].append({
                            'file': file_id,
                            'subject': subj
                        })
                        return None

                # Access raw metadata
                raw = ds.raw
                if raw is None:
                    dropped['raw_load_failed'].append({
                        'file': file_id,
                        'subject': subj
                    })
                    return None

                # Check duration
                dur = raw.times[-1]
                if dur < self.hparams.window_size_s:
                    dropped['duration_too_short'].append({
                        'file': file_id,
                        'subject': subj,
                        'duration': dur,
                        'required': self.hparams.window_size_s,
                        'shortfall': self.hparams.window_size_s - dur
                    })
                    return None

                # Check channels
                n_chans = len(raw.ch_names)
                if n_chans != self.hparams.n_channels:
                    dropped['wrong_channel_count'].append({
                        'file': file_id,
                        'subject': subj,
                        'channels': n_chans,
                        'expected': self.hparams.n_channels
                    })
                    return None

                return ds
            except Exception as e:
                dropped['other_error'].append({
                    'file': file_id if 'file_id' in locals() else 'unknown',
                    'error': str(e)
                })
                return None

        # Use threading for I/O-bound metadata checks
        logger.info("  Checking durations and channels...")
        filtered = Parallel(n_jobs=-1, backend='threading')(
            delayed(check_dataset)(ds) for ds in tqdm(
                datasets.datasets,
                desc="  Filtering datasets",
                unit="recording",
                ncols=80
            )
        )

        filtered = [ds for ds in filtered if ds is not None]
        filtered_datasets = BaseConcatDataset(filtered)

        # Log detailed filtering statistics
        total_dropped = sum(len(v) for v in dropped.values())
        if total_dropped > 0:
            logger.warning(
                f"[yellow]Dropped {total_dropped} recording(s) due to quality filters:[/yellow]"
            )

            if dropped['excluded_subject']:
                logger.warning(
                    f"  • Excluded subjects: {len(dropped['excluded_subject'])}")
                for info in dropped['excluded_subject'][:5]:  # Show first 5
                    logger.warning(f"    - Subject {info['subject']}")
                if len(dropped['excluded_subject']) > 5:
                    logger.warning(
                        f"    ... and {len(dropped['excluded_subject']) - 5} more")

            if dropped['nan_externalizing']:
                logger.warning(
                    f"  • NaN externalizing (Challenge 2): {len(dropped['nan_externalizing'])}")
                for info in dropped['nan_externalizing'][:3]:
                    logger.warning(f"    - Subject {info['subject']}")
                if len(dropped['nan_externalizing']) > 3:
                    logger.warning(
                        f"    ... and {len(dropped['nan_externalizing']) - 3} more")

            if dropped['duration_too_short']:
                logger.warning(
                    f"  • Duration too short: {len(dropped['duration_too_short'])}")
                for info in dropped['duration_too_short'][:3]:
                    logger.warning(
                        f"    - {info['file']}: {info['duration']:.2f}s < {info['required']:.2f}s "
                        f"(shortfall: {info['shortfall']:.2f}s)"
                    )
                if len(dropped['duration_too_short']) > 3:
                    logger.warning(
                        f"    ... and {len(dropped['duration_too_short']) - 3} more")

            if dropped['wrong_channel_count']:
                logger.warning(
                    f"  • Wrong channel count: {len(dropped['wrong_channel_count'])}")
                for info in dropped['wrong_channel_count'][:3]:
                    logger.warning(
                        f"    - {info['file']}: {info['channels']} channels (expected {info['expected']})"
                    )
                if len(dropped['wrong_channel_count']) > 3:
                    logger.warning(
                        f"    ... and {len(dropped['wrong_channel_count']) - 3} more")

            if dropped['raw_load_failed']:
                logger.warning(
                    f"  • Raw load failed: {len(dropped['raw_load_failed'])}")

            if dropped['other_error']:
                logger.warning(
                    f"  • Other errors: {len(dropped['other_error'])}")
                for info in dropped['other_error'][:3]:
                    logger.warning(f"    - {info['file']}: {info['error']}")
                if len(dropped['other_error']) > 3:
                    logger.warning(
                        f"    ... and {len(dropped['other_error']) - 3} more")

        logger.info(
            f"\n[green]✓ Kept {len(filtered_datasets.datasets)}/{len(datasets.datasets)} recordings[/green]")

        return filtered_datasets

    def _create_challenge1_windows(self, datasets: BaseConcatDataset) -> BaseConcatDataset:
        """Create stimulus-locked windows for Challenge 1.

        Note: datasets have already been filtered by _filter_datasets for general
        quality checks (duration, channels, excluded subjects).
        """
        logger.info("Creating Challenge 1 stimulus-locked windows...")

        # Preprocess: Annotate trials with target RT and add anchors
        transformation_offline = [
            Preprocessor(
                annotate_trials_with_target,
                target_field="rt_from_stimulus",
                epoch_length=self.hparams.epoch_len_s,
                require_stimulus=True,
                require_response=True,
                apply_on_array=False,
            ),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ]

        logger.info("Annotating trials with target RT and adding anchors...")
        preprocess(datasets, transformation_offline, n_jobs=-1)

        # Create stimulus-locked windows
        logger.info(
            f"Window: [{self.hparams.shift_after_stim}s, "
            f"{self.hparams.shift_after_stim + self.hparams.window_len}s] relative to stimulus"
        )
        logger.info(
            f"Window length: {self.hparams.window_len}s "
            f"({int(self.hparams.window_len * self.hparams.sfreq)} samples)"
        )

        single_windows = create_windows_from_events(
            datasets,
            mapping={self.hparams.anchor: 0},
            trial_start_offset_samples=int(
                self.hparams.shift_after_stim * self.hparams.sfreq),
            trial_stop_offset_samples=int(
                (self.hparams.shift_after_stim +
                 self.hparams.window_len) * self.hparams.sfreq
            ),
            window_size_samples=int(
                self.hparams.epoch_len_s * self.hparams.sfreq),
            window_stride_samples=self.hparams.sfreq,
            preload=True,
        )

        logger.info(f"[bold]Created {len(single_windows)} windows[/bold]")

        # Inject metadata
        single_windows = add_extras_columns(
            single_windows,
            datasets,
            desc=self.hparams.anchor,
            keys=("target", "rt_from_stimulus", "rt_from_trialstart",
                  "stimulus_onset", "response_onset", "correct", "response_type")
        )

        return single_windows

    def _create_challenge2_windows(self, datasets: BaseConcatDataset) -> BaseConcatDataset:
        """Create fixed-length windows for Challenge 2."""
        logger.info("Creating Challenge 2 fixed-length windows...")
        logger.info(
            f"Window size: {self.hparams.window_size_s}s "
            f"({int(self.hparams.window_size_s * self.hparams.sfreq)} samples)"
        )
        logger.info(
            f"Window stride: {self.hparams.window_stride_s}s "
            f"({int(self.hparams.window_stride_s * self.hparams.sfreq)} samples)"
        )

        windows = create_fixed_length_windows(
            datasets,
            window_size_samples=int(
                self.hparams.window_size_s * self.hparams.sfreq),
            window_stride_samples=int(
                self.hparams.window_stride_s * self.hparams.sfreq),
            drop_last_window=True,
            preload=True
        )

        logger.info(f"[bold]Created {len(windows)} windows[/bold]")
        return windows

    def _create_pretrain_windows(self, datasets: BaseConcatDataset) -> BaseConcatDataset:
        """Create fixed-length windows for pre-training."""
        logger.info("Creating pre-training fixed-length windows...")
        logger.info(
            f"Window size: {self.hparams.window_size_s}s "
            f"({int(self.hparams.window_size_s * self.hparams.sfreq)} samples)"
        )
        logger.info(
            f"Window stride: {self.hparams.window_stride_s}s "
            f"({int(self.hparams.window_stride_s * self.hparams.sfreq)} samples)"
        )

        windows = create_fixed_length_windows(
            datasets,
            window_size_samples=int(
                self.hparams.window_size_s * self.hparams.sfreq),
            window_stride_samples=int(
                self.hparams.window_stride_s * self.hparams.sfreq),
            drop_last_window=True,
            preload=True
        )

        logger.info(f"[bold]Created {len(windows)} windows[/bold]")
        return windows

    def _create_splits(self, single_windows, metadata):
        """Create subject-level train/val/test splits.

        In "dev" mode: Split into train/val/test
        In "submission" mode: Use all subjects for training, no val/test
        """
        logger.info("\n[bold cyan]SPLITTING DATA[/bold cyan]")

        subjects = metadata["subject"].unique()
        subjects = [
            s for s in subjects if s not in self.hparams.excluded_subjects]
        logger.info(
            f"[bold]Total subjects (after exclusion):[/bold] {len(subjects)}")

        if self.hparams.mode == "submission":
            # Submission mode: Train on ALL subjects (no val/test split from training releases)
            logger.info(
                "[yellow]Submission mode: Using ALL subjects for training[/yellow]")
            train_subj = subjects
            valid_subj = []
            test_subj = []
        else:
            # Dev mode: Split train/val/test
            # Split: train / (val + test)
            train_subj, valid_test_subj = train_test_split(
                subjects,
                test_size=(self.hparams.val_frac + self.hparams.test_frac),
                random_state=check_random_state(self.hparams.seed),
                shuffle=True
            )

            # Split: val / test
            valid_subj, test_subj = train_test_split(
                valid_test_subj,
                test_size=self.hparams.test_frac /
                (self.hparams.val_frac + self.hparams.test_frac),
                random_state=check_random_state(self.hparams.seed + 1),
                shuffle=True
            )

            # Sanity check
            assert (set(valid_subj) | set(test_subj) |
                    set(train_subj)) == set(subjects)

        logger.info(f"Train subjects: {len(train_subj)}")
        logger.info(f"Val subjects: {len(valid_subj)}")
        logger.info(
            f"Test subjects: {len(test_subj)} {'(will be replaced by R5)' if self.hparams.test_on_r5 else ''}")

        # Create splits
        subject_split = single_windows.split("subject")
        self.train_set = BaseConcatDataset(
            [subject_split[s] for s in train_subj if s in subject_split]
        )

        if self.hparams.mode == "submission":
            # Submission mode: No val/test from training releases (will use R5 for test)
            self.val_set = None
            self.test_set = None
        else:
            # Dev mode: Create val and test sets
            self.val_set = BaseConcatDataset(
                [subject_split[s] for s in valid_subj if s in subject_split]
            )
            if not self.hparams.test_on_r5:
                # Only create local test set if not using R5
                self.test_set = BaseConcatDataset(
                    [subject_split[s] for s in test_subj if s in subject_split]
                )

        logger.info(f"\n[bold]Window counts:[/bold]")
        logger.info(f"  Train: {len(self.train_set)}")
        if self.val_set is not None:
            logger.info(f"  Val: {len(self.val_set)}")
        if self.test_set is not None and not self.hparams.test_on_r5:
            logger.info(f"  Test: {len(self.test_set)}")

    def _load_r5_test(self):
        """Load R5 as test set, matching local_scoring.py preprocessing.

        This method replicates the exact preprocessing pipeline from
        startkit/local_scoring.py (lines 136-195) to ensure test evaluation
        matches the competition scoring environment.

        Returns:
            BaseConcatDataset: R5 windows ready for testing
        """
        # R5 cache key (separate from training releases, includes data_req)
        windowing_params = f"shift{int(self.hparams.shift_after_stim*10)}_len{int(self.hparams.window_len*10)}"
        r5_cache_key = f"windows_{self.data_req}_R5_{windowing_params}_mini{self.hparams.use_mini}.pkl"
        r5_cache_path = self.r5_cache_dir / r5_cache_key

        # Try loading from cache
        if r5_cache_path.exists():
            logger.info(
                f"[green]✓[/green] Loading R5 from cache: {r5_cache_key}")
            with open(r5_cache_path, "rb") as f:
                r5_windows = pickle.load(f)
            logger.info(
                f"[green]✓[/green] Loaded {len(r5_windows)} R5 windows from cache")
        else:
            # Cache miss - load R5 with identical preprocessing to local_scoring.py
            logger.info(
                "[yellow]Loading R5 (competition validation set)[/yellow]")
            logger.info(
                "[yellow]This matches local_scoring.py preprocessing exactly[/yellow]")

            # Load R5 dataset (identical to local_scoring.py lines 136-144)
            dataset_r5 = EEGChallengeDataset(
                release="R5",
                mini=self.hparams.use_mini,
                query=dict(task="contrastChangeDetection"),
                cache_dir=str(self.r5_data_dir),
                description_fields=[
                    "subject", "session", "run", "task", "age", "sex", "p_factor",
                ],
            )

            # Preload raws
            logger.info("Preloading R5 raw data...")
            raws = Parallel(n_jobs=os.cpu_count())(
                delayed(lambda d: d.raw)(d) for d in dataset_r5.datasets
            )
            logger.info("[green]Done loading R5 raw data[/green]")

            # Preprocess (identical to local_scoring.py lines 147-158)
            logger.info("Preprocessing R5...")
            preprocessors = [
                Preprocessor(
                    annotate_trials_with_target,
                    apply_on_array=False,
                    target_field="rt_from_stimulus",
                    epoch_length=self.hparams.epoch_len_s,
                    require_stimulus=True,
                    require_response=True,
                ),
                Preprocessor(add_aux_anchors, apply_on_array=False),
            ]
            preprocess(dataset_r5, preprocessors, n_jobs=-1)

            # Keep only recordings with stimulus anchors (line 166)
            dataset_r5 = keep_only_recordings_with(
                self.hparams.anchor, dataset_r5)
            logger.info(
                f"[bold]R5 recordings with stimulus anchors:[/bold] {len(dataset_r5.datasets)}")

            # Filter out recordings where stimulus anchors violate window boundaries
            dataset_r5 = filter_recordings_with_valid_windows(
                dataset_r5,
                self.hparams.anchor,
                self.hparams.shift_after_stim,
                self.hparams.window_len,
                self.hparams.sfreq,
            )
            logger.info(
                f"[bold]R5 recordings with valid windows:[/bold] {len(dataset_r5.datasets)}")

            # Create windows (identical to local_scoring.py lines 169-179)
            logger.info("Creating R5 windows...")
            r5_windows = create_windows_from_events(
                dataset_r5,
                mapping={self.hparams.anchor: 0},
                trial_start_offset_samples=int(
                    self.hparams.shift_after_stim * self.hparams.sfreq),
                trial_stop_offset_samples=int(
                    (self.hparams.shift_after_stim +
                     self.hparams.window_len) * self.hparams.sfreq
                ),
                window_size_samples=int(
                    self.hparams.epoch_len_s * self.hparams.sfreq),
                window_stride_samples=self.hparams.sfreq,
                preload=True,
            )

            # Add metadata (identical to local_scoring.py lines 182-195)
            r5_windows = add_extras_columns(
                r5_windows,
                dataset_r5,
                desc=self.hparams.anchor,
                keys=(
                    "target",
                    "rt_from_stimulus",
                    "rt_from_trialstart",
                    "stimulus_onset",
                    "response_onset",
                    "correct",
                    "response_type",
                ),
            )

            logger.info(f"[bold]Created {len(r5_windows)} R5 windows[/bold]")

            # Cache R5 windows
            logger.info(
                f"[yellow]⚠[/yellow] Caching R5 windows to: {r5_cache_key}")
            with open(r5_cache_path, "wb") as f:
                pickle.dump(r5_windows, f)

        # Display R5 statistics
        r5_metadata = r5_windows.get_metadata()
        logger.info(f"\n[bold]R5 Test Set Statistics:[/bold]")
        logger.info(f"  Windows: {len(r5_windows)}")
        logger.info(f"  Subjects: {len(r5_metadata['subject'].unique())}")
        logger.info(f"  Mean RT: {r5_metadata['target'].mean():.4f}s")
        logger.info(f"  Std RT: {r5_metadata['target'].std():.4f}s")

        return r5_windows

    def train_dataloader(self):
        """Create training DataLoader with persistence."""
        # Wrap with CroppedWindowDataset for Challenge 2 and pre-training
        if self.data_req in ["challenge2", "pretrain"]:
            crop_len = int(self.hparams.epoch_len_s * self.hparams.sfreq)
            dataset = CroppedWindowDataset(
                self.train_set,
                crop_len=crop_len,
                sfreq=self.hparams.sfreq,
                mode="train",
                data_req=self.data_req
            )
        else:
            dataset = self.train_set

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation DataLoader with persistence.

        Returns None in submission mode (no validation split).
        """
        if self.val_set is None:
            return None

        # Wrap with CroppedWindowDataset for Challenge 2 and pre-training
        if self.data_req in ["challenge2", "pretrain"]:
            crop_len = int(self.hparams.epoch_len_s * self.hparams.sfreq)
            dataset = CroppedWindowDataset(
                self.val_set,
                crop_len=crop_len,
                sfreq=self.hparams.sfreq,
                mode="val",
                data_req=self.data_req
            )
        else:
            dataset = self.val_set

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test DataLoader.

        Uses R5 if test_on_r5=True, otherwise uses local test split.
        """
        if self.test_set is None:
            raise RuntimeError(
                "test_set is None. Did you call setup()? "
                "In submission mode, test_on_r5 must be True."
            )

        # Wrap with CroppedWindowDataset for Challenge 2 and pre-training
        if self.data_req in ["challenge2", "pretrain"]:
            crop_len = int(self.hparams.epoch_len_s * self.hparams.sfreq)
            dataset = CroppedWindowDataset(
                self.test_set,
                crop_len=crop_len,
                sfreq=self.hparams.sfreq,
                mode="test",
                data_req=self.data_req
            )
        else:
            dataset = self.test_set

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
