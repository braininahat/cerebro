"""HBN Dataset: Loading and filtering (no windowing)."""

import logging
import os
from pathlib import Path
from typing import List, Optional

from braindecode.datasets import BaseConcatDataset
from eegdash.dataset import EEGChallengeDataset
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HBNDataset:
    """Load and filter HBN recordings without windowing.

    This is Layer 1 of the composable data architecture:
    - Loads EEGChallengeDataset from specified releases
    - Applies quality filters (duration, channels, excluded subjects)
    - Returns filtered recordings (NO windowing - that's task layer)

    Args:
        data_dir: Root directory containing HBN-EEG releases
        releases: List of release names (e.g., ["R1", "R2", ...])
        tasks: List of task names to load (e.g., ["contrastChangeDetection"])
        excluded_subjects: List of subject IDs to exclude
        use_mini: Use mini dataset for fast prototyping
        description_fields: Fields to load from participants.tsv
        min_duration_s: Minimum recording duration in seconds
        expected_n_channels: Expected number of channels
        num_workers: Number of parallel workers for filtering

    Example:
        >>> dataset = HBNDataset(
        ...     data_dir="data",
        ...     releases=["R1", "R2"],
        ...     tasks=["contrastChangeDetection"],
        ...     use_mini=True
        ... )
        >>> recordings = dataset.load()
        >>> # recordings is BaseConcatDataset of filtered Raw objects
    """

    def __init__(
        self,
        data_dir: str,
        releases: List[str],
        tasks: List[str],
        excluded_subjects: Optional[List[str]] = None,
        use_mini: bool = False,
        description_fields: Optional[List[str]] = None,
        min_duration_s: float = 2.0,
        expected_n_channels: int = 129,
        num_workers: int = 8,
    ):
        self.data_dir = Path(data_dir)
        self.releases = releases
        self.tasks = tasks
        self.excluded_subjects = excluded_subjects or []
        self.use_mini = use_mini
        self.description_fields = description_fields or ["subject", "session", "run", "task"]
        self.min_duration_s = min_duration_s
        self.expected_n_channels = expected_n_channels
        self.num_workers = num_workers

        # R5 guard: Prevent training on competition validation set
        if "R5" in releases:
            raise ValueError(
                "R5 is the COMPETITION VALIDATION SET and must NEVER be used for training! "
                f"Got releases={releases}. Use releases from [R1, R2, R3, R4, R6, R7, R8, R9, R10, R11] only."
            )

    def load(self) -> BaseConcatDataset:
        """Load and filter HBN recordings.

        Returns:
            BaseConcatDataset of filtered recordings (no windowing applied)
        """
        logger.info("[bold cyan]LOADING HBN DATA[/bold cyan]")

        # Load all release/task combinations
        all_datasets_list = []
        total_combinations = len(self.releases) * len(self.tasks)

        with tqdm(total=total_combinations, desc="Loading datasets", unit="dataset") as pbar:
            for release in self.releases:
                for task in self.tasks:
                    try:
                        pbar.set_postfix_str(f"{release}/{task}")
                        dataset = EEGChallengeDataset(
                            task=task,
                            release=release,
                            cache_dir=str(self.data_dir),
                            mini=self.use_mini,
                            description_fields=self.description_fields,
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
        logger.info(f"[bold]Total recordings:[/bold] {len(all_datasets.datasets)}")

        # Preload raws in parallel
        logger.info("Preloading raw data...")
        raws = Parallel(n_jobs=os.cpu_count())(
            delayed(lambda d: d.raw)(d) for d in all_datasets.datasets
        )
        logger.info("[green]Done loading raw data[/green]")

        # Apply quality filters
        logger.info("\n[bold cyan]FILTERING DATA[/bold cyan]")
        filtered_datasets = self._filter_datasets(all_datasets)

        return filtered_datasets

    def _filter_datasets(self, datasets: BaseConcatDataset) -> BaseConcatDataset:
        """Apply quality filters to datasets with detailed logging."""
        logger.info(f"Filtering {len(datasets.datasets)} datasets...")

        dropped = {
            'excluded_subject': [],
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
                if subj in self.excluded_subjects:
                    dropped['excluded_subject'].append({
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
                if dur < self.min_duration_s:
                    dropped['duration_too_short'].append({
                        'file': file_id,
                        'subject': subj,
                        'duration': dur,
                        'required': self.min_duration_s,
                        'shortfall': self.min_duration_s - dur
                    })
                    return None

                # Check channels
                n_chans = len(raw.ch_names)
                if n_chans != self.expected_n_channels:
                    dropped['wrong_channel_count'].append({
                        'file': file_id,
                        'subject': subj,
                        'channels': n_chans,
                        'expected': self.expected_n_channels
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
        filtered = Parallel(n_jobs=self.num_workers, backend='threading')(
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
