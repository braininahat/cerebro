"""Challenge 1 DataModule: Convenience wrapper for HBNDataset + Challenge1Task."""

from pathlib import Path
from typing import List, Optional

from .base import BaseTaskDataModule
from .datasets import HBNDataset
from .tasks import Challenge1Task


# Default excluded subjects from startkit
DEFAULT_EXCLUDED_SUBJECTS = [
    "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
    "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
    "NDARBA381JGH"
]


class Challenge1DataModule(BaseTaskDataModule):
    """Challenge 1: Response Time Prediction from CCD Task.

    Convenience wrapper that combines HBNDataset + Challenge1Task.

    This module handles:
    1. Loading HBN CCD task data
    2. Creating stimulus-locked windows
    3. Subject-level train/val/test splits

    Args:
        data_dir: Root directory containing HBN-EEG releases
        releases: List of releases to use (e.g., ["R1", "R2", ...])
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for parallel data loading
        excluded_subjects: List of subject IDs to exclude
        shift_after_stim: Start window X seconds after stimulus (default: 0.5)
        window_len: Window length in seconds (default: 2.0)
        use_mini: Use mini dataset for fast prototyping
        val_frac: Fraction of subjects for validation
        test_frac: Fraction of subjects for test
        seed: Random seed for splits
        sfreq: Sampling frequency in Hz
        epoch_len_s: Epoch length for preprocessing in seconds
        anchor: Anchor name for windowing

    Example:
        >>> dm = Challenge1DataModule(
        ...     data_dir="data",
        ...     releases=["R1", "R2"],
        ...     batch_size=64,
        ...     use_mini=True
        ... )
        >>> dm.setup("fit")
        >>> train_loader = dm.train_dataloader()
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
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 2025,
        sfreq: int = 100,
        epoch_len_s: float = 2.0,
        anchor: str = "stimulus_anchor",
    ):
        # Use default excluded subjects if none provided
        if excluded_subjects is None:
            excluded_subjects = DEFAULT_EXCLUDED_SUBJECTS

        # Create dataset (loading + filtering)
        dataset = HBNDataset(
            data_dir=data_dir,
            releases=releases,
            tasks=["contrastChangeDetection"],  # Challenge 1 uses CCD task
            excluded_subjects=excluded_subjects,
            use_mini=use_mini,
            description_fields=["subject", "session", "run", "task", "age", "sex", "p_factor"],
            min_duration_s=window_len,
            expected_n_channels=129,
            num_workers=num_workers,
        )

        # Create task (windowing)
        task = Challenge1Task(
            window_len=window_len,
            shift_after_stim=shift_after_stim,
            sfreq=sfreq,
            epoch_len_s=epoch_len_s,
            anchor=anchor,
        )

        # Initialize base with composed dataset + task
        super().__init__(
            dataset=dataset,
            task=task,
            batch_size=batch_size,
            num_workers=num_workers,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            excluded_subjects=excluded_subjects,
        )
