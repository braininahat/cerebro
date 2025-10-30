"""Challenge 2 Task: Fixed windows with random crops for externalizing prediction."""

import logging

from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows

logger = logging.getLogger(__name__)


class Challenge2Task:
    """Create fixed-length windows for Challenge 2 (externalizing prediction).

    This is Layer 2 of the composable data architecture:
    - Takes filtered recordings from Dataset layer
    - Creates fixed-length windows with stride
    - Returns WindowsDataset ready for training

    Note: Random cropping is handled by CroppedWindowDataset wrapper in DataModule.

    Args:
        window_size_s: Window size in seconds (default: 4.0)
        window_stride_s: Window stride in seconds (default: 2.0)
        sfreq: Sampling frequency in Hz (default: 100)

    Example:
        >>> task = Challenge2Task(window_size_s=4.0, window_stride_s=2.0)
        >>> windows = task.create_windows(filtered_recordings)
        >>> # windows is BaseConcatDataset of fixed-length windows
    """

    def __init__(
        self,
        window_size_s: float = 4.0,
        window_stride_s: float = 2.0,
        sfreq: int = 100,
    ):
        self.window_size_s = window_size_s
        self.window_stride_s = window_stride_s
        self.sfreq = sfreq

    def create_windows(self, recordings: BaseConcatDataset) -> BaseConcatDataset:
        """Create fixed-length windows from filtered recordings.

        Args:
            recordings: BaseConcatDataset of filtered recordings from HBNDataset

        Returns:
            BaseConcatDataset of windowed data
        """
        logger.info("[bold cyan]CHALLENGE 2 WINDOWING[/bold cyan]")
        logger.info("Creating fixed-length windows...")
        logger.info(
            f"Window size: {self.window_size_s}s "
            f"({int(self.window_size_s * self.sfreq)} samples)"
        )
        logger.info(
            f"Window stride: {self.window_stride_s}s "
            f"({int(self.window_stride_s * self.sfreq)} samples)"
        )

        windows = create_fixed_length_windows(
            recordings,
            window_size_samples=int(self.window_size_s * self.sfreq),
            window_stride_samples=int(self.window_stride_s * self.sfreq),
            drop_last_window=True,
            preload=True
        )

        logger.info(f"[bold]Created {len(windows)} windows[/bold]")

        # Add subject-level targets to windows metadata
        # For Challenge 2, target is the externalizing score from subject metadata
        self._add_subject_targets(windows)

        # Filter out subjects with NaN externalizing scores
        windows = self._filter_nan_targets(windows)

        return windows

    def _add_subject_targets(self, windows: BaseConcatDataset):
        """Add subject-level externalizing scores as targets to window metadata.

        For Challenge 2, all windows from the same subject share the same target
        (externalizing score from participants.tsv).

        Args:
            windows: BaseConcatDataset of windowed data
        """
        logger.info("Adding subject-level targets (externalizing scores)...")

        for win_ds in windows.datasets:
            metadata = win_ds.metadata.copy()

            # Get externalizing score from window metadata (comes from subject description)
            # If 'externalizing' column exists, use it as target
            if 'externalizing' in metadata.columns:
                metadata['target'] = metadata['externalizing']
            else:
                # Fallback: try to get from description
                if hasattr(win_ds, 'description') and 'externalizing' in win_ds.description:
                    target_value = win_ds.description['externalizing']
                    metadata['target'] = target_value
                else:
                    # No externalizing score available - set to NaN
                    logger.warning(f"No externalizing score for dataset {win_ds.description.get('subject', 'unknown')}")
                    import numpy as np
                    metadata['target'] = np.nan

            # Update metadata
            win_ds.metadata = metadata.reset_index(drop=True)

            # Update y attribute (needed for training)
            if hasattr(win_ds, 'y'):
                import numpy as np
                y_np = win_ds.metadata['target'].astype(float).to_numpy()
                win_ds.y = y_np[:, None]  # (N, 1)

        logger.info("✓ Targets added to all windows")

    def _filter_nan_targets(self, windows: BaseConcatDataset) -> BaseConcatDataset:
        """Filter out subjects with NaN externalizing scores.

        Args:
            windows: BaseConcatDataset with targets added

        Returns:
            Filtered BaseConcatDataset excluding subjects with NaN targets
        """
        import math
        import numpy as np

        initial_count = len(windows.datasets)

        # Filter datasets where target is not NaN
        valid_datasets = []
        for ds in windows.datasets:
            # Check if this dataset has valid (non-NaN) targets
            if hasattr(ds, 'description') and 'externalizing' in ds.description:
                target_value = ds.description['externalizing']
                # Keep only if target is not NaN
                if not (isinstance(target_value, float) and math.isnan(target_value)):
                    valid_datasets.append(ds)
            elif hasattr(ds, 'y') and ds.y is not None:
                # Check y values directly
                if not np.isnan(ds.y).any():
                    valid_datasets.append(ds)

        filtered_windows = BaseConcatDataset(valid_datasets)
        removed_count = initial_count - len(valid_datasets)

        logger.info(
            f"Filtered out {removed_count} datasets with NaN targets "
            f"({len(valid_datasets)} datasets remaining)"
        )

        return filtered_windows
