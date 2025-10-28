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
        return windows
