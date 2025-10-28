"""Challenge 1 Task: Stimulus-locked windows for RT prediction."""

import logging

from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from eegdash.hbn.windows import (
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
)

logger = logging.getLogger(__name__)


class Challenge1Task:
    """Create stimulus-locked windows for Challenge 1 (RT prediction).

    This is Layer 2 of the composable data architecture:
    - Takes filtered recordings from Dataset layer
    - Annotates trials with target RT
    - Creates stimulus-locked windows
    - Returns WindowsDataset ready for training

    Args:
        window_len: Window length in seconds (default: 2.0)
        shift_after_stim: Start window X seconds after stimulus (default: 0.5)
        sfreq: Sampling frequency in Hz (default: 100)
        epoch_len_s: Epoch length for preprocessing (default: 2.0)
        anchor: Anchor event name (default: "stimulus_anchor")

    Example:
        >>> task = Challenge1Task(window_len=2.0, shift_after_stim=0.5)
        >>> windows = task.create_windows(filtered_recordings)
        >>> # windows is BaseConcatDataset of stimulus-locked windows with RT targets
    """

    def __init__(
        self,
        window_len: float = 2.0,
        shift_after_stim: float = 0.5,
        sfreq: int = 100,
        epoch_len_s: float = 2.0,
        anchor: str = "stimulus_anchor",
    ):
        self.window_len = window_len
        self.shift_after_stim = shift_after_stim
        self.sfreq = sfreq
        self.epoch_len_s = epoch_len_s
        self.anchor = anchor

    def create_windows(self, recordings: BaseConcatDataset) -> BaseConcatDataset:
        """Create stimulus-locked windows from filtered recordings.

        Args:
            recordings: BaseConcatDataset of filtered recordings from HBNDataset

        Returns:
            BaseConcatDataset of windowed data with RT targets
        """
        logger.info("[bold cyan]CHALLENGE 1 WINDOWING[/bold cyan]")
        logger.info("Creating stimulus-locked windows...")

        # Preprocess: Annotate trials with target RT and add anchors
        transformation_offline = [
            Preprocessor(
                annotate_trials_with_target,
                target_field="rt_from_stimulus",
                epoch_length=self.epoch_len_s,
                require_stimulus=True,
                require_response=True,
                apply_on_array=False,
            ),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ]

        logger.info("Annotating trials with target RT and adding anchors...")
        preprocess(recordings, transformation_offline, n_jobs=-1)

        # Create stimulus-locked windows
        logger.info(
            f"Window: [{self.shift_after_stim}s, "
            f"{self.shift_after_stim + self.window_len}s] relative to stimulus"
        )
        logger.info(
            f"Window length: {self.window_len}s "
            f"({int(self.window_len * self.sfreq)} samples)"
        )

        single_windows = create_windows_from_events(
            recordings,
            mapping={self.anchor: 0},
            trial_start_offset_samples=int(self.shift_after_stim * self.sfreq),
            trial_stop_offset_samples=int(
                (self.shift_after_stim + self.window_len) * self.sfreq
            ),
            window_size_samples=int(self.epoch_len_s * self.sfreq),
            window_stride_samples=self.sfreq,
            preload=True,
        )

        logger.info(f"[bold]Created {len(single_windows)} windows[/bold]")

        # Inject metadata
        single_windows = add_extras_columns(
            single_windows,
            recordings,
            desc=self.anchor,
            keys=("target", "rt_from_stimulus", "rt_from_trialstart",
                  "stimulus_onset", "response_onset", "correct", "response_type")
        )

        return single_windows
