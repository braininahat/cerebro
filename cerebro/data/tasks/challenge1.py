"""Challenge 1 Task: Stimulus-locked windows for RT prediction."""

import logging

import mne
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

    def _filter_boundary_annotations(
        self, recordings: BaseConcatDataset
    ) -> BaseConcatDataset:
        """Filter annotations too close to recording boundaries.

        Removes annotations where the windowing would extend past recording end.
        This prevents ValueError during create_windows_from_events.

        Args:
            recordings: BaseConcatDataset with annotations

        Returns:
            BaseConcatDataset with filtered annotations and valid recordings only
        """
        # Calculate required buffer from annotation onset to window end
        # Buffer = shift_after_stim + window_len
        buffer_samples = int((self.shift_after_stim + self.window_len) * self.sfreq)

        valid_recordings = []
        removed_count = 0
        total_annotations_before = 0
        total_annotations_after = 0

        for dataset in recordings.datasets:
            raw = dataset.raw
            annotations = raw.annotations

            # Get recording duration in samples
            recording_duration_samples = raw.n_times

            total_annotations_before += len(annotations)

            # Filter annotations: keep only those with enough buffer
            # annotation onset (in seconds) + buffer (in seconds) must be < recording duration
            valid_mask = []
            for ann in annotations:
                onset_samples = int(ann['onset'] * self.sfreq)
                end_samples = onset_samples + buffer_samples
                valid_mask.append(end_samples <= recording_duration_samples)

            # Create new annotations with only valid entries
            if any(valid_mask):
                # Keep annotations that pass the boundary check
                # IMPORTANT: Preserve extras field which contains RT metadata
                valid_indices = [i for i, valid in enumerate(valid_mask) if valid]

                # Filter extras if they exist (MNE uses .extras, not .metadata)
                extras = None
                if hasattr(annotations, 'extras') and annotations.extras is not None:
                    extras = [annotations.extras[i] for i in valid_indices]

                filtered_annotations = mne.Annotations(
                    onset=[annotations[i]['onset'] for i in valid_indices],
                    duration=[annotations[i]['duration'] for i in valid_indices],
                    description=[annotations[i]['description'] for i in valid_indices],
                    orig_time=annotations.orig_time,
                )

                # Set extras if they exist
                if extras:
                    filtered_annotations.extras = extras

                raw.set_annotations(filtered_annotations)
                valid_recordings.append(dataset)
                total_annotations_after += len(filtered_annotations)
            else:
                # No valid annotations - drop this recording
                removed_count += 1

        logger.info(
            f"Boundary filtering: {total_annotations_before} annotations → "
            f"{total_annotations_after} annotations "
            f"(removed {total_annotations_before - total_annotations_after})"
        )
        logger.info(
            f"Kept {len(valid_recordings)}/{len(recordings.datasets)} recordings "
            f"(removed {removed_count} with no valid annotations)"
        )

        if not valid_recordings:
            raise ValueError(
                "No recordings remain after boundary filtering. "
                f"Window parameters (shift={self.shift_after_stim}s, "
                f"length={self.window_len}s) may be too large."
            )

        return BaseConcatDataset(valid_recordings)

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

        # Filter out recordings without stimulus anchors (critical!)
        from eegdash.hbn.windows import keep_only_recordings_with
        recordings_filtered = keep_only_recordings_with(self.anchor, recordings)
        logger.info(f"Kept {len(recordings_filtered.datasets)} recordings with {self.anchor}")

        # Filter annotations too close to recording boundaries
        recordings_clean = self._filter_boundary_annotations(recordings_filtered)

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
            recordings_clean,
            mapping={self.anchor: 0},
            trial_start_offset_samples=int(self.shift_after_stim * self.sfreq),
            trial_stop_offset_samples=int(
                (self.shift_after_stim + self.window_len) * self.sfreq
            ),
            window_size_samples=int(self.epoch_len_s * self.sfreq),
            window_stride_samples=self.sfreq,
            preload=True,
            drop_bad_windows=False,  # Already filtered boundary violations above
            verbose=False,  # Suppress "Used Annotations" logging spam
        )

        logger.info(f"[bold]Created {len(single_windows)} windows[/bold]")

        # Inject metadata
        single_windows = add_extras_columns(
            single_windows,
            recordings_clean,
            desc=self.anchor,
            keys=("target", "rt_from_stimulus", "rt_from_trialstart",
                  "stimulus_onset", "response_onset", "correct", "response_type")
        )

        return single_windows
