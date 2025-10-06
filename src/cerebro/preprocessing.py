import torch
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from eegdash.hbn.windows import (
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
    keep_only_recordings_with,
)

from .constants import *


def prepare_dataset(dataset, target_field="rt_from_stimulus", epoch_length=EPOCH_LEN_S):
    """Apply preprocessing transformations to dataset.

    Args:
        dataset: EEGChallengeDataset or similar
        target_field: Field to use as target for annotation
        epoch_length: Length of epochs in seconds

    Returns:
        Preprocessed dataset
    """
    transformation_offline = [
        Preprocessor(
            annotate_trials_with_target,
            target_field=target_field,
            epoch_length=epoch_length,
            require_stimulus=True,
            require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    preprocess(dataset, transformation_offline, n_jobs=1)
    return dataset


def filter_by_anchor(dataset, anchor=ANCHOR):
    """Filter dataset to keep only recordings with specified anchor.

    Args:
        dataset: Dataset to filter
        anchor: Anchor name to filter by

    Returns:
        Filtered dataset
    """
    return keep_only_recordings_with(anchor, dataset)


def create_single_windows(
    dataset,
    anchor=ANCHOR,
    shift_after_stim=SHIFT_AFTER_STIM,
    window_len=WINDOW_LEN,
    epoch_len_s=EPOCH_LEN_S,
    sfreq=SFREQ,
):
    """Create windowed data from dataset.

    Args:
        dataset: Preprocessed dataset
        anchor: Anchor to use for windowing
        shift_after_stim: Time shift after stimulus in seconds
        window_len: Window length in seconds
        epoch_len_s: Epoch length in seconds
        sfreq: Sampling frequency

    Returns:
        Windowed dataset
    """
    windows = create_windows_from_events(
        dataset,
        mapping={anchor: 0},
        trial_start_offset_samples=int(shift_after_stim * sfreq),
        trial_stop_offset_samples=int((shift_after_stim + window_len) * sfreq),
        window_size_samples=int(epoch_len_s * sfreq),
        window_stride_samples=sfreq,
        preload=True,
    )
    return windows


def remove_high_amplitude_channels(signal, z_threshold=5.0):
    """Return a boolean mask flagging channels whose z-score exceeds the threshold.

    Args:
        signal: Tensor with shape (n_channels, n_times)
        z_threshold: Z-score threshold for outlier detection

    Returns:
        Boolean tensor of shape (n_channels,) with True for channels to drop.
    """
    if signal.numel() == 0:
        return torch.zeros(signal.size(0), dtype=torch.bool)
    mean = signal.mean(dim=1, keepdim=True)
    std = signal.std(dim=1, keepdim=True)
    std = torch.clamp(std, min=1e-6)
    centered = signal - mean
    rms = torch.sqrt((centered ** 2).mean(dim=1))
    return rms >= z_threshold


def add_metadata(windows, dataset, anchor=ANCHOR):
    """Add metadata columns to windowed data.

    Args:
        windows: Windowed dataset
        dataset: Original dataset (for metadata)
        anchor: Anchor description

    Returns:
        Windows with added metadata
    """
    keys = (
        "target",
        "rt_from_stimulus",
        "rt_from_trialstart",
        "stimulus_onset",
        "response_onset",
        "correct",
        "response_type",
    )
    return add_extras_columns(windows, dataset, desc=anchor, keys=keys)
