import torch

from cerebro.preprocessing import remove_high_amplitude_channels


def test_mark_bad_channels_flags_threshold():
    signal = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 100.0],
    ])
    mask = remove_high_amplitude_channels(signal, z_threshold=2.0)
    assert mask.tolist() == [False, True]


def test_mark_bad_channels_safe_for_empty_signal():
    empty = torch.zeros((0, 10))
    mask = remove_high_amplitude_channels(empty)
    assert mask.numel() == 0
