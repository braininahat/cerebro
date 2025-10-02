import torch

from cerebro.pretraining import RandomCropWindowsDataset


class DummyWindowsDataset:
    def __init__(self, signal, description, crop_start_stop):
        self._signal = signal
        self.description = description
        self._crop = crop_start_stop

    def __len__(self):  # pragma: no cover - trivial
        return 1

    def __getitem__(self, index):  # pragma: no cover - simple stub
        return self._signal, torch.tensor(0.0), self._crop


def test_random_crop_returns_expected_shape():
    signal = torch.arange(0, 40, dtype=torch.float32).view(1, -1)
    dataset = DummyWindowsDataset(
        signal,
        description={"subject": "S1", "task": "RestingState", "release": "R1"},
        crop_start_stop=(0, 0, signal.shape[-1]),
    )
    wrapper = RandomCropWindowsDataset(dataset, crop_size_samples=10, seed=0)
    crop, info = wrapper[0]
    assert crop.shape == (1, 10)
    assert info["subject"] == "S1"
    assert info["task"] == "RestingState"


def test_random_crop_handles_short_windows():
    signal = torch.arange(0, 10, dtype=torch.float32).view(1, -1)
    dataset = DummyWindowsDataset(
        signal,
        description={"subject": "S1", "task": "RestingState", "release": "R1"},
        crop_start_stop=(0, 0, signal.shape[-1]),
    )
    wrapper = RandomCropWindowsDataset(dataset, crop_size_samples=20, seed=0)
    crop, _ = wrapper[0]
    assert crop.shape == (1, 10)
