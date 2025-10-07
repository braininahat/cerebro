import torch

from torch.utils.data import DataLoader, Dataset

from cerebro.pretraining import (
    RandomCropWindowsDataset,
    RandomMaskingDataset,
    create_masked_modeling_dataloader,
)
from fm.tasks import masked_mse_loss


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
    assert isinstance(crop, torch.Tensor)


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


def test_random_crop_can_return_multiple_views():
    signal = torch.arange(0, 40, dtype=torch.float32).view(1, -1)
    dataset = DummyWindowsDataset(
        signal,
        description={"subject": "S1", "task": "RestingState", "release": "R1"},
        crop_start_stop=(0, 0, signal.shape[-1]),
    )
    wrapper = RandomCropWindowsDataset(dataset, crop_size_samples=10, seed=0, views=2)
    crops, info = wrapper[0]
    assert crops.shape == (2, 1, 10)
    assert isinstance(crops, torch.Tensor)
    assert info["release"] == "R1"


def test_random_masking_dataset_applies_mask():
    signal = torch.ones(3, 10, dtype=torch.float32)

    class Dummy(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return signal.clone(), 0.0, (0, 0, signal.shape[-1])

    dataset = RandomMaskingDataset(Dummy(), time_mask_fraction=0.2, channel_mask_fraction=1 / 3, seed=0)
    masked, target, mask = dataset[0]
    assert masked.shape == target.shape == mask.shape
    assert torch.all(target == 1.0)
    assert torch.all(masked[~mask] == 1.0)
    assert torch.all(masked[mask] == 0.0)


def test_masked_mse_loss_only_penalises_masked_entries():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    mask = torch.tensor([[True, True], [False, False]])
    loss = masked_mse_loss(pred, target, mask)
    expected = torch.mean((pred[:1] - target[:1]) ** 2)
    assert torch.allclose(loss, expected)


def test_create_masked_modeling_dataloader_wraps_base(monkeypatch):
    class DummyBase(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return torch.zeros(129, 20), 0.0, (0, 0, 20)

    def fake_loader(**kwargs):
        assert kwargs["dataset_variant"] == "mini"
        ds = DummyBase()
        loader = DataLoader(ds, batch_size=2)
        return loader

    monkeypatch.setattr("cerebro.pretraining.create_pretraining_dataloader", fake_loader)

    loader = create_masked_modeling_dataloader(
        tasks=["task"],
        releases=["R1"],
        batch_size=2,
        time_mask_fraction=0.2,
        channel_mask_fraction=0.2,
    )

    batch = next(iter(loader))
    masked, target, mask = batch
    assert masked.shape == target.shape == mask.shape
