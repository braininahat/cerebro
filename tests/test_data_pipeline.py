import math
from typing import Dict, Iterable, List

import pytest
import torch

from cerebro import preprocessing
from cerebro.data import create_dataloaders, remove_subjects, split_subjects, split_windows_by_subject
from cerebro.preprocessing import filter_by_anchor


class DummyDataset:
    """Minimal dataset that mimics a window dataset."""

    def __init__(self, subject: str, values: Iterable[int]):
        self.subject = subject
        self._values = list(values)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._values)

    def __getitem__(self, idx):  # pragma: no cover - trivial
        value = self._values[idx]
        # mimic (X, y) tuple
        return torch.tensor([value], dtype=torch.float32), torch.tensor([value], dtype=torch.float32)


class DummyWindows:
    """Lightweight stand-in for Braindecode's BaseConcatDataset.split."""

    def __init__(self, mapping: Dict[str, DummyDataset]):
        self._mapping = mapping

    def split(self, key: str) -> Dict[str, DummyDataset]:  # pragma: no cover - simple getter
        assert key == "subject"
        return self._mapping


def test_remove_subjects_excludes_bad_list():
    subjects = ["A", "B", "C"]
    excluded = ["B"]
    filtered = remove_subjects(subjects, subjects_to_remove=excluded)
    assert filtered == ["A", "C"]


def test_split_subjects_reproducible_seed():
    subjects = [f"S{i}" for i in range(10)]
    train, valid, test = split_subjects(subjects, valid_frac=0.2, test_frac=0.2, seed=42)
    assert len(train) + len(valid) + len(test) == len(subjects)
    assert set(train) & set(valid) == set()
    assert set(train) & set(test) == set()
    assert set(valid) & set(test) == set()
    # Deterministic under same seed
    train2, valid2, test2 = split_subjects(subjects, valid_frac=0.2, test_frac=0.2, seed=42)
    assert train == train2
    assert valid == valid2
    assert test == test2


def test_split_windows_by_subject_preserves_counts():
    mapping = {
        "S1": DummyDataset("S1", range(5)),
        "S2": DummyDataset("S2", range(3)),
        "S3": DummyDataset("S3", range(2)),
    }
    windows = DummyWindows(mapping)
    train, valid, test = split_windows_by_subject(
        windows,
        train_subjects=["S1"],
        valid_subjects=["S2"],
        test_subjects=["S3"],
    )
    assert len(train.datasets) == 1
    assert len(valid.datasets) == 1
    assert len(test.datasets) == 1
    # Flatten dataset lengths
    assert sum(len(ds) for ds in train.datasets) == 5
    assert sum(len(ds) for ds in valid.datasets) == 3
    assert sum(len(ds) for ds in test.datasets) == 2


def test_create_dataloaders_uses_common_batch_params():
    dataset = DummyDataset("S1", range(8))
    loaders = create_dataloaders(dataset, dataset, dataset, batch_size=4, num_workers=0)
    train_loader, valid_loader, test_loader = loaders
    assert len(train_loader) == math.ceil(len(dataset) / 4)
    assert len(valid_loader) == math.ceil(len(dataset) / 4)
    assert len(test_loader) == math.ceil(len(dataset) / 4)


def test_filter_by_anchor_calls_keep_only_recordings(monkeypatch):
    called = {}

    def fake_keep_only_recordings(anchor, dataset):
        called["anchor"] = anchor
        called["dataset"] = dataset
        return "filtered"

    monkeypatch.setattr(preprocessing, "keep_only_recordings_with", fake_keep_only_recordings)
    result = filter_by_anchor(dataset="dummy", anchor="stimulus_anchor")
    assert result == "filtered"
    assert called == {"anchor": "stimulus_anchor", "dataset": "dummy"}
