"""Helpers for creating datasets and dataloaders for self-supervised pretraining."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows
from torch.utils.data import DataLoader, Dataset

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    MINI_DATASET_ROOT,
    N_CHANS,
    SFREQ,
    SUBJECTS_TO_REMOVE,
)
from .data import load_mini_dataset


@dataclass
class WindowInfo:
    subject: str
    task: str
    release: str


class RandomCropWindowsDataset(Dataset):
    """Wrap a windows dataset to produce random crops of a fixed length."""

    def __init__(self, dataset, crop_size_samples: int, seed: int | None = None):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self._rng = random.Random(seed)
        self._description = dataset.description

    def __len__(self):  # pragma: no cover - simple proxy
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]
        i_window_in_trial, i_start, i_stop = crop_inds
        available = i_stop - i_start
        if available < self.crop_size_samples:
            start_offset = 0
        else:
            start_offset = self._rng.randint(0, available - self.crop_size_samples)
        start = start_offset
        stop = min(start_offset + self.crop_size_samples, X.shape[-1])
        X_crop = X[:, start:stop]

        info = {
            "subject": self._description.get("subject", ""),
            "task": self._description.get("task", ""),
            "release": self._description.get("release", ""),
        }
        return X_crop, info


def _filter_recordings(datasets, min_samples: int) -> List:
    filtered = []
    for ds in datasets:
        desc = ds.description
        subject = desc.get("subject", "")
        if subject in SUBJECTS_TO_REMOVE:
            continue
        raw = ds.raw
        if raw.n_times < min_samples:
            continue
        if len(raw.ch_names) != N_CHANS:
            continue
        filtered.append(ds)
    return filtered


def create_pretraining_dataloader(
    tasks: Sequence[str],
    releases: Sequence[str],
    window_size_s: float = 4.0,
    stride_s: float = 2.0,
    crop_size_s: float = 2.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int | None = None,
) -> DataLoader:
    """Build a dataloader of random crops across tasks/releases for self-supervision."""

    datasets = []
    description_fields = ["subject", "task", "run", "release"]
    for release in releases:
        for task in tasks:
            ds = load_mini_dataset(
                task=task,
                release=release,
                mini=True,
                cache_dir=MINI_DATASET_ROOT,
                description_fields=description_fields,
            )
            for base_ds in ds.datasets:
                for field in description_fields:
                    if field not in base_ds.description.index:
                        base_ds.description.loc[field] = ""
                base_ds.description.loc["release"] = release
            datasets.extend(ds.datasets)

    min_samples = int(window_size_s * SFREQ)
    filtered = _filter_recordings(datasets, min_samples=min_samples)
    concat = BaseConcatDataset(filtered)

    windows_ds = create_fixed_length_windows(
        concat,
        window_size_samples=min_samples,
        window_stride_samples=int(stride_s * SFREQ),
        drop_last_window=True,
    )

    crop_samples = int(crop_size_s * SFREQ)
    wrapped = BaseConcatDataset(
        [
            RandomCropWindowsDataset(ds, crop_samples, seed=(seed or 0) + idx)
            for idx, ds in enumerate(windows_ds.datasets)
        ]
    )

    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    return loader


__all__ = [
    "create_pretraining_dataloader",
    "RandomCropWindowsDataset",
]
