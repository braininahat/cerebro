"""Helpers for creating datasets and dataloaders for self-supervised pretraining."""

from __future__ import annotations

import random
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

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
from .data import load_challenge_dataset


@dataclass
class WindowInfo:
    subject: str
    task: str
    release: str


class RandomCropWindowsDataset(Dataset):
    """Wrap a windows dataset to produce random crops of a fixed length.

    Args:
        dataset: Windowed dataset returning ``(X, y, crop_inds)`` tuples.
        crop_size_samples: Desired crop length in samples.
        seed: Optional seed used for deterministic shuffling per view.
        views: Number of random crops to return per window.
    """

    def __init__(
        self,
        dataset,
        crop_size_samples: int,
        seed: int | None = None,
        views: int = 1,
    ):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self._rng = random.Random(seed)
        self._description = dataset.description
        if views < 1:
            raise ValueError("views must be >= 1")
        self.views = views

    def __len__(self):  # pragma: no cover - simple proxy
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]
        i_window_in_trial, i_start, i_stop = crop_inds
        available = i_stop - i_start
        crops: list[torch.Tensor] = []
        for _ in range(self.views):
            if available < self.crop_size_samples:
                start_offset = 0
            else:
                start_offset = self._rng.randint(
                    0, available - self.crop_size_samples
                )
            start = start_offset
            stop = min(start_offset + self.crop_size_samples, X.shape[-1])
            X_crop = X[:, start:stop]
            if not isinstance(X_crop, torch.Tensor):
                X_crop = torch.as_tensor(X_crop)
            crops.append(X_crop)

        info = {
            "subject": self._description.get("subject", ""),
            "task": self._description.get("task", ""),
            "release": self._description.get("release", ""),
        }
        if self.views == 1:
            return crops[0], info
        stacked = torch.stack(crops, dim=0)
        return stacked, info


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
    views: int = 1,
    val_fraction: float | None = None,
    dataset_variant: str = "mini",
) -> DataLoader | Tuple[DataLoader, DataLoader]:
    """Build dataloaders of random crops across tasks/releases for self-supervision.

    When ``val_fraction`` is provided a second dataloader is returned containing the
    held-out recordings for validation loss monitoring.
    """

    datasets = []
    description_fields = ["subject", "task", "run", "release"]
    windows_datasets = []
    cache_root = Path(MINI_DATASET_ROOT) / "cached_windows" / dataset_variant

    for release in releases:
        for task in tasks:
            cache_file = cache_root / f"windows_{task}_{release}.pt"
            if cache_file.exists():
                windows = torch.load(cache_file)
            else:
                ds = load_challenge_dataset(
                    task=task,
                    release=release,
                    variant=dataset_variant,
                    cache_dir=MINI_DATASET_ROOT,
                    description_fields=description_fields,
                )
                for base_ds in ds.datasets:
                    for field in description_fields:
                        if field not in base_ds.description.index:
                            base_ds.description.loc[field] = ""
                    base_ds.description.loc["release"] = release
                datasets = ds.datasets
                min_samples = int(window_size_s * SFREQ)
                filtered = _filter_recordings(datasets, min_samples=min_samples)
                concat = BaseConcatDataset(filtered)
                windows = create_fixed_length_windows(
                    concat,
                    window_size_samples=min_samples,
                    window_stride_samples=int(stride_s * SFREQ),
                    drop_last_window=True,
                )
            for base_ds in windows.datasets:
                if "release" not in base_ds.description.index:
                    base_ds.description.loc["release"] = release
                if "task" not in base_ds.description.index:
                    base_ds.description.loc["task"] = task
            windows_datasets.extend(windows.datasets)

    windows_ds = BaseConcatDataset(windows_datasets)

    crop_samples = int(crop_size_s * SFREQ)
    wrapped = BaseConcatDataset(
        [
            RandomCropWindowsDataset(
                ds,
                crop_samples,
                seed=(seed or 0) + idx,
                views=views,
            )
            for idx, ds in enumerate(windows_ds.datasets)
        ]
    )

    if val_fraction:
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        n_total = len(wrapped.datasets)
        if n_total <= 1:
            val_fraction = None
        if val_fraction:
            n_val = max(1, int(n_total * val_fraction))
            rng = random.Random(seed)
            indices = list(range(n_total))
            rng.shuffle(indices)
            val_indices = set(indices[:n_val])
            train_datasets = [
                ds for idx, ds in enumerate(wrapped.datasets) if idx not in val_indices
            ]
            val_datasets = [
                ds for idx, ds in enumerate(wrapped.datasets) if idx in val_indices
            ]
            train_concat = BaseConcatDataset(train_datasets)
            val_concat = BaseConcatDataset(val_datasets)

            train_loader = DataLoader(
                train_concat,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
            )
            val_loader = DataLoader(
                val_concat,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            )
            return train_loader, val_loader

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


class RandomMaskingDataset(Dataset):
    """Apply random channel/time masking to windows for masked modeling."""

    def __init__(
        self,
        dataset,
        time_mask_fraction: float = 0.1,
        channel_mask_fraction: float = 0.1,
        mask_value: float = 0.0,
        seed: int | None = None,
    ):
        if not 0 < time_mask_fraction <= 1:
            raise ValueError("time_mask_fraction must be in (0, 1]")
        if not 0 < channel_mask_fraction <= 1:
            raise ValueError("channel_mask_fraction must be in (0, 1]")
        self.dataset = dataset
        self.time_mask_fraction = time_mask_fraction
        self.channel_mask_fraction = channel_mask_fraction
        self.mask_value = mask_value
        self.seed = seed or 0

    def __len__(self):  # pragma: no cover - proxy
        return len(self.dataset)

    def __getitem__(self, index):
        X, *_ = self.dataset[index]
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)
        X = X.float()
        rng = random.Random(self.seed + index)

        masked = X.clone()
        mask = torch.zeros_like(X, dtype=torch.bool)

        n_channels, n_times = X.shape
        n_mask_channels = max(1, int(self.channel_mask_fraction * n_channels))
        channel_indices = rng.sample(range(n_channels), n_mask_channels)
        mask[channel_indices, :] = True

        n_mask_times = max(1, int(self.time_mask_fraction * n_times))
        start = rng.randint(0, max(0, n_times - n_mask_times))
        mask[:, start : start + n_mask_times] = True

        masked[mask] = self.mask_value
        return masked, X, mask


def create_masked_modeling_dataloader(
    tasks: Sequence[str],
    releases: Sequence[str],
    window_size_s: float = 4.0,
    stride_s: float = 2.0,
    crop_size_s: float = 2.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int | None = None,
    time_mask_fraction: float = 0.1,
    channel_mask_fraction: float = 0.1,
    mask_value: float = 0.0,
    val_fraction: float | None = None,
    dataset_variant: str = "mini",
) -> DataLoader | Tuple[DataLoader, DataLoader]:
    """Build dataloaders for masked time-channel reconstruction."""

    base_loader = create_pretraining_dataloader(
        tasks=tasks,
        releases=releases,
        window_size_s=window_size_s,
        stride_s=stride_s,
        crop_size_s=crop_size_s,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        views=1,
        val_fraction=val_fraction,
        dataset_variant=dataset_variant,
    )

    def wrap(base_loader):
        return DataLoader(
            RandomMaskingDataset(
                base_loader.dataset,
                time_mask_fraction=time_mask_fraction,
                channel_mask_fraction=channel_mask_fraction,
                mask_value=mask_value,
                seed=seed,
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )

    if isinstance(base_loader, tuple):
        train_loader, val_loader = base_loader
        return (
            wrap(train_loader),
            DataLoader(
                RandomMaskingDataset(
                    val_loader.dataset,
                    time_mask_fraction=time_mask_fraction,
                    channel_mask_fraction=channel_mask_fraction,
                    mask_value=mask_value,
                    seed=seed,
                ),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            ),
        )

    return wrap(base_loader)


__all__.extend([
    "RandomMaskingDataset",
    "create_masked_modeling_dataloader",
])
