"""DataModule-style wrapper around the pretraining dataloader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from torch.utils.data import DataLoader

from cerebro.pretraining import create_pretraining_dataloader


@dataclass
class MultiTaskPretrainConfig:
    tasks: Sequence[str]
    releases: Sequence[str]
    window_size_s: float = 4.0
    stride_s: float = 2.0
    crop_size_s: float = 2.0
    batch_size: int = 128
    num_workers: int = 8
    seed: int | None = None
    views: int = 1
    val_fraction: float | None = None
    dataset_variant: str = "mini"


class MultiTaskPretrainDataModule:
    """Thin wrapper that exposes a train dataloader for multitask pretraining."""

    def __init__(self, config: MultiTaskPretrainConfig):
        self.config = config
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self.summary: Dict[str, Dict[str, Any]] = {}

    def setup(self):  # pragma: no cover - trivial guard
        if self._train_loader is None:
            loader = create_pretraining_dataloader(
                tasks=self.config.tasks,
                releases=self.config.releases,
                window_size_s=self.config.window_size_s,
                stride_s=self.config.stride_s,
                crop_size_s=self.config.crop_size_s,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                seed=self.config.seed,
                views=self.config.views,
                val_fraction=self.config.val_fraction,
                dataset_variant=self.config.dataset_variant,
            )
            if isinstance(loader, tuple):
                self._train_loader, self._val_loader = loader
            else:
                self._train_loader = loader
            self.summary["train"] = _summarize_loader(self._train_loader)
            if self._val_loader is not None:
                self.summary["val"] = _summarize_loader(self._val_loader)
            self.summary["variant"] = self.config.dataset_variant

    def train_dataloader(self) -> DataLoader:
        self.setup()
        assert self._train_loader is not None
        return self._train_loader

    def val_dataloader(self) -> DataLoader | None:
        self.setup()
        return self._val_loader


def _summarize_loader(loader: DataLoader | None) -> dict[str, Any]:
    if loader is None or not hasattr(loader, "dataset"):
        return {}
    dataset = loader.dataset
    bases = getattr(dataset, "datasets", [dataset])
    subjects: set[str] = set()
    tasks: set[str] = set()
    releases: set[str] = set()
    n_windows = 0
    for base in bases:
        desc = getattr(base, "_description", None)
        if desc is None and hasattr(base, "description"):
            desc = base.description

        def fetch(key: str) -> str:
            if desc is None:
                return ""
            if isinstance(desc, dict):
                return str(desc.get(key, ""))
            if hasattr(desc, "get"):
                value = desc.get(key, "")
                return "" if value is None else str(value)
            return ""

        subj = fetch("subject")
        task = fetch("task")
        release = fetch("release")
        if subj:
            subjects.add(subj)
        if task:
            tasks.add(task)
        if release:
            releases.add(release)
        try:
            n_windows += len(base)
        except TypeError:
            pass

    return {
        "n_recordings": len(bases),
        "n_windows": n_windows,
        "subjects": sorted(subjects),
        "tasks": sorted(tasks),
        "releases": sorted(releases),
        "batches_per_epoch": len(loader) if hasattr(loader, "__len__") else None,
    }
