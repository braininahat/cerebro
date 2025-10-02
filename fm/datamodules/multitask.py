"""DataModule-style wrapper around the pretraining dataloader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


class MultiTaskPretrainDataModule:
    """Thin wrapper that exposes a train dataloader for multitask pretraining."""

    def __init__(self, config: MultiTaskPretrainConfig):
        self.config = config
        self._train_loader: DataLoader | None = None

    def setup(self):  # pragma: no cover - trivial guard
        if self._train_loader is None:
            self._train_loader = create_pretraining_dataloader(
                tasks=self.config.tasks,
                releases=self.config.releases,
                window_size_s=self.config.window_size_s,
                stride_s=self.config.stride_s,
                crop_size_s=self.config.crop_size_s,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                seed=self.config.seed,
            )

    def train_dataloader(self) -> DataLoader:
        self.setup()
        assert self._train_loader is not None
        return self._train_loader
