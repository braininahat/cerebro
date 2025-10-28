"""Data modules for EEG datasets."""

from .base import BaseTaskDataModule
from .challenge1 import Challenge1DataModule
from .challenge2 import Challenge2DataModule

# Layer 1: Datasets
from .datasets import HBNDataset

# Layer 2: Tasks
from .tasks import Challenge1Task, Challenge2Task

__all__ = [
    # DataModules (Layer 3)
    "BaseTaskDataModule",
    "Challenge1DataModule",
    "Challenge2DataModule",
    # Datasets (Layer 1)
    "HBNDataset",
    # Tasks (Layer 2)
    "Challenge1Task",
    "Challenge2Task",
]
