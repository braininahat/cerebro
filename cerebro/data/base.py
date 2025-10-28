"""Base DataModule for composable Dataset + Task architecture."""

import logging
from typing import Optional

import lightning as L
import torch
from braindecode.datasets import BaseConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Collate function that handles (x, y) tuples."""
    x_batch = torch.stack([item[0] for item in batch], dim=0)
    y_batch = torch.stack([item[1] for item in batch], dim=0)
    return x_batch, y_batch


class BaseTaskDataModule(L.LightningDataModule):
    """Composable DataModule: Dataset + Task.

    This is Layer 3 of the composable data architecture:
    - Composes a Dataset (loading/filtering) and Task (windowing)
    - Handles subject-level splits
    - Provides Lightning dataloaders

    Args:
        dataset: Dataset instance (e.g., HBNDataset)
        task: Task instance (e.g., Challenge1Task)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for parallel data loading
        val_frac: Fraction of subjects for validation
        test_frac: Fraction of subjects for test
        seed: Random seed for splits

    Example:
        >>> from cerebro.data.datasets import HBNDataset
        >>> from cerebro.data.tasks import Challenge1Task
        >>>
        >>> dataset = HBNDataset(data_dir="data", releases=["R1"], tasks=["contrastChangeDetection"])
        >>> task = Challenge1Task(window_len=2.0, shift_after_stim=0.5)
        >>> dm = BaseTaskDataModule(dataset, task, batch_size=64)
        >>> dm.setup("fit")
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        dataset,
        task,
        batch_size: int = 512,
        num_workers: int = 8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 2025,
        excluded_subjects: Optional[list[str]] = None,
    ):
        super().__init__()
        # Save hyperparameters (ignore dataset/task objects)
        self.save_hyperparameters(ignore=["dataset", "task"])

        self.dataset = dataset
        self.task = task
        self.excluded_subjects = excluded_subjects or []

        # Will be populated in setup()
        self.train_set = None
        self.val_set = None
        self.test_set = None

    @property
    def batch_size(self) -> int:
        """Batch size for DataLoaders (Lightning compatibility)."""
        return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        """Update batch size (modified by Lightning's batch size scaler)."""
        self.hparams.batch_size = value

    def setup(self, stage: Optional[str] = None):
        """Setup datasets: load → window → split.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict' (unused, loads all)
        """
        logger.info("\n" + "="*60)
        logger.info("[bold cyan]DATA SETUP[/bold cyan]")
        logger.info("="*60)

        # Step 1: Load recordings (Dataset layer)
        recordings = self.dataset.load()

        # Step 2: Create windows (Task layer)
        windows = self.task.create_windows(recordings)

        # Inspect metadata
        metadata = windows.get_metadata()
        logger.info(f"\n[bold]Metadata columns:[/bold] {len(list(metadata.columns))} columns")
        logger.info(f"\n[bold]Sample metadata:[/bold]")
        logger.info(metadata.head().to_string())

        # Log target statistics if available
        if 'target' in metadata.columns:
            logger.info(f"\n[bold]Target Statistics:[/bold]")
            logger.info(f"  Mean: {metadata['target'].mean():.4f}")
            logger.info(f"  Std: {metadata['target'].std():.4f}")
            logger.info(f"  Min: {metadata['target'].min():.4f}")
            logger.info(f"  Max: {metadata['target'].max():.4f}")

        # Step 3: Split at subject level
        self._create_splits(windows, metadata)

    def _create_splits(self, windows: BaseConcatDataset, metadata):
        """Create subject-level train/val/test splits."""
        logger.info("\n[bold cyan]SPLITTING DATA[/bold cyan]")

        subjects = metadata["subject"].unique()
        subjects = [s for s in subjects if s not in self.excluded_subjects]
        logger.info(f"[bold]Total subjects (after exclusion):[/bold] {len(subjects)}")

        # Split: train / (val + test)
        train_subj, valid_test_subj = train_test_split(
            subjects,
            test_size=(self.hparams.val_frac + self.hparams.test_frac),
            random_state=check_random_state(self.hparams.seed),
            shuffle=True
        )

        # Split: val / test
        valid_subj, test_subj = train_test_split(
            valid_test_subj,
            test_size=self.hparams.test_frac / (self.hparams.val_frac + self.hparams.test_frac),
            random_state=check_random_state(self.hparams.seed + 1),
            shuffle=True
        )

        # Sanity check
        assert (set(valid_subj) | set(test_subj) | set(train_subj)) == set(subjects)

        logger.info(f"Train subjects: {len(train_subj)}")
        logger.info(f"Val subjects: {len(valid_subj)}")
        logger.info(f"Test subjects: {len(test_subj)}")

        # Create splits
        subject_split = windows.split("subject")
        self.train_set = BaseConcatDataset(
            [subject_split[s] for s in train_subj if s in subject_split]
        )
        self.val_set = BaseConcatDataset(
            [subject_split[s] for s in valid_subj if s in subject_split]
        )
        self.test_set = BaseConcatDataset(
            [subject_split[s] for s in test_subj if s in subject_split]
        )

        logger.info(f"\n[bold]Window counts:[/bold]")
        logger.info(f"  Train: {len(self.train_set)}")
        logger.info(f"  Val: {len(self.val_set)}")
        logger.info(f"  Test: {len(self.test_set)}")

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=True,
        )
