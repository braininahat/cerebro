from pathlib import Path
from typing import Optional, Tuple

from braindecode.datasets import BaseConcatDataset
from eegdash.dataset import EEGChallengeDataset
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader

from .constants import *
from .preprocessing import *


def load_mini_dataset(
    task: str,
    release: str = "R1",
    mini: bool = True,
    cache_dir: Path = MINI_DATASET_ROOT,
) -> EEGChallengeDataset:
    """Load EEG challenge dataset for a specific task.

    Args:
        task: Task name (e.g., "contrastChangeDetection")
        release: Release version (default "R1")
        mini: Whether to use mini dataset (default True)
        cache_dir: Directory containing cached data

    Returns:
        EEGChallengeDataset object
    """
    return EEGChallengeDataset(
        task=task,
        release=release,
        mini=mini,
        cache_dir=cache_dir,
    )


def download_all_raws(dataset, n_jobs=-1):
    """Download all raw data files in parallel.

    Args:
        dataset: EEGChallengeDataset
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        List of raw MNE objects
    """
    return Parallel(n_jobs=n_jobs)(
        delayed(lambda d: d.raw)(d) for d in dataset.datasets
    )


def get_unique_subjects(windows):
    """Get unique subject IDs from windowed dataset.

    Args:
        windows: Windowed dataset with metadata

    Returns:
        Array of unique subject IDs
    """
    return windows.description["subject"].unique()


def remove_subjects(subjects, subjects_to_remove=SUBJECTS_TO_REMOVE):
    """Remove problematic subjects from list.

    Args:
        subjects: List of subject IDs
        subjects_to_remove: List of subjects to exclude

    Returns:
        Filtered list of subjects
    """
    return [s for s in subjects if s not in subjects_to_remove]


def split_subjects(
    subjects,
    valid_frac: float = VALID_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = SEED,
) -> Tuple[list, list, list]:
    """Split subjects into train/valid/test sets.

    Args:
        subjects: List of subject IDs
        valid_frac: Fraction for validation set
        test_frac: Fraction for test set
        seed: Random seed

    Returns:
        Tuple of (train_subjects, valid_subjects, test_subjects)
    """
    train_subj, valid_test_subject = train_test_split(
        subjects,
        test_size=(valid_frac + test_frac),
        random_state=check_random_state(seed),
        shuffle=True,
    )

    valid_subj, test_subj = train_test_split(
        valid_test_subject,
        test_size=test_frac / (valid_frac + test_frac),
        random_state=check_random_state(seed + 1),
        shuffle=True,
    )

    return train_subj, valid_subj, test_subj


def split_windows_by_subject(
    windows, train_subjects, valid_subjects, test_subjects
) -> Tuple[BaseConcatDataset, BaseConcatDataset, BaseConcatDataset]:
    """Split windowed data by subject IDs.

    Args:
        windows: Windowed dataset
        train_subjects: List of training subject IDs
        valid_subjects: List of validation subject IDs
        test_subjects: List of test subject IDs

    Returns:
        Tuple of (train_set, valid_set, test_set)
    """
    subject_split = windows.split("subject")
    train_set = []
    valid_set = []
    test_set = []

    for s in subject_split:
        if s in train_subjects:
            train_set.append(subject_split[s])
        elif s in valid_subjects:
            valid_set.append(subject_split[s])
        elif s in test_subjects:
            test_set.append(subject_split[s])

    return (
        BaseConcatDataset(train_set),
        BaseConcatDataset(valid_set),
        BaseConcatDataset(test_set),
    )


def create_dataloaders(
    train_set: BaseConcatDataset,
    valid_set: BaseConcatDataset,
    test_set: BaseConcatDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders from datasets.

    Args:
        train_set: Training dataset
        valid_set: Validation dataset
        test_set: Test dataset
        batch_size: Batch size for all loaders
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader


def prepare_data_pipeline(
    task: str = "contrastChangeDetection",
    release: str = "R1",
    remove_bad_subjects: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Complete data pipeline from loading to dataloaders.

    Args:
        task: Task name
        release: Release version
        remove_bad_subjects: Whether to remove problematic subjects
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    # Load dataset
    dataset = load_mini_dataset(task, release)

    # Preprocess
    dataset = prepare_dataset(dataset)
    dataset = filter_by_anchor(dataset)

    # Create windows
    windows = create_single_windows(dataset)
    windows = add_metadata(windows, dataset)

    # Split subjects
    subjects = get_unique_subjects(windows)
    if remove_bad_subjects:
        subjects = remove_subjects(subjects)

    train_subj, valid_subj, test_subj = split_subjects(subjects)

    # Split data
    train_set, valid_set, test_set = split_windows_by_subject(
        windows, train_subj, valid_subj, test_subj
    )

    # Create dataloaders
    return create_dataloaders(train_set, valid_set, test_set, batch_size=batch_size, num_workers=num_workers)
