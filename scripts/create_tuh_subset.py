#!/usr/bin/env python3
"""Create a prototype subset of TUH EEG dataset for local testing.

Creates symlinks to a subset of subjects to avoid data duplication.
Full pipeline can be tested on subset, then scaled to full dataset on cluster.

Usage:
    uv run python scripts/create_tuh_subset.py --n-subjects 50
"""

import argparse
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def create_tuh_subset(
    tuh_root: Path,
    output_root: Path,
    n_subjects: int = 50,
    seed: int = 42
):
    """Create a subset of TUH dataset via symlinks.

    Args:
        tuh_root: Root of TUH dataset (e.g., /home/varun/datasets/TUH/tuh_eeg)
        output_root: Where to create subset (e.g., /home/varun/datasets/TUH/tuh_eeg_subset)
        n_subjects: Number of subjects to include
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Find all subjects
    edf_root = tuh_root / "v2.0.1" / "edf"
    if not edf_root.exists():
        raise ValueError(f"TUH EDF root not found: {edf_root}")

    # Get all subject directories (format: /edf/{age_group}/{subject_id})
    all_subjects = []
    for age_group_dir in sorted(edf_root.iterdir()):
        if age_group_dir.is_dir():
            for subject_dir in age_group_dir.iterdir():
                if subject_dir.is_dir():
                    all_subjects.append(subject_dir)

    logger.info(f"Found {len(all_subjects)} total subjects in {tuh_root.name}")

    # Sample random subjects
    if n_subjects > len(all_subjects):
        logger.warning(f"Requested {n_subjects} subjects but only {len(all_subjects)} available")
        n_subjects = len(all_subjects)

    sampled_subjects = random.sample(all_subjects, n_subjects)
    logger.info(f"Sampled {len(sampled_subjects)} subjects for subset")

    # Create output structure
    output_edf = output_root / "v2.0.1" / "edf"
    output_edf.mkdir(parents=True, exist_ok=True)

    # Create symlinks
    total_sessions = 0
    total_files = 0

    for subject_dir in sampled_subjects:
        # Relative path: {age_group}/{subject_id}
        age_group = subject_dir.parent.name
        subject_id = subject_dir.name

        # Create age group dir if needed
        output_age_group = output_edf / age_group
        output_age_group.mkdir(exist_ok=True)

        # Create symlink to subject directory
        output_subject = output_age_group / subject_id
        if not output_subject.exists():
            output_subject.symlink_to(subject_dir, target_is_directory=True)

        # Count sessions and files
        for session_dir in subject_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('s'):
                total_sessions += 1
                for rec_type_dir in session_dir.iterdir():
                    if rec_type_dir.is_dir():
                        total_files += sum(1 for f in rec_type_dir.glob("*.edf"))

    logger.info(f"✓ Created subset at {output_root}")
    logger.info(f"  Subjects: {len(sampled_subjects)}")
    logger.info(f"  Sessions: {total_sessions}")
    logger.info(f"  EDF files: {total_files}")
    logger.info(f"  Size: ~{total_files * 20 / 1000:.1f} GB (estimated)")

    # Create README
    readme = output_root / "README.md"
    readme.write_text(f"""# TUH EEG Subset for Prototyping

Created by: scripts/create_tuh_subset.py
Seed: {seed}

## Contents
- Subjects: {len(sampled_subjects)} (sampled from {len(all_subjects)} total)
- Sessions: {total_sessions}
- EDF files: {total_files}
- Estimated size: ~{total_files * 20 / 1000:.1f} GB

## Structure
This subset uses **symlinks** to the original dataset, so it uses minimal disk space.
The full directory structure is preserved for compatibility with TUH loaders.

## Usage
Use this subset exactly like the full dataset:

```python
from braindecode.datasets import TUH

# Use subset for prototyping
tuh = TUH(path="{output_root}", n_jobs=8)

# When ready, use full dataset (on cluster)
# tuh = TUH(path="/path/to/full/tuh_eeg", n_jobs=32)
```

## Scaling to Full Dataset
All code tested on this subset will work on the full dataset by changing the path.
""")

    logger.info(f"✓ Created README at {readme}")


def main():
    parser = argparse.ArgumentParser(description="Create TUH EEG subset for prototyping")
    parser.add_argument(
        "--tuh-root",
        type=Path,
        default=Path("/home/varun/datasets/TUH/tuh_eeg"),
        help="Path to full TUH dataset"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/varun/datasets/TUH/tuh_eeg_subset"),
        help="Where to create subset"
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=50,
        help="Number of subjects to sample"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    create_tuh_subset(
        tuh_root=args.tuh_root,
        output_root=args.output,
        n_subjects=args.n_subjects,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
