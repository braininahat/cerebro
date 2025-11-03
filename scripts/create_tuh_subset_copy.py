#!/usr/bin/env python3
"""Create a proper TUH subset by COPYING files (not symlinking).

Creates a subset that preserves exact directory structure for braindecode compatibility.
Uses hard links if possible (same filesystem), otherwise copies files.

Usage:
    uv run python scripts/create_tuh_subset_copy.py --n-subjects 50

    # Use hard links (saves space if same filesystem)
    uv run python scripts/create_tuh_subset_copy.py --n-subjects 50 --use-hardlinks
"""

import argparse
import random
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def create_tuh_subset_with_copies(
    tuh_root: Path,
    output_root: Path,
    n_subjects: int = 50,
    use_hardlinks: bool = False,
    seed: int = 42
):
    """Create a subset of TUH dataset by copying files (preserves structure).

    Args:
        tuh_root: Root of TUH dataset (e.g., /home/varun/datasets/TUH/tuh_eeg)
        output_root: Where to create subset (e.g., /home/varun/datasets/TUH/tuh_eeg_subset)
        n_subjects: Number of subjects to include
        use_hardlinks: Use hard links instead of copying (saves space, requires same filesystem)
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

    # Create output structure (preserve version/edf structure)
    output_edf = output_root / "v2.0.1" / "edf"
    output_edf.mkdir(parents=True, exist_ok=True)

    # Copy/link files
    total_sessions = 0
    total_files = 0
    total_size_mb = 0

    link_method = "hard link" if use_hardlinks else "copy"
    logger.info(f"Using {link_method} for files...")

    for subject_dir in sampled_subjects:
        # Relative path: {age_group}/{subject_id}
        age_group = subject_dir.parent.name
        subject_id = subject_dir.name

        # Create age group dir
        output_age_group = output_edf / age_group
        output_age_group.mkdir(exist_ok=True)

        # Create subject directory
        output_subject = output_age_group / subject_id
        output_subject.mkdir(exist_ok=True)

        # Copy entire subject tree (sessions → recording types → EDFs)
        for session_dir in subject_dir.iterdir():
            if not session_dir.is_dir() or not session_dir.name.startswith('s'):
                continue

            total_sessions += 1
            output_session = output_subject / session_dir.name
            output_session.mkdir(exist_ok=True)

            # Copy recording type directories
            for rec_type_dir in session_dir.iterdir():
                if not rec_type_dir.is_dir():
                    continue

                output_rec_type = output_session / rec_type_dir.name
                output_rec_type.mkdir(exist_ok=True)

                # Copy/link EDF files
                for edf_file in rec_type_dir.glob("*.edf"):
                    output_edf_file = output_rec_type / edf_file.name

                    if use_hardlinks:
                        try:
                            # Hard link (same filesystem only)
                            output_edf_file.hardlink_to(edf_file)
                        except OSError:
                            # Fall back to copy if hard link fails
                            logger.warning(f"Hard link failed for {edf_file.name}, using copy")
                            shutil.copy2(edf_file, output_edf_file)
                    else:
                        # Regular copy
                        shutil.copy2(edf_file, output_edf_file)

                    total_files += 1
                    total_size_mb += edf_file.stat().st_size / (1024 * 1024)

    logger.info(f"✓ Created subset at {output_root}")
    logger.info(f"  Subjects: {len(sampled_subjects)}")
    logger.info(f"  Sessions: {total_sessions}")
    logger.info(f"  EDF files: {total_files}")
    logger.info(f"  Total size: {total_size_mb / 1024:.2f} GB")
    logger.info(f"  Method: {link_method}")

    # Create README
    readme = output_root / "README.md"
    readme.write_text(f"""# TUH EEG Subset for Prototyping

Created by: scripts/create_tuh_subset_copy.py
Seed: {seed}
Method: {link_method}

## Contents
- Subjects: {len(sampled_subjects)} (sampled from {len(all_subjects)} total)
- Sessions: {total_sessions}
- EDF files: {total_files}
- Size: {total_size_mb / 1024:.2f} GB

## Structure
This subset preserves the exact directory structure of the full TUH dataset:
```
tuh_eeg_subset/
└── v2.0.1/
    └── edf/
        ├── 020/  (age group)
        │   └── subject_id/
        │       └── s001_2004/  (session)
        │           └── 02_tcp_le/  (recording type)
        │               └── *.edf
        └── 021/
            └── ...
```

## Usage
Use this subset exactly like the full dataset:

```python
from braindecode.datasets import TUH

# Use subset for prototyping
tuh = TUH(path="{output_root}", n_jobs=8)

# When ready, use full dataset (on cluster)
# tuh = TUH(path="/path/to/full/tuh_eeg", n_jobs=32)
```

## Braindecode Compatibility
✅ Preserves exact TUH directory structure
✅ braindecode can parse file paths correctly
✅ All metadata (age, gender, session dates) intact

## Scaling to Full Dataset
All code tested on this subset will work on the full dataset by changing the path.
On cluster, set TUH_ROOT environment variable to point to full dataset.
""")

    logger.info(f"✓ Created README at {readme}")

    # Save subject list for reproducibility
    subject_list_file = output_root / "sampled_subjects.txt"
    subject_list_file.write_text("\n".join([s.name for s in sampled_subjects]))
    logger.info(f"✓ Saved subject list to {subject_list_file}")


def main():
    parser = argparse.ArgumentParser(description="Create TUH EEG subset by copying files")
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
        "--use-hardlinks",
        action="store_true",
        help="Use hard links instead of copying (saves space, requires same filesystem)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Confirm if output exists
    if args.output.exists():
        response = input(f"{args.output} already exists. Remove and recreate? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            return
        import shutil
        shutil.rmtree(args.output)
        logger.info(f"Removed existing {args.output}")

    create_tuh_subset_with_copies(
        tuh_root=args.tuh_root,
        output_root=args.output,
        n_subjects=args.n_subjects,
        use_hardlinks=args.use_hardlinks,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
