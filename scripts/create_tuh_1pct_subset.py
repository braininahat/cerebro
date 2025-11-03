#!/usr/bin/env python3
"""Create a 1% subset of ALL TUH datasets for prototyping.

Samples 1% of EDF files from each subdataset using hard links.
Preserves exact directory structure for compatibility.

Usage:
    uv run python scripts/create_tuh_1pct_subset.py

    # Custom percentage
    uv run python scripts/create_tuh_1pct_subset.py --percentage 0.5  # 0.5%
"""

import argparse
import random
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def sample_files_from_dataset(
    dataset_root: Path,
    output_root: Path,
    percentage: float,
    use_hardlinks: bool,
    seed: int
):
    """Sample percentage of files from a single TUH subdataset.

    Args:
        dataset_root: Root of subdataset (e.g., tuh_eeg, tuh_eeg_seizure)
        output_root: Where to create subset
        percentage: Percentage of files to sample (e.g., 1.0 for 1%)
        use_hardlinks: Use hard links instead of copying
        seed: Random seed

    Returns:
        Tuple of (n_files_sampled, total_size_mb)
    """
    random.seed(seed)

    # Find all EDF files
    edf_files = list(dataset_root.rglob("*.edf"))

    if not edf_files:
        logger.warning(f"No EDF files found in {dataset_root.name}")
        return 0, 0.0

    # Sample files
    n_to_sample = max(1, int(len(edf_files) * (percentage / 100)))
    sampled_files = random.sample(edf_files, n_to_sample)

    logger.info(f"  {dataset_root.name}: sampling {n_to_sample}/{len(edf_files)} files ({percentage}%)")

    total_size = 0
    link_method = "hard link" if use_hardlinks else "copy"

    for edf_file in sampled_files:
        # Compute relative path from dataset root
        rel_path = edf_file.relative_to(dataset_root)
        output_file = output_root / rel_path

        # Create parent directories
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Copy or hard link
        if use_hardlinks:
            try:
                output_file.hardlink_to(edf_file)
            except OSError:
                logger.warning(f"Hard link failed for {edf_file.name}, using copy")
                shutil.copy2(edf_file, output_file)
        else:
            shutil.copy2(edf_file, output_file)

        total_size += edf_file.stat().st_size

    return n_to_sample, total_size / (1024 * 1024)  # Convert to MB


def create_tuh_subset_all(
    tuh_root: Path,
    output_root: Path,
    percentage: float = 1.0,
    use_hardlinks: bool = True,
    seed: int = 42
):
    """Create subset of all TUH subdatasets.

    Args:
        tuh_root: Root directory containing all TUH subdatasets
        output_root: Where to create unified subset
        percentage: Percentage of files to sample from each dataset
        use_hardlinks: Use hard links to save space
        seed: Random seed
    """
    logger.info(f"Creating {percentage}% subset of all TUH datasets")
    logger.info(f"Source: {tuh_root}")
    logger.info(f"Output: {output_root}")
    logger.info(f"Method: {'hard links' if use_hardlinks else 'copy'}")

    # Find all TUH subdatasets (directories starting with "tuh_")
    tuh_datasets = [d for d in tuh_root.iterdir()
                    if d.is_dir() and d.name.startswith("tuh_") and d.name != "tuh_eeg_subset"]

    logger.info(f"\nFound {len(tuh_datasets)} TUH subdatasets:")
    for ds in tuh_datasets:
        n_files = len(list(ds.rglob("*.edf")))
        logger.info(f"  - {ds.name}: {n_files} files")

    # Create output root
    output_root.mkdir(parents=True, exist_ok=True)

    # Sample from each dataset
    logger.info(f"\nSampling {percentage}% from each dataset...")

    total_files = 0
    total_size_mb = 0
    dataset_stats = {}

    for dataset_dir in tuh_datasets:
        output_dataset = output_root / dataset_dir.name
        n_files, size_mb = sample_files_from_dataset(
            dataset_root=dataset_dir,
            output_root=output_dataset,
            percentage=percentage,
            use_hardlinks=use_hardlinks,
            seed=seed + hash(dataset_dir.name)  # Different seed per dataset
        )

        total_files += n_files
        total_size_mb += size_mb
        dataset_stats[dataset_dir.name] = {"files": n_files, "size_mb": size_mb}

    logger.info(f"\n✓ Created subset at {output_root}")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Total size: {total_size_mb / 1024:.2f} GB")
    logger.info(f"  Method: {'hard links' if use_hardlinks else 'copy'}")

    # Create README
    readme_content = f"""# TUH 1% Subset for Prototyping

Created by: scripts/create_tuh_1pct_subset.py
Seed: {seed}
Percentage: {percentage}%
Method: {'hard links' if use_hardlinks else 'copy'}

## Contents

Total: {total_files} EDF files ({total_size_mb / 1024:.2f} GB)

"""

    for ds_name in sorted(dataset_stats.keys()):
        stats = dataset_stats[ds_name]
        readme_content += f"- **{ds_name}**: {stats['files']} files ({stats['size_mb']:.1f} MB)\n"

    readme_content += f"""
## Structure

Each subdataset preserves its original directory structure:

```
tuh_subset/
├── tuh_eeg/
│   └── v2.0.1/edf/...
├── tuh_eeg_seizure/
│   └── v2.0.0/edf/...
├── tuh_eeg_abnormal/
│   └── v2.0.0/edf/...
└── ...
```

## Usage

```python
# Process all subdatasets
from pathlib import Path

tuh_subset = Path("{output_root}")
for dataset_dir in tuh_subset.glob("tuh_*"):
    print(f"Processing {{dataset_dir.name}}")
    # Convert to cache, etc.
```

## Scaling to Full Dataset

All code tested on this subset will work on the full dataset by changing the path.
On cluster, set TUH_ROOT environment variable to point to full dataset.

## Hard Links Note

This subset uses hard links to the original data. The files will remain accessible
even after deleting the original directories, but the space won't be freed until
both the original and subset are deleted.
"""

    readme_path = output_root / "README.md"
    readme_path.write_text(readme_content)
    logger.info(f"✓ Created README at {readme_path}")

    return dataset_stats


def main():
    parser = argparse.ArgumentParser(
        description="Create 1% subset of all TUH datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tuh-root",
        type=Path,
        default=Path("/home/varun/datasets/TUH"),
        help="Path to TUH root (contains tuh_eeg, tuh_eeg_seizure, etc.)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/varun/datasets/TUH/tuh_subset"),
        help="Where to create subset"
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=1.0,
        help="Percentage of files to sample (default: 1.0 for 1%%)"
    )
    parser.add_argument(
        "--use-hardlinks",
        action="store_true",
        default=True,
        help="Use hard links instead of copying (default: True)"
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
        shutil.rmtree(args.output)
        logger.info(f"Removed existing {args.output}")

    create_tuh_subset_all(
        tuh_root=args.tuh_root,
        output_root=args.output,
        percentage=args.percentage,
        use_hardlinks=args.use_hardlinks,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
