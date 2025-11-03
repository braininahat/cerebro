#!/usr/bin/env python3
"""Test TUH dataset caching on subset.

Tests the full pipeline: EDF → Zarr → Windows → Training

Usage:
    # Test raw cache building
    uv run python scripts/test_tuh_cache.py --stage raw

    # Test windowing
    uv run python scripts/test_tuh_cache.py --stage windows

    # Test end-to-end (all stages)
    uv run python scripts/test_tuh_cache.py --stage all
"""

import argparse
import logging
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_raw_cache():
    """Test building raw cache from TUH subset."""
    import os
    from cerebro.data.unified_cache import UniversalCacheManager

    logger.info("=" * 80)
    logger.info("STAGE 1: Building Raw Cache from TUH Subset")
    logger.info("=" * 80)

    # Get TUH root from environment variable
    tuh_root = os.getenv("TUH_ROOT")
    if not tuh_root:
        raise ValueError("TUH_ROOT environment variable not set! Check your .env file")

    tuh_subset_path = Path(tuh_root) / "tuh_eeg_subset"
    logger.info(f"Using TUH subset at: {tuh_subset_path}")

    cache_mgr = UniversalCacheManager(
        cache_root="data/cache/tuh_test",
        preprocessing_params={
            "sfreq": 100,  # Resample to 100Hz
            "bandpass": [0.5, 50],  # 0.5-50Hz bandpass
            "standardize": False,  # Optional z-score normalization
            "n_channels": None,  # TUH has variable channels
        }
    )

    # Build cache for TUH subset
    cache_mgr.build_raw(
        dataset="tuh",
        releases=["subset"],  # Not used, but kept for interface
        tasks="all",  # Include all recording types (01_tcp_ar, 02_tcp_le, etc.)
        mini=False,
        tuh_path=str(tuh_subset_path)
    )

    logger.info("\n" + "=" * 80)
    logger.info("Raw Cache Status:")
    logger.info("=" * 80)
    cache_mgr.print_status()

    return cache_mgr


def test_windowing(cache_mgr=None):
    """Test windowing from raw cache."""
    from cerebro.data.unified_cache import UniversalCacheManager

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: Creating Windowed Dataset")
    logger.info("=" * 80)

    if cache_mgr is None:
        cache_mgr = UniversalCacheManager(
            cache_root="data/cache/tuh_test",
            preprocessing_params={
                "sfreq": 100,
                "bandpass": [0.5, 50],
                "standardize": False,
                "n_channels": None,
            }
        )

    # Query raw cache
    recordings = cache_mgr.query_raw(
        dataset="tuh",
        releases=None,  # All releases
        subjects=None,  # All subjects
        tasks=None,     # All tasks
    )

    logger.info(f"Found {len(recordings)} cached recordings")
    if len(recordings) == 0:
        logger.error("No recordings found! Run --stage raw first")
        return None

    # Create windowed dataset
    try:
        train_ds = cache_mgr.get_windowed_dataset(
            recordings=recordings,
            window_len_s=10.0,  # 10-second windows
            stride_s=5.0,       # 50% overlap
            crop_len_s=None,    # No temporal cropping
            mode='train'
        )

        logger.info("\n" + "=" * 80)
        logger.info("Windowed Dataset Stats:")
        logger.info("=" * 80)
        stats = train_ds.get_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        return train_ds

    except Exception as e:
        logger.error(f"Failed to create windowed dataset: {e}", exc_info=True)
        return None


def test_training(train_ds=None):
    """Test PyTorch DataLoader integration."""
    import torch
    from torch.utils.data import DataLoader

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: Testing PyTorch DataLoader")
    logger.info("=" * 80)

    if train_ds is None:
        logger.error("No dataset provided! Run --stage windows first")
        return

    # Create DataLoader
    loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"DataLoader created: {len(loader)} batches")

    # Test loading a few batches
    logger.info("Testing batch loading...")
    for i, batch in enumerate(loader):
        if i >= 3:  # Test first 3 batches
            break
        logger.info(f"  Batch {i+1}: shape={batch.shape}, dtype={batch.dtype}")

    logger.info("✓ DataLoader test complete!")


def test_space_savings():
    """Calculate space savings from compression."""
    import os
    import subprocess

    logger.info("\n" + "=" * 80)
    logger.info("Space Savings Analysis")
    logger.info("=" * 80)

    # Get TUH root from environment
    tuh_root = os.getenv("TUH_ROOT", "/home/varun/datasets/TUH")
    tuh_subset_path = Path(tuh_root) / "tuh_eeg_subset"

    # Original EDF size
    try:
        result = subprocess.run(
            ["du", "-sh", str(tuh_subset_path)],
            capture_output=True,
            text=True
        )
        original_size = result.stdout.split()[0]
        logger.info(f"Original EDF size: {original_size}")
    except Exception as e:
        logger.warning(f"Could not get original size: {e}")
        original_size = "unknown"

    # Cached Zarr size
    try:
        result = subprocess.run(
            ["du", "-sh", "data/cache/tuh_test/raw"],
            capture_output=True,
            text=True
        )
        zarr_size = result.stdout.split()[0]
        logger.info(f"Zarr cache size: {zarr_size}")
    except Exception as e:
        logger.warning(f"Could not get cache size: {e}")
        zarr_size = "unknown"

    # Windowed memmap size (if exists)
    try:
        result = subprocess.run(
            ["du", "-sh", "data/cache/tuh_test/windowed"],
            capture_output=True,
            text=True
        )
        window_size = result.stdout.split()[0]
        logger.info(f"Window cache size: {window_size}")
    except Exception as e:
        logger.info("Window cache not yet built")


def main():
    parser = argparse.ArgumentParser(description="Test TUH caching pipeline")
    parser.add_argument(
        "--stage",
        choices=["raw", "windows", "training", "all"],
        default="all",
        help="Which stage to test"
    )

    args = parser.parse_args()

    cache_mgr = None
    train_ds = None

    if args.stage in ["raw", "all"]:
        cache_mgr = test_raw_cache()
        test_space_savings()

    if args.stage in ["windows", "all"]:
        train_ds = test_windowing(cache_mgr)

    if args.stage in ["training", "all"]:
        test_training(train_ds)

    logger.info("\n" + "=" * 80)
    logger.info("✓ All tests complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
