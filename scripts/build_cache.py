#!/usr/bin/env python3
"""Universal cache builder for HBN dataset with flexible windowing.

Builds unified cache with release metadata for runtime split flexibility.

Usage:
    # Challenge 1: Fixed-length windows (for comparison)
    uv run python scripts/build_cache.py --challenge 1 --window-len 2.5 --stride 0.5

    # Challenge 1: Raw only (for stimulus-locked windowing at runtime)
    uv run python scripts/build_cache.py --challenge 1 --raw-only

    # Challenge 2: Multi-task with fixed windows
    uv run python scripts/build_cache.py --challenge 2 --window-len 2.0 --stride 1.0
"""

import argparse
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from cerebro.data.unified_cache import UniversalCacheManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


def build_cache(
    challenge: int,
    window_len_s: float = None,
    stride_s: float = None,
    raw_only: bool = False,
    mini: bool = False,
):
    """Build unified cache for specified challenge.

    Args:
        challenge: Challenge number (1 or 2)
        window_len_s: Window length in seconds (ignored if raw_only=True)
        stride_s: Window stride in seconds (ignored if raw_only=True)
        raw_only: If True, only build raw cache (Level 1)
        mini: If True, use mini dataset for testing
    """
    # Configuration
    HBN_ROOT = Path(os.getenv("HBN_ROOT", "/media/varun/OS/Users/varun/DATASETS/HBN"))
    CACHE_ROOT = Path(os.getenv("CACHE_PATH", "/home/varun/repos/cerebro/cache"))

    # All available releases (R1-R11, excluding R5 test set)
    ALL_RELEASES = ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"]

    # Challenge-specific configuration
    if challenge == 1:
        TASKS = ["contrastChangeDetection"]
        cache_name = "unified_challenge1"
        if window_len_s is None:
            window_len_s = 2.5
        if stride_s is None:
            stride_s = 0.5
    elif challenge == 2:
        TASKS = [
            "contrastChangeDetection",
            "DespicableMe", "DiaryOfAWimpyKid", "ThePresent",
            "FunwithFractals",
            "RestingState",
            "surroundSupp",
            "symbolSearch"
        ]
        cache_name = "unified_challenge2"
        if window_len_s is None:
            window_len_s = 2.0
        if stride_s is None:
            stride_s = 1.0
    else:
        raise ValueError(f"Challenge must be 1 or 2, got {challenge}")

    CACHE_PATH = CACHE_ROOT / cache_name

    logger.info("=" * 80)
    logger.info(f"Building Unified Cache for Challenge {challenge}")
    logger.info("=" * 80)
    logger.info(f"Data root: {HBN_ROOT}")
    logger.info(f"Cache root: {CACHE_PATH}")
    logger.info(f"Releases: {', '.join(ALL_RELEASES)}")
    logger.info(f"Tasks: {', '.join(TASKS)}")
    logger.info(f"Mini dataset: {mini}")
    if raw_only:
        logger.info("Mode: Raw cache only (Level 1)")
    else:
        logger.info(f"Mode: Raw + Windowed cache (Levels 1 + 2)")
        logger.info(f"Window: {window_len_s}s, Stride: {stride_s}s")
    logger.info("=" * 80)

    # Initialize cache manager
    cache = UniversalCacheManager(
        cache_root=str(CACHE_PATH),
        data_dir=str(HBN_ROOT),
        preprocessing_params={
            "sfreq": 100,  # Challenge data is 100Hz
            "bandpass": None,  # Already filtered (0.5-50Hz)
            "n_channels": 129,
            "standardize": False,
        }
    )

    # Step 1: Build raw cache for all releases
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Building Raw Cache (Level 1)")
    logger.info("=" * 80)
    logger.info("This will download and preprocess all recordings (~15-20 minutes first time)")

    cache.build_raw(
        dataset="hbn",
        releases=ALL_RELEASES,
        tasks=TASKS,
        mini=mini
    )

    # Query all recordings to get counts
    all_recordings = cache.query_raw(
        releases=ALL_RELEASES,
        tasks=TASKS,
        mini=mini
    )
    logger.info(f"✅ Raw cache complete: {len(all_recordings)} recordings across {len(ALL_RELEASES)} releases")

    # Step 2: Build window cache (optional)
    if not raw_only:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Building Window Cache (Level 2)")
        logger.info("=" * 80)
        logger.info(f"Creating windows: {window_len_s}s length, {stride_s}s stride")
        logger.info("This creates a memory-mapped cache for instant loading")

        # Build windowed dataset (this creates the memmap cache)
        windowed_ds = cache.get_windowed_dataset(
            recordings=all_recordings,
            window_len_s=window_len_s,
            stride_s=stride_s
        )

        logger.info(f"✅ Window cache complete: {len(windowed_ds)} windows")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ CACHE BUILD COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Cache location: {CACHE_PATH}")
    logger.info(f"Total recordings: {len(all_recordings)}")
    if not raw_only:
        logger.info(f"Total windows: {len(windowed_ds)}")
    logger.info("\nRelease breakdown:")
    for release in ALL_RELEASES:
        release_recs = cache.query_raw(releases=[release], tasks=TASKS, mini=mini)
        logger.info(f"  {release}: {len(release_recs)} recordings")

    logger.info("\nNext steps:")
    logger.info("1. Training notebooks load data via UniversalCacheManager")
    logger.info("2. Instant data loading (<10s) for all future runs!")
    logger.info("3. Change train/val splits by filtering releases at runtime")
    logger.info(f"4. Example: cache.query_raw(releases=['R1', 'R2'], tasks={TASKS})")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Build unified cache for HBN challenges with release metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Challenge 1: Raw only (for stimulus-locked windowing at runtime)
  python scripts/build_cache.py --challenge 1 --raw-only

  # Challenge 1: With fixed windows (for comparison)
  python scripts/build_cache.py --challenge 1 --window-len 2.5 --stride 0.5

  # Challenge 2: Multi-task with fixed windows
  python scripts/build_cache.py --challenge 2 --window-len 2.0 --stride 1.0

  # Mini dataset for testing
  python scripts/build_cache.py --challenge 1 --mini --raw-only
        """
    )

    parser.add_argument(
        "--challenge",
        type=int,
        required=True,
        choices=[1, 2],
        help="Challenge number (1 or 2)"
    )
    parser.add_argument(
        "--window-len",
        type=float,
        default=None,
        help="Window length in seconds (default: 2.5 for C1, 2.0 for C2)"
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=None,
        help="Window stride in seconds (default: 0.5 for C1, 1.0 for C2)"
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only build raw cache (Level 1), skip windowing"
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Use mini dataset for testing"
    )

    args = parser.parse_args()

    build_cache(
        challenge=args.challenge,
        window_len_s=args.window_len,
        stride_s=args.stride,
        raw_only=args.raw_only,
        mini=args.mini,
    )


if __name__ == "__main__":
    main()
