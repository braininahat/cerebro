#!/usr/bin/env python3
"""Clear failed recordings from cache manifests.

Run this to clean up failed entries and retry them on next run.

Usage:
    python scripts/clear_failed_cache.py [data_dir]

If data_dir not provided, uses $EEG2025_DATA_ROOT or ./data
"""

import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cerebro.data.unified_cache import UniversalCacheManager

def main():
    """Clear failed recordings from both raw and window manifests."""

    # Get data directory
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path(os.getenv("EEG2025_DATA_ROOT", "./data"))

    cache_root = data_dir / "cache"
    if not cache_root.exists():
        print(f"❌ Cache directory not found: {cache_root}")
        print(f"\nUsage: python scripts/clear_failed_cache.py [data_dir]")
        return

    cache_mgr = UniversalCacheManager(
        cache_root=str(cache_root),
        preprocessing_params={
            "sfreq": 100,
            "bandpass": None,
            "n_channels": 129,
            "standardize": False,
        }
    )

    print("="*60)
    print("CLEARING FAILED CACHE ENTRIES")
    print("="*60)

    # Clear raw cache failures
    print("\n📁 Raw Cache Manifest:")
    cache_mgr.raw_manifest.print_status("Before Cleanup")
    cache_mgr.raw_manifest.clear_failed_recordings()
    cache_mgr.raw_manifest.print_status("After Cleanup")

    # Clear window cache failures
    print("\n📁 Window Cache Manifest:")
    cache_mgr.window_manifest.print_status("Before Cleanup")
    cache_mgr.window_manifest.clear_failed_recordings()
    cache_mgr.window_manifest.print_status("After Cleanup")

    print("\n✓ Cleanup complete! Failed recordings will be retried on next run.")

if __name__ == "__main__":
    main()
