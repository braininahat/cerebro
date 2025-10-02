"""Utility to cache all EEG2025 mini and full releases.

Run with the desired cache root via `EEG2025_DATA_DIR`. The script will:
1. Download all mini releases (EEGLAB `.set` format, ~3-4 GB each)
2. Download all full releases (larger BIDS sets, tens of GB each)
"""

import os
from pathlib import Path

from eegdash.dataset import EEGChallengeDataset


def main() -> None:
    data_root = Path(os.environ.get("EEG2025_DATA_DIR", "data")).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    releases = [f"R{i}" for i in range(1, 12)]
    task = "contrastChangeDetection"

    print("=== Caching mini releases (EEGLAB .set) ===")
    for release in releases:
        print(f"Downloading {release}_mini_L100 …")
        EEGChallengeDataset(
            task=task,
            release=release,
            mini=True,
            cache_dir=data_root,
        )

    full_root = data_root / "full"
    full_root.mkdir(parents=True, exist_ok=True)
    print("\n=== Caching full releases (BIDS) ===")
    for release in releases:
        print(f"Downloading {release} full release …")
        EEGChallengeDataset(
            task=task,
            release=release,
            mini=False,
            cache_dir=full_root,
        )

    print("\nAll requested releases should now be cached.")


if __name__ == "__main__":
    main()
