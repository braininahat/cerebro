"""Universal two-level cache system for EEG datasets.

This module provides a unified caching architecture that works across multiple datasets
(HBN, TUH, MOABB) with two levels:

Level 1 (Raw Cache):
- Stores preprocessed raw recordings per (subject × task × release)
- Enables flexible querying and composition
- Serves as source of truth for all window configurations

Level 2 (Window Cache):
- Stores windowed data derived from Level 1
- Multiple window configs coexist (2s, 4s, 8s, etc.)
- Zero-copy memory-mapped access for maximum I/O performance

Key Features:
- Subject-level splitting (prevents data leakage)
- Composable queries (arbitrary release/task/subject combinations)
- Memory efficient (zero-copy memory mapping)
- Fault tolerant (checkpoint manifests)
- Pre-buildable (CLI tool for cache generation)

Usage:
    from cerebro.data.unified_cache import UniversalCacheManager

    cache = UniversalCacheManager("data/cache")

    # Build raw cache
    cache.build_raw(
        dataset="hbn",
        releases=["R1", "R2"],
        tasks=["restingState"],
        mini=True
    )

    # Query and window
    recordings = cache.query_raw(releases=["R1"], tasks=["restingState"])
    train_ds = cache.get_windowed_dataset(recordings, window_len_s=2.0, stride_s=1.0)
"""

from cerebro.data.unified_cache.cache_manager import UniversalCacheManager
from cerebro.data.unified_cache.lazy_dataset import MemmapWindowDataset

__all__ = ["UniversalCacheManager", "MemmapWindowDataset"]
