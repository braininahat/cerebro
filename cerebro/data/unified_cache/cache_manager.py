"""Universal cache manager - main orchestration layer.

Provides high-level API for two-level caching:
- Level 1: Raw preprocessed recordings
- Level 2: Windowed data derived from Level 1
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from cerebro.data.unified_cache.manifest import (
    ManifestManager,
    RAW_MANIFEST_SCHEMA,
    WINDOW_MANIFEST_SCHEMA,
)

logger = logging.getLogger(__name__)


class UniversalCacheManager:
    """Universal cache manager for multi-level EEG data caching.

    Manages two-level cache architecture:
    - Level 1 (Raw): Preprocessed raw recordings per (subject × task × release)
    - Level 2 (Windows): Windowed data derived from Level 1

    Args:
        cache_root: Root directory for cache storage
        preprocessing_params: Dict of preprocessing parameters (sfreq, bandpass, etc.)

    Example:
        >>> cache = UniversalCacheManager("data/cache")
        >>> cache.build_raw(dataset="hbn", releases=["R1"], tasks=["restingState"], mini=True)
        >>> recordings = cache.query_raw(releases=["R1"], tasks=["restingState"])
        >>> train_ds = cache.get_windowed_dataset(recordings, window_len_s=2.0, stride_s=1.0)
    """

    def __init__(
        self,
        cache_root: str,
        preprocessing_params: Optional[Dict[str, Any]] = None,
    ):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Default preprocessing params
        if preprocessing_params is None:
            preprocessing_params = {
                "sfreq": 100,
                "bandpass": None,  # [0.5, 50] or None
                "n_channels": 129,
                "standardize": False,
            }
        self.preprocessing_params = preprocessing_params
        self.preprocessing_hash = self._compute_preprocessing_hash()

        # Create cache subdirectories
        self.raw_cache_dir = self.cache_root / "raw"
        self.window_cache_dir = self.cache_root / "windowed"
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        self.window_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize manifest managers
        self.raw_manifest = ManifestManager(
            self.raw_cache_dir / "raw_manifest.parquet",
            schema=RAW_MANIFEST_SCHEMA
        )
        self.window_manifest = ManifestManager(
            self.window_cache_dir / "window_manifest.parquet",
            schema=WINDOW_MANIFEST_SCHEMA
        )

        logger.info(f"Initialized UniversalCacheManager")
        logger.info(f"  Cache root: {self.cache_root}")
        logger.info(f"  Preprocessing hash: {self.preprocessing_hash}")

    def _compute_preprocessing_hash(self) -> str:
        """Compute hash of preprocessing parameters.

        Returns:
            MD5 hash (first 8 chars)
        """
        # Sort for deterministic hashing
        sorted_params = sorted(self.preprocessing_params.items())
        param_str = str(sorted_params)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    def _get_recording_id(
        self,
        dataset: str,
        release: str,
        subject: str,
        task: str,
        mini: bool
    ) -> str:
        """Generate unique recording ID.

        Args:
            dataset: Dataset name (e.g., "hbn", "tuh")
            release: Release ID (e.g., "R1")
            subject: Subject ID
            task: Task name
            mini: Mini dataset flag

        Returns:
            Unique recording ID string
        """
        mini_str = "mini" if mini else "full"
        return f"{dataset}_{release}_{subject}_{task}_{mini_str}"

    def _get_raw_zarr_path(self, recording_id: str) -> Path:
        """Get Zarr path for raw recording.

        Args:
            recording_id: Unique recording identifier

        Returns:
            Path to Zarr directory
        """
        filename = f"{recording_id}_{self.preprocessing_hash}.zarr"
        return self.raw_cache_dir / "recordings" / filename

    def _get_window_config_id(
        self,
        window_len_s: float,
        stride_s: float,
        mini: bool
    ) -> str:
        """Generate window configuration ID.

        Args:
            window_len_s: Window length in seconds
            stride_s: Stride in seconds
            mini: Mini dataset flag (included to prevent mini/full collision)

        Returns:
            Config ID string
        """
        # Format to avoid float precision issues
        win_str = f"{int(window_len_s*10)}".replace(".", "p")
        stride_str = f"{int(stride_s*10)}".replace(".", "p")
        mini_str = "mini" if mini else "full"
        return f"windows_{self.preprocessing_hash}_win{win_str}_stride{stride_str}_{mini_str}"

    def _get_window_zarr_path(self, window_config_id: str) -> Path:
        """Get Zarr path for windowed data.

        Args:
            window_config_id: Window configuration ID

        Returns:
            Path to Zarr directory
        """
        return self.window_cache_dir / "windows" / f"{window_config_id}.zarr"

    def _get_window_metadata_path(self, window_config_id: str) -> Path:
        """Get metadata path for windowed data.

        Args:
            window_config_id: Window configuration ID

        Returns:
            Path to metadata Parquet file
        """
        return self.window_cache_dir / "windows" / f"{window_config_id}_meta.parquet"

    # ==================== Level 1: Raw Cache ====================

    def build_raw(
        self,
        dataset: str,
        releases: List[str],
        tasks: List[str],
        mini: bool = False,
        **kwargs
    ):
        """Build raw cache for specified data.

        Args:
            dataset: Dataset name ("hbn", "tuh", etc.)
            releases: List of release IDs
            tasks: List of task names or "all"
            mini: Use mini dataset
            **kwargs: Additional dataset-specific parameters
        """
        from cerebro.data.unified_cache.raw_cache import RawCacheBuilder

        builder = RawCacheBuilder(self)
        builder.build(dataset, releases, tasks, mini, **kwargs)

    def query_raw(
        self,
        dataset: Optional[str] = None,
        releases: Optional[List[str]] = None,
        subjects: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        mini: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Query raw cache manifest.

        Args:
            dataset: Dataset name filter
            releases: Release ID filter
            subjects: Subject ID filter
            tasks: Task name filter
            mini: Mini dataset filter

        Returns:
            DataFrame of matching recordings
        """
        return self.raw_manifest.query(
            dataset=dataset,
            releases=releases,
            subjects=subjects,
            tasks=tasks,
            mini=mini,
            status="complete"
        )

    # ==================== Level 2: Window Cache ====================

    def get_windowed_dataset(
        self,
        recordings: pd.DataFrame,
        window_len_s: float,
        stride_s: float,
        crop_len_s: Optional[float] = None,
        mode: str = 'train'
    ):
        """Get windowed dataset (builds from raw cache if missing).

        Args:
            recordings: DataFrame from query_raw()
            window_len_s: Window length in seconds
            stride_s: Stride in seconds
            crop_len_s: Optional crop length for augmentation
            mode: 'train' or 'val' (affects cropping strategy)

        Returns:
            LazyZarrWindowDataset instance
        """
        from cerebro.data.unified_cache.window_cache import WindowCacheBuilder

        builder = WindowCacheBuilder(self)
        return builder.get_or_build(recordings, window_len_s, stride_s, crop_len_s, mode)

    # ==================== Status & Utilities ====================

    def print_status(self):
        """Print cache status summary."""
        self.raw_manifest.print_status("Raw Cache Status")
        self.window_manifest.print_status("Window Cache Status")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with raw and window cache stats
        """
        return {
            "raw": self.raw_manifest.get_stats(),
            "window": self.window_manifest.get_stats(),
            "preprocessing_hash": self.preprocessing_hash,
            "cache_root": str(self.cache_root),
        }
