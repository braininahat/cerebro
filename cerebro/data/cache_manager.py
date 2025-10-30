"""Granular cache manager for per-release EEG data caching.

This module provides infrastructure for reusable per-release caching that avoids
regenerating cache when expanding datasets (e.g., R1-R4 → R1-R11).

Key Design Principles:
- Cache at release level (not monolithic combinations)
- Cache keys based on preprocessing params only (not data selection like releases/seed)
- Manifest-based tracking for fault tolerance and status monitoring
- Generic interface usable across all data modules (LaBraM, JEPA, Movies, etc.)

Example Usage:
    cache_mgr = GranularCacheManager(
        cache_root="data/cache/labram_pretrain",
        preprocessing_params={
            "sfreq": 100,
            "tasks": ["restingState"],
            "mini": True,
            "n_channels": 129
        }
    )

    # Check what needs loading
    missing = cache_mgr.get_missing_releases(["R1", "R2", "R3"])

    # Load from cache
    for release in ["R1", "R2"]:
        if cache_mgr.is_cached(release):
            data = cache_mgr.load_release(release)

    # Save new data
    cache_mgr.save_release("R3", raw_objects)
    cache_mgr.mark_complete("R3", metadata={"n_recordings": 20})
"""

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class GranularCacheManager:
    """Manages per-release caching with manifest tracking.

    Args:
        cache_root: Root directory for this module's cache (e.g., "data/cache/labram_pretrain")
        preprocessing_params: Dict of preprocessing parameters that affect cache compatibility
                              (e.g., {"sfreq": 100, "tasks": ["restingState"], "mini": True})
                              NOTE: Should NOT include data selection params (releases, seed, splits)
    """

    def __init__(
        self,
        cache_root: str,
        preprocessing_params: Dict[str, Any]
    ):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.preprocessing_params = preprocessing_params
        self.cache_key_base = self._generate_cache_key()

        # Manifest tracking
        self.manifest_path = self.cache_root / "manifest.json"
        self.manifest = self._load_manifest()

    def _generate_cache_key(self) -> str:
        """Generate cache key from preprocessing params.

        Returns:
            Cache key suffix (e.g., "sfreq100_tasks{hash}_miniTrue")
        """
        # Sort for deterministic hashing
        sorted_params = sorted(self.preprocessing_params.items())

        # Build key parts
        parts = []
        for key, value in sorted_params:
            if key == "tasks" and isinstance(value, list):
                # Hash task list for brevity
                tasks_str = "_".join(sorted(value))
                tasks_hash = hashlib.md5(tasks_str.encode()).hexdigest()[:8]
                parts.append(f"tasks{tasks_hash}")
            elif isinstance(value, bool):
                parts.append(f"{key}{value}")
            else:
                parts.append(f"{key}{value}")

        return "_".join(parts)

    def _get_release_cache_path(self, release: str) -> Path:
        """Get cache file path for a specific release.

        Args:
            release: Release name (e.g., "R1", "R2")

        Returns:
            Path to cache file (e.g., "R1_sfreq100_tasks{hash}_miniTrue.pkl")
        """
        filename = f"{release}_{self.cache_key_base}.pkl"
        return self.cache_root / filename

    def _load_manifest(self) -> Dict:
        """Load cache manifest from disk.

        Returns:
            Manifest dict with structure:
            {
                "cache_key": "sfreq100_tasks{hash}_miniTrue",
                "preprocessing_params": {...},
                "releases": {
                    "R1": {"status": "complete", "timestamp": "...", "n_recordings": 20},
                    "R2": {"status": "failed", "error": "..."},
                    ...
                }
            }
        """
        if not self.manifest_path.exists():
            return {
                "cache_key": self.cache_key_base,
                "preprocessing_params": self.preprocessing_params,
                "releases": {}
            }

        try:
            with open(self.manifest_path, "r") as f:
                manifest = json.load(f)

            # Validate cache key compatibility
            if manifest.get("cache_key") != self.cache_key_base:
                logger.warning(
                    f"Cache key mismatch! Manifest has '{manifest.get('cache_key')}', "
                    f"but current params generate '{self.cache_key_base}'. "
                    f"Manifest will be updated."
                )
                manifest["cache_key"] = self.cache_key_base
                manifest["preprocessing_params"] = self.preprocessing_params

            return manifest

        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}. Creating new manifest.")
            return {
                "cache_key": self.cache_key_base,
                "preprocessing_params": self.preprocessing_params,
                "releases": {}
            }

    def _save_manifest(self):
        """Save manifest to disk."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def is_cached(self, release: str) -> bool:
        """Check if release is cached and complete.

        Args:
            release: Release name (e.g., "R1")

        Returns:
            True if release is cached and marked complete
        """
        # Check manifest status
        release_info = self.manifest["releases"].get(release, {})
        if release_info.get("status") != "complete":
            return False

        # Verify file exists
        cache_path = self._get_release_cache_path(release)
        return cache_path.exists()

    def get_missing_releases(self, requested_releases: List[str]) -> List[str]:
        """Get list of releases that need to be loaded/cached.

        Args:
            requested_releases: List of release names (e.g., ["R1", "R2", "R3"])

        Returns:
            List of releases not yet cached (e.g., ["R3"])
        """
        return [r for r in requested_releases if not self.is_cached(r)]

    def get_cached_releases(self, requested_releases: List[str]) -> List[str]:
        """Get list of releases that are already cached.

        Args:
            requested_releases: List of release names

        Returns:
            List of releases that are cached
        """
        return [r for r in requested_releases if self.is_cached(r)]

    def load_release(self, release: str) -> Any:
        """Load cached data for a release.

        Args:
            release: Release name (e.g., "R1")

        Returns:
            Cached data (typically list of MNE Raw objects)

        Raises:
            FileNotFoundError: If release is not cached
        """
        if not self.is_cached(release):
            raise FileNotFoundError(
                f"Release {release} not cached. "
                f"Run setup() to load and cache it first."
            )

        cache_path = self._get_release_cache_path(release)
        logger.info(f"Loading {release} from cache: {cache_path.name}")

        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def save_release(self, release: str, data: Any):
        """Save data for a release to cache.

        Args:
            release: Release name (e.g., "R1")
            data: Data to cache (typically list of MNE Raw objects)
        """
        cache_path = self._get_release_cache_path(release)
        logger.info(f"Caching {release} to: {cache_path.name}")

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def mark_complete(self, release: str, metadata: Optional[Dict] = None):
        """Mark a release as successfully cached.

        Args:
            release: Release name (e.g., "R1")
            metadata: Optional metadata (e.g., {"n_recordings": 20, "n_samples": 1000000})
        """
        self.manifest["releases"][release] = {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        self._save_manifest()
        logger.info(f"✓ Marked {release} as complete in manifest")

    def mark_failed(self, release: str, error: str):
        """Mark a release as failed during caching.

        Args:
            release: Release name (e.g., "R1")
            error: Error message
        """
        self.manifest["releases"][release] = {
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "error": error
        }
        self._save_manifest()
        logger.warning(f"✗ Marked {release} as failed in manifest: {error}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics:
            {
                "total_releases": 10,
                "cached_complete": 8,
                "cached_failed": 1,
                "missing": 1,
                "cache_key": "sfreq100_tasks{hash}_miniTrue"
            }
        """
        releases = self.manifest["releases"]
        complete = sum(1 for r in releases.values() if r.get("status") == "complete")
        failed = sum(1 for r in releases.values() if r.get("status") == "failed")

        return {
            "total_releases": len(releases),
            "cached_complete": complete,
            "cached_failed": failed,
            "missing": len(releases) - complete - failed,
            "cache_key": self.cache_key_base
        }

    def print_status(self, requested_releases: Optional[List[str]] = None):
        """Print cache status to logger.

        Args:
            requested_releases: Optional list of releases to check status for
        """
        logger.info(f"Cache key: {self.cache_key_base}")
        logger.info(f"Cache root: {self.cache_root}")

        if requested_releases:
            cached = self.get_cached_releases(requested_releases)
            missing = self.get_missing_releases(requested_releases)

            logger.info(f"Requested: {len(requested_releases)} releases")
            logger.info(f"  ✓ Cached: {len(cached)} releases: {cached}")
            logger.info(f"  ✗ Missing: {len(missing)} releases: {missing}")
        else:
            stats = self.get_cache_stats()
            logger.info(f"Total tracked: {stats['total_releases']} releases")
            logger.info(f"  ✓ Complete: {stats['cached_complete']}")
            logger.info(f"  ✗ Failed: {stats['cached_failed']}")
            logger.info(f"  ? Missing: {stats['missing']}")
