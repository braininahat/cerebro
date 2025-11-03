"""Unit tests for GranularCacheManager.

Tests the per-release caching infrastructure including:
- Cache key generation (preprocessing params only)
- Per-release cache file paths
- Manifest tracking and status
- Cache hit/miss detection
- Save/load functionality
"""

import json
import pickle
import tempfile
from pathlib import Path

import pytest

from cerebro.data.cache_manager import GranularCacheManager


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_preprocessing_params():
    """Sample preprocessing parameters."""
    return {
        "sfreq": 100,
        "tasks": ["restingState", "contrastChangeDetection"],
        "mini": True,
        "n_channels": 129,
    }


@pytest.fixture
def cache_mgr(temp_cache_dir, sample_preprocessing_params):
    """Create GranularCacheManager instance."""
    return GranularCacheManager(
        cache_root=str(temp_cache_dir),
        preprocessing_params=sample_preprocessing_params
    )


def test_cache_key_generation(cache_mgr):
    """Test that cache key is generated from preprocessing params only."""
    # Cache key should not change if we create another manager with same params
    cache_mgr2 = GranularCacheManager(
        cache_root=cache_mgr.cache_root,
        preprocessing_params=cache_mgr.preprocessing_params
    )
    assert cache_mgr.cache_key_base == cache_mgr2.cache_key_base

    # Cache key SHOULD change if preprocessing params change
    different_params = cache_mgr.preprocessing_params.copy()
    different_params["sfreq"] = 200  # Different preprocessing
    cache_mgr3 = GranularCacheManager(
        cache_root=cache_mgr.cache_root,
        preprocessing_params=different_params
    )
    assert cache_mgr.cache_key_base != cache_mgr3.cache_key_base


def test_cache_key_deterministic(sample_preprocessing_params, temp_cache_dir):
    """Test that cache key is deterministic for same params."""
    mgr1 = GranularCacheManager(
        cache_root=str(temp_cache_dir),
        preprocessing_params=sample_preprocessing_params
    )
    mgr2 = GranularCacheManager(
        cache_root=str(temp_cache_dir),
        preprocessing_params=sample_preprocessing_params
    )
    assert mgr1.cache_key_base == mgr2.cache_key_base


def test_release_cache_path(cache_mgr):
    """Test per-release cache file path generation."""
    r1_path = cache_mgr._get_release_cache_path("R1")
    r2_path = cache_mgr._get_release_cache_path("R2")

    # Different releases should have different paths
    assert r1_path != r2_path

    # Paths should include release name and cache key
    assert "R1" in r1_path.name
    assert "R2" in r2_path.name
    assert cache_mgr.cache_key_base in r1_path.name
    assert cache_mgr.cache_key_base in r2_path.name

    # Should be .pkl files
    assert r1_path.suffix == ".pkl"
    assert r2_path.suffix == ".pkl"


def test_initial_manifest_empty(cache_mgr):
    """Test that newly created cache manager has empty manifest."""
    assert cache_mgr.manifest["releases"] == {}
    assert cache_mgr.manifest["cache_key"] == cache_mgr.cache_key_base


def test_is_cached_false_initially(cache_mgr):
    """Test that releases are not cached initially."""
    assert not cache_mgr.is_cached("R1")
    assert not cache_mgr.is_cached("R2")


def test_get_missing_releases(cache_mgr):
    """Test getting list of missing (uncached) releases."""
    requested = ["R1", "R2", "R3"]
    missing = cache_mgr.get_missing_releases(requested)
    assert set(missing) == {"R1", "R2", "R3"}

    # After marking R1 complete
    cache_mgr.mark_complete("R1")
    # Create dummy cache file so is_cached returns True
    cache_path = cache_mgr._get_release_cache_path("R1")
    cache_path.touch()

    missing = cache_mgr.get_missing_releases(requested)
    assert set(missing) == {"R2", "R3"}


def test_get_cached_releases(cache_mgr):
    """Test getting list of cached releases."""
    requested = ["R1", "R2", "R3"]

    # Initially none cached
    cached = cache_mgr.get_cached_releases(requested)
    assert cached == []

    # Mark R1 complete and create file
    cache_mgr.mark_complete("R1")
    cache_path = cache_mgr._get_release_cache_path("R1")
    cache_path.touch()

    cached = cache_mgr.get_cached_releases(requested)
    assert cached == ["R1"]


def test_save_and_load_release(cache_mgr):
    """Test saving and loading release data."""
    test_data = [{"sample": "data1"}, {"sample": "data2"}]

    # Save
    cache_mgr.save_release("R1", test_data)

    # Check file exists
    cache_path = cache_mgr._get_release_cache_path("R1")
    assert cache_path.exists()

    # Mark complete (required for is_cached to return True)
    cache_mgr.mark_complete("R1")

    # Load
    assert cache_mgr.is_cached("R1")
    loaded_data = cache_mgr.load_release("R1")
    assert loaded_data == test_data


def test_load_uncached_release_raises(cache_mgr):
    """Test that loading uncached release raises error."""
    with pytest.raises(FileNotFoundError):
        cache_mgr.load_release("R_nonexistent")


def test_mark_complete_updates_manifest(cache_mgr):
    """Test that marking complete updates manifest."""
    metadata = {"n_recordings": 20, "n_samples": 1000000}
    cache_mgr.mark_complete("R1", metadata=metadata)

    # Check manifest
    assert "R1" in cache_mgr.manifest["releases"]
    assert cache_mgr.manifest["releases"]["R1"]["status"] == "complete"
    assert cache_mgr.manifest["releases"]["R1"]["n_recordings"] == 20
    assert cache_mgr.manifest["releases"]["R1"]["n_samples"] == 1000000
    assert "timestamp" in cache_mgr.manifest["releases"]["R1"]

    # Check manifest file exists
    assert cache_mgr.manifest_path.exists()

    # Reload manifest from disk to verify persistence
    with open(cache_mgr.manifest_path, "r") as f:
        disk_manifest = json.load(f)
    assert disk_manifest["releases"]["R1"]["status"] == "complete"


def test_mark_failed_updates_manifest(cache_mgr):
    """Test that marking failed updates manifest."""
    error_msg = "Connection timeout"
    cache_mgr.mark_failed("R1", error_msg)

    # Check manifest
    assert "R1" in cache_mgr.manifest["releases"]
    assert cache_mgr.manifest["releases"]["R1"]["status"] == "failed"
    assert cache_mgr.manifest["releases"]["R1"]["error"] == error_msg
    assert "timestamp" in cache_mgr.manifest["releases"]["R1"]


def test_cache_stats(cache_mgr):
    """Test cache statistics."""
    # Initial state
    stats = cache_mgr.get_cache_stats()
    assert stats["total_releases"] == 0
    assert stats["cached_complete"] == 0
    assert stats["cached_failed"] == 0

    # Add some releases
    cache_mgr.mark_complete("R1")
    cache_mgr.mark_complete("R2")
    cache_mgr.mark_failed("R3", "error")

    stats = cache_mgr.get_cache_stats()
    assert stats["total_releases"] == 3
    assert stats["cached_complete"] == 2
    assert stats["cached_failed"] == 1


def test_manifest_persistence(cache_mgr, sample_preprocessing_params):
    """Test that manifest persists across manager instances."""
    # Mark releases in first manager
    cache_mgr.mark_complete("R1", metadata={"n_recordings": 10})
    cache_mgr.mark_failed("R2", "error")

    # Create new manager with same cache root
    cache_mgr2 = GranularCacheManager(
        cache_root=str(cache_mgr.cache_root),
        preprocessing_params=sample_preprocessing_params
    )

    # Should load manifest from disk
    assert "R1" in cache_mgr2.manifest["releases"]
    assert "R2" in cache_mgr2.manifest["releases"]
    assert cache_mgr2.manifest["releases"]["R1"]["status"] == "complete"
    assert cache_mgr2.manifest["releases"]["R2"]["status"] == "failed"


def test_cache_key_mismatch_warning(cache_mgr, sample_preprocessing_params, temp_cache_dir):
    """Test that cache key mismatch generates warning and updates manifest."""
    # Create manifest with old cache key
    cache_mgr.mark_complete("R1")

    # Create new manager with different preprocessing params (different cache key)
    different_params = sample_preprocessing_params.copy()
    different_params["sfreq"] = 200  # Different sfreq -> different cache key

    cache_mgr2 = GranularCacheManager(
        cache_root=str(temp_cache_dir),
        preprocessing_params=different_params
    )

    # Should update manifest with new cache key
    assert cache_mgr2.manifest["cache_key"] == cache_mgr2.cache_key_base
    assert cache_mgr2.manifest["cache_key"] != cache_mgr.cache_key_base


def test_is_cached_requires_file_and_manifest(cache_mgr):
    """Test that is_cached requires both manifest entry AND file existence."""
    # Mark complete but don't create file
    cache_mgr.mark_complete("R1")
    assert not cache_mgr.is_cached("R1")  # No file yet

    # Create file
    cache_path = cache_mgr._get_release_cache_path("R1")
    cache_path.touch()
    assert cache_mgr.is_cached("R1")  # Now both exist


def test_tasks_list_hashing(temp_cache_dir):
    """Test that task list order doesn't affect cache key (sorted internally)."""
    params1 = {
        "sfreq": 100,
        "tasks": ["restingState", "contrastChangeDetection"],
        "mini": True,
    }
    params2 = {
        "sfreq": 100,
        "tasks": ["contrastChangeDetection", "restingState"],  # Different order
        "mini": True,
    }

    mgr1 = GranularCacheManager(cache_root=str(temp_cache_dir), preprocessing_params=params1)
    mgr2 = GranularCacheManager(cache_root=str(temp_cache_dir), preprocessing_params=params2)

    # Cache keys should be identical (tasks are sorted before hashing)
    assert mgr1.cache_key_base == mgr2.cache_key_base


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
