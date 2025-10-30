"""Manifest operations for tracking cache status.

Provides Parquet-based manifest tracking for both raw and window caches.
Manifests enable efficient querying, fault tolerance, and progress tracking.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ManifestManager:
    """Manages Parquet manifest files for cache tracking.

    Provides operations for:
    - Loading/saving manifests
    - Querying with complex filters
    - Marking completion/failure
    - Tracking progress

    Args:
        manifest_path: Path to manifest Parquet file
        schema: Dict defining column names and types
    """

    def __init__(self, manifest_path: Path, schema: Optional[Dict[str, Any]] = None):
        self.manifest_path = Path(manifest_path)
        self.schema = schema or {}
        self.manifest = self._load_or_create()

    def _load_or_create(self) -> pd.DataFrame:
        """Load manifest from disk or create empty DataFrame."""
        if self.manifest_path.exists():
            try:
                manifest = pd.read_parquet(self.manifest_path)
                logger.info(f"Loaded manifest: {self.manifest_path.name} ({len(manifest)} entries)")
                return manifest
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}. Creating new manifest.")

        # Create empty DataFrame with schema
        return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in self.schema.items()})

    def save(self):
        """Save manifest to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.to_parquet(self.manifest_path, index=False)

    def query(
        self,
        dataset: Optional[str] = None,
        releases: Optional[List[str]] = None,
        subjects: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        mini: Optional[bool] = None,
        status: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query manifest with filters.

        Args:
            dataset: Dataset name (e.g., "hbn", "tuh")
            releases: List of release IDs (e.g., ["R1", "R2"])
            subjects: List of subject IDs
            tasks: List of task names
            mini: Mini dataset flag
            status: Entry status ("complete", "failed", etc.)

        Returns:
            Filtered DataFrame
        """
        mask = pd.Series([True] * len(self.manifest))

        if dataset and "dataset" in self.manifest.columns:
            mask &= self.manifest["dataset"] == dataset

        if releases and "release" in self.manifest.columns:
            mask &= self.manifest["release"].isin(releases)

        if subjects and "subject" in self.manifest.columns:
            mask &= self.manifest["subject"].isin(subjects)

        if tasks and "task" in self.manifest.columns:
            mask &= self.manifest["task"].isin(tasks)

        if mini is not None and "mini" in self.manifest.columns:
            mask &= self.manifest["mini"] == mini

        if status and "status" in self.manifest.columns:
            mask &= self.manifest["status"] == status

        return self.manifest[mask].copy()

    def is_processed(self, recording_id: str) -> bool:
        """Check if recording has been processed.

        Args:
            recording_id: Unique recording identifier

        Returns:
            True if recording exists in manifest with status='complete'
        """
        if "recording_id" not in self.manifest.columns:
            return False

        mask = (
            (self.manifest["recording_id"] == recording_id) &
            (self.manifest["status"] == "complete")
        )
        return mask.any()

    def mark_complete(self, entry: Dict[str, Any]):
        """Mark entry as complete in manifest.

        Deduplicates by removing any existing entries for this recording_id
        before appending. This prevents duplicate rows when retrying failed recordings.

        Args:
            entry: Dict with manifest columns + values
        """
        entry["status"] = "complete"
        entry["timestamp"] = datetime.now().isoformat()

        # Remove any existing entries for this recording_id to prevent duplicates
        if "recording_id" in entry and "recording_id" in self.manifest.columns:
            self.manifest = self.manifest[
                self.manifest["recording_id"] != entry["recording_id"]
            ]

        # Append to manifest
        new_row = pd.DataFrame([entry])
        self.manifest = pd.concat([self.manifest, new_row], ignore_index=True)

    def mark_failed(self, entry: Dict[str, Any], error: str):
        """Mark entry as failed in manifest.

        Deduplicates by removing any existing entries for this recording_id
        before appending. This prevents duplicate rows when retrying failed recordings.

        Args:
            entry: Dict with manifest columns + values
            error: Error message
        """
        entry["status"] = "failed"
        entry["error_msg"] = str(error)[:500]  # Truncate long errors
        entry["timestamp"] = datetime.now().isoformat()

        # Remove any existing entries for this recording_id to prevent duplicates
        if "recording_id" in entry and "recording_id" in self.manifest.columns:
            self.manifest = self.manifest[
                self.manifest["recording_id"] != entry["recording_id"]
            ]

        # Append to manifest
        new_row = pd.DataFrame([entry])
        self.manifest = pd.concat([self.manifest, new_row], ignore_index=True)

    def get_missing_recordings(
        self,
        all_recording_ids: List[str]
    ) -> List[str]:
        """Get list of recordings not yet processed.

        Args:
            all_recording_ids: List of all recording IDs to check

        Returns:
            List of missing recording IDs
        """
        if "recording_id" not in self.manifest.columns:
            return all_recording_ids

        completed = set(
            self.manifest[self.manifest["status"] == "complete"]["recording_id"].values
        )
        return [rid for rid in all_recording_ids if rid not in completed]

    def clear_failed_recordings(self):
        """Remove all failed entries from manifest to enable retry.

        This cleans up the manifest by removing failed recordings, allowing
        them to be retried on the next run. Successful recordings are preserved.
        """
        if "status" not in self.manifest.columns:
            return

        n_failed_before = (self.manifest["status"] == "failed").sum()
        if n_failed_before == 0:
            logger.info("No failed recordings to clear")
            return

        # Keep only complete recordings
        self.manifest = self.manifest[self.manifest["status"] == "complete"].copy()

        logger.info(f"Cleared {n_failed_before} failed recordings from manifest")
        logger.info(f"They will be retried on next run")

        # Save cleaned manifest
        self.save()

    def get_stats(self) -> Dict[str, Any]:
        """Get manifest statistics.

        Returns:
            Dict with counts of total/complete/failed entries
        """
        total = len(self.manifest)

        if "status" not in self.manifest.columns:
            return {"total": total, "complete": 0, "failed": 0, "missing": total}

        complete = (self.manifest["status"] == "complete").sum()
        failed = (self.manifest["status"] == "failed").sum()

        return {
            "total": total,
            "complete": complete,
            "failed": failed,
            "missing": total - complete - failed
        }

    def print_status(self, title: str = "Cache Status"):
        """Print manifest status summary.

        Args:
            title: Title to display
        """
        stats = self.get_stats()
        logger.info(f"\n{'='*60}")
        logger.info(f"{title}")
        logger.info(f"{'='*60}")
        logger.info(f"  Total entries:    {stats['total']}")
        logger.info(f"  ✓ Complete:       {stats['complete']}")
        logger.info(f"  ✗ Failed:         {stats['failed']}")
        logger.info(f"  ? Missing:        {stats['missing']}")
        logger.info(f"{'='*60}\n")


# Schema definitions for different manifest types

RAW_MANIFEST_SCHEMA = {
    "dataset": "string",
    "release": "string",
    "subject": "string",
    "task": "string",
    "mini": "bool",
    "recording_id": "string",
    "sfreq": "int64",
    "n_channels": "int64",
    "n_samples": "int64",
    "duration_s": "float64",
    "preprocessing_hash": "string",
    "raw_zarr_path": "string",
    "status": "string",
    "error_msg": "string",
    "timestamp": "string",
}

WINDOW_MANIFEST_SCHEMA = {
    "window_config_id": "string",
    "raw_config_hash": "string",
    "window_len_s": "float64",
    "stride_s": "float64",
    "n_windows": "int64",
    "window_zarr_path": "string",
    "metadata_path": "string",
    "status": "string",
    "error_msg": "string",
    "timestamp": "string",
}
