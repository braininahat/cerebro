"""Level 2: Window cache builder.

Builds windowed datasets from Level 1 raw cache.
Multiple window configurations can coexist.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from tqdm import tqdm

from cerebro.data.unified_cache.lazy_dataset import LazyZarrWindowDataset
from cerebro.data.unified_cache.raw_cache import GracefulInterruptHandler

logger = logging.getLogger(__name__)


class WindowCacheBuilder:
    """Builds Level 2 window cache from Level 1 raw cache.

    Args:
        cache_manager: Parent UniversalCacheManager instance
    """

    def __init__(self, cache_manager):
        self.cache_mgr = cache_manager
        self.window_manifest = self.cache_mgr.window_manifest

    def get_or_build(
        self,
        recordings: pd.DataFrame,
        window_len_s: float,
        stride_s: float,
        crop_len_s: Optional[float] = None,
        mode: str = 'train'
    ) -> LazyZarrWindowDataset:
        """Get windowed dataset (builds if missing).

        Args:
            recordings: DataFrame from query_raw()
            window_len_s: Window length in seconds
            stride_s: Stride in seconds
            crop_len_s: Optional crop length for augmentation

        Returns:
            LazyZarrWindowDataset instance
        """
        # Get window config ID
        window_config_id = self.cache_mgr._get_window_config_id(window_len_s, stride_s)
        zarr_path = self.cache_mgr._get_window_zarr_path(window_config_id)
        metadata_path = self.cache_mgr._get_window_metadata_path(window_config_id)

        # Check if window cache exists
        if zarr_path.exists() and metadata_path.exists():
            logger.info(f"Loading windowed dataset from cache: {window_config_id}")
            metadata = pd.read_parquet(metadata_path)

            # Filter metadata to match requested recordings
            metadata_filtered = metadata[
                metadata["recording_id"].isin(recordings["recording_id"].values)
            ].reset_index(drop=True)

            # Debug: Check for NaN after filtering
            if metadata_filtered["zarr_index"].isna().any():
                logger.error(f"NaN zarr_index after filtering!")
                logger.error(f"  Requested recordings: {len(recordings)}")
                logger.error(f"  Filtered metadata: {len(metadata_filtered)}")
                logger.error(f"  NaN count: {metadata_filtered['zarr_index'].isna().sum()}")
                # Drop NaN rows as defensive fix
                logger.warning(f"Dropping {metadata_filtered['zarr_index'].isna().sum()} rows with NaN zarr_index")
                metadata_filtered = metadata_filtered.dropna(subset=["zarr_index"]).reset_index(drop=True)

            return LazyZarrWindowDataset(
                zarr_path=zarr_path,
                metadata=metadata_filtered,
                crop_len_s=crop_len_s,
                sfreq=self.cache_mgr.preprocessing_params["sfreq"],
                mode=mode
            )

        # Build window cache
        logger.info(f"Building windowed dataset: {window_config_id}")
        return self._build_window_cache(
            recordings, window_len_s, stride_s, crop_len_s,
            window_config_id, zarr_path, metadata_path, mode
        )

    def _build_window_cache(
        self,
        recordings: pd.DataFrame,
        window_len_s: float,
        stride_s: float,
        crop_len_s: Optional[float],
        window_config_id: str,
        zarr_path: Path,
        metadata_path: Path,
        mode: str = 'train'
    ) -> LazyZarrWindowDataset:
        """Build window cache from raw cache.

        Args:
            recordings: DataFrame of recordings to window
            window_len_s: Window length in seconds
            stride_s: Stride in seconds
            crop_len_s: Optional crop length
            window_config_id: Config ID for this window setup
            zarr_path: Path to save Zarr array
            metadata_path: Path to save metadata

        Returns:
            LazyZarrWindowDataset
        """
        sfreq = self.cache_mgr.preprocessing_params["sfreq"]
        window_samples = int(window_len_s * sfreq)
        stride_samples = int(stride_s * sfreq)

        # Get n_channels from first recording
        first_raw_path = recordings.iloc[0]["raw_zarr_path"]
        first_raw = zarr.open(str(first_raw_path), mode='r')
        n_channels = first_raw.shape[0]

        # Create Zarr array for windows (use v2 format for compatibility)
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        window_zarr = zarr.open(
            str(zarr_path),
            mode='w',
            shape=(0, n_channels, window_samples),
            chunks=(100, n_channels, window_samples),
            dtype=np.float32,
            compressor=compressor,
            zarr_format=2  # Use v2 format for compressor API compatibility
        )

        # Process each recording
        metadata_rows = []
        total_windows = 0

        with GracefulInterruptHandler() as handler:
            for _, rec in tqdm(recordings.iterrows(), total=len(recordings), desc="Windowing", unit="rec"):
                if handler.interrupted:
                    break

                try:
                    # Load raw from Level 1 cache
                    raw_zarr = zarr.open(str(rec["raw_zarr_path"]), mode='r')
                    data = np.array(raw_zarr)  # Shape: (n_channels, n_samples)

                    # Create windows
                    n_samples = data.shape[1]
                    n_windows = max(0, (n_samples - window_samples) // stride_samples + 1)

                    if n_windows == 0:
                        logger.warning(f"Recording {rec['recording_id']} too short for windowing")
                        continue

                    # Extract windows
                    windows = np.zeros((n_windows, n_channels, window_samples), dtype=np.float32)
                    for i in range(n_windows):
                        start = i * stride_samples
                        end = start + window_samples
                        windows[i] = data[:, start:end]

                    # Append to Zarr
                    zarr_start = window_zarr.shape[0]
                    window_zarr.append(windows, axis=0)

                    # Track metadata
                    for i in range(n_windows):
                        metadata_rows.append({
                            "window_idx": total_windows + i,
                            "recording_id": rec["recording_id"],
                            "dataset": rec["dataset"],
                            "release": rec["release"],
                            "subject": rec["subject"],
                            "task": rec["task"],
                            "mini": rec["mini"],
                            "zarr_index": zarr_start + i,
                            "time_offset_s": i * stride_s
                        })

                    total_windows += n_windows

                except Exception as e:
                    logger.error(f"Failed to window {rec['recording_id']}: {e}")
                    continue

        # Save metadata
        metadata = pd.DataFrame(metadata_rows)
        metadata.to_parquet(metadata_path, index=False)

        # Update window manifest
        self.window_manifest.mark_complete({
            "window_config_id": window_config_id,
            "raw_config_hash": self.cache_mgr.preprocessing_hash,
            "window_len_s": window_len_s,
            "stride_s": stride_s,
            "n_windows": total_windows,
            "window_zarr_path": str(zarr_path),
            "metadata_path": str(metadata_path),
            "error_msg": ""
        })
        self.window_manifest.save()

        logger.info(f"✓ Created {total_windows} windows")

        return LazyZarrWindowDataset(
            zarr_path=zarr_path,
            metadata=metadata,
            crop_len_s=crop_len_s,
            sfreq=sfreq,
            mode=mode
        )
