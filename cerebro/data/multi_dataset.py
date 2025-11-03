"""Multi-dataset wrapper for joint training with channel projection.

Wraps multiple EEG datasets (e.g., HBN, TUH) and projects them to a unified
channel space for joint pretraining. Handles datasets with different:
- Channel counts (TUH: 21 channels → HBN: 129 channels)
- Channel names (TUH: "Fp1" → HBN: "FP1")
- Sampling rates (handled in raw cache: all resampled to 100Hz)
"""

import logging
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset

from cerebro.utils.channel_mapping import get_tuh_projection_indices
from cerebro.utils.electrode_locations import load_hbn_chs_info

logger = logging.getLogger(__name__)


class MultiDatasetWrapper(Dataset):
    """Wraps multiple datasets with channel projection to unified HBN space.

    Projects all datasets to HBN 129-channel space. For datasets with fewer
    channels (e.g., TUH with 21), unmapped positions are:
    1. Filled with zeros
    2. Marked via unmapped_mask (returned with sample)
    3. Always masked during training (pre-masking)

    Note: Assumes all datasets are already at 100Hz (handled in raw cache).

    Args:
        dataset_list: List of (dataset_name, Dataset) tuples
        projection_enabled: If True, project non-HBN datasets to HBN space

    Example:
        >>> from cerebro.data.multi_dataset import MultiDatasetWrapper
        >>> hbn_ds = ... # HBN MemmapWindowDataset
        >>> tuh_ds = ... # TUH MemmapWindowDataset
        >>> multi_ds = MultiDatasetWrapper([
        ...     ("hbn", hbn_ds),
        ...     ("tuh", tuh_ds)
        ... ])
        >>> x, unmapped_mask = multi_ds[0]
        >>> x.shape  # (129, T) - always in HBN space
        >>> unmapped_mask  # None for HBN, (129,) boolean for TUH
    """

    def __init__(
        self,
        dataset_list: List[Tuple[str, Dataset]],
        projection_enabled: bool = True
    ):
        self.datasets = dataset_list
        self.projection_enabled = projection_enabled

        # Build cumulative index for efficient dataset selection
        self.cumulative_sizes = []
        total = 0
        for ds_name, ds in dataset_list:
            total += len(ds)
            self.cumulative_sizes.append(total)

        logger.info(f"Multi-dataset wrapper: {total} total windows across {len(dataset_list)} datasets")

        # Precompute TUH→HBN projection indices (cached per unique channel layout)
        self.tuh_projections = {}

        if projection_enabled:
            self.hbn_chs_info = load_hbn_chs_info()
            logger.info("Channel projection enabled (TUH → HBN 129-channel space)")

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int):
        """Get sample with optional channel projection.

        Returns:
            x: EEG tensor (129, T) - always in HBN space at 100Hz
            unmapped_mask: Optional tensor (129,) boolean - True = always mask this channel
                          (None for HBN samples, mask for TUH samples)
        """
        # Find which dataset this index belongs to (binary search would be faster)
        ds_idx = 0
        for i, cumsum in enumerate(self.cumulative_sizes):
            if idx < cumsum:
                ds_idx = i
                break

        # Adjust index to be relative to selected dataset
        if ds_idx > 0:
            idx_in_ds = idx - self.cumulative_sizes[ds_idx - 1]
        else:
            idx_in_ds = idx

        ds_name, dataset = self.datasets[ds_idx]

        # Get sample from dataset
        sample = dataset[idx_in_ds]

        # Handle different return formats (tensor vs tuple)
        if isinstance(sample, tuple):
            x = sample[0]  # (C, T)
        else:
            x = sample  # (C, T)

        # Apply projection if needed
        if ds_name == "hbn":
            # HBN: Already 129 channels at 100Hz, no projection needed
            return x, None

        elif ds_name == "tuh" and self.projection_enabled:
            # TUH: Already resampled to 100Hz in raw cache
            # Project 21 channels → 129 channels
            projected, unmapped_mask = self._project_tuh_to_hbn(x, dataset, idx_in_ds)
            return projected, unmapped_mask

        else:
            # Unknown dataset or projection disabled
            logger.warning(f"Unknown dataset '{ds_name}' or projection disabled")
            return x, None

    def _get_tuh_channel_names(self, dataset, idx_in_ds: int) -> List[str]:
        """Get TUH channel names for a specific sample.

        Tries to get from dataset metadata, falls back to standard 21-channel montage.

        Args:
            dataset: Source dataset
            idx_in_ds: Index within dataset

        Returns:
            List of TUH channel names
        """
        # Try to get from metadata if available
        if hasattr(dataset, 'metadata') and 'ch_names' in dataset.metadata.columns:
            ch_names = dataset.metadata.iloc[idx_in_ds]['ch_names']
            if isinstance(ch_names, list):
                return ch_names

        # Fallback: Standard 21-channel TUH montage (AR referential)
        # Order matches typical TUH dataset channel arrangement
        return [
            "A1", "A2",  # Reference channels
            "Fp1", "Fp2", "F3", "F4", "C3", "C4",
            "P3", "P4", "O1", "O2", "F7", "F8",
            "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
        ]

    def _project_tuh_to_hbn(
        self,
        tuh_window: torch.Tensor,
        dataset,
        idx_in_ds: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project TUH (C_tuh, T) to HBN (129, T) space.

        Uses cached projection indices for efficiency.

        Args:
            tuh_window: TUH EEG tensor (C_tuh, T) - typically (21, T)
            dataset: Source dataset for metadata
            idx_in_ds: Index within dataset

        Returns:
            projected: HBN-space tensor (129, T) with zeros in unmapped positions
            unmapped_mask: Boolean mask (129,) of unmapped positions (True = always mask)
        """
        # Get channel names for this recording
        tuh_ch_names = self._get_tuh_channel_names(dataset, idx_in_ds)

        # Cache projection for this channel layout (different TUH recordings may have different channels)
        ch_key = tuple(tuh_ch_names)
        if ch_key not in self.tuh_projections:
            src_idx, tgt_idx, unmapped = get_tuh_projection_indices(
                tuh_ch_names, self.hbn_chs_info
            )
            self.tuh_projections[ch_key] = (src_idx, tgt_idx, unmapped)

        src_idx, tgt_idx, unmapped_mask = self.tuh_projections[ch_key]

        # Create 129-channel tensor (zeros for unmapped channels)
        C, T = tuh_window.shape
        projected = torch.zeros(129, T, dtype=tuh_window.dtype, device=tuh_window.device)

        # Copy mapped channels to their HBN positions
        projected[tgt_idx] = tuh_window[src_idx]

        return projected, torch.from_numpy(unmapped_mask).bool()

    def get_dataset_counts(self) -> dict:
        """Get count of samples per dataset.

        Returns:
            Dictionary mapping dataset name to sample count
        """
        counts = {}
        prev_cumsum = 0
        for (ds_name, _), cumsum in zip(self.datasets, self.cumulative_sizes):
            counts[ds_name] = cumsum - prev_cumsum
            prev_cumsum = cumsum
        return counts
