"""Channel mapping utilities for multi-dataset training.

Provides functions to map channels from different EEG datasets (e.g., TUH) to a
unified channel space (HBN 129-channel montage) for joint pretraining.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


def load_tuh_to_hbn_mapping() -> Dict[str, str]:
    """Load TUH→HBN channel mapping from docs/tuh_to_hbn.json.

    Returns:
        Dictionary mapping TUH channel names to HBN channel names

    Example:
        >>> mapping = load_tuh_to_hbn_mapping()
        >>> mapping["Fp1"]
        'FP1'
        >>> mapping["T3"]
        'T9'
    """
    # Resolve path relative to repo root
    repo_root = Path(__file__).parent.parent.parent
    mapping_path = repo_root / "docs" / "tuh_to_hbn.json"

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Channel mapping file not found: {mapping_path}\n"
            f"Expected path: docs/tuh_to_hbn.json (relative to repo root)"
        )

    with open(mapping_path, 'r') as f:
        data = json.load(f)

    if "channel_map" not in data:
        raise ValueError(f"Invalid mapping file: missing 'channel_map' key")

    logger.info(f"Loaded TUH→HBN mapping: {len(data['channel_map'])} channels")

    return data["channel_map"]


def get_tuh_projection_indices(
    tuh_ch_names: List[str],
    target_chs_info: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get indices for projecting TUH channels to HBN 129-channel space.

    This function determines which TUH channels can be mapped to HBN channels,
    and returns the indices needed for efficient array copying during projection.

    Args:
        tuh_ch_names: List of TUH channel names from recording (typically 21 channels)
        target_chs_info: HBN 129-channel chs_info (from load_hbn_chs_info())

    Returns:
        source_indices: Indices in TUH array to copy from (e.g., [0, 1, 2, ...])
        target_indices: Corresponding indices in HBN array (e.g., [15, 47, 3, ...])
        unmapped_mask: Boolean array (129,) where True = channel not in TUH

    Example:
        >>> from cerebro.utils.electrode_locations import load_hbn_chs_info
        >>> tuh_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4"]
        >>> hbn_chs = load_hbn_chs_info()
        >>> src_idx, tgt_idx, unmapped = get_tuh_projection_indices(tuh_names, hbn_chs)
        >>> # src_idx: [0, 1, 2, 3, 4, 5]
        >>> # tgt_idx: [21, 11, 24, 108, 36, 104]  # Indices in HBN 129-channel array
        >>> # unmapped.sum(): 123  # 123 HBN channels not in TUH
    """
    # Load TUH→HBN mapping
    channel_map = load_tuh_to_hbn_mapping()

    # Build HBN name→index mapping
    target_names = [ch['ch_name'] for ch in target_chs_info]
    name_to_idx = {name: i for i, name in enumerate(target_names)}

    # Map TUH channels to HBN indices
    source_indices = []
    target_indices = []
    missing_channels = []

    for src_idx, tuh_name in enumerate(tuh_ch_names):
        # Look up mapped HBN channel name
        hbn_name = channel_map.get(tuh_name)

        if hbn_name and hbn_name in name_to_idx:
            source_indices.append(src_idx)
            target_indices.append(name_to_idx[hbn_name])
        else:
            # Channel not found in mapping or HBN montage
            if not hbn_name:
                logger.debug(f"TUH channel '{tuh_name}' not in mapping file")
            else:
                logger.debug(f"TUH channel '{tuh_name}' maps to '{hbn_name}' but not found in HBN montage")
            missing_channels.append(tuh_name)

    # Create unmapped mask (True = this HBN channel is not represented in TUH)
    unmapped_mask = np.ones(129, dtype=bool)
    unmapped_mask[target_indices] = False

    logger.info(
        f"TUH→HBN projection: {len(source_indices)}/{len(tuh_ch_names)} channels mapped, "
        f"{unmapped_mask.sum()}/129 HBN positions unmapped"
    )

    if missing_channels:
        logger.warning(f"Unmapped TUH channels: {missing_channels}")

    return (
        np.array(source_indices, dtype=np.int64),
        np.array(target_indices, dtype=np.int64),
        unmapped_mask
    )
