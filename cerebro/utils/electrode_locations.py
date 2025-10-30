"""Utilities for loading electrode location information for EEG models.

This module provides functions to load electrode coordinate files (e.g., .sfp format)
and convert them to formats required by various deep learning frameworks.
"""

from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_hbn_chs_info(sfp_path: str = "docs/GSN_HydroCel_129_AdjustedLabels.sfp") -> List[Dict[str, Any]]:
    """Load HBN 129-channel electrode locations as MNE chs_info format.

    This function reads the GSN-HydroCel-129 electrode coordinate file and converts
    it to the MNE channel info format required by braindecode models like SignalJEPA.

    Args:
        sfp_path: Path to .sfp electrode coordinate file (relative to repo root)

    Returns:
        List of channel info dicts compatible with braindecode.models.SignalJEPA.
        Each dict contains:
            - 'ch_name': Channel name (str)
            - 'loc': 12-element array with 3D position (first 3 are X,Y,Z)
            - 'kind': Channel type (int)
            - 'coil_type': Coil type (int)
            - 'unit': Unit (int)
            - 'coord_frame': Coordinate frame (int)

    Example:
        >>> chs_info = load_hbn_chs_info()
        >>> len(chs_info)
        129
        >>> chs_info[0]['ch_name']
        'E1'
        >>> chs_info[0]['loc'][:3]  # X, Y, Z coordinates
        array([1.479, 5.687, 6.813])

    Note:
        The .sfp file contains 132 entries total (3 fiducials + 129 EEG channels).
        This function automatically excludes the fiducial points (NZ, LPA, RPA)
        and returns only the 129 EEG channels.
    """
    import mne

    # Resolve path relative to repo root (3 levels up from this file)
    repo_root = Path(__file__).parent.parent.parent
    full_sfp_path = repo_root / sfp_path

    if not full_sfp_path.exists():
        raise FileNotFoundError(
            f"Electrode coordinate file not found: {full_sfp_path}\n"
            f"Expected path: {sfp_path} (relative to repo root)"
        )

    logger.info(f"Loading electrode locations from: {full_sfp_path}")

    # Load montage from .sfp file
    montage = mne.channels.read_custom_montage(str(full_sfp_path))

    # Filter out fiducial points (NZ, LPA, RPA) and keep only EEG channels
    fiducials = {'NZ', 'LPA', 'RPA'}
    ch_names = [name for name in montage.ch_names if name not in fiducials]

    # Ensure we have exactly 129 channels for HBN dataset
    if len(ch_names) < 129:
        raise ValueError(
            f"Expected at least 129 EEG channels, but found {len(ch_names)} "
            f"after excluding fiducials from {full_sfp_path}"
        )

    # Take first 129 channels (in case there are extra)
    ch_names = ch_names[:129]

    logger.info(f"Loaded {len(ch_names)} EEG channel locations")
    logger.debug(f"Channel names: {ch_names[:5]}... (showing first 5)")

    # Create MNE Info object with montage
    # Use sfreq=100 as default for HBN dataset
    info = mne.create_info(ch_names=ch_names, sfreq=100, ch_types='eeg')
    info.set_montage(montage, on_missing='ignore')

    # Extract chs_info (list of channel info dicts)
    chs_info = info['chs']

    # Validate that we got the expected number of channels
    assert len(chs_info) == 129, f"Expected 129 channels, got {len(chs_info)}"

    # Validate that each channel has location information
    for ch in chs_info:
        if 'loc' not in ch or len(ch['loc']) != 12:
            raise ValueError(
                f"Channel {ch.get('ch_name', 'unknown')} missing valid location info. "
                f"Expected 'loc' array of length 12."
            )

    # FIX: MNE puts coordinates in loc[0:3], but braindecode's SignalJEPA
    # expects them in loc[3:6]. Copy coordinates to the expected location.
    for ch in chs_info:
        ch['loc'][3:6] = ch['loc'][0:3].copy()

    logger.info("✓ Successfully loaded and validated 129 channel locations")

    return chs_info
