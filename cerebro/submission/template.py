"""Submission template for Codabench.

This file is copied into submission.zip as submission.py.
It uses braindecode models directly (no cerebro imports) to keep the zip small.

The Submission class must implement:
- get_model_challenge_1(): Return model for response time prediction
- get_model_challenge_2(): Return model for externalizing score prediction
"""

import os
from pathlib import Path

import torch
from braindecode.models import EEGNeX


def resolve_path(filename: str) -> str:
    """Resolve path for Codabench compatibility.

    Args:
        filename: Name of file in submission directory

    Returns:
        Absolute path to file
    """
    return str(Path(__file__).parent / filename)


class Submission:
    """Submission class for NeurIPS 2025 EEG Foundation Challenge.

    This class loads trained models and provides them to the scoring system.
    Model weights are loaded from:
    - weights_challenge_1.pt: Challenge 1 model (response time prediction)
    - weights_challenge_2.pt: Challenge 2 model (externalizing score prediction)

    Args:
        sfreq: Sampling frequency in Hz (provided by scoring system)
        device: Device to load models on ('cpu' or 'cuda')
    """

    def __init__(self, sfreq: int = 100, device: str = "cpu"):
        self.sfreq = sfreq
        self.device = device

    def get_model_challenge_1(self):
        """Load Challenge 1 model: Response time prediction from CCD task.

        Input: (batch, 129 channels, 200 time points)  # 2.0s at 100 Hz
        Output: (batch, 1)  # Response time in seconds

        Returns:
            Trained PyTorch model for Challenge 1
        """
        # Create model architecture
        # NOTE: Architecture must match training configuration
        model = EEGNeX(
            n_chans=129,           # 128 EEG channels + 1 reference (Cz)
            n_outputs=1,           # Single regression output (response time)
            sfreq=100,             # Sampling frequency in Hz
            n_times=200,           # 2.0s window at 100 Hz
            # Default EEGNeX parameters (no do_spatial_filter in braindecode 0.9.1)
            # do_spatial_filter=True,  # REMOVED - not in braindecode API
            # depth_multiplier=2,  # Use defaults
            # kernel_size=64,
            # n_filters=8,
            # drop_prob=0.25,
        )

        # Load trained weights
        state_dict = torch.load(
            resolve_path("weights_challenge_1.pt"),
            map_location=self.device
        )
        model.load_state_dict(state_dict)

        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        return model

    def get_model_challenge_2(self):
        """Load Challenge 2 model: Externalizing score prediction from multi-task EEG.

        Input: (batch, 129 channels, 400 time points)  # 4.0s at 100 Hz
        Output: (batch, 1)  # Externalizing score (p_factor)

        Returns:
            Trained PyTorch model for Challenge 2
        """
        # Create model architecture
        # NOTE: Architecture must match training configuration
        model = EEGNeX(
            n_chans=129,           # 128 EEG channels + 1 reference (Cz)
            n_outputs=1,           # Single regression output (externalizing score)
            sfreq=100,             # Sampling frequency in Hz
            n_times=200,           # 2.0s window at 100 Hz (CORRECTED from 400)
            # Default EEGNeX parameters (no do_spatial_filter in braindecode 0.9.1)
            # do_spatial_filter=True,  # REMOVED - not in braindecode API
            # depth_multiplier=2,  # Use defaults
            # kernel_size=64,
            # n_filters=8,
            # drop_prob=0.25,
        )

        # Load trained weights
        state_dict = torch.load(
            resolve_path("weights_challenge_2.pt"),
            map_location=self.device
        )
        model.load_state_dict(state_dict)

        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        return model
