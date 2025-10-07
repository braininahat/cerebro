"""Model components for the foundation-model pipeline."""

from .jepa import EEGJEPAModel
from .masked import EEGMaskedAutoencoder

__all__ = ["EEGJEPAModel", "EEGMaskedAutoencoder"]
