"""Pretraining objectives for the foundation model."""

from .jepa import jepa_loss
from .masked import masked_mse_loss

__all__ = ["jepa_loss", "masked_mse_loss"]
