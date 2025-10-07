"""Loss helpers for masked time-channel modeling."""

from __future__ import annotations

import torch


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error computed only over masked positions.

    Args:
        pred: Predicted reconstruction (batch, channels, time).
        target: Ground truth signal (same shape as pred).
        mask: Boolean mask (True at masked positions).

    Returns:
        Scalar tensor with average squared error on masked entries.
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred, target, and mask must share the same shape")

    if mask.dtype != torch.bool:
        mask = mask.bool()

    masked_elements = mask.sum()
    if masked_elements == 0:
        return torch.tensor(0.0, device=pred.device)

    diff = (pred - target)[mask]
    return torch.mean(diff * diff)


__all__ = ["masked_mse_loss"]
