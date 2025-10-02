"""Loss helpers for JEPA-style latent prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_loss(context: torch.Tensor, target: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Compute symmetric InfoNCE between context and target embeddings."""
    if context.shape != target.shape:
        raise ValueError("Context and target must have matching shapes")

    context = F.normalize(context, dim=-1)
    target = F.normalize(target, dim=-1)

    logits = context @ target.t() / temperature
    labels = torch.arange(context.size(0), device=context.device)

    loss_ct = F.cross_entropy(logits, labels)
    loss_tc = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ct + loss_tc)
