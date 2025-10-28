"""Loss functions for various training objectives.

This module provides loss functions for different training strategies:
- InfoNCE for contrastive learning (triplet and all-pairs variants)
- NRMSE for evaluation metrics
- Custom losses for specific tasks
"""

import torch
import torch.nn.functional as F

__all__ = [
    "info_nce_loss",           # Alias for info_nce_triplet (backward compat)
    "info_nce_triplet",        # Triplet-based InfoNCE (1 negative per anchor)
    "info_nce_all_pairs",      # All-pairs InfoNCE (batch_size-1 negatives)
    "info_nce_loss_multi_negative",  # Multiple negatives per anchor
    "normalized_mse_loss",
]


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """InfoNCE loss for triplet-based contrastive learning.

    Computes the noise-contrastive estimation loss that pulls positive
    pairs together while pushing negative pairs apart in embedding space.

    Args:
        anchor: (batch_size, embedding_dim) anchor embeddings
        positive: (batch_size, embedding_dim) positive embeddings
        negative: (batch_size, embedding_dim) negative embeddings
        temperature: Temperature scaling parameter. Lower values create
            harder negatives. Typical range: 0.05-0.5

    Returns:
        Scalar loss value

    Note:
        Embeddings are L2-normalized before computing similarities to
        ensure cosine similarity is in [-1, 1] range.
    """
    # L2 normalize embeddings to unit sphere
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)

    # Compute cosine similarities
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature  # (batch_size,)
    neg_sim = torch.sum(anchor * negative, dim=1) / temperature  # (batch_size,)

    # Stack similarities: [pos, neg] for each sample
    # Label is 0 (first position) for all samples since positive is always first
    logits = torch.stack([pos_sim, neg_sim], dim=1)  # (batch_size, 2)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    # Cross-entropy loss: -log(exp(pos) / (exp(pos) + exp(neg)))
    return F.cross_entropy(logits, labels)


def info_nce_all_pairs(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE loss with all-pairs negatives (SimCLR style).

    Uses all other samples in batch as negatives for stronger contrastive signal.
    Assumes batch structure: (anchor_i, positive_i) are corresponding pairs.

    This is more efficient than triplet-based sampling when batch sizes are large
    (256+), as it leverages all batch samples as negatives.

    Args:
        anchor: Anchor embeddings (batch_size, embedding_dim)
        positive: Positive embeddings (batch_size, embedding_dim)
        temperature: Temperature scaling factor. Lower values create harder negatives.

    Returns:
        Scalar loss value

    Example:
        >>> # Batch of 256 samples
        >>> anchor_emb = model(anchor_batch)     # (256, 128)
        >>> positive_emb = model(positive_batch) # (256, 128)
        >>> loss = info_nce_all_pairs(anchor_emb, positive_emb)
        >>> # Each anchor uses 255 negatives (all other samples in batch)
    """
    # L2 normalize embeddings
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)

    batch_size = anchor.shape[0]

    # Compute similarity matrix: anchor vs all positives
    # Shape: (batch_size, batch_size)
    similarity_matrix = torch.matmul(anchor, positive.T) / temperature

    # Create labels: diagonal elements are the positive pairs
    # For sample i, the i-th column in row i is the positive
    labels = torch.arange(batch_size, device=anchor.device)

    # Standard cross-entropy: treats row i's i-th element as positive,
    # all other elements in row as negatives
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss


def info_nce_triplet(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """InfoNCE loss with single sampled negative (triplet style).

    Simpler version that samples one negative per anchor. Works well with
    smaller batch sizes and is memory efficient.

    Args:
        anchor: Anchor embeddings (batch_size, embedding_dim)
        positive: Positive embeddings (batch_size, embedding_dim)
        negative: Negative embeddings (batch_size, embedding_dim)
        temperature: Temperature scaling parameter. Lower values create
            harder negatives. Typical range: 0.05-0.5

    Returns:
        Scalar loss value

    Note:
        Embeddings are L2-normalized before computing similarities to
        ensure cosine similarity is in [-1, 1] range.
    """
    # L2 normalize embeddings to unit sphere
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)

    # Compute cosine similarities
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature
    neg_sim = torch.sum(anchor * negative, dim=1) / temperature

    # Stack and compute cross-entropy
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


# Backward compatibility alias
info_nce_loss = info_nce_triplet


def info_nce_loss_multi_negative(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """InfoNCE loss with multiple negatives per anchor.

    Args:
        anchor: (batch_size, embedding_dim) anchor embeddings
        positive: (batch_size, embedding_dim) positive embeddings
        negatives: (batch_size, n_negatives, embedding_dim) negative embeddings
        temperature: Temperature scaling parameter

    Returns:
        Scalar loss value
    """
    batch_size = anchor.shape[0]
    n_negatives = negatives.shape[1]

    # L2 normalize embeddings
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negatives = F.normalize(negatives, p=2, dim=2)

    # Compute positive similarity
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature  # (batch_size,)

    # Compute negative similarities (batch matrix multiplication)
    neg_sim = torch.bmm(
        negatives,  # (batch_size, n_negatives, embedding_dim)
        anchor.unsqueeze(2)  # (batch_size, embedding_dim, 1)
    ).squeeze(2) / temperature  # (batch_size, n_negatives)

    # Concatenate positive and negative similarities
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, 1 + n_negatives)

    # Labels: positive is always at index 0
    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)


def normalized_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    normalize_by_std: bool = True
) -> torch.Tensor:
    """MSE loss optionally normalized by target standard deviation.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        normalize_by_std: If True, normalize by target std (creating NRMSE)

    Returns:
        Scalar loss value
    """
    mse = F.mse_loss(predictions, targets)

    if normalize_by_std:
        target_std = torch.std(targets)
        if target_std > 0:
            mse = mse / (target_std ** 2)

    return mse