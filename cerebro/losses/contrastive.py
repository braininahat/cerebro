"""Contrastive learning losses for multi-phase training.

Implements InfoNCE and related contrastive objectives for EEG representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
    reduction: str = 'mean',
) -> torch.Tensor:
    """InfoNCE (Noise Contrastive Estimation) loss.

    Pulls anchor closer to positive, pushes away from negatives.

    Args:
        anchor: Anchor embeddings (batch, dim)
        positive: Positive embeddings (batch, dim)
        negative: Negative embeddings (batch * (N-1), dim) or None
            If None, uses in-batch negatives (all other samples)
        temperature: Temperature parameter for softmax (default: 0.07)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        InfoNCE loss

    Example:
        >>> anchor = F.normalize(torch.randn(32, 128), dim=1)
        >>> positive = F.normalize(torch.randn(32, 128), dim=1)
        >>> loss = info_nce_loss(anchor, positive, temperature=0.07)
    """
    batch_size = anchor.size(0)

    # Normalize embeddings
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)

    # Positive similarity: (batch,)
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature

    if negative is not None:
        # Explicit negatives provided
        negative = F.normalize(negative, dim=1)

        # Negative similarity: (batch, num_negatives)
        neg_sim = torch.matmul(anchor, negative.T) / temperature

        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    else:
        # In-batch negatives: all other samples in batch
        # Similarity matrix: (batch, batch)
        sim_matrix = torch.matmul(anchor, positive.T) / temperature

        # Mask out diagonal (positive pairs)
        mask = torch.eye(batch_size, device=anchor.device, dtype=torch.bool)
        logits = sim_matrix.masked_fill(mask, float('-inf'))

        # Positive similarity is the diagonal
        pos_sim = torch.diagonal(sim_matrix)

        # For cross-entropy, we need logits where first column is positive
        # Concatenate positive sim with negative sims
        neg_sim = sim_matrix[~mask].view(batch_size, batch_size - 1)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

    # Labels: positive is always index 0
    labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels, reduction=reduction)

    return loss


class InfoNCE(nn.Module):
    """InfoNCE loss module.

    Args:
        temperature: Temperature parameter (default: 0.07)
        reduction: 'mean', 'sum', or 'none'

    Example:
        >>> loss_fn = InfoNCE(temperature=0.07)
        >>> anchor = torch.randn(32, 128)
        >>> positive = torch.randn(32, 128)
        >>> loss = loss_fn(anchor, positive)
    """

    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return info_nce_loss(
            anchor, positive, negative,
            temperature=self.temperature,
            reduction=self.reduction
        )


class MovieISCLoss(nn.Module):
    """Movie Inter-Subject Correlation (ISC) contrastive loss.

    Treats same movie timestamp across different subjects as positive pairs.

    Args:
        temperature: Temperature parameter (default: 0.07)
        reduction: 'mean', 'sum', or 'none'

    Example:
        >>> # Subject embeddings for same movie timestamp
        >>> subject_embeddings = torch.randn(5, 128)  # 5 subjects watching same scene
        >>> loss_fn = MovieISCLoss(temperature=0.07)
        >>> loss = loss_fn(subject_embeddings)
    """

    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute ISC loss for multiple subjects viewing same stimulus.

        Args:
            embeddings: Subject embeddings (num_subjects, dim)
                All subjects watched same movie timestamp

        Returns:
            ISC loss (scalar)
        """
        num_subjects = embeddings.size(0)

        if num_subjects < 2:
            # Need at least 2 subjects for ISC
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute all pairwise similarities
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask out diagonal (self-similarity)
        mask = torch.eye(num_subjects, device=embeddings.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # For each subject, all others are positives (same movie timestamp)
        # Use symmetric loss: each subject vs all others
        loss = 0.0
        for i in range(num_subjects):
            # Subject i's similarities to all others
            subject_sim = sim_matrix[i]  # (num_subjects,)

            # Softmax over all similarities (including self, which is masked)
            # Ideally, all should be high (uniform distribution)
            # But we want to maximize similarity, so use negative log softmax

            # Alternative: Use mean similarity as proxy for ISC
            # Higher ISC = higher correlation across subjects
            loss += -subject_sim.mean()

        # Average across subjects
        loss = loss / num_subjects

        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * num_subjects
        else:
            return loss


class ContrastiveProjectionHead(nn.Module):
    """Projection head for contrastive learning.

    Projects encoder outputs to normalized embeddings for contrastive loss.

    Args:
        input_dim: Encoder output dimension
        hidden_dim: Hidden layer dimension (default: 512)
        output_dim: Projection dimension (default: 128)
        num_layers: Number of layers (default: 2)

    Example:
        >>> proj_head = ContrastiveProjectionHead(input_dim=512, output_dim=128)
        >>> encoder_out = torch.randn(32, 512)
        >>> projection = proj_head(encoder_out)  # (32, 128), normalized
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder output (batch, input_dim)

        Returns:
            Normalized projection (batch, output_dim)
        """
        projection = self.projection(x)
        # L2 normalization
        return F.normalize(projection, dim=1)
