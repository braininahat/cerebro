"""
Diversity-preserving loss functions for preventing representation collapse in regression tasks.

Key ideas:
1. Variance preservation: Ensure predictions match target distribution variance
2. Batch decorrelation: Prevent all samples from collapsing to same representation
3. Attention entropy bonus: Reward diverse attention patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class DiversityPreservingLoss(nn.Module):
    """
    Loss function that prevents representation collapse in regression tasks.

    Combines:
    - Primary regression loss (MAE or MSE)
    - Variance penalty to match target distribution
    - Batch diversity penalty to prevent identical predictions
    - Optional attention entropy bonus

    Based on VICReg and other self-supervised learning techniques adapted for regression.
    """

    def __init__(
        self,
        primary_loss: str = "mae",
        lambda_variance: float = 0.1,
        lambda_diversity: float = 0.05,
        lambda_entropy: float = 0.01,
        target_std: Optional[float] = None,
        min_std_ratio: float = 0.5,
        epsilon: float = 1e-6
    ):
        """
        Args:
            primary_loss: "mae" or "mse" for primary regression loss
            lambda_variance: Weight for variance preservation penalty
            lambda_diversity: Weight for batch diversity penalty
            lambda_entropy: Weight for attention entropy bonus
            target_std: Expected standard deviation of targets (computed from data if None)
            min_std_ratio: Minimum acceptable ratio of pred_std/target_std before penalty
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.primary_loss = primary_loss
        self.lambda_variance = lambda_variance
        self.lambda_diversity = lambda_diversity
        self.lambda_entropy = lambda_entropy
        self.target_std = target_std
        self.min_std_ratio = min_std_ratio
        self.epsilon = epsilon

        # Track statistics for monitoring
        self.last_stats = {}

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute diversity-preserving loss.

        Args:
            predictions: Model predictions (batch_size, ...)
            targets: Ground truth targets (batch_size, ...)
            attention_weights: Optional attention weights for entropy bonus (batch, heads, seq, seq)
            return_components: If True, return individual loss components for monitoring

        Returns:
            Total loss, optionally with component breakdown
        """
        batch_size = predictions.shape[0]

        # Flatten predictions and targets if needed
        pred_flat = predictions.view(batch_size, -1)
        target_flat = targets.view(batch_size, -1)

        # 1. Primary regression loss
        if self.primary_loss == "mae":
            primary = F.l1_loss(pred_flat, target_flat)
        elif self.primary_loss == "mse":
            primary = F.mse_loss(pred_flat, target_flat)
        else:
            raise ValueError(f"Unknown primary loss: {self.primary_loss}")

        # 2. Variance preservation penalty
        # Penalize if prediction variance is too low compared to target variance
        pred_std = pred_flat.std() + self.epsilon

        if self.target_std is None:
            target_std = target_flat.std() + self.epsilon
        else:
            target_std = self.target_std

        # Only apply penalty if pred_std is below threshold
        std_ratio = pred_std / target_std
        variance_penalty = torch.tensor(0.0, device=predictions.device)

        if std_ratio < self.min_std_ratio:
            # Quadratic penalty that increases as variance drops
            variance_penalty = self.lambda_variance * ((target_std * self.min_std_ratio - pred_std) ** 2)

        # 3. Batch diversity penalty (prevent all samples collapsing to same value)
        # Based on VICReg covariance regularization
        diversity_penalty = torch.tensor(0.0, device=predictions.device)

        if batch_size > 1:
            # Center predictions per feature
            pred_centered = pred_flat - pred_flat.mean(dim=0, keepdim=True)

            # Compute covariance matrix (batch x batch)
            # We want diagonal to be large (variance) and off-diagonal small (decorrelation)
            cov = (pred_centered @ pred_centered.T) / (batch_size - 1)

            # Penalize high correlation between different samples
            # Extract off-diagonal elements
            off_diagonal = cov.flatten()[1:].view(batch_size - 1, batch_size + 1)[:, :-1].flatten()
            diversity_penalty = self.lambda_diversity * (off_diagonal ** 2).mean()

            # Alternative: penalize low variance across batch dimension
            batch_var = pred_centered.var(dim=0).mean()
            if batch_var < 0.01:  # Critical threshold
                diversity_penalty += self.lambda_diversity * ((0.01 - batch_var) ** 2)

        # 4. Attention entropy bonus (optional - rewards diverse attention)
        entropy_bonus = torch.tensor(0.0, device=predictions.device)

        if attention_weights is not None and self.lambda_entropy > 0:
            # Compute entropy of attention weights
            # Higher entropy = more uniform attention = better diversity
            # attention_weights shape: (batch, heads, seq, seq) or (batch, seq, seq)

            if attention_weights.dim() == 4:
                # Multi-head attention
                attn_probs = attention_weights.mean(dim=1)  # Average over heads
            else:
                attn_probs = attention_weights

            # Compute entropy: -sum(p * log(p))
            # Add epsilon for numerical stability
            attn_probs = attn_probs.clamp(min=self.epsilon)
            entropy = -(attn_probs * attn_probs.log()).sum(dim=-1).mean()

            # Normalize by sequence length for consistent scale
            seq_len = attn_probs.shape[-1]
            max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float32))
            normalized_entropy = entropy / max_entropy

            # Bonus for high entropy (diverse attention)
            entropy_bonus = self.lambda_entropy * normalized_entropy

        # Combine all components
        total_loss = primary + variance_penalty + diversity_penalty - entropy_bonus

        # Store statistics for monitoring
        self.last_stats = {
            'primary_loss': primary.item(),
            'variance_penalty': variance_penalty.item(),
            'diversity_penalty': diversity_penalty.item(),
            'entropy_bonus': entropy_bonus.item() if attention_weights is not None else 0.0,
            'pred_std': pred_std.item(),
            'target_std': target_std.item() if not isinstance(target_std, float) else target_std,
            'std_ratio': std_ratio.item() if not isinstance(std_ratio, float) else std_ratio,
            'batch_variance': pred_centered.var(dim=0).mean().item() if batch_size > 1 else 0.0
        }

        if return_components:
            return total_loss, self.last_stats
        else:
            return total_loss


class AdaptiveDiversityLoss(DiversityPreservingLoss):
    """
    Adaptive version that adjusts penalty weights based on training progress.

    Early training: Focus on primary task
    Mid training: Increase diversity penalties as needed
    Late training: Reduce penalties to allow convergence
    """

    def __init__(self, *args, warmup_epochs: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.base_lambda_variance = self.lambda_variance
        self.base_lambda_diversity = self.lambda_diversity

    def set_epoch(self, epoch: int):
        """Update current epoch for adaptive weighting."""
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            # During warmup, gradually increase diversity penalties
            warmup_factor = epoch / self.warmup_epochs
            self.lambda_variance = self.base_lambda_variance * warmup_factor
            self.lambda_diversity = self.base_lambda_diversity * warmup_factor
        else:
            # After warmup, use full penalties
            self.lambda_variance = self.base_lambda_variance
            self.lambda_diversity = self.base_lambda_diversity


def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention weights as a measure of diversity.

    Args:
        attention_weights: Attention weights tensor (batch, heads, seq, seq) or (batch, seq, seq)

    Returns:
        Scalar entropy value (higher = more diverse attention)
    """
    epsilon = 1e-8

    if attention_weights.dim() == 4:
        # Multi-head: average over heads
        attn = attention_weights.mean(dim=1)
    else:
        attn = attention_weights

    # Ensure valid probability distribution
    attn = attn.clamp(min=epsilon, max=1.0)

    # Compute entropy: -sum(p * log(p))
    entropy = -(attn * attn.log()).sum(dim=-1).mean()

    return entropy


def diagnose_collapse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    embeddings: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Diagnose various types of representation collapse.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        embeddings: Optional intermediate embeddings to analyze

    Returns:
        Dictionary of diagnostic metrics
    """
    diagnostics = {}

    # 1. Prediction diversity
    pred_std = predictions.std().item()
    target_std = targets.std().item()
    diagnostics['pred_std'] = pred_std
    diagnostics['target_std'] = target_std
    diagnostics['std_ratio'] = pred_std / (target_std + 1e-8)

    # 2. Batch correlation (are all samples getting same prediction?)
    if predictions.shape[0] > 1:
        pred_flat = predictions.view(predictions.shape[0], -1)
        if pred_flat.shape[0] > 1 and pred_flat.shape[1] > 0:
            # Compute correlation matrix
            pred_centered = pred_flat - pred_flat.mean(dim=0, keepdim=True)
            pred_norm = pred_centered / (pred_centered.norm(dim=1, keepdim=True) + 1e-8)
            corr_matrix = pred_norm @ pred_norm.T

            # Average off-diagonal correlation (excluding self-correlation)
            mask = ~torch.eye(corr_matrix.shape[0], dtype=torch.bool, device=corr_matrix.device)
            avg_correlation = corr_matrix[mask].mean().item()
            diagnostics['batch_correlation'] = avg_correlation
        else:
            diagnostics['batch_correlation'] = 0.0

    # 3. Embedding analysis (if provided)
    if embeddings is not None:
        emb_flat = embeddings.view(embeddings.shape[0], -1)

        # Effective rank (number of significant singular values)
        if emb_flat.shape[0] > 1 and emb_flat.shape[1] > 1:
            try:
                svd = torch.svd(emb_flat)
                singular_values = svd.S
                # Normalize singular values
                sv_normalized = singular_values / (singular_values.sum() + 1e-8)
                # Effective rank: exp(entropy of singular values)
                sv_entropy = -(sv_normalized * (sv_normalized + 1e-8).log()).sum()
                effective_rank = sv_entropy.exp().item()
                diagnostics['embedding_rank'] = effective_rank
                diagnostics['embedding_max_sv'] = singular_values[0].item()
            except:
                diagnostics['embedding_rank'] = -1
                diagnostics['embedding_max_sv'] = -1

        # Embedding std
        diagnostics['embedding_std'] = emb_flat.std().item()

    # 4. Collapse severity score (0=healthy, 1=complete collapse)
    collapse_score = 0.0

    # Low prediction diversity
    if diagnostics['std_ratio'] < 0.1:
        collapse_score += 0.3
    elif diagnostics['std_ratio'] < 0.3:
        collapse_score += 0.15

    # High batch correlation
    if 'batch_correlation' in diagnostics:
        if abs(diagnostics['batch_correlation']) > 0.9:
            collapse_score += 0.3
        elif abs(diagnostics['batch_correlation']) > 0.7:
            collapse_score += 0.15

    # Low embedding rank
    if 'embedding_rank' in diagnostics and diagnostics['embedding_rank'] > 0:
        max_rank = min(embeddings.shape[0], embeddings.shape[-1]) if embeddings is not None else 1
        rank_ratio = diagnostics['embedding_rank'] / max_rank
        if rank_ratio < 0.1:
            collapse_score += 0.4
        elif rank_ratio < 0.3:
            collapse_score += 0.2

    diagnostics['collapse_severity'] = min(collapse_score, 1.0)

    return diagnostics