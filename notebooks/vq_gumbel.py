# vq_gumbel.py
"""
Gumbel-Softmax Vector Quantization following LaBraM.
Provides differentiable quantization for better gradient flow during training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizerGumbel(nn.Module):
    """
    Gumbel-Softmax VQ following LaBraM.
    Uses Gumbel-Softmax for differentiable sampling during training
    and hard assignment during evaluation.
    """

    def __init__(self, num_codes=8192, dim=256, temperature=1.0,
                 kl_weight=0.01, commitment_weight=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.commitment_weight = commitment_weight

        # Codebook
        self.codebook = nn.Embedding(num_codes, dim)
        nn.init.normal_(self.codebook.weight, std=1/np.sqrt(dim))

    def forward(self, z):
        """
        z: (B, C, P, D)
        Returns:
          z_q: quantized z (straight-through)  (B, C, P, D)
          vq_loss: commitment + KL losses
          codes: (B, C, P) long
        """
        B, C, P, D = z.shape
        flat = z.reshape(-1, D)  # (N, D) where N = B*C*P

        # Compute logits (similarity to codebook entries)
        logits = torch.matmul(flat, self.codebook.weight.t())  # (N, num_codes)

        if self.training:
            # Gumbel-Softmax for differentiable sampling
            soft_one_hot = F.gumbel_softmax(
                logits, tau=self.temperature, hard=False, dim=-1
            )  # (N, num_codes)
            z_q = torch.matmul(soft_one_hot, self.codebook.weight)  # (N, D)

            # KL divergence regularization (encourage uniform codebook usage)
            qy = F.softmax(logits, dim=-1)
            # KL(q || uniform) = sum(q * log(q * num_codes))
            kl_loss = self.kl_weight * torch.sum(
                qy * torch.log(qy * self.num_codes + 1e-10), dim=-1
            ).mean()
        else:
            # Hard assignment for inference
            idx = logits.argmax(dim=-1)
            z_q = self.codebook(idx)
            kl_loss = torch.tensor(0.0, device=z.device)

        # Reshape back
        z_q = z_q.reshape(B, C, P, D)
        codes = logits.argmax(dim=-1).reshape(B, C, P)

        # Commitment loss (encourage encoder to commit to codebook)
        commit_loss = self.commitment_weight * F.mse_loss(z_q.detach(), z)

        # Straight-through estimator for gradients
        z_q = z + (z_q - z).detach()

        total_loss = commit_loss + kl_loss

        return z_q, total_loss, codes

    def get_codebook_usage(self, codes):
        """
        Compute codebook usage statistics.
        codes: (B, C, P) long tensor
        Returns: dict with usage stats
        """
        flat_codes = codes.reshape(-1)
        unique_codes = torch.unique(flat_codes)
        usage_rate = len(unique_codes) / self.num_codes

        return {
            'unique_codes': len(unique_codes),
            'usage_rate': usage_rate,
            'total_codes': self.num_codes
        }


class VectorQuantizerEMA(nn.Module):
    """
    Improved EMA-based VQ with proper initialization and decay schedule.
    Alternative to Gumbel-Softmax approach.
    """

    def __init__(self, num_codes=8192, dim=256, decay=0.99,
                 eps=1e-5, commitment_weight=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.decay = decay
        self.eps = eps
        self.commitment_weight = commitment_weight

        # Codebook (learned via EMA updates)
        self.codebook = nn.Parameter(torch.randn(num_codes, dim))
        nn.init.normal_(self.codebook.weight if hasattr(
            self.codebook, 'weight') else self.codebook, std=1/np.sqrt(dim))

        # EMA buffers
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_mean", torch.zeros(num_codes, dim))
        self.register_buffer("_codebook_initialized", torch.tensor(0))

    def _init_codebook(self, z):
        """Initialize codebook with k-means++ on first batch"""
        if self._codebook_initialized.item() == 1:
            return

        B, C, P, D = z.shape
        flat = z.reshape(-1, D).detach()

        # Use k-means++ initialization
        n = min(flat.shape[0], self.num_codes * 10)
        indices = torch.randperm(flat.shape[0], device=z.device)[:n]
        samples = flat[indices]

        # Simple random sampling for initialization
        if samples.shape[0] >= self.num_codes:
            idx = torch.randperm(samples.shape[0], device=z.device)[
                :self.num_codes]
            self.codebook.data.copy_(samples[idx])
            self.ema_mean.copy_(samples[idx])
            self.ema_count.fill_(1.0)

        self._codebook_initialized.fill_(1)

    def forward(self, z):
        """
        z: (B, C, P, D)
        Returns:
          z_q: quantized z (straight-through)  (B, C, P, D)
          vq_loss: commitment + codebook losses
          codes: (B, C, P) long
        """
        B, C, P, D = z.shape
        z_flat = z.reshape(-1, D)  # (N, D), N=B*C*P

        # Initialize codebook on first forward pass
        if self.training:
            self._init_codebook(z)

        # L2 distance to codes
        codebook = self.codebook
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ codebook.t()
            + codebook.pow(2).sum(dim=1, keepdim=False)[None, :]
        )  # (N, num_codes)

        codes = dist.argmin(dim=1)  # (N,)
        z_q = codebook[codes].view(B, C, P, D)

        if self.training:
            # EMA update of codebook
            with torch.no_grad():
                # One-hot encode assignments
                # (N, num_codes)
                onehot = F.one_hot(codes, self.num_codes).float()

                # Update counts
                n = onehot.sum(dim=0)  # (num_codes,)
                self.ema_count.mul_(self.decay).add_(n, alpha=1 - self.decay)

                # Update means
                dw = onehot.t() @ z_flat  # (num_codes, D)
                self.ema_mean.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                # Normalize and update codebook
                n_stable = self.ema_count + self.eps
                self.codebook.data.copy_(self.ema_mean / n_stable.unsqueeze(1))

                # Reset unused codes (random restart)
                usage_mask = self.ema_count < 1.0
                if usage_mask.any():
                    # Reinitialize unused codes with random samples from current batch
                    n_reset = usage_mask.sum().item()
                    if z_flat.shape[0] >= n_reset:
                        rand_idx = torch.randperm(
                            z_flat.shape[0], device=z.device)[:n_reset]
                        self.codebook.data[usage_mask] = z_flat[rand_idx].detach(
                        )
                        self.ema_count[usage_mask] = 1.0
                        self.ema_mean[usage_mask] = z_flat[rand_idx].detach()

        # Commitment loss
        commit_loss = self.commitment_weight * F.mse_loss(z_q.detach(), z)

        # Codebook loss (for EMA, this is mainly for monitoring)
        codebook_loss = F.mse_loss(z_q, z.detach())

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        total_loss = commit_loss + 0.25 * codebook_loss

        return z_q, total_loss, codes
