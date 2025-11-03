# tokenizer_vq_snp.py
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# -------------------------
# Vector-Quantized codebook (EMA)
# -------------------------
class VectorQuantEMA(nn.Module):
    def __init__(self, num_codes: int = 8192, dim: int = 256, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_codes, self.dim = num_codes, dim
        self.decay, self.eps = decay, eps

        # Codebook (learned via EMA updates)
        self.codebook = nn.Parameter(torch.randn(num_codes, dim))
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_mean", torch.zeros(num_codes, dim))

    def forward(self, z: torch.Tensor):
        """
        z: (B, C, P, D)
        Returns:
          z_q: quantized z (straight-through)  (B, C, P, D)
          vq_loss: commitment + codebook losses
          codes: (B, C, P) long
        """
        B, C, P, D = z.shape
        z_flat = z.reshape(-1, D)  # (N, D), N=B*C*P

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
            onehot = F.one_hot(codes, self.num_codes).to(
                z_flat.dtype)  # (N, K)
            # EMA updates
            self.ema_count.mul_(self.decay).add_(
                onehot.sum(0), alpha=1 - self.decay)
            ema_mean_add = onehot.t() @ z_flat  # (K, D)
            self.ema_mean.mul_(self.decay).add_(
                ema_mean_add, alpha=1 - self.decay)

            n = self.ema_count.sum()
            # Laplace smoothing
            cluster_size = (self.ema_count + self.eps) / \
                (n + self.num_codes * self.eps) * n
            # Update codebook
            new_codebook = self.ema_mean / cluster_size[:, None]
            self.codebook.data.copy_(new_codebook)

        # Commitment + codebook (straight-through)
        # (use 0.25 commitment weight like VQ-VAE2-style defaults)
        vq_loss = F.mse_loss(z_q.detach(), z) + 0.25 * \
            F.mse_loss(z_q, z.detach())
        z_q_st = z_q.detach() + (z - z.detach())  # straight-through

        return z_q_st, vq_loss, codes.view(B, C, P)


# -------------------------
# FFT-based spectral targets (LaBraM-style VQ-NSP)
# Supports (B,C,L) and (B,C,P,L) inputs → returns matching (… , K)
# -------------------------
def _pow2_pad_len(n: int) -> int:
    # next power-of-two ≥ n
    return 1 << (n - 1).bit_length()


@torch.no_grad()
def fft_targets(
    x: torch.Tensor,               # (B, C, L) or (B, C, P, L)
    n_bins: int | None = None,     # if set, interpolate to this many freq bins
    use_hann: bool = True,
    one_sided: bool = True,
    chunk_n: int = 65536,          # number of 1D FFTs per chunk (B*C*[P])
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Window + complex FFT + amplitude/phase → sin/cos (float32), robust on Hopper/GH200.

    Returns:
      amp : (B, C, K)   or (B, C, P, K)
      sin : (B, C, K)   or (B, C, P, K)
      cos : (B, C, K)   or (B, C, P, K)
    """
    assert x.dim() in (
        3, 4), f"x must be (B,C,L) or (B,C,P,L), got {tuple(x.shape)}"
    device = x.device
    if x.dim() == 3:
        B, C, L = x.shape
        P = None
    else:
        B, C, P, L = x.shape

    # Optional Hann window
    if use_hann:
        hann = torch.hann_window(
            L, periodic=True, device=device, dtype=x.dtype)
        if P is None:
            xw = x * hann.view(1, 1, L)
        else:
            xw = x * hann.view(1, 1, 1, L)
    else:
        xw = x

    # Pad to power-of-two (more stable/faster cuFFT plans)
    pad_len = _pow2_pad_len(L)
    Fpos = pad_len // 2 + 1 if one_sided else pad_len

    # Flatten to (N, L) for batched 1D FFT
    if P is None:
        x2 = xw.contiguous().to(torch.float32).view(B * C, L)  # fp32 for FFT
        N = B * C
    else:
        x2 = xw.contiguous().to(torch.float32).view(B * C * P, L)
        N = B * C * P

    # Chunked FFT to avoid cuFFT internal errors on massive batches
    outs = []
    # ensure FFT runs in fp32 even if trainer uses AMP/bf16-mixed
    with torch.cuda.amp.autocast(enabled=False):
        for s in range(0, N, chunk_n):
            e = min(N, s + chunk_n)
            chunk = x2[s:e]  # (M, L), fp32
            # (M, pad_len) complex64
            X = torch.fft.fft(chunk, n=pad_len, dim=-1)
            if one_sided:
                X = X[..., :Fpos]  # (M, Fpos)
            outs.append(X)
    X = torch.cat(outs, dim=0)  # (N, F)

    # Amplitude and phase
    amp = X.abs()
    phs = torch.angle(X)
    sin = torch.sin(phs)
    cos = torch.cos(phs)

    # Optional interpolation to fixed #bins
    if n_bins is not None and amp.shape[-1] != n_bins:
        def _resize(t: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t.unsqueeze(1), size=n_bins, mode="linear", align_corners=False).squeeze(1)
        amp = _resize(amp)
        sin = _resize(sin)
        cos = _resize(cos)

    # Log-compress amplitude (common & stable)
    amp = torch.log1p(amp)

    K = amp.shape[-1]
    if P is None:
        amp = amp.view(B, C, K)
        sin = sin.view(B, C, K)
        cos = cos.view(B, C, K)
    else:
        amp = amp.view(B, C, P, K)
        sin = sin.view(B, C, P, K)
        cos = cos.view(B, C, P, K)

    return amp.to(torch.float32), sin.to(torch.float32), cos.to(torch.float32)


# -------------------------
# Tokenizer: VQ-NSP (Encoder → VQ → predict spectral targets)
# -------------------------
class TokenizerVQSNP(pl.LightningModule):
    """
    LaBraM-style tokenizer:
      - splits (B,C,T) into (B,C,P,L) patches
      - encodes each (channel×patch) to a D-dim latent
      - VQ-EMA codebook quantizes the latent to a discrete code
      - predicts spectral targets (amp, sin, cos) for each patch

    Heads output (B, C, P, K) to match spectral targets (B, C, P, K).
    """

    def __init__(
        self,
        n_chans: int = 129,
        crop_len: int = 200,
        patch_len: int = 20,
        dim: int = 256,
        num_codes: int = 8192,
        n_fft_bins: int = 129,  # one-sided bins for pad_len=256; change to taste
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert crop_len % patch_len == 0, f"crop_len({crop_len}) must be divisible by patch_len({patch_len})"
        self.P = crop_len // patch_len
        self.patch_len = patch_len

        # Tiny conv encoder over time per (channel×patch)
        # Input will be shaped as (B, 1, C*P, L) to share weights across rows
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(128, dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
        )

        self.vq = VectorQuantEMA(num_codes=num_codes, dim=dim)

        K = n_fft_bins
        self.head_amp = nn.Linear(dim, K)
        self.head_sin = nn.Linear(dim, K)
        self.head_cos = nn.Linear(dim, K)

        self.lr = lr
        self.weight_decay = weight_decay

    # --------- helpers ----------
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) → (B, C, P, L)
        """
        B, C, T = x.shape
        L = self.patch_len
        P = T // L
        return x.view(B, C, P, L)

    # --------- forward ----------
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, T)
        Returns:
          z   : (B, C, P, D) encoder output
          z_q : (B, C, P, D) quantized (straight-through)
          vq_loss: scalar
          codes: (B, C, P) long
          xp  : (B, C, P, L) raw patches (for spectral targets)
        """
        B, C, T = x.shape
        xp = self.patchify(x)  # (B,C,P,L)

        # reshape for conv2d: treat (C×P) as "height" and L as "width"
        # enc outputs: (B, D, C*P, L) → mean over time → (B, D, C*P)
        xp2 = xp.reshape(B, 1, C * self.P, self.patch_len)
        z = self.enc(xp2).mean(dim=-1)               # (B, D, C*P)
        z = z.permute(0, 2, 1).reshape(B, C, self.P, -1)  # (B, C, P, D)

        z_q, vq_loss, codes = self.vq(z)             # quantized latents

        return z, z_q, vq_loss, codes, xp

    # --------- training ----------
    def training_step(self, batch, batch_idx):
        """
        batch: (B, C, T) float tensor
        """
        x = batch  # (B, C, T)
        z, z_q, vq_loss, codes, xp = self.forward(x)  # xp: (B, C, P, L)

        # Build spectral targets per (channel×patch)
        amp_t, sin_t, cos_t = fft_targets(
            xp,                                # (B, C, P, L)
            n_bins=self.hparams.n_fft_bins,
            use_hann=True,
            one_sided=True,
            chunk_n=65536,                     # reduce if you still see cuFFT issues
        )  # each: (B, C, P, K)

        # Predict spectral targets from quantized latents
        amp_h = self.head_amp(z_q)            # (B, C, P, K)
        sin_h = self.head_sin(z_q)            # (B, C, P, K)
        cos_h = self.head_cos(z_q)            # (B, C, P, K)

        loss = (
            F.mse_loss(amp_h, amp_t) +
            F.mse_loss(sin_h, sin_t) +
            F.mse_loss(cos_h, cos_t) +
            vq_loss
        )

        self.log("tok_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        z, z_q, vq_loss, codes, xp = self.forward(x)
        amp_t, sin_t, cos_t = fft_targets(
            xp, n_bins=self.hparams.n_fft_bins, use_hann=True, one_sided=True)
        amp_h = self.head_amp(z_q)
        sin_h = self.head_sin(z_q)
        cos_h = self.head_cos(z_q)
        loss = (
            F.mse_loss(amp_h, amp_t) +
            F.mse_loss(sin_h, sin_t) +
            F.mse_loss(cos_h, cos_t) +
            vq_loss
        )
        self.log("val_tok_loss", loss, on_epoch=True,
                 prog_bar=True, batch_size=x.size(0))
        return loss

    # --------- optim ----------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=getattr(self.trainer, "max_epochs", 100), eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
