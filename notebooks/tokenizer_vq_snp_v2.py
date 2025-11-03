# tokenizer_vq_snp_v2.py
"""
Improved LaBraM-style tokenizer with:
- Better encoder architecture (from labram_encoder.py)
- Gumbel-Softmax VQ option (from vq_gumbel.py)
- Frequency band-based FFT targets
- Better hyperparameters
"""
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from labram_encoder import LaBraMEncoder
from vq_gumbel import VectorQuantizerGumbel, VectorQuantizerEMA


# -------------------------
# FFT-based spectral targets with frequency bands (LaBraM-style)
# -------------------------
@torch.no_grad()
def fft_targets_bands(
    x: torch.Tensor,               # (B, C, P, L)
    sfreq: int = 100,
    freq_bands: list = None,
    use_hann: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute band-specific FFT targets following LaBraM.

    Args:
        x: (B, C, P, L) input patches
        sfreq: sampling frequency
        freq_bands: list of (low, high) frequency bands in Hz
        use_hann: whether to apply Hann window

    Returns:
        amp_t: (B, C, P, n_bands) - amplitude per frequency band
        sin_t: (B, C, P, n_bands) - sin(phase) per band
        cos_t: (B, C, P, n_bands) - cos(phase) per band
    """
    if freq_bands is None:
        # EEG standard frequency bands
        freq_bands = [
            (0.5, 4),    # Delta
            (4, 8),      # Theta
            (8, 13),     # Alpha
            (13, 30),    # Beta
            (30, 50),    # Gamma
        ]

    B, C, P, L = x.shape

    # Apply Hann window
    if use_hann:
        window = torch.hann_window(
            L, device=x.device, dtype=x.dtype).view(1, 1, 1, L)
        x = x * window

    # Reshape for FFT: (B*C*P, L)
    x_flat = x.reshape(B * C * P, L)

    # Full FFT in fp32 - using torch.fft.fft instead of rfft for consistency
    with torch.amp.autocast('cuda', enabled=False):
        X = torch.fft.fft(x_flat.float(), dim=-1)  # (B*C*P, L) complex
        # Take only positive frequencies (one-sided spectrum)
        X = X[..., :L//2 + 1]  # (B*C*P, L//2+1)

    freqs = torch.fft.rfftfreq(L, 1/sfreq, device=x.device)

    # Extract band-specific features
    amp_bands = []
    sin_bands = []
    cos_bands = []

    for low, high in freq_bands:
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() == 0:
            # Handle empty band
            amp_bands.append(torch.zeros(B * C * P, device=x.device))
            sin_bands.append(torch.zeros(B * C * P, device=x.device))
            cos_bands.append(torch.zeros(B * C * P, device=x.device))
            continue

        X_band = X[:, mask]  # (B*C*P, n_freqs_in_band)

        # Average amplitude and phase in band
        amp = X_band.abs().mean(dim=-1)  # (B*C*P,)
        phs = torch.angle(X_band).mean(dim=-1)

        amp_bands.append(amp)
        sin_bands.append(torch.sin(phs))
        cos_bands.append(torch.cos(phs))

    amp_t = torch.stack(amp_bands, dim=-1)  # (B*C*P, n_bands)
    sin_t = torch.stack(sin_bands, dim=-1)
    cos_t = torch.stack(cos_bands, dim=-1)

    # Log-compress amplitude
    amp_t = torch.log1p(amp_t)

    # Reshape back to (B, C, P, n_bands)
    n_bands = len(freq_bands)
    amp_t = amp_t.reshape(B, C, P, n_bands)
    sin_t = sin_t.reshape(B, C, P, n_bands)
    cos_t = cos_t.reshape(B, C, P, n_bands)

    return amp_t.float(), sin_t.float(), cos_t.float()


# Original FFT targets function (for backward compatibility)
def _pow2_pad_len(n: int) -> int:
    return 1 << (n - 1).bit_length()


@torch.no_grad()
def fft_targets(
    x: torch.Tensor,
    n_bins: int | None = None,
    use_hann: bool = True,
    one_sided: bool = True,
    chunk_n: int = 65536,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Original FFT targets implementation for backward compatibility"""
    assert x.dim() in (
        3, 4), f"x must be (B,C,L) or (B,C,P,L), got {tuple(x.shape)}"
    device = x.device
    if x.dim() == 3:
        B, C, L = x.shape
        P = None
    else:
        B, C, P, L = x.shape

    if use_hann:
        hann = torch.hann_window(
            L, periodic=True, device=device, dtype=x.dtype)
        if P is None:
            xw = x * hann.view(1, 1, L)
        else:
            xw = x * hann.view(1, 1, 1, L)
    else:
        xw = x

    pad_len = _pow2_pad_len(L)
    Fpos = pad_len // 2 + 1 if one_sided else pad_len

    if P is None:
        x2 = xw.contiguous().to(torch.float32).view(B * C, L)
        N = B * C
    else:
        x2 = xw.contiguous().to(torch.float32).view(B * C * P, L)
        N = B * C * P

    outs = []
    with torch.amp.autocast('cuda', enabled=False):
        for s in range(0, N, chunk_n):
            e = min(N, s + chunk_n)
            chunk = x2[s:e]
            X = torch.fft.fft(chunk, n=pad_len, dim=-1)
            if one_sided:
                X = X[..., :Fpos]
            outs.append(X)
    X = torch.cat(outs, dim=0)

    amp = X.abs()
    phs = torch.angle(X)
    sin = torch.sin(phs)
    cos = torch.cos(phs)

    if n_bins is not None and amp.shape[-1] != n_bins:
        def _resize(t):
            return F.interpolate(t.unsqueeze(1), size=n_bins, mode="linear", align_corners=False).squeeze(1)
        amp = _resize(amp)
        sin = _resize(sin)
        cos = _resize(cos)

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
# Improved Tokenizer: VQ-NSP with LaBraM architecture
# -------------------------
class TokenizerVQSNP(pl.LightningModule):
    """
    Improved LaBraM-style tokenizer:
      - Uses LaBraMEncoder for better feature extraction
      - Supports both Gumbel-Softmax and EMA-based VQ
      - Uses frequency band-based spectral targets
      - Better training dynamics
    """

    def __init__(
        self,
        n_chans: int = 129,
        crop_len: int = 200,
        patch_len: int = 20,
        dim: int = 256,
        num_codes: int = 8192,
        n_fft_bins: int = 5,  # number of frequency bands
        sfreq: int = 100,
        use_band_fft: bool = True,  # use frequency bands vs raw FFT
        vq_type: str = "gumbel",  # "gumbel" or "ema"
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 1.0,
        kl_weight: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert crop_len % patch_len == 0, f"crop_len({crop_len}) must be divisible by patch_len({patch_len})"
        self.P = crop_len // patch_len
        self.patch_len = patch_len
        self.n_chans = n_chans
        self.sfreq = sfreq
        self.use_band_fft = use_band_fft

        # Use improved LaBraM encoder
        self.enc = LaBraMEncoder(n_chans=n_chans, patch_len=patch_len, dim=dim)

        # Vector Quantizer
        if vq_type == "gumbel":
            self.vq = VectorQuantizerGumbel(
                num_codes=num_codes,
                dim=dim,
                temperature=temperature,
                kl_weight=kl_weight
            )
        elif vq_type == "ema":
            self.vq = VectorQuantizerEMA(num_codes=num_codes, dim=dim)
        else:
            raise ValueError(f"Unknown vq_type: {vq_type}")

        # Prediction heads
        K = n_fft_bins
        self.head_amp = nn.Linear(dim, K)
        self.head_sin = nn.Linear(dim, K)
        self.head_cos = nn.Linear(dim, K)

        self.lr = lr
        self.weight_decay = weight_decay

    # --------- helpers ----------
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, C, P, L)"""
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

        # Encode using LaBraM encoder
        z = self.enc(xp)  # (B, C, P, D)

        # Vector quantization
        z_q, vq_loss, codes = self.vq(z)

        return z, z_q, vq_loss, codes, xp    # --------- training ----------

    def training_step(self, batch, batch_idx):
        """batch: (B, C, T) float tensor"""
        x = batch  # (B, C, T)
        z, z_q, vq_loss, codes, xp = self.forward(x)  # xp: (B, C, P, L)

        # Build spectral targets
        if self.use_band_fft:
            amp_t, sin_t, cos_t = fft_targets_bands(
                xp, sfreq=self.sfreq, use_hann=True
            )  # each: (B, C, P, K)
        else:
            amp_t, sin_t, cos_t = fft_targets(
                xp, n_bins=self.hparams.n_fft_bins, use_hann=True, one_sided=True
            )

        # Predict spectral targets from quantized latents
        amp_h = self.head_amp(z_q)
        sin_h = self.head_sin(z_q)
        cos_h = self.head_cos(z_q)

        # Reconstruction loss
        recon_loss = (
            F.mse_loss(amp_h, amp_t) +
            F.mse_loss(sin_h, sin_t) +
            F.mse_loss(cos_h, cos_t)
        )

        total_loss = recon_loss + vq_loss

        # Logging
        self.log("tok_loss", total_loss, on_step=True,
                 on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log("recon_loss", recon_loss, on_step=True,
                 on_epoch=True, batch_size=x.size(0))
        self.log("vq_loss", vq_loss, on_step=True,
                 on_epoch=True, batch_size=x.size(0))

        # Log codebook usage if using Gumbel VQ
        if hasattr(self.vq, 'get_codebook_usage'):
            if batch_idx % 100 == 0:
                usage_stats = self.vq.get_codebook_usage(codes)
                self.log(
                    "codebook_usage", usage_stats['usage_rate'], prog_bar=False, batch_size=x.size(0))

        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        z, z_q, vq_loss, codes, xp = self.forward(x)

        if self.use_band_fft:
            amp_t, sin_t, cos_t = fft_targets_bands(
                xp, sfreq=self.sfreq, use_hann=True)
        else:
            amp_t, sin_t, cos_t = fft_targets(
                xp, n_bins=self.hparams.n_fft_bins, use_hann=True, one_sided=True)

        amp_h = self.head_amp(z_q)
        sin_h = self.head_sin(z_q)
        cos_h = self.head_cos(z_q)

        recon_loss = (
            F.mse_loss(amp_h, amp_t) +
            F.mse_loss(sin_h, sin_t) +
            F.mse_loss(cos_h, cos_t)
        )

        total_loss = recon_loss + vq_loss

        self.log("val_tok_loss", total_loss, on_epoch=True,
                 prog_bar=True, batch_size=x.size(0))
        self.log("val_recon_loss", recon_loss,
                 on_epoch=True, batch_size=x.size(0))
        self.log("val_vq_loss", vq_loss, on_epoch=True, batch_size=x.size(0))

        return total_loss

    # --------- optim ----------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=getattr(self.trainer, "max_epochs", 100), eta_min=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch"}
        }
