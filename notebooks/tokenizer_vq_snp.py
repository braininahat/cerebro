# tokenizer_vq_snp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class VectorQuantEMA(nn.Module):
    def __init__(self, num_codes=8192, dim=256, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_codes, self.dim = num_codes, dim
        self.decay, self.eps = decay, eps
        self.codebook = nn.Parameter(torch.randn(num_codes, dim))
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_mean", torch.zeros(num_codes, dim))

    def forward(self, z):  # (B,C,P,D)
        B, C, P, D = z.shape
        flat = z.reshape(-1, D)                           # (B*C*P,D)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2*flat @ self.codebook.T
                + self.codebook.pow(2).sum(1)[None, :])    # (N,num_codes)
        idx = dist.argmin(1)                               # (N,)
        z_q = self.codebook[idx].view(B, C, P, D)             # quantized
        if self.training:
            onehot = F.one_hot(idx, self.num_codes).float()
            self.ema_count.mul_(self.decay).add_(
                onehot.sum(0), alpha=1-self.decay)
            mean = (onehot.T @ flat)
            self.ema_mean.mul_(self.decay).add_(mean, alpha=1-self.decay)
            n = self.ema_count.sum()
            cluster_size = (self.ema_count + self.eps) / \
                (n + self.num_codes*self.eps) * n
            self.codebook.data.copy_(self.ema_mean / cluster_size[:, None])
        # commitment loss
        loss = F.mse_loss(z_q.detach(), z) + 0.25 * F.mse_loss(z_q, z.detach())
        return z_q.detach() + (z - z.detach()), loss, idx.view(B, C, P)


def rfft_targets(x_patches, n_bins=None):
    # x_patches: (B,C,P,L)
    X = torch.fft.rfft(x_patches, dim=-1)
    amp = X.abs()
    ph = torch.angle(X)
    sin = torch.sin(ph)
    cos = torch.cos(ph)
    if n_bins and amp.size(-1) > n_bins:
        amp = amp[..., :n_bins]
        sin = sin[..., :n_bins]
        cos = cos[..., :n_bins]
    return amp, sin, cos


class TokenizerVQSNP(pl.LightningModule):
    def __init__(self, n_chans=129, crop_len=200, patch_len=20,
                 dim=256, num_codes=8192, n_fft_bins=11, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.P = crop_len // patch_len
        self.patch_len = patch_len
        # tiny convs over time per channel×patch
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3), padding=(0, 1)), nn.GELU(),
            nn.Conv2d(64, 128, (1, 3), padding=(0, 1)), nn.GELU(),
            nn.Conv2d(128, dim, (1, 3), padding=(0, 1)), nn.GELU(),
        )  # input will be (B,1,C*P,L), see forward
        self.vq = VectorQuantEMA(num_codes=num_codes, dim=dim)
        # prediction heads: amp, sin, cos for K fft bins
        K = n_fft_bins
        self.head_amp = nn.Linear(dim, K)
        self.head_sin = nn.Linear(dim, K)
        self.head_cos = nn.Linear(dim, K)

    def patchify(self, x):  # (B,C,T)->(B,C,P,L)
        B, C, T = x.shape
        L = self.patch_len
        P = T//L
        return x.view(B, C, P, L)

    def forward(self, x):   # returns codes (B,C,P)
        B, C, T = x.shape
        xp = self.patchify(x)                       # (B,C,P,L)
        # reshape for conv2d: treat channel×patch as spatial rows
        xp2 = xp.reshape(B, 1, C*self.P, self.patch_len)
        # (B,dim,C*P) pooled over time
        z = self.enc(xp2).mean(-1)
        z = z.permute(0, 2, 1).reshape(B, C, self.P, -1)  # (B,C,P,dim)
        zq, vq_loss, codes = self.vq(z)             # quantized
        return z, zq, vq_loss, codes, xp

    def training_step(self, batch, _):
        x = batch  # (B,C,T)
        z, zq, vq_loss, codes, xp = self.forward(x)
        amp_t, sin_t, cos_t = rfft_targets(xp, n_bins=self.hparams.n_fft_bins)
        amp_h = self.head_amp(zq).squeeze(-2)   # (B,C,P,K)
        sin_h = self.head_sin(zq).squeeze(-2)
        cos_h = self.head_cos(zq).squeeze(-2)
        loss = (F.mse_loss(amp_h, amp_t) +
                F.mse_loss(sin_h, sin_t) +
                F.mse_loss(cos_h, cos_t) +
                vq_loss)
        self.log("tok_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
