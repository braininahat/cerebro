# masked_code_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MaskedCodeModel(pl.LightningModule):
    def __init__(self, num_codes=8192, C=129, P=10, D=256,
                 L=8, H=8, mask_ratio=0.5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        N = C*P
        self.embed = nn.Embedding(num_codes+2, D)    # +MASK +PAD
        self.mask_token_id = num_codes
        self.pad_token_id = num_codes+1
        self.pos = nn.Parameter(torch.randn(1, N, D))  # learnable 1D pos
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=H, batch_first=True)
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=L)
        self.head = nn.Linear(D, num_codes)          # logits over codes
        self.C, self.P = C, P
        self.mask_ratio = mask_ratio

    def make_mask(self, B):
        N = self.C*self.P
        k = max(1, int(round(self.mask_ratio*N)))
        m = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        for b in range(B):
            idx = torch.randperm(N, device=self.device)[:k]
            m[b, idx] = True
        return m

    def training_step(self, batch, _):
        codes = batch["codes"].to(self.device)  # (B,C,P) ints
        B, C, P = codes.shape
        N = C*P
        seq = codes.view(B, N)
        mask = self.make_mask(B)
        seq_masked = seq.clone()
        seq_masked[mask] = self.mask_token_id

        x = self.embed(seq_masked) + self.pos          # (B,N,D)
        z = self.tr(x)                                  # (B,N,D)
        logits = self.head(z)                           # (B,N,V)
        loss = F.cross_entropy(logits[mask], seq[mask])  # CE on masked only
        self.log("mc_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
