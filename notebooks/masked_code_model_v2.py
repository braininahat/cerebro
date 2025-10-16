# masked_code_model_v2.py
"""
Improved masked code modeling with spatial awareness.
Considers channel structure when masking, following LaBraM's approach.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SpatialMaskedCodeModel(pl.LightningModule):
    """
    Masked code modeling with spatial (channel) awareness.

    Improvements over simple random masking:
    - Separate channel and temporal position embeddings
    - Spatially-aware masking (mask contiguous channel-patch blocks)
    - Better transformer architecture
    - Improved prediction head with layer norm
    """

    def __init__(
        self,
        num_codes=8192,
        C=129,  # number of channels
        P=10,   # number of patches per channel
        D=256,  # embedding dimension
        n_layers=8,
        n_heads=8,
        mask_ratio=0.5,
        lr=1e-3,
        dropout=0.1,
        mask_strategy="spatial",  # "spatial", "random", or "channel"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.C = C
        self.P = P
        self.D = D
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy

        # Embeddings
        self.code_embed = nn.Embedding(num_codes + 2, D)  # +MASK +PAD
        self.mask_token_id = num_codes
        self.pad_token_id = num_codes + 1

        # Separate channel and temporal position embeddings (following LaBraM)
        self.channel_embed = nn.Embedding(C, D)
        self.patch_embed = nn.Embedding(P, D)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=n_heads,
            dim_feedforward=4*D,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # Prediction head with layer norm
        self.head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D, num_codes)
        )

        self.lr = lr

    def make_spatial_mask(self, B, device):
        """
        Create spatially-aware mask (mask contiguous channel-patch blocks).
        This encourages the model to learn spatial relationships.
        """
        N = self.C * self.P
        k = max(1, int(round(self.mask_ratio * N)))

        mask = torch.zeros(B, self.C, self.P, dtype=torch.bool, device=device)

        for b in range(B):
            # Randomly select channel blocks to mask
            n_channel_blocks = max(1, k // self.P)
            n_channel_blocks = min(n_channel_blocks, self.C)

            channels_to_mask = torch.randperm(self.C, device=device)[
                :n_channel_blocks]

            for ch in channels_to_mask:
                # Mask random contiguous patches in this channel
                n_patches = min(self.P, max(1, k // len(channels_to_mask)))

                # Random contiguous span
                if n_patches >= self.P:
                    mask[b, ch, :] = True
                else:
                    start_idx = torch.randint(
                        0, self.P - n_patches + 1, (1,), device=device).item()
                    mask[b, ch, start_idx:start_idx + n_patches] = True

        return mask.view(B, -1)

    def make_channel_mask(self, B, device):
        """
        Mask entire channels at a time.
        This encourages learning cross-channel relationships.
        """
        n_channels_to_mask = max(1, int(round(self.mask_ratio * self.C)))

        mask = torch.zeros(B, self.C, self.P, dtype=torch.bool, device=device)

        for b in range(B):
            channels = torch.randperm(self.C, device=device)[
                :n_channels_to_mask]
            mask[b, channels, :] = True

        return mask.view(B, -1)

    def make_random_mask(self, B, device):
        """Standard random masking."""
        N = self.C * self.P
        k = max(1, int(round(self.mask_ratio * N)))

        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(N, device=device)[:k]
            mask[b, idx] = True

        return mask

    def make_mask(self, B, device):
        """Create mask based on strategy."""
        if self.mask_strategy == "spatial":
            return self.make_spatial_mask(B, device)
        elif self.mask_strategy == "channel":
            return self.make_channel_mask(B, device)
        else:  # "random"
            return self.make_random_mask(B, device)

    def forward(self, codes, return_attention=False):
        """
        codes: (B, C, P) long tensor
        Returns:
            loss: scalar
            logits: (B, C*P, num_codes)
            mask: (B, C*P) boolean mask
        """
        B, C, P = codes.shape
        codes_flat = codes.view(B, -1)  # (B, C*P)

        # Create mask
        mask = self.make_mask(B, codes.device)

        # Apply mask
        codes_masked = codes_flat.clone()
        codes_masked[mask] = self.mask_token_id

        # Code embeddings
        code_emb = self.code_embed(codes_masked)  # (B, C*P, D)

        # Add positional information
        channel_ids = torch.arange(C, device=codes.device).repeat_interleave(P)
        channel_ids = channel_ids.unsqueeze(0).expand(B, -1)  # (B, C*P)

        patch_ids = torch.arange(P, device=codes.device).repeat(C)
        patch_ids = patch_ids.unsqueeze(0).expand(B, -1)  # (B, C*P)

        pos_emb = self.channel_embed(channel_ids) + self.patch_embed(patch_ids)

        x = code_emb + pos_emb  # (B, C*P, D)

        # Transformer
        z = self.transformer(x)  # (B, C*P, D)

        # Predict codes
        logits = self.head(z)  # (B, C*P, num_codes)

        # Compute loss only on masked positions
        loss = F.cross_entropy(
            logits[mask].reshape(-1, logits.shape[-1]),
            codes_flat[mask].reshape(-1)
        )

        return loss, logits, mask

    def training_step(self, batch, batch_idx):
        codes = batch["codes"]  # (B, C, P)
        loss, logits, mask = self.forward(codes)

        # Calculate accuracy on masked positions
        with torch.no_grad():
            codes_flat = codes.view(codes.size(0), -1)
            preds = logits.argmax(dim=-1)
            acc = (preds[mask] == codes_flat[mask]).float().mean()

        self.log("mc_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("mc_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("mask_ratio", mask.float().mean(),
                 on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        codes = batch["codes"]  # (B, C, P)
        loss, logits, mask = self.forward(codes)

        # Calculate accuracy
        with torch.no_grad():
            codes_flat = codes.view(codes.size(0), -1)
            preds = logits.argmax(dim=-1)
            acc = (preds[mask] == codes_flat[mask]).float().mean()

        self.log("val_mc_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_mc_acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )

        # Cosine annealing with warmup
        def lr_lambda(current_step):
            max_steps = self.trainer.max_epochs
            warmup_steps = max(1, int(0.1 * max_steps))

            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / \
                float(max(1, max_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item()))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }


# Backward compatibility with original MaskedCodeModel
class MaskedCodeModel(SpatialMaskedCodeModel):
    """Alias for backward compatibility"""

    def __init__(self, num_codes=8192, C=129, P=10, D=256,
                 L=8, H=8, mask_ratio=0.5, lr=1e-3):
        super().__init__(
            num_codes=num_codes,
            C=C,
            P=P,
            D=D,
            n_layers=L,
            n_heads=H,
            mask_ratio=mask_ratio,
            lr=lr,
            mask_strategy="random"  # Use random for backward compatibility
        )
