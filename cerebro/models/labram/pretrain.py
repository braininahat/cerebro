# --------------------------------------------------------
# LaBraM-style Masked EEG Modeling (MEM) with PyTorch Lightning
# - Dynamic pos/time embeddings (n_chans_hint, max_time_window_hint)
# - TemporalConv front-end with projection to embed_dim
# - Random masking + symmetric objective (mask and inverse)
# - Uses external VQ tokenizer for labels via get_codebook_indices()
# --------------------------------------------------------

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from einops import rearrange
from torch.nn.init import trunc_normal_ as __call_trunc_normal_


# Reuse your existing Transformer blocks from LaBraM finetune code
# (must provide `Block` and `_cfg` in modeling_finetune.py)
from .finetune import Block, _cfg


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


# -------------------------------
# Temporal front-end (tokenizer-like patch embed over time)
# -------------------------------
class TemporalConv(nn.Module):
    """Temporal conv stack used as tokenizer/patch embed for raw EEG patches."""

    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        # conv1 downsamples time by stride=8
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(
            1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)

        self.conv2 = nn.Conv2d(out_chans, out_chans,
                               kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)

        self.conv3 = nn.Conv2d(out_chans, out_chans,
                               kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        # x: [B, N, A, T]
        x = rearrange(x, 'B N A T -> B (N A) T')
        x = x.unsqueeze(1)               # [B, 1, N*A, T]
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA L -> B NA (L C)')
        return x


# -------------------------------
# Student MEM backbone (masked modeling head)
# -------------------------------
class NeuralTransformerForMaskedEEGModeling(nn.Module):
    """
    - Accepts raw EEG patches [B, N, A, T]
    - TemporalConv extracts per-token temporal features (dim may != embed_dim)
    - Optional Linear projection aligns token dim to embed_dim
    - Dynamic pos/time embeddings via hints and runtime matching
    - Masking with a learnable mask token
    """

    def __init__(
        self,
        EEG_size=1600,
        patch_size=200,
        in_chans=1,
        out_chans=8,
        vocab_size=8192,
        embed_dim=200,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_norm=None,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=None,
        init_values=None,
        attn_head_dim=None,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        init_std=0.02,
        # NEW: hints for dynamic embeddings
        n_chans_hint=129,
        max_time_window_hint=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.init_std = init_std

        # Temporal front-end
        self.patch_embed = TemporalConv(out_chans=out_chans)

        # infer default time windows if not provided
        if max_time_window_hint is None:
            max_time_window_hint = max(1, EEG_size // patch_size)

        # projection to embed_dim if needed
        # after conv1: L' = floor((T - 1)/8) + 1
        Lp = (self.patch_size - 1) // 8 + 1
        temporal_dim = Lp * out_chans
        self._pre_embed_proj = None
        if temporal_dim != embed_dim:
            self._pre_embed_proj = nn.Linear(temporal_dim, embed_dim)

        # learned tokens/embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, n_chans_hint + 1, embed_dim))
        else:
            self.pos_embed = None

        self.time_embed = nn.Parameter(
            torch.zeros(1, max_time_window_hint, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, init_values=init_values,
                window_size=None, attn_head_dim=attn_head_dim,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # head
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        # init
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.time_embed, std=self.init_std)
        trunc_normal_(self.cls_token,  std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self._fix_init_weight()

    def _fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    # runtime padding/truncation (prevents 259 vs 257 errors)
    def _match_pos_embed(self, pos_embed, need_tokens):
        cur = pos_embed.shape[1]
        if cur == need_tokens:
            return pos_embed
        if cur > need_tokens:
            return pos_embed[:, :need_tokens, :]
        pad = need_tokens - cur
        last = pos_embed[:, -1:, :].expand(1, pad, pos_embed.size(-1))
        return torch.cat([pos_embed, last], dim=1)

    def _match_time_embed(self, time_embed, need_T):
        cur = time_embed.shape[1]
        if cur >= need_T:
            return time_embed[:, :need_T, :]
        pad = need_T - cur
        last = time_embed[:, -1:, :].expand(1, pad, time_embed.size(-1))
        return torch.cat([time_embed, last], dim=1)

    def forward_features(self, x, input_chans, bool_masked_pos):
        """
        x: [B, N, A, T]  (T==patch_size ideally), N channels, A time windows
        bool_masked_pos: [B, N*A] over patch tokens (no CLS)
        """
        B, N, A, T = x.shape
        x = self.patch_embed(x)                      # [B, N*A, temporal_dim]
        if self._pre_embed_proj is not None:
            x = self._pre_embed_proj(x)              # [B, N*A, embed_dim]
        seq_len = x.size(1)
        assert seq_len == N * \
            A, f"token count mismatch: got {seq_len}, expected {N*A}"

        cls_tokens = self.cls_token.expand(B, 1, -1)
        mask_token = self.mask_token.expand(B, seq_len, -1)

        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros(
                (B, N*A), dtype=torch.bool, device=x.device)
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)  # [B, N*A, 1]
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)       # [B, 1+N*A, D]
        L = x.size(1)                # total tokens with CLS
        L_no_cls = L - 1             # tokens excluding CLS

        # ------- CHANNEL POSITIONAL (CLS-safe, per-channel then expand over A) -------
        if self.pos_embed is not None:
            if input_chans is not None:
                idx = torch.as_tensor(
                    input_chans, device=self.pos_embed.device).long()   # [N]
                idx = idx + 1                              # shift for CLS at 0
                # [1+N] with CLS first
                idx = torch.cat(
                    [torch.zeros(1, dtype=idx.dtype, device=idx.device), idx], dim=0)
                pos_used = self.pos_embed[:, idx]          # [1, 1+N, D]
            else:
                pos_used = self.pos_embed                  # [1, 1+N_hint, D]

            # ensure we have at least 1+N tokens in pos_used (pad/truncate to 1+N)
            pos_used = self._match_pos_embed(
                pos_used, need_tokens=1 + N)   # <-- only 1+N here

            cls_pos = pos_used[:, :1, :].expand(
                B, -1, -1)                 # [B, 1, D]
            # [B, N*A, D]
            chan_pos = pos_used[:, 1:1+N,
                                :].unsqueeze(2).expand(B, N, A, -1).flatten(1, 2)
            chan_pos = chan_pos[:, :L_no_cls, :]
            # [B, 1+N*A, D]
            pos_seq = torch.cat((cls_pos, chan_pos), dim=1)
            x = x + pos_seq

        # ------- TIME POSITIONAL (per-window, then tile across N) -------
        # [1, A, D] (pad/trunc to A)
        te = self._match_time_embed(self.time_embed, need_T=A)
        te = te.unsqueeze(1).expand(B, N, -1, -1).flatten(1, 2)  # [B, N*A, D]
        te = te[:, :L_no_cls, :]
        x[:, 1:, :] += te

        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)
        return self.norm(x)

    def forward(self, x, input_chans=None, bool_masked_pos=None, return_all_patch_tokens=False):
        feats = self.forward_features(x, input_chans, bool_masked_pos)
        if return_all_patch_tokens:
            return feats
        patch_tokens = feats[:, 1:]                 # [B, N*A, D]
        logits = F.linear(patch_tokens, self.lm_head.weight,
                          self.lm_head.bias)  # [B, N*A, vocab]
        return logits


# -------------------------------
# Symmetric MEM wrapper (student-only head)
# -------------------------------
class NeuralTransformerForMEM(nn.Module):
    """
    Calls student twice (mask and inverse) like LaBraM MEM.
    Returns logits for masked tokens (x_rec) and inverse masked tokens (x_rec_sym).
    """

    def __init__(
        self,
        EEG_size=1600,
        patch_size=200,
        in_chans=1,
        out_chans=8,
        vocab_size=8192,
        embed_dim=200,
        depth=12,
        num_heads=10,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_norm=None,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=None,
        init_values=None,
        attn_head_dim=None,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        init_std=0.02,
        n_chans_hint=129,
        max_time_window_hint=None,
        **kwargs
    ):
        super().__init__()
        self.patch_size = patch_size
        self.student = NeuralTransformerForMaskedEEGModeling(
            EEG_size=EEG_size,
            patch_size=patch_size,
            in_chans=in_chans,
            out_chans=out_chans,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer or partial(nn.LayerNorm, eps=1e-6),
            init_values=init_values,
            attn_head_dim=attn_head_dim,
            use_abs_pos_emb=use_abs_pos_emb,
            use_rel_pos_bias=use_rel_pos_bias,
            use_shared_rel_pos_bias=use_shared_rel_pos_bias,
            init_std=init_std,
            n_chans_hint=n_chans_hint,
            max_time_window_hint=max_time_window_hint,
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        trunc_normal_(self.lm_head.weight, std=init_std)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'student.cls_token', 'student.pos_embed', 'student.time_embed'}

    def forward(self, x, input_chans=None, bool_masked_pos=None):
        B, N, A, T = x.shape
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros(
                (B, N*A), dtype=torch.bool, device=x.device)

        feats_masked = self.student(
            x, input_chans, bool_masked_pos, return_all_patch_tokens=True)  # [B, 1+N*A, D]
        patch_tokens = feats_masked[:, 1:]               # [B, N*A, D]
        # [#masked, vocab]
        x_rec = self.lm_head(patch_tokens[bool_masked_pos])

        feats_sym = self.student(
            x, input_chans, ~bool_masked_pos, return_all_patch_tokens=True)
        patch_tokens_sym = feats_sym[:, 1:]
        x_rec_sym = self.lm_head(
            patch_tokens_sym[~bool_masked_pos])  # [#unmasked, vocab]

        return x_rec, x_rec_sym


def random_masking(x_flat: torch.Tensor, mask_ratio: float) -> torch.BoolTensor:
    """
    x_flat: [B, L, D] (sequence already flattened to L=N*A tokens)
    Returns: mask [B, L] with True = masked
    """
    B, L, _ = x_flat.shape
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(B, L, device=x_flat.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    # create mask (1 masked, 0 keep), then unshuffle to original order
    mask = torch.ones(B, L, device=x_flat.device)
    mask[:, :len_keep] = 0.0
    mask = torch.gather(mask, dim=1, index=ids_restore).to(torch.bool)
    return mask


class MEMPretrainModule(pl.LightningModule):
    """
    LightningModule that trains MEM with labels from a VQ tokenizer:
      - Constructs NeuralTransformerForMEM internally
      - Constructs VQNSP tokenizer internally
      - x expected from dataloader as [B, N, T_total]; we chunk to [B, N, A, T]
    """

    def __init__(
        self,
        # Model architecture parameters (NeuralTransformerForMEM)
        EEG_size: int = 200,
        patch_size: int = 100,
        in_chans: int = 1,
        out_chans: int = 8,
        vocab_size: int = 8192,
        embed_dim: int = 200,
        depth: int = 12,
        num_heads: int = 10,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm=None,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_std: float = 0.02,
        n_chans: int = 129,
        max_time_window_hint: Optional[int] = None,
        # Tokenizer parameters (VQNSP)
        pretrained: bool = False,
        pretrained_weight: Optional[str] = None,
        n_code: int = 8192,
        code_dim: int = 32,
        encoder_depth: int = 24,
        decoder_depth: int = 3,
        decay: float = 0.99,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        betas: Tuple[float, float] = (0.9, 0.95),
        warmup_epochs: int = 5,          # Linear warmup epochs (matches original)
        mask_ratio: float = 0.5,
        scale_input: bool = False,       # divide by 100 like engine
        scale_eeg: bool = False,         # alias for scale_input (from YAML)
        chunk_pad: bool = True,         # pad/truncate T_total to multiple of patch_size
        use_cosine_per_step: bool = True,
        eta_min: float = 1e-6,
        **kwargs
    ):
        super().__init__()

        # Handle scale_eeg alias
        if scale_eeg:
            scale_input = scale_eeg

        # Build the student model (NeuralTransformerForMEM)
        self.model = NeuralTransformerForMEM(
            EEG_size=EEG_size,
            patch_size=patch_size,
            in_chans=in_chans,
            out_chans=out_chans,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=None,
            attn_head_dim=None,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            init_std=init_std,
            n_chans_hint=n_chans,
            max_time_window_hint=max_time_window_hint or (
                EEG_size // patch_size),
        )

        # Build the tokenizer (VQNSP) - uses simplified interface
        from .tokenizer import VQNSP

        self.tokenizer = VQNSP(
            pretrained=pretrained,
            as_tokenizer=pretrained and pretrained_weight is not None,
            pretrained_weight=pretrained_weight,
            n_code=n_code,
            code_dim=code_dim,
            EEG_size=EEG_size,
            patch_size=patch_size,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            decay=decay,
            n_chans=n_chans,
            max_time_window_hint=max_time_window_hint,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )

        self.tokenizer.eval()  # tokenizer in eval mode

        # Store hyperparameters (exclude constructed modules)
        self.save_hyperparameters(ignore=['model', 'tokenizer'])

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.scale_input = scale_input
        self.chunk_pad = chunk_pad
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_epochs = warmup_epochs
        self.use_cosine_per_step = use_cosine_per_step
        self.eta_min = eta_min

        self.loss_fn = nn.CrossEntropyLoss()

    # ---- helpers ----
    def _chunk_to_patches(self, x_2d: torch.Tensor) -> torch.Tensor:
        """
        x_2d: [B, N, T_total] -> [B, N, A, T], T=self.patch_size.
        Crop or pad (zeros) to nearest multiple of T.
        """
        B, N, Ttotal = x_2d.shape
        T = self.patch_size
        if Ttotal < T:
            pad = T - Ttotal
            x_2d = F.pad(x_2d, (0, pad))
            Ttotal = T
        A = Ttotal // T
        rem = Ttotal - A * T
        if rem != 0:
            if self.chunk_pad:
                # pad up to next multiple
                pad = T - rem
                x_2d = F.pad(x_2d, (0, pad))
                A = (Ttotal + pad) // T
            else:
                # crop down to multiple
                x_2d = x_2d[..., :A*T]
        x_4d = rearrange(x_2d, 'B N (A T) -> B N A T', A=A, T=T)
        return x_4d

    def _get_input_chans(self, x_4d: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Optionally provide an index tensor for channel positions, or None to use all.
        Here we return None (match original engine default).
        """
        return None

    def _mem_step(self, x: torch.Tensor, split: str):
        """
        One forward+loss pass for train/val.
        x: [B, N, T_total]
        """
        x = x.float()
        if self.scale_input:
            x = x / 100.0

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self._chunk_to_patches(x)                        # [B, N, A, T]
        B, N, A, T = x.shape

        # build mask over L=N*A tokens
        x_flat = rearrange(x, 'B N A T -> B (N A) T')
        mask = random_masking(x_flat, mask_ratio=self.mask_ratio)  # [B, N*A]
        input_chans = self._get_input_chans(x)

        # code labels from tokenizer (no grad)
        with torch.no_grad():
            input_ids = self.tokenizer.get_codebook_indices(
                x, input_chans)  # [B, N*A]
            labels = input_ids[mask]
            labels_sym = input_ids[~mask]

        # forward student twice (mask and inverse)
        x_rec, x_rec_sym = self.model(
            x, input_chans=input_chans, bool_masked_pos=mask)
        loss_rec = self.loss_fn(x_rec, labels)
        loss_rec_sym = self.loss_fn(x_rec_sym, labels_sym)
        loss = loss_rec + loss_rec_sym

        # metrics: accuracy on masked/unmasked predictions
        mlm_acc = (x_rec.argmax(-1) == labels).float().mean(
        ) if x_rec.numel() else torch.tensor(0., device=x.device)
        mlm_acc_sym = (x_rec_sym.argmax(-1) == labels_sym).float().mean(
        ) if x_rec_sym.numel() else torch.tensor(0., device=x.device)

        # logging
        self.log(f"{split}/loss", loss, on_step=(split ==
                 "train"), on_epoch=True, prog_bar=True)
        self.log(f"{split}/loss_rec_half", loss_rec / 2.0,
                 on_step=(split == "train"), on_epoch=True)
        self.log(f"{split}/mlm_acc", mlm_acc, on_step=(split ==
                 "train"), on_epoch=True, prog_bar=True)
        self.log(f"{split}/mlm_acc_sym", mlm_acc_sym,
                 on_step=(split == "train"), on_epoch=True)

        # for checkpoint monitor
        if split == "val":
            self.log("val_mem_loss", loss, on_step=False,
                     on_epoch=True, prog_bar=True)

        return loss

    # ---- Lightning hooks ----
    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self._mem_step(x, split="train")

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self._mem_step(x, split="val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        if self.use_cosine_per_step:
            # step-wise warmup + cosine like original LaBraM
            total_steps = getattr(
                self.trainer, "estimated_stepping_batches", None)
            if total_steps is None:
                # fallback: per-epoch cosine without warmup
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=self.trainer.max_epochs, eta_min=self.eta_min)
                return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

            # Calculate warmup steps
            steps_per_epoch = total_steps // self.trainer.max_epochs
            warmup_steps = self.warmup_epochs * steps_per_epoch

            # Warmup: Linear from ~0 to base_lr (matches original)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=1e-10,      # Nearly 0 (avoid division by zero)
                end_factor=1.0,          # Reach base_lr
                total_iters=warmup_steps
            )

            # Main: Cosine from base_lr to eta_min
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=total_steps - warmup_steps,
                eta_min=self.eta_min
            )

            # Chain warmup → cosine (matches original LaBraM)
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )

            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"},
            }
        return opt


# -------------------------------
# Timmmable builders (optional)
# -------------------------------
def labram_base_patch100_200_8k_vocab(pretrained: bool = False, **kwargs):
    """
    Base MEM student for 100-sample patches on 2s windows at 100 Hz.
    -> EEG_size = 200, patch_size = 100, A=2
    """
    kwargs.pop("num_classes", None)
    vocab_size = kwargs.pop("vocab_size", 8192)

    # Hints to make pos/time embeddings line up with your montage and A=2
    n_chans_hint = kwargs.pop("n_chans_hint", 129)
    max_time_window_hint = kwargs.pop("max_time_window_hint", 2)

    model = NeuralTransformerForMEM(
        EEG_size=200,
        patch_size=100,
        embed_dim=200,
        depth=12,
        num_heads=10,
        mlp_ratio=4,
        qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        out_chans=8,
        vocab_size=vocab_size,
        n_chans_hint=n_chans_hint,
        max_time_window_hint=max_time_window_hint,
        **kwargs,
    )
    model.default_cfg = _cfg()

    if pretrained:
        ckpt = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
    return model
