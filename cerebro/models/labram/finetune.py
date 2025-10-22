# --------------------------------------------------------
# LaBraM-style Transformer for EEG Finetuning (Regression, PL)
# - 100 Hz, 2 s windows (T_total=200) → patch_size=100, A=2
# - TemporalConv front-end + projection to embed_dim (if needed)
# - Dynamic pos/time embeddings with safe pad/truncate
# - LightningModule wrapper for regression (CCD reaction times)
# --------------------------------------------------------

from typing import Tuple
from torchmetrics import MeanMetric
from torchmetrics.regression import MeanSquaredError
import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from einops import rearrange
from torch.nn.init import trunc_normal_
from pathlib import Path

# -----------------------
# Timm-style config helper
# -----------------------


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1, 'input_size': (1, 129, 200), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5,), 'std': (0.5,),
        **kwargs
    }
# === add this utility anywhere above the factories ===


def _flexible_state_dict_load(
    module: nn.Module,
    src: dict,
    ignore_mlp_head: bool = True,
    strict: bool = False,
    verbose: bool = True,
):
    """
    Load weights with common key/layout variations handled:
      - src can be a full checkpoint or a raw state_dict
      - accepts keys under 'model', 'state_dict', etc.
      - strips 'module.' prefix
      - optionally drops final head weights if shapes don't match
    """
    # 1) pull the right dict
    if any(k in src for k in ("model", "state_dict", "ema")):
        state = src.get("model") or src.get("state_dict") or src.get("ema")
    else:
        state = src

    # 2) normalize keys (strip leading "module.")
    fixed = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        fixed[k] = v

    if ignore_mlp_head:
        # find head params present in current model and drop mismatch tensors
        head_keys = [k for k in fixed.keys() if k.startswith("head.")]
        for k in head_keys:
            if k in module.state_dict():
                # keep only if same shape; otherwise drop
                if module.state_dict()[k].shape != fixed[k].shape:
                    if verbose:
                        print(f"[load] drop mismatched head param: {k} "
                              f"{tuple(fixed[k].shape)} -> expected {tuple(module.state_dict()[k].shape)}")
                    fixed.pop(k, None)
            else:
                fixed.pop(k, None)

    missing, unexpected = module.load_state_dict(fixed, strict=strict)
    if verbose:
        if missing:
            print(f"[load] missing keys: {len(missing)} (e.g., {missing[:4]})")
        if unexpected:
            print(
                f"[load] unexpected keys: {len(unexpected)} (e.g., {unexpected[:4]})")


# -----------------------
# Blocks / Attention / MLP
# -----------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None,
                 attn_drop=0., proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = (attn_head_dim or (dim // num_heads))
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(
            all_head_dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(
            all_head_dim)) if qkv_bias else None

        self.q_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.k_norm = qk_norm(head_dim) if qk_norm is not None else None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias,
                                  torch.zeros_like(
                                      self.v_bias, requires_grad=False),
                                  self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_qkv:
            return x, qkv
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., init_values=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_norm, qk_scale,
                              attn_drop, drop, window_size, attn_head_dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x),
                                   rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 *
                                   self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# -----------------------
# Front-end tokenization
# -----------------------
class PatchEmbed(nn.Module):
    """Linear patch embed (used for decoder/code inputs)."""

    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = 62 * (EEG_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(
            1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        # x is expected as [B, C, H, W] for this path (not used in this finetune head)
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TemporalConv(nn.Module):
    """Temporal tokenizer for raw EEG: [B, N, A, T] -> [B, N*A, L'*C]"""

    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
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
        x = x.unsqueeze(1)                          # [B, 1, N*A, T]
        x = self.gelu1(self.norm1(self.conv1(x)))   # downsample time by ~8x
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA L -> B NA (L C)')
        return x


# -----------------------
# Backbone (finetune)
# -----------------------
class NeuralTransformer(nn.Module):
    """
    Backbone used for finetuning:
    - TemporalConv → (optional Linear proj) → Transformer → pooled → regression head
    - Dynamic pos_embed/time_embed sized by n_chans_hint and max_time_window_hint
    """

    def __init__(self, EEG_size=200, patch_size=100, in_chans=1, out_chans=8, num_classes=1, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, n_chans_hint=129, max_time_window_hint=2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Temporal tokenizer
        self.patch_embed = TemporalConv(out_chans=out_chans) if in_chans == 1 else PatchEmbed(
            EEG_size=EEG_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # Determine output temporal dim from conv1 path: L' = floor((T + 7)/8)
        self._pre_embed_proj = None
        if isinstance(self.patch_embed, TemporalConv):
            Lp = (self.patch_size + 7) // 8
            temporal_dim = Lp * out_chans          # e.g., T=100 → L'=13 → 13*8=104
            if temporal_dim != embed_dim:
                self._pre_embed_proj = nn.Linear(temporal_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Absolute positional + time embeddings (match dynamically)
        self.pos_embed = nn.Parameter(torch.zeros(
            1, n_chans_hint + 1, embed_dim)) if use_abs_pos_emb else None
        self.time_embed = nn.Parameter(
            torch.zeros(1, max_time_window_hint, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                  init_values=init_values, window_size=None)
            for i in range(depth)
        ])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        # Regression head for CCD response time
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        ) if num_classes > 0 else nn.Identity()

        # Init
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.fix_init_weight()

    @classmethod
    def from_pretrained(cls,
                        init_ckpt: str | Path | dict,
                        map_location: str = "cpu",
                        ignore_mlp_head: bool = True,
                        strict: bool = False,
                        verbose: bool = True,
                        **kwargs):
        """
        Build a backbone and load weights.
        `init_ckpt` can be a path or a state_dict-like object.
        """
        model = cls(**kwargs)
        if isinstance(init_ckpt, (str, Path)):
            ckpt = torch.load(str(init_ckpt), map_location=map_location)
        else:
            ckpt = init_ckpt
        _flexible_state_dict_load(
            model, ckpt, ignore_mlp_head=ignore_mlp_head, strict=strict, verbose=verbose)
        return model

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    # ---- pad/truncate helpers (avoid off-by-one errors) ----
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

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False):
        """
        x: [B, N, A, T] with T==self.patch_size (here 100), A==2 for 2s @ 100Hz
        """
        B, N, A, T = x.shape
        input_time_window = A if T == self.patch_size else T

        # [B, N*A, L'*C] or [B, N*A, embed_dim]
        x = self.patch_embed(x)
        if self._pre_embed_proj is not None:
            x = self._pre_embed_proj(x)          # [B, N*A, embed_dim]

        cls_tokens = self.cls_token.expand(B, 1, -1)
        x = torch.cat((cls_tokens, x), dim=1)    # [B, 1+N*A, D]

        pos_embed_used = self.pos_embed[:, input_chans] if (
            self.pos_embed is not None and input_chans is not None) else self.pos_embed
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(
                B, -1, input_time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:, 0:1, :].expand(
                B, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            nc = N if T == self.patch_size else A
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(
                1).expand(B, nc, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, input_chans=None, **kwargs):
        feats = self.forward_features(
            x, input_chans=input_chans, **kwargs)  # [B, D]
        return self.head(feats)  # [B, 1]


# -----------------------
# Lightning wrapper
# -----------------------


class EEGRegressorPL(pl.LightningModule):
    """
    Finetuning wrapper for regression:
      - Expects batch as (x, y) with x:[B,N,T_total] (T_total=200), y:[B] or [B,1]
      - Chunks to [B,N,A=2,T=100], scales by /100 like pretrain
      - Logs global RMSE and NRMSE (= RMSE / std(y_true)) for train/val/test
    """

    def __init__(self,
                 # NeuralTransformer architecture parameters
                 EEG_size: int = 200,
                 patch_size: int = 100,
                 in_chans: int = 1,
                 out_chans: int = 8,
                 num_classes: int = 1,
                 embed_dim: int = 200,
                 depth: int = 12,
                 num_heads: int = 10,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 init_values: Optional[float] = None,
                 use_abs_pos_emb: bool = True,
                 use_rel_pos_bias: bool = False,
                 use_shared_rel_pos_bias: bool = False,
                 use_mean_pooling: bool = True,
                 init_scale: float = 0.001,
                 n_chans_hint: int = 129,
                 max_time_window_hint: int = 2,
                 # Pretrained weights
                 pretrained: bool = False,
                 pretrained_weight: Optional[str] = None,
                 # Training parameters
                 lr: float = 1e-4,
                 weight_decay: float = 0.05,
                 betas: Tuple[float, float] = (0.9, 0.95),
                 use_cosine_per_step: bool = True,
                 eta_min: float = 1e-6,
                 scale_input: bool = True):
        super().__init__()
        self.save_hyperparameters()

        # Build backbone from arguments (use nn.LayerNorm internally)
        if pretrained and pretrained_weight:
            self.model = NeuralTransformer.from_pretrained(
                init_ckpt=pretrained_weight,
                EEG_size=EEG_size,
                patch_size=patch_size,
                in_chans=in_chans,
                out_chans=out_chans,
                num_classes=num_classes,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=None,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=nn.LayerNorm,
                init_values=init_values,
                use_abs_pos_emb=use_abs_pos_emb,
                use_rel_pos_bias=use_rel_pos_bias,
                use_shared_rel_pos_bias=use_shared_rel_pos_bias,
                use_mean_pooling=use_mean_pooling,
                init_scale=init_scale,
                n_chans_hint=n_chans_hint,
                max_time_window_hint=max_time_window_hint,
                ignore_mlp_head=True,
                strict=False,
            )
        else:
            self.model = NeuralTransformer(
                EEG_size=EEG_size,
                patch_size=patch_size,
                in_chans=in_chans,
                out_chans=out_chans,
                num_classes=num_classes,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=None,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=nn.LayerNorm,
                init_values=init_values,
                use_abs_pos_emb=use_abs_pos_emb,
                use_rel_pos_bias=use_rel_pos_bias,
                use_shared_rel_pos_bias=use_shared_rel_pos_bias,
                use_mean_pooling=use_mean_pooling,
                init_scale=init_scale,
                n_chans_hint=n_chans_hint,
                max_time_window_hint=max_time_window_hint,
            )

        self.patch_size = patch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.use_cosine_per_step = use_cosine_per_step
        self.eta_min = eta_min
        self.scale_input = scale_input
        self.loss_fn = nn.MSELoss()

        # ---- TorchMetrics for global RMSE/NRMSE (per stage) ----
        # Train
        self.train_rmse = MeanSquaredError(squared=False)
        self.train_y_mean = MeanMetric()
        self.train_y_sq_mean = MeanMetric()
        # Val
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_y_mean = MeanMetric()
        self.val_y_sq_mean = MeanMetric()
        # Test
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_y_mean = MeanMetric()
        self.test_y_sq_mean = MeanMetric()

    # -------- data shaping --------
    def _chunk_to_patches(self, x_2d: torch.Tensor) -> torch.Tensor:
        # [B, N, T_total] -> [B, N, A, T]
        B, N, Ttotal = x_2d.shape
        T = self.patch_size
        if Ttotal < T:
            x_2d = F.pad(x_2d, (0, T - Ttotal))
            Ttotal = T
        A = Ttotal // T
        rem = Ttotal - A * T
        if rem != 0:
            x_2d = x_2d[..., :A * T]
        x_4d = rearrange(x_2d, 'B N (A T) -> B N A T', A=A, T=T)
        return x_4d  # expect A==2 for 2s@100Hz with T=100

    def forward(self, x):
        x = self._chunk_to_patches(x)
        if self.scale_input:
            x = x / 100.0
        return self.model(x)  # [B, 1]

    # -------- training --------
    def training_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            x, y = batch
        else:
            raise ValueError("Batch must be (x, y) for regression finetune.")
        y = y.view(-1, 1).float()
        preds = self(x)
        loss = self.loss_fn(preds, y)

        # Update global train metrics
        y_flat = y.squeeze(-1)
        p_flat = preds.squeeze(-1)
        self.train_rmse.update(p_flat, y_flat)
        self.train_y_mean.update(y_flat)
        self.train_y_sq_mean.update(y_flat ** 2)

        # Usual loss logging (batch + epoch-avg)
        self.log("train/loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        # Compute true epoch-level RMSE and NRMSE
        rmse = self.train_rmse.compute()
        Ey = self.train_y_mean.compute()
        Ey2 = self.train_y_sq_mean.compute()
        y_std = torch.sqrt(torch.clamp(Ey2 - Ey**2, min=1e-12))
        nrmse = rmse / y_std

        self.log("train/rmse_epoch", rmse.float(),
                 prog_bar=True, sync_dist=True)
        self.log("train/nrmse_epoch", nrmse.float(),
                 prog_bar=True, sync_dist=True)

        # Reset for next epoch
        self.train_rmse.reset()
        self.train_y_mean.reset()
        self.train_y_sq_mean.reset()

    # -------- validation --------
    def validation_step(self, batch, batch_idx):
        x, y = batch if isinstance(batch, (tuple, list)) else (batch, None)
        y = y.view(-1, 1).float()
        preds = self(x)
        loss = self.loss_fn(preds, y)

        # Update global val metrics
        y_flat = y.squeeze(-1)
        p_flat = preds.squeeze(-1)
        self.val_rmse.update(p_flat, y_flat)
        self.val_y_mean.update(y_flat)
        self.val_y_sq_mean.update(y_flat ** 2)

        # Per-epoch loss (averaged by Lightning)
        self.log("val/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        rmse = self.val_rmse.compute()
        Ey = self.val_y_mean.compute()
        Ey2 = self.val_y_sq_mean.compute()
        y_std = torch.sqrt(torch.clamp(Ey2 - Ey**2, min=1e-12))
        nrmse = rmse / y_std

        self.log("val/rmse_epoch", rmse.float(), prog_bar=True, sync_dist=True)
        self.log("val/nrmse_epoch", nrmse.float(),
                 prog_bar=True, sync_dist=True)

        self.val_rmse.reset()
        self.val_y_mean.reset()
        self.val_y_sq_mean.reset()

    # -------- test --------
    def test_step(self, batch, batch_idx):
        x, y = batch if isinstance(batch, (tuple, list)) else (batch, None)
        y = y.view(-1, 1).float()
        preds = self(x)
        loss = self.loss_fn(preds, y)

        # Update global test metrics
        y_flat = y.squeeze(-1)
        p_flat = preds.squeeze(-1)
        self.test_rmse.update(p_flat, y_flat)
        self.test_y_mean.update(y_flat)
        self.test_y_sq_mean.update(y_flat ** 2)

        self.log("test/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        rmse = self.test_rmse.compute()
        Ey = self.test_y_mean.compute()
        Ey2 = self.test_y_sq_mean.compute()
        y_std = torch.sqrt(torch.clamp(Ey2 - Ey**2, min=1e-12))
        nrmse = rmse / y_std

        self.log("test/rmse_epoch", rmse.float(),
                 prog_bar=True, sync_dist=True)
        self.log("test/nrmse_epoch", nrmse.float(),
                 prog_bar=True, sync_dist=True)

        self.test_rmse.reset()
        self.test_y_mean.reset()
        self.test_y_sq_mean.reset()

    # -------- optim --------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if self.use_cosine_per_step and total_steps:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_steps, eta_min=self.eta_min)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "name": "cosine_step"}}
        return opt
