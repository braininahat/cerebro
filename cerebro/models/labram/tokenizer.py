import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from torch.nn.init import trunc_normal_
import lightning.pytorch as pl

from .modeling_finetune import NeuralTransformer
from .norm_ema_quantizer import NormEMAVectorQuantizer


class VQNSP(pl.LightningModule):
    def __init__(self,
                 # Model architecture control
                 pretrained=False,
                 as_tokenizer=False,
                 pretrained_weight=None,
                 # Codebook parameters
                 # Number of codebook vectors (n_embed)
                 n_code=8192,
                 # Codebook embedding dimension (embed_dim)
                 code_dim=32,
                 # Input configuration
                 EEG_size=200,             # Input size in samples (2s @ 100Hz)
                 patch_size=100,           # Patch size in samples (1s @ 100Hz)
                 encoder_depth=24,         # Encoder transformer depth
                 decoder_depth=3,          # Decoder transformer depth
                 # EMA quantizer parameters
                 decay=0.99,               # EMA decay factor
                 quantize_kmeans_init=True,
                 # Channel and temporal hints
                 n_chans=None,        # Number of EEG channels (e.g., 129)
                 max_time_window_hint=None,  # Maximum time window in seconds
                 # Legacy parameters (kept for compatibility)
                 encoder_config=None,
                 decoder_config=None,
                 decoder_out_dim=None,     # computed from patch_len if None
                 smooth_l1_loss=False,
                 patch_len=None,           # explicit patch size (samples)
                 learning_rate=1e-4,
                 weight_decay=0.05,
                 betas=(0.9, 0.95),
                 log_lr=True,              # log lr/min_lr per step like engine
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.opt_betas = betas
        self.log_lr = log_lr
        print(kwargs)

        # Map new parameter names to legacy internal names
        n_embed = n_code
        embed_dim = code_dim

        # Build encoder/decoder configs (ignores passed encoder_config/decoder_config)
        encoder_config, decoder_config = get_model_default_params(), get_model_default_params()
        encoder_config['EEG_size'] = EEG_size
        encoder_config['num_classes'] = 0
        encoder_config['patch_size'] = patch_size
        encoder_config['depth'] = encoder_depth
        decoder_config['EEG_size'] = EEG_size // encoder_config['patch_size']
        decoder_config['patch_size'] = 1
        decoder_config['in_chans'] = code_dim
        decoder_config['num_classes'] = 0
        decoder_config['depth'] = decoder_depth

        if as_tokenizer:
            assert pretrained and pretrained_weight is not None
            weights = torch.load(
                pretrained_weight, map_location='cpu')
            weights = weights.get('model', weights.get('state_dict', weights))
            for k in list(weights.keys()):
                if k.startswith(("loss", "teacher", "scaling")):
                    del weights[k]
            self.load_state_dict(weights, strict=False)

        # ---------- Encoder / decoder ----------
        if decoder_config['in_chans'] != embed_dim:
            print(
                f"Rewrite the in_chans in decoder from {decoder_config['in_chans']} to {embed_dim}")
            decoder_config['in_chans'] = embed_dim

        if patch_len is not None:
            encoder_config['patch_size'] = patch_len
        assert 'patch_size' in encoder_config, "encoder_config must define 'patch_size'"
        self.patch_len = int(encoder_config['patch_size'])

        print('Final encoder config', encoder_config)
        self.encoder = NeuralTransformer(**encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder = NeuralTransformer(**decoder_config)

        # ---------- VQ ----------
        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0,
            kmeans_init=quantize_kmeans_init, decay=decay
        )

        self.patch_size = self.patch_len
        self.n_chans_hint = n_chans
        self.token_shape = None

        if decoder_out_dim is None:
            decoder_out_dim = self.patch_len
        self.decoder_out_dim = int(decoder_out_dim)

        # ---------- Heads ----------
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'],
                      encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim)
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'],
                      decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.decode_task_layer_angle = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'],
                      decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )

        self.kwargs = kwargs

        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)
        self.decode_task_layer_angle.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

    # ---------- LaBraM engine parity hooks ----------
    def on_train_epoch_start(self):
        # Reset codebook usage stats before each epoch (mirrors engine)
        if hasattr(self, "quantize"):
            try:
                self.quantize.reset_cluster_size(self.device)  # preferred
            except Exception:
                pass
        # (engine also sets per-iter LR externally; Lightning schedulers handle that here)

    def on_validation_epoch_start(self):
        if hasattr(self, "quantize"):
            try:
                self.quantize.reset_cluster_size(self.device)
            except Exception:
                pass

    def on_train_epoch_end(self):
        # Log unused code count like engine_for_vqnsp
        zero_cnt = self._unused_code_count()
        if zero_cnt is not None:
            # engine prints "Unused code in codebook: {zero_cnt}"
            self.log("train/unused_code", float(zero_cnt),
                     prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        zero_cnt = self._unused_code_count()
        if zero_cnt is not None:
            self.log("val/unused_code", float(zero_cnt),
                     prog_bar=True, logger=True)

    def _unused_code_count(self):
        # engine checks both ._codebook.cluster_size and .cluster_size
        try:
            cluster = getattr(self.quantize, "_codebook",
                              self.quantize).cluster_size
            return (cluster == 0).sum().item()
        except Exception:
            return None

    # ---------- core layers ----------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed',
            'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed'
        }

    @property
    def device(self):
        return self.decoder.cls_token.device

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, input_chans=None, **kwargs):
        quantize, embed_ind, loss = self.encode(data, input_chans=input_chans)
        output = {}
        output['token'] = embed_ind.view(data.shape[0], -1)
        output['input_img'] = data
        output['quantize'] = rearrange(quantize, 'b d a c -> b (a c) d')
        return output

    def encode(self, x, input_chans=None):
        # x: (B, N, A, T)
        _, n, a, _ = x.shape
        self.token_shape = (n, a)
        encoder_features = self.encoder(
            x, input_chans, return_patch_tokens=True)
        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(
                encoder_features.type_as(self.encode_task_layer[-1].weight)
            )
        N = to_quantizer_features.shape[1]
        h, w = n, N // n
        to_quantizer_features = rearrange(
            to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w)
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)
        return quantize, embed_ind, loss

    def decode(self, quantize, input_chans=None, **kwargs):
        decoder_features = self.decoder(
            quantize, input_chans, return_patch_tokens=True)
        rec = self.decode_task_layer(decoder_features)
        rec_angle = self.decode_task_layer_angle(decoder_features)
        return rec, rec_angle

    def get_codebook_indices(self, x, input_chans=None, **kwargs):
        return self.get_tokens(x, input_chans, **kwargs)['token']

    def calculate_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c')
        return self.loss_fn(rec, target)

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        return (x - mean) / (std + 1e-6)

    def _chunk_to_patches(self, x_2d):
        # (B, N, T_total) -> (B, N, A, T=self.patch_len), crop/pad as needed
        B, N, T_total = x_2d.shape
        T = self.patch_len
        A = T_total // T
        if A == 0:
            pad = T - T_total
            x_2d = F.pad(x_2d, (0, pad))
            A = 1
        else:
            T_keep = A * T
            if T_keep < T_total:
                x_2d = x_2d[..., :T_keep]
        x_4d = rearrange(x_2d, 'B N (A T) -> B N A T', A=A, T=T)
        return x_4d, A, T

    # ---------- forward ----------
    def forward(self, x, input_chans=None, **kwargs):
        # hygiene first
        x = x.to(self.device, non_blocking=True).to(torch.float32)

        x, A, T = self._chunk_to_patches(x)

        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = self.std_norm(torch.abs(x_fft))
        angle = self.std_norm(torch.angle(x_fft))

        quantize, embed_ind, emb_loss = self.encode(x, input_chans)
        xrec, xrec_angle = self.decode(quantize, input_chans)

        rec_loss = self.calculate_rec_loss(xrec, amplitude)
        rec_angle_loss = self.calculate_rec_loss(xrec_angle, angle)
        loss = emb_loss + rec_loss + rec_angle_loss

        split = "train" if self.training else "val"
        log = {
            f'{split}/quant_loss': emb_loss.detach().mean(),
            f'{split}/rec_loss': rec_loss.detach().mean(),
            f'{split}/rec_angle_loss': rec_angle_loss.detach().mean(),
            f'{split}/total_loss': loss.detach().mean(),
        }
        return loss, log

    # ---------- Lightning train/val with LaBraM behavior ----------
    def _scale_input_like_engine(self, x):
        # engine divides by 100 before forward
        return x.float() / 100.0

    def _log_lr_from_optim(self):
        # mimic engine's lr and min_lr logging
        try:
            opt = self.optimizers()
            lrs = [g['lr'] for g in opt.param_groups]
            if lrs:
                self.log("opt/lr", max(lrs), on_step=True,
                         prog_bar=False, logger=True)
                self.log("opt/min_lr", min(lrs), on_step=True,
                         prog_bar=False, logger=True)
                # also log weight decay (first group with wd>0)
                wd = next((g['weight_decay'] for g in opt.param_groups if g.get(
                    'weight_decay', 0) > 0), 0.0)
                self.log("opt/weight_decay", wd, on_step=True,
                         prog_bar=False, logger=True)
        except Exception:
            pass

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        try:
            import torch.backends.cuda as cuda_back
            cuda_back.cufft_plan_cache.clear()
            cuda_back.cufft_plan_cache.max_size = 0   # disable plan caching this epoch
        except Exception:
            pass

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        try:
            import torch.backends.cuda as cuda_back
            cuda_back.cufft_plan_cache.clear()
            cuda_back.cufft_plan_cache.max_size = 0
        except Exception:
            pass

    def training_step(self, batch, batch_idx):
        # batch may be x or (x, input_chans)
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        input_chans = None

        x = self._scale_input_like_engine(x)   # /100 as in engine
        loss, log = self(x, input_chans)

        # log like engine (both detailed and total)
        for k, v in log.items():
            self.log(k, v, on_step=True, on_epoch=True,
                     prog_bar=('total_loss' in k), logger=True)

        # extra: plain keys without split prefix (engine strips prefix)
        self.log("quant_loss", log[f"{'train' if self.training else 'val'}/quant_loss"],
                 on_step=True, on_epoch=True, logger=True)
        self.log("rec_loss",   log[f"{'train' if self.training else 'val'}/rec_loss"],
                 on_step=True, on_epoch=True, logger=True)
        self.log("rec_angle_loss", log[f"{'train' if self.training else 'val'}/rec_angle_loss"],
                 on_step=True, on_epoch=True, logger=True)

        # lr/min_lr per step
        if self.log_lr:
            self._log_lr_from_optim()

        # Lightning handles AMP/scaler; engine uses loss_scaler+clip_grad externally
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        input_chans = None
        x = self._scale_input_like_engine(x)   # /100 as in engine
        loss, log = self(x, input_chans)

        # Log a dedicated monitor key like your PL callback expects
        self.log("val_tok_loss", loss.detach(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        for k, v in log.items():
            self.log(k, v, on_step=False, on_epoch=True,
                     prog_bar=('total_loss' in k), logger=True)

        # also plain, unprefixed versions (engine prints these)
        self.log("quant_loss", log["val/quant_loss"],
                 on_step=False, on_epoch=True, logger=True)
        self.log("rec_loss",   log["val/rec_loss"],
                 on_step=False, on_epoch=True, logger=True)
        self.log("rec_angle_loss", log["val/rec_angle_loss"],
                 on_step=False, on_epoch=True, logger=True)

        return loss

    def setup(self, stage=None):
        if stage == "fit":
            self.total_steps = self.trainer.estimated_stepping_batches

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.opt_betas,
            weight_decay=self.weight_decay,
        )

        # self.total_steps can be set in setup("fit") or computed dynamically
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.total_steps,  # total number of training steps
            eta_min=1e-6             # final learning rate
        )

        # 3️⃣ Return both to Lightning
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # <— update every iteration, not per epoch
                "frequency": 1,
                "name": "cosine_step"
            }
        }


def get_model_default_params():
    return dict(EEG_size=1600, patch_size=200, in_chans=1, num_classes=1000, embed_dim=200, depth=12, num_heads=10,
                mlp_ratio=4., qkv_bias=True,  qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., use_abs_pos_emb=True,
                use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)
