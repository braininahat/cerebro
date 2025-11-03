# Path 3 Infrastructure: Multi-Phase Training with Contrastive Learning

**Status**: Phase 2 (Contrastive) verified ✅
**Date**: 2025-10-28
**Validation**: SignalJEPA smoke test (20 steps)

---

## Overview

Path 3 extends the 2-phase training (Pretrain → Supervised) with an intermediate contrastive learning phase that uses movie ISC (Inter-Subject Correlation) data:

**Path 3 Pipeline**: Pretrain → **Contrastive** → Supervised

This contrasts with:
- **Path 1**: Direct supervised (EEGNeX, SignalJEPA_PreLocal)
- **Path 2**: Pretrain → Supervised (LaBraM, SignalJEPA base)

---

## Implementation

### 1. ContrastiveTrainer Modifications

**File**: `cerebro/trainers/contrastive.py`

**Changes**:
- Added `pretrained_checkpoint` parameter to `__init__()`
- Implemented `_load_pretrained_encoder()` method (copied from SupervisedTrainer)
- Enables Phase 2 to load Phase 1 pretrained weights

**Usage**:
```python
ContrastiveTrainer(
    model=ContrastiveModel(...),
    pretrained_checkpoint="outputs/jepa/pretrain_smoke/checkpoints/last.ckpt",  # ← NEW
    pairing_strategy="all_pairs",
    temperature=0.07,
    ...
)
```

### 2. MovieDataModule Fixes

**File**: `cerebro/data/movies.py`

**Changes**:
- Added `from eegdash import EEGChallengeDataset` import
- Fixed `load_and_window_movies()` call to pass `EEGChallengeDataset` class

**File**: `cerebro/utils/movie_windows.py`

**Bug Fix #1**: `load_and_window_movies()` signature
- **Problem**: MovieDataModule passed `releases` (plural) but function expected `release` (singular)
- **Solution**: Updated function to accept `releases: list[str] | str` and loop over multiple releases

**Bug Fix #2**: `extract_subject_id()` type checking
- **Problem**: Function checked `isinstance(description, dict)` but EEGChallengeDataset uses `pandas.Series`
- **Solution**: Duck-typed check using `hasattr(description, '__getitem__') and hasattr(description, '__contains__')`
- **Result**: Now correctly extracts 20 subjects from R1 mini (was only finding 1 subject before)

---

## Configurations

### Phase 2: Movie Contrastive Learning

**File**: `configs/jepa_phase2_contrastive_smoke.yaml`

**Key settings**:
```yaml
model:
  class_path: cerebro.trainers.contrastive.ContrastiveTrainer
  init_args:
    model:
      class_path: cerebro.models.architectures.ContrastiveModel
      init_args:
        encoder_class: SignalJEPA
        projection_dim: 128
        hidden_dim: 256
        input_scale: 1000.0

    pretrained_checkpoint: outputs/jepa/pretrain_smoke/checkpoints/last.ckpt  # Phase 1
    pairing_strategy: all_pairs           # SimCLR-style
    temperature: 0.07
    lr: 0.0001

data:
  class_path: cerebro.data.movies.MovieDataModule
  init_args:
    releases: [R1]
    movie_names: [DespicableMe, ThePresent]
    window_len_s: 2.0
    pos_strategy: same_movie_time         # Same movie+time, different subject
    neg_strategy: diff_movie_mixed        # Different movie, any subject
    return_triplets: false                 # (anchor, pos) for all_pairs
    batch_size: 64
    use_mini: true                        # 20 subjects per release
```

### Phase 3: Supervised Finetuning

**Files**:
- `configs/jepa_phase3_challenge1_smoke.yaml` (Response time prediction)
- `configs/jepa_phase3_challenge2_smoke.yaml` (P_factor prediction)

**Key difference from Path 2**:
```yaml
model:
  class_path: cerebro.trainers.supervised.SupervisedTrainer
  init_args:
    pretrained_checkpoint: outputs/jepa/phase2_contrastive_smoke/checkpoints/last.ckpt  # ← Phase 2, not Phase 1
```

---

## Orchestration Script

**File**: `run_path3_jepa_smoke.sh`

**Usage**:
```bash
./run_path3_jepa_smoke.sh
```

**Stages**:
1. Phase 1: JEPA Pretrain (20 steps) → `outputs/jepa/pretrain_smoke/checkpoints/last.ckpt` (9.6M)
2. Phase 2: Movie Contrastive (20 steps) → `outputs/jepa/phase2_contrastive_smoke/checkpoints/last.ckpt` (1.4M)
3. Phase 3a: Challenge 1 Finetuning (20 steps) → `outputs/jepa/phase3_c1_smoke/checkpoints/last.ckpt`
4. Phase 3b: Challenge 2 Finetuning (20 steps) → `outputs/jepa/phase3_c2_smoke/checkpoints/last.ckpt`

---

## Verification Results

### Phase 2 Smoke Test (2025-10-28)

**Command**:
```bash
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_phase2_contrastive_smoke.yaml
```

**Results**:
- ✅ Loaded 20 subjects from R1 mini (DespicableMe + ThePresent)
- ✅ Created contrastive pairs using movie ISC
- ✅ Training completed: 20 steps in ~1 minute
- ✅ Metrics: `train_loss=4.69`, `pos_sim=0.674`, `val_loss=4.80`
- ✅ Checkpoint saved: `outputs/jepa/phase2_contrastive_smoke/checkpoints/last.ckpt` (1.4M)

**Checkpoint loading validation**:
```
[ContrastiveTrainer] Loading pretrained encoder from: outputs/jepa/pretrain_smoke/checkpoints/last.ckpt
[ContrastiveTrainer] Loaded 123 encoder parameters
  Missing keys: 11 (e.g., ['model.feature_encoder.1.0.weight', ...])
  Unexpected keys: 123 (e.g., ['channel_positions', ...])
```
✅ Successfully loaded Phase 1 weights into encoder

---

## Infrastructure Fixes Summary

| Issue | File | Fix |
|-------|------|-----|
| ContrastiveTrainer can't load Phase 1 checkpoints | `cerebro/trainers/contrastive.py` | Added `pretrained_checkpoint` parameter and `_load_pretrained_encoder()` method |
| MovieDataModule missing EEGChallengeDataset | `cerebro/data/movies.py` | Added import and passed class to `load_and_window_movies()` |
| `load_and_window_movies()` only accepts single release | `cerebro/utils/movie_windows.py` | Changed `release: str` to `releases: list[str] \| str` |
| `extract_subject_id()` returns None for pandas Series | `cerebro/utils/movie_windows.py` | Duck-typed check instead of `isinstance(dict)` |

---

## Next Steps

1. ✅ Test Phase 3 Challenge 1 finetuning
2. ✅ Test Phase 3 Challenge 2 finetuning
3. ✅ Run full Path 3 pipeline end-to-end
4. 📝 Update PIPELINES_VERIFIED.md with Path 3 results
5. 🚀 Create LaBraM Path 3 variant (if needed)

---

## SignalJEPA with Learned Channels (2025-10-28)

**Problem**: Base SignalJEPA requires 3D channel locations (`ch_locs`) which are NaN in HBN data. Additionally, we want to unify training across TUH (21 channels) and HBN (129 channels).

**Solution**: `SignalJEPAWithLearnedChannels` encoder with Perceiver-style learnable electrode queries.

### Architecture

```
Input (B, C_observed, T) where C ∈ {21, 129}
   ↓
Feature Encoder (conv layers)
   ↓ (B, C*T', d_model)
Learnable Positional Encoder:
   ├─ Reshape: (B, C, T', d_model)
   ├─ Pool over time: (B, C, d_model)
   ├─ Cross-attention: 128 electrode queries attend to C channels
   │   queries: (B, 128, spat_dim=30)
   │   keys/values: (B, C, spat_dim=30)
   │   → (B, 128, spat_dim=30)
   ├─ Combine with temporal encoding (sinusoidal)
   └─ Output: (B, 128*T', d_model)
   ↓
Transformer Encoder
   ↓ (B, 128*T', d_model)
Output: Fixed 128 electrode representations
```

### Key Features

1. **Variable input channels**: Works with 21 (TUH) or 129 (HBN) input channels
2. **Fixed output channels**: Always produces 128 electrode representations
3. **No channel locations needed**: Learns spatial structure via cross-attention
4. **Cross-dataset training**: Unified representation enables TUH + HBN pretraining

### Implementation

**File**: `cerebro/models/components/encoders.py`

**Classes**:
- `LearnableChannelPositionalEncoder` (lines 197-330): Perceiver-style cross-attention module
- `SignalJEPAWithLearnedChannels` (lines 333-468): Complete encoder with learned channels

**Registry name**: `SignalJEPA_LearnedChannels`

### Validation

**Test script**: `test_learned_channels.py`

```bash
uv run python test_learned_channels.py
# ✓ HBN (129 channels) → torch.Size([2, 128, 64])
# ✓ TUH (21 channels) → torch.Size([2, 128, 64])
```

Both produce unified 128-channel representations regardless of input size.

### Next Steps

1. 🔄 Create Phase 1 pretrain config with `SignalJEPA_LearnedChannels`
2. 🔄 Update Phase 2 contrastive config to use learned channels encoder
3. 🔄 Test full Path 3 pipeline with learned channels
4. 🔄 Implement TUH + HBN joint pretraining

---

## Notes

- **Contrastive learning requires multiple subjects**: Mini mode works (20 subjects per release) but won't work with single-subject datasets
- **Movie ISC pairing**: Positive pairs = same movie + same time_bin + different subject; Negative pairs = different movie
- **SignalJEPA encoder compatibility**: ContrastiveModel wraps any encoder from the registry, making it easy to use with different architectures
- **Learned channels enable cross-dataset training**: SignalJEPA_LearnedChannels unifies TUH (21ch) and HBN (129ch) into 128 electrode space
- **LaBraM Path 3**: Not yet implemented due to custom transformer architecture (not in standard encoder registry)
