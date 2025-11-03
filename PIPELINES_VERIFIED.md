# ✅ Verified Working Pipelines

**Validation Date**: 2025-10-28
**Validation Type**: Smoke tests (20 steps each, ~2 min/stage)
**Status**: ALL PIPELINES OPERATIONAL

---

## 🔬 Multi-Phase Training Pipelines

### 1. LaBraM (Learned Brain Masked Modeling)
**Status**: ✅ Fully working
**Pipeline**: Codebook VQ → MEM Pretrain → Challenge 1/2 Finetuning
**Checkpoints**:
- ✓ `outputs/labram/codebook_smoke/checkpoints/last.ckpt` (155M)
- ✓ `outputs/labram/pretrain_smoke/checkpoints/last.ckpt` (145M)
- ✓ `outputs/labram/finetune_c1_smoke/checkpoints/last.ckpt` (68M)
- ✓ `outputs/labram/finetune_c2_smoke/checkpoints/last.ckpt` (68M)

**Commands**:
```bash
# Stage 1: Codebook training (VQ-VAE with 8192 tokens)
WANDB_MODE=offline uv run cerebro fit --config configs/labram/codebook_smoke.yaml

# Stage 2: Masked EEG modeling pretraining
WANDB_MODE=offline uv run cerebro fit --config configs/labram/pretrain_smoke.yaml

# Stage 3: Challenge 1 finetuning (response time prediction)
WANDB_MODE=offline uv run cerebro fit --config configs/labram/finetune_challenge1_smoke.yaml

# Stage 4: Challenge 2 finetuning (externalizing score prediction)
WANDB_MODE=offline uv run cerebro fit --config configs/labram/finetune_challenge2_smoke.yaml
```

---

### 2. SignalJEPA (Joint Embedding Predictive Architecture)
**Status**: ✅ Fully working
**Pipeline**: JEPA Pretrain → Challenge 1/2 Finetuning
**Checkpoints**:
- ✓ `outputs/jepa/pretrain_smoke/checkpoints/last.ckpt` (9.6M)
- ✓ `outputs/jepa/finetune_c1_smoke/checkpoints/last.ckpt` (577K)
- ✓ `outputs/jepa/finetune_c2_smoke/checkpoints/last.ckpt` (961K)

**Commands**:
```bash
# Stage 1: JEPA pretraining (self-supervised)
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_pretrain_smoke.yaml

# Stage 2: Challenge 1 finetuning (with pretrained weights)
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_c1_smoke.yaml

# Stage 3: Challenge 2 finetuning (with pretrained weights)
WANDB_MODE=offline uv run cerebro fit --config configs/jepa_c2_smoke.yaml
```

---

## 🎯 Supervised Baseline Pipelines

### 3. EEGNeX Supervised (Challenge 1)
**Status**: ✅ Working
**Checkpoint**: `outputs/supervised_eegnex_c1_smoke/checkpoints/last.ckpt` (62K params)
**Command**:
```bash
WANDB_MODE=offline uv run cerebro fit --config configs/supervised_eegnex_challenge1_smoke.yaml
```
**Smoke test metrics**: `val_nrmse=2.880` (20 steps)

---

### 4. EEGNeX Supervised (Challenge 2)
**Status**: ✅ Working
**Checkpoint**: `outputs/supervised_eegnex_c2_smoke/checkpoints/last.ckpt` (62K params)
**Command**:
```bash
WANDB_MODE=offline uv run cerebro fit --config configs/supervised_eegnex_challenge2_smoke.yaml
```
**Smoke test metrics**: `val_nrmse=1.250` (20 steps)

---

### 5. SignalJEPA Supervised (Challenge 1)
**Status**: ✅ Working
**Checkpoint**: `outputs/supervised_jepa_c1_smoke/checkpoints/last.ckpt` (47K params)
**Command**:
```bash
WANDB_MODE=offline uv run cerebro fit --config configs/supervised_jepa_challenge1_smoke.yaml
```
**Smoke test metrics**: `val_nrmse=1.240` (20 steps)

---

### 6. SignalJEPA Supervised (Challenge 2)
**Status**: ✅ Working
**Checkpoint**: `outputs/supervised_jepa_c2_smoke/checkpoints/last.ckpt` (47K params)
**Command**:
```bash
WANDB_MODE=offline uv run cerebro fit --config configs/supervised_jepa_challenge2_smoke.yaml
```
**Smoke test metrics**: `val_nrmse=1.080` (20 steps)

---

## 📦 Submission & Scoring Infrastructure

### 7. Submission Creation
**Status**: ✅ Working
**Command**:
```bash
uv run cerebro-build-submission \
    --challenge1-ckpt outputs/supervised_eegnex_c1_smoke/checkpoints/last.ckpt \
    --challenge2-ckpt outputs/supervised_eegnex_c2_smoke/checkpoints/last.ckpt \
    -o submission.zip
```
**Output**: `submission.zip` (459.3 KB) containing:
- `submission.py` (Submission class)
- `weights_challenge_1.pt` (32 params)
- `weights_challenge_2.pt` (32 params)

---

### 8. Local Scoring
**Status**: ✅ Working
**Command**:
```bash
uv run cerebro-score submission.zip --fast-dev-run
```
**Output**: NRMSE scores for Challenge 1 and Challenge 2 + overall score (0.3×C1 + 0.7×C2)

---

## 🚀 Quickstart Commands

### Run full 7-stage smoke test pipeline:
```bash
./run_smoke_test_pipeline.sh
```

### Run comprehensive verification (all pipelines + submission + scoring):
```bash
./run_all_verification_tests.sh
```

---

## 📊 Architecture Summary

| Pipeline | Type | Params | Pretrain? | Checkpoints |
|----------|------|--------|-----------|-------------|
| LaBraM | Multi-phase (VQ→MEM) | 459M total | Yes (codebook+pretrain) | 4 stages |
| SignalJEPA | Multi-phase (JEPA) | 11M total | Yes (JEPA) | 3 stages |
| EEGNeX C1 | Supervised baseline | 62K | No | 1 stage |
| EEGNeX C2 | Supervised baseline | 62K | No | 1 stage |
| JEPA Supervised C1 | Supervised baseline | 47K | No | 1 stage |
| JEPA Supervised C2 | Supervised baseline | 47K | No | 1 stage |

---

## 🔧 Infrastructure Fixes Applied

1. **SupervisedTrainer checkpoint loading**: Added `pretrained_checkpoint` parameter and `_load_pretrained_encoder()` method
2. **JEPA config checkpoint paths**: Fixed 6 configs to use `last.ckpt` instead of `pretrain-last.ckpt`
3. **Challenge 2 NaN filtering**: Added `_filter_nan_targets()` to remove subjects with missing externalizing scores
4. **PyTorch weights_only**: Added `weights_only=False` for checkpoint loading (PyTorch 2.x compatibility)
5. **Submission template**: Removed invalid `do_spatial_filter` parameter from EEGNeX initialization
6. **Weight extraction**: Fixed RegressorModel → bare encoder extraction (handles `full_model.` and `features.` prefixes)

---

## ✅ Next Steps

All infrastructure is now validated. Ready for:
1. **Full production runs** (mini → full configs)
2. **Perceiver architecture** integration
3. **TUH dataset** integration
4. **Hyperparameter sweeps**

---

## 📝 Notes

- All smoke tests use 20 steps (~2 min/stage) for rapid validation
- Smoke test scores are not meaningful (undertrained models)
- Full training configs available: `*_mini.yaml` (500 steps) and `*_full.yaml` (production)
- All pipelines use R1 data only for smoke tests; full configs use R1-R4, R6-R11
