# Foundation Model Pretraining + Finetuning Guide

Complete pipeline for training LaBraM and SignalJEPA foundation models on HBN, then finetuning for competition challenges.

---

## Overview

**Two foundation models:**
1. **LaBraM** - 3-phase: Codebook → MEM Pretrain → Finetune
2. **SignalJEPA** - 2-phase: JEPA Pretrain → Finetune

**Two competition challenges:**
- **Challenge 1** (30%): Response time (RT) prediction from Contrast Change Detection task
- **Challenge 2** (70%): Externalizing psychopathology prediction from multi-task EEG

---

## Training Pipeline

### **LaBraM (3 Phases)**

#### Phase 1: Codebook Training (~24-48 hours)
Learn discrete EEG token vocabulary via vector quantization.

```bash
# Train tokenizer on full HBN dataset (R1-R4, R6-R11)
uv run cerebro fit --config configs/labram/codebook_full.yaml
```

**Output:** `outputs/labram/codebook_full/checkpoints/tokenizer-last.ckpt`

**Architecture:**
- 8192 discrete tokens
- 32-dim embeddings per token
- Encoder depth: 24 layers
- Decoder depth: 3 layers

---

#### Phase 2: MEM Pretraining (~48-72 hours)
Masked EEG Modeling - predict masked tokens (like BERT for EEG).

**Prerequisites:** Completed Phase 1 codebook training

```bash
# 1. Update pretrained_weight path in config:
#    configs/labram/pretrain_full.yaml:
#    pretrained_weight: outputs/labram/codebook_full/checkpoints/tokenizer-last.ckpt

# 2. Run MEM pretraining
uv run cerebro fit --config configs/labram/pretrain_full.yaml
```

**Output:** `outputs/labram/pretrain_full/checkpoints/pretrain-last.ckpt`

**Architecture:**
- Student model: 12-layer transformer
- Embedding dim: 200
- Attention heads: 10
- Mask ratio: 50%
- Dataset: Passive tasks (movies, resting state, etc.)

---

#### Phase 3a: Challenge 1 Finetuning (~12-24 hours)
Supervised training on RT prediction.

**Prerequisites:** Completed Phase 2 MEM pretraining

```bash
# 1. Update pretrained_weight path in config:
#    configs/labram/finetune_challenge1_full.yaml:
#    pretrained_weight: outputs/labram/pretrain_full/checkpoints/pretrain-last.ckpt

# 2. Run Challenge 1 finetuning
uv run cerebro fit --config configs/labram/finetune_challenge1_full.yaml
```

**Output:** `outputs/labram/finetune_c1_full/checkpoints/challenge1-best.ckpt`

**Task:** Stimulus-locked windows (2s) → RT prediction

---

#### Phase 3b: Challenge 2 Finetuning (~12-24 hours)
Supervised training on externalizing prediction.

**Prerequisites:** Completed Phase 2 MEM pretraining

```bash
# 1. Update pretrained_weight path in config:
#    configs/labram/finetune_challenge2_full.yaml:
#    pretrained_weight: outputs/labram/pretrain_full/checkpoints/pretrain-last.ckpt

# 2. Run Challenge 2 finetuning
uv run cerebro fit --config configs/labram/finetune_challenge2_full.yaml
```

**Output:** `outputs/labram/finetune_c2_full/checkpoints/challenge2-best.ckpt`

**Task:** Fixed windows (4s, 2s stride) → Externalizing score prediction

---

### **SignalJEPA (2 Phases)**

#### Phase 1: JEPA Pretraining (~36-48 hours)
Self-supervised learning via multi-scale temporal prediction.

```bash
# Train JEPA on full HBN dataset (R1-R4, R6-R11)
uv run cerebro fit --config configs/jepa_phase1_pretrain_full.yaml
```

**Output:** `outputs/jepa/pretrain_full/checkpoints/pretrain-last.ckpt`

**Architecture:**
- 96-dim latent: trait(24) + state(36) + event(36)
- FNO spatial encoder (50 Fourier modes)
- Mamba temporal encoder (4 layers)
- Multi-scale prediction losses
- Dataset: All tasks (movies, resting, active tasks)

---

#### Phase 2a: Challenge 1 Finetuning (~8-16 hours)
Supervised training on RT prediction.

**Prerequisites:** Completed Phase 1 JEPA pretraining

```bash
# 1. Update pretrained_checkpoint path in config:
#    configs/jepa_finetune_challenge1_full.yaml:
#    pretrained_checkpoint: outputs/jepa/pretrain_full/checkpoints/pretrain-last.ckpt

# 2. Run Challenge 1 finetuning
uv run cerebro fit --config configs/jepa_finetune_challenge1_full.yaml
```

**Output:** `outputs/jepa/finetune_c1_full/checkpoints/challenge1-best.ckpt`

**Task:** Stimulus-locked windows (2s) → RT prediction

---

#### Phase 2b: Challenge 2 Finetuning (~12-24 hours)
Supervised training on externalizing prediction.

**Prerequisites:** Completed Phase 1 JEPA pretraining

```bash
# 1. Update pretrained_checkpoint path in config:
#    configs/jepa_finetune_challenge2_full.yaml:
#    pretrained_checkpoint: outputs/jepa/pretrain_full/checkpoints/pretrain-last.ckpt

# 2. Run Challenge 2 finetuning
uv run cerebro fit --config configs/jepa_finetune_challenge2_full.yaml
```

**Output:** `outputs/jepa/finetune_c2_full/checkpoints/challenge2-best.ckpt`

**Task:** Fixed windows (4s, 2s stride) → Externalizing score prediction

---

## Timeline Summary

### **LaBraM Total: ~96-168 hours (4-7 days)**
1. Codebook: 24-48h
2. MEM Pretrain: 48-72h
3. Challenge 1 Finetune: 12-24h
4. Challenge 2 Finetune: 12-24h

### **SignalJEPA Total: ~56-88 hours (2.5-4 days)**
1. JEPA Pretrain: 36-48h
2. Challenge 1 Finetune: 8-16h
3. Challenge 2 Finetune: 12-24h

**Note:** Phases 3a/3b (LaBraM) and 2a/2b (SignalJEPA) can run in parallel if you have multiple GPUs.

---

## Key Design Decisions

### **LaBraM:**
- ✅ Discrete tokenization → interpretable latent space
- ✅ Published architecture (proven on EEG)
- ❌ Requires 2-stage pretraining (slower)
- ❌ More complex pipeline

### **SignalJEPA:**
- ✅ End-to-end continuous representations
- ✅ Faster pretraining (1 phase vs 2)
- ✅ Multi-scale temporal dynamics (trait/state/event)
- ❌ Less interpretable latent space

---

## Dataset Details

### **Pretraining Data:**
- **Releases:** R1-R4, R6-R11 (10 releases, ~1000+ subjects)
- **R5 EXCLUDED** - Competition validation set!
- **Tasks:** Movies, resting state, active tasks
- **Channels:** 129 (128 + Cz reference)
- **Sampling:** 100 Hz

### **Finetuning Data:**
- **Challenge 1:** contrastChangeDetection (stimulus-locked)
- **Challenge 2:** contrastChangeDetection + optionally more tasks
- **Windowing:** Task-specific (see configs)

---

## Monitoring Training

All configs use WandB for logging (offline mode by default).

**View logs:**
```bash
# Sync to cloud (optional)
wandb sync outputs/<model>/<phase>/wandb/offline-*

# Or view locally
wandb offline
```

**Key metrics to monitor:**
- **Codebook:** `val_tok_loss` (reconstruction quality)
- **MEM Pretrain:** `val_mem_loss` (masked prediction accuracy)
- **JEPA Pretrain:** `val_loss` (multi-scale prediction loss)
- **Finetuning:** `val_nrmse` (normalized RMSE, competition metric)

---

## Quick Start (Mini Versions)

For testing the pipeline on small data before full training:

### LaBraM Mini
```bash
# 1. Codebook (R1 only)
uv run cerebro fit --config configs/labram/codebook.yaml

# 2. MEM Pretrain (R1 only)
#    Update pretrained_weight path first!
uv run cerebro fit --config configs/labram/pretrain.yaml

# 3. Challenge 2 Finetune (R1 only)
#    Update pretrained_weight path first!
uv run cerebro fit --config configs/labram/finetune.yaml
```

### SignalJEPA Mini
```bash
# 1. JEPA Pretrain (R1 only, limit batches)
uv run cerebro fit --config configs/jepa_phase1_pretrain_mini.yaml

# 2. Challenge 1/2 Finetune (R1 only)
uv run cerebro fit --config configs/supervised_jepa_challenge1_mini.yaml
uv run cerebro fit --config configs/supervised_jepa_challenge2_mini.yaml
```

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in config
- Use `precision: "bf16-mixed"` (already default in most configs)
- Reduce `num_workers` if CPU memory limited

### Checkpoint Loading Errors
- Ensure `pretrained_weight` / `pretrained_checkpoint` paths are correct
- Check that encoder architecture matches between pretrain and finetune

### Slow Data Loading
- Increase `num_workers` (currently 16, try 32-64 if you have CPU cores)
- Data is cached after first load - subsequent runs will be faster
- Check disk I/O (use SSD for data directory)

### WandB Issues
- Set `mode: offline` in logger config (already default)
- Logs saved to `outputs/<model>/<phase>/wandb/`

---

## Next Steps

After completing all training phases:

1. **Evaluate on local test set:**
   ```bash
   uv run cerebro test --config <finetune_config> --ckpt_path <best_checkpoint>
   ```

2. **Generate submissions:**
   - Use `cerebro/cli/build_submission.py` to create submission CSVs
   - Submit to competition leaderboard for R5 validation

3. **Compare models:**
   - LaBraM vs SignalJEPA
   - With/without pretraining
   - Different finetuning strategies

---

## Config Files Reference

### LaBraM
- `configs/labram/codebook_full.yaml` - Phase 1: Tokenizer training
- `configs/labram/pretrain_full.yaml` - Phase 2: MEM pretraining
- `configs/labram/finetune_challenge1_full.yaml` - Phase 3a: Challenge 1
- `configs/labram/finetune_challenge2_full.yaml` - Phase 3b: Challenge 2

### SignalJEPA
- `configs/jepa_phase1_pretrain_full.yaml` - Phase 1: JEPA pretraining
- `configs/jepa_finetune_challenge1_full.yaml` - Phase 2a: Challenge 1
- `configs/jepa_finetune_challenge2_full.yaml` - Phase 2b: Challenge 2

---

## Contact

For questions or issues:
- GitHub Issues: https://github.com/eeg2025/cerebro/issues
- Competition Discord: [link]
