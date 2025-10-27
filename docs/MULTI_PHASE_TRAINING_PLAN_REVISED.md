# Multi-Phase Training Plan: LaBraM First, Then SignalJEPA

**Strategy**: Validate the multi-phase training framework with LaBraM (already implemented), then port to SignalJEPA.

---

## Why LaBraM First?

### 1. **Already Implemented**
Current codebase has:
- ✅ `cerebro/models/labram/tokenizer.py` - VQNSP tokenizer with vector quantization
- ✅ `cerebro/models/labram/pretrain.py` - MEMPretrainModule (masked EEG modeling)
- ✅ `cerebro/models/labram/finetune.py` - EEGRegressorPL for regression
- ✅ `configs/labram/` - Pretrain, finetune, codebook configs
- ✅ Works on HBN (129 channels, 100 Hz)

### 2. **Lower Risk**
- No need to learn SignalJEPA API first
- Can focus on multi-phase framework design
- Faster iteration cycles
- Proven architecture (published paper)

### 3. **Framework Validation**
Once we validate with LaBraM:
- Channel adaptation (21ch → 128ch)
- Contrastive learning infrastructure
- Auxiliary demographic tasks
- Phase transition pipeline

Then porting to SignalJEPA is straightforward.

---

## SignalJEPA Components (braindecode v1.2.0)

### Available Models

| Model | Purpose | Components |
|-------|---------|------------|
| `SignalJEPA` | Self-supervised pretraining | Feature encoder + position encoder + encoder-decoder transformer |
| `SignalJEPA_PreLocal` | Classification (pre-local predictor) | Above + pre-local prediction head |
| `SignalJEPA_PostLocal` | Classification (post-local predictor) | Above + post-local prediction head |
| `SignalJEPA_Contextual` | Classification (contextual predictor) | Above + contextual prediction head |

### "Full" SignalJEPA Model

The **full SignalJEPA** for our use case would be:
1. **Pretraining**: Base `SignalJEPA` model
2. **Downstream tasks**: Any of the three predictor variants depending on task requirements

**Key architecture**:
```python
SignalJEPA(
    n_chans=128,
    n_times=200,  # 2s @ 100Hz
    sfreq=100,
    # Feature encoder (temporal conv downsampling)
    feature_encoder__conv_layers_spec=((8, 32, 8), (16, 2, 2), (32, 2, 2), (64, 2, 2), (64, 2, 2)),
    # Position encoder (spatial + temporal embeddings)
    pos_encoder__spat_dim=30,
    pos_encoder__time_dim=34,
    # Transformer
    transformer__d_model=64,
    transformer__num_encoder_layers=8,
    transformer__num_decoder_layers=4,
    transformer__nhead=8
)
```

---

## Phase 1: LaBraM Multi-Phase Training

### Architecture: LaBraM

**Components**:
1. **VQNSP Tokenizer** (already implemented)
   - Vector quantization with EMA
   - Codebook: 8192 codes, 32-dim embeddings
   - Encodes EEG → discrete tokens

2. **MEMPretrainModule** (already implemented)
   - Masked EEG Modeling
   - Randomly masks tokens + predicts
   - Symmetric objective (mask and inverse)

3. **EEGRegressorPL** (already implemented)
   - Fine-tuning for regression tasks
   - Loads pretrained weights
   - Challenge 1/2 compatible

### What Needs Implementation (LaBraM)

#### 1.1 Channel Adaptation
- [ ] **TUH integration**: Add TUH data loader with channel adapter
- [ ] **Perceiver cross-attention**: 21ch → 128ch mapping
- [ ] **Mixed dataset**: Combine TUH + HBN with sampling weights

**Location**: `cerebro/data/loaders/mixed.py`, `cerebro/models/components/channel_adapter.py`

#### 1.2 Contrastive Learning (Phase 2)
- [ ] **Contrastive data module**: Positive/negative pair generation
  - Movie ISC (same movie/time across subjects)
  - Resting state (eyes open/closed)
  - Task-specific contrastives
- [ ] **Contrastive training module**: InfoNCE loss
- [ ] **Projection head**: 2-layer MLP for contrastive learning

**Location**: `cerebro/data/augmentation/contrastive.py`, `cerebro/training/contrastive_labram.py`

#### 1.3 Auxiliary Demographic Tasks (Joint with Phase 1)
- [ ] **Demographic data loading**: age, sex, p_factor, etc.
- [ ] **Auxiliary heads**: Regression/classification heads
- [ ] **Multi-task loss**: Weighted combination

**Location**: `cerebro/models/components/heads/`, `cerebro/losses/composite.py`

#### 1.4 Phase Transition Infrastructure
- [ ] **Checkpoint manager**: Load Phase N weights → Phase N+1
- [ ] **Freezing/unfreezing**: Freeze encoder for K epochs, then fine-tune
- [ ] **Evaluation probes**: Linear classifier to track representation quality

**Location**: `cerebro/callbacks/phase_transition.py`, `cerebro/evaluation/probing.py`

### LaBraM Training Pipeline

```
Phase 1a: Codebook Training (Existing)
└─> VQNSP tokenizer training on HBN

Phase 1b: MEM Pretraining (Enhanced)
├─> Input: TUH + HBN mixed (70% HBN, 30% TUH)
├─> Masking: Random token masking (50%)
├─> Auxiliary tasks: age, sex, p_factor (joint training)
└─> Output: Pretrained encoder checkpoint

Phase 2: Contrastive Learning (NEW)
├─> Input: HBN only (movies, resting, tasks)
├─> Load: Phase 1b checkpoint
├─> Contrastive formulations:
│   ├─> Movie ISC
│   ├─> Resting state (eyes open/closed)
│   └─> Task-specific contrastives
├─> Loss: InfoNCE with temperature=0.07
└─> Output: Contrastive-enhanced checkpoint

Phase 3: Supervised Fine-Tuning (Existing, Enhanced)
├─> Load: Phase 2 checkpoint
├─> Freeze encoder: 5 epochs
├─> Fine-tune end-to-end: remaining epochs
├─> Challenge 1: Response time prediction
├─> Challenge 2: P-factor prediction
└─> Output: Task-specific models
```

---

## Phase 2: SignalJEPA Multi-Phase Training

**After validating with LaBraM**, port to SignalJEPA.

### Architecture: SignalJEPA

**Components**:
1. **Feature Encoder** (conv layers)
   - Temporal downsampling via conv
   - Default: 5 conv layers with stride

2. **Position Encoder**
   - Spatial embeddings (channel positions)
   - Temporal embeddings (time positions)

3. **Transformer**
   - Encoder layers (context encoding)
   - Decoder layers (prediction)

4. **Predictors** (for downstream tasks)
   - Pre-local: Local temporal prediction
   - Post-local: Refined prediction
   - Contextual: Global context aggregation

### SignalJEPA Training Pipeline

```
Phase 1a: Self-Supervised Pretraining (NEW)
├─> Input: TUH + HBN mixed
├─> Architecture: Base SignalJEPA
├─> Masking: Spatial + temporal + spatiotemporal
├─> Auxiliary tasks: age, sex, p_factor
└─> Output: Pretrained SignalJEPA checkpoint

Phase 2: Contrastive Learning (Port from LaBraM)
├─> Same as LaBraM Phase 2
├─> Use SignalJEPA encoder instead
└─> Output: Contrastive-enhanced checkpoint

Phase 3: Supervised Fine-Tuning (NEW)
├─> Load: Phase 2 checkpoint
├─> Architecture: SignalJEPA_PreLocal (or others)
├─> Challenge 1/2 training
└─> Output: Task-specific models
```

### Key Differences: LaBraM vs. SignalJEPA

| Aspect | LaBraM | SignalJEPA |
|--------|--------|------------|
| **Pretraining** | Vector quantization (discrete tokens) | Direct representation learning (continuous) |
| **Masking** | Token-level masking | Patch-level masking (spatial/temporal) |
| **Codebook** | Requires codebook training phase | No codebook needed |
| **Predictor** | Single decoder | Three predictor variants (pre-local, post-local, contextual) |
| **Complexity** | Higher (VQ + MEM) | Lower (end-to-end) |
| **Proven** | Published on EEG | Published on EEG |

---

## Implementation Timeline (Revised)

### Track 1: LaBraM Multi-Phase (Proof of Concept)

#### Session 1: Foundation (2 days)
- [ ] Channel adapter (Perceiver cross-attention)
- [ ] TUH + HBN mixed dataset
- [ ] Resampling (TUH 250 Hz → 100 Hz)
- [ ] Test: Can load TUH + HBN together with unified 128-channel representation

#### Session 2: Auxiliary Tasks (1 day)
- [ ] Demographic data loading (participants.tsv parsing)
- [ ] Auxiliary heads (age, sex, p_factor, attention, internalizing, externalizing)
- [ ] Multi-task loss (weighted combination)
- [ ] Test: MEMPretrainModule + auxiliary tasks train correctly

#### Session 3: Phase 1b - Enhanced MEM Pretraining (1 day)
- [ ] Update MEMPretrainModule to support auxiliary tasks
- [ ] Config: `configs/labram/pretrain_multitask.yaml`
- [ ] Train: Mixed TUH + HBN with demographics
- [ ] Eval: Aux task performance, linear probe on Challenge 1 val

#### Session 4: Phase 2 - Contrastive Learning (2-3 days)
- [ ] Contrastive pair generators (movie ISC, resting, tasks)
- [ ] Contrastive data module
- [ ] InfoNCE loss
- [ ] Contrastive training module (load Phase 1b checkpoint)
- [ ] Config: `configs/labram/contrastive.yaml`
- [ ] Train: HBN contrastive learning
- [ ] Eval: ISC scores, alignment metrics, linear probe

#### Session 5: Phase 3 - Supervised Fine-Tuning (1 day)
- [ ] Update EEGRegressorPL to load Phase 2 checkpoint
- [ ] Add encoder freezing for first K epochs
- [ ] Config: `configs/labram/finetune_phase3.yaml`
- [ ] Train: Challenge 1 + Challenge 2
- [ ] Eval: NRMSE, competition score

#### Session 6: Ablations & Evaluation (1-2 days)
- [ ] Full pipeline: Phase 1b → 2 → 3
- [ ] Ablations:
  - No contrastive (Phase 1b → 3 directly)
  - No auxiliary tasks (Phase 1b without demographics)
  - Supervised only (Phase 3 from scratch)
- [ ] Compare performance
- [ ] Document results: `docs/LABRAM_EVALUATION_RESULTS.md`

**Total: ~1.5 weeks**

### Track 2: SignalJEPA Multi-Phase (After LaBraM Validation)

#### Session 7: SignalJEPA Pretraining (2-3 days)
- [ ] Wrap braindecode `SignalJEPA` in Lightning module
- [ ] Implement spatial/temporal/spatiotemporal masking
- [ ] Add auxiliary tasks
- [ ] Config: `configs/signaljepa/pretrain.yaml`
- [ ] Train & evaluate

#### Session 8: SignalJEPA Contrastive (1 day)
- [ ] Port contrastive infrastructure from LaBraM
- [ ] Use SignalJEPA encoder instead
- [ ] Config: `configs/signaljepa/contrastive.yaml`
- [ ] Train & evaluate

#### Session 9: SignalJEPA Supervised (1 day)
- [ ] Wrap `SignalJEPA_PreLocal` (or others) in Lightning
- [ ] Load Phase 2 checkpoint
- [ ] Config: `configs/signaljepa/finetune.yaml`
- [ ] Train & evaluate

#### Session 10: Comparison & Analysis (1 day)
- [ ] LaBraM vs. SignalJEPA performance
- [ ] Architectural insights
- [ ] Documentation: `docs/LABRAM_VS_SIGNALJEPA.md`

**Total: ~1 week**

---

## Success Metrics

### LaBraM Track (Proof of Concept)
- ✅ Phase 1b → 3 outperforms supervised-only baseline (>5% improvement)
- ✅ Phase 2 contrastive improves over Phase 1b alone (>3% improvement)
- ✅ Auxiliary tasks improve representation quality (linear probe accuracy)
- ✅ Full pipeline (1b → 2 → 3) achieves best performance
- ✅ Ablations quantify each phase's contribution

### SignalJEPA Track (After Validation)
- ✅ Match or exceed LaBraM performance
- ✅ Compare training efficiency (time, compute)
- ✅ Identify architectural advantages/disadvantages

---

## Immediate Next Steps

### Option A: Start LaBraM Track Now
1. Create branch: `git checkout -b feat/labram-multiphase`
2. Scaffold directories (see Session 1)
3. Implement channel adapter
4. Test mixed TUH + HBN dataset

### Option B: Review Plan First
- Discuss channel adaptation strategy
- Clarify auxiliary task integration with LaBraM
- Adjust timeline/priorities

### Option C: Hybrid Approach
- Start with minimal channel adapter (zero-padding for TUH)
- Focus on contrastive + auxiliary tasks first
- Add Perceiver later if needed

---

## Questions to Clarify

1. **TUH data priority**: Do you have TUH data downloaded and ready? If not, start HBN-only with channel adaptation deferred.

2. **Auxiliary task integration**: Should auxiliary tasks be:
   - Separate heads on LaBraM encoder output? OR
   - Integrated into MEMPretrainModule directly?

3. **Contrastive architecture**: Should we:
   - Add projection head to existing LaBraM encoder? OR
   - Fine-tune encoder directly with InfoNCE?

4. **Baseline urgency**: Should we run supervised-only baselines (Challenge 1/2) first to establish performance floor?

5. **Compute budget**: How many GPUs? Training time constraints? This affects whether we do full TUH pretraining or HBN-only.

---

## Key Design Principles

### 1. **Incremental Complexity**
Start simple, add complexity only when validated:
- Session 1: Basic mixed dataset (even with zero-padding)
- Session 2: Add auxiliary tasks
- Session 3: Add contrastive learning
- Session 4: Add Perceiver (if needed)

### 2. **Reuse LaBraM Code**
Minimize changes to existing LaBraM implementation:
- Extend `MEMPretrainModule`, don't rewrite
- Add new loss terms, don't replace
- New configs, don't modify existing

### 3. **Test at Every Step**
- Unit tests for new components
- Integration tests for pipeline
- Sanity checks (loss decreases, metrics improve)

### 4. **Document Learnings**
- Checkpoint after each session
- Document what works, what doesn't
- Update plan based on findings

---

**Document Version**: 1.0 (Revised)
**Last Updated**: 2025-10-27
**Author**: Varun (with Claude assistance)
**Supersedes**: SIGNALJEPA_TRAINING_PLAN.md (now a reference)

---

## Appendix: Code Reuse Opportunities

### From Existing LaBraM Implementation

**Can reuse directly**:
- `cerebro/models/labram/tokenizer.py` - VQNSP for Phase 1a
- `cerebro/models/labram/pretrain.py` - MEMPretrainModule as base
- `cerebro/models/labram/finetune.py` - EEGRegressorPL for Phase 3
- `cerebro/models/labram/norm_ema_quantizer.py` - VQ codebook
- `configs/labram/` - Existing configs as templates

**Needs enhancement**:
- `MEMPretrainModule`: Add auxiliary task heads and losses
- `EEGRegressorPL`: Add Phase 2 checkpoint loading and encoder freezing
- Configs: New configs for multitask pretrain, contrastive, enhanced finetune

**Needs new implementation**:
- Channel adapter (Perceiver)
- Mixed TUH + HBN dataset
- Contrastive pair generators
- Contrastive training module
- Evaluation probing callback

---

## Appendix: SignalJEPA Details

### Model Instantiation (when we get there)

```python
from braindecode.models import SignalJEPA, SignalJEPA_PreLocal

# Phase 1: Pretraining
model_pretrain = SignalJEPA(
    n_chans=128,
    n_times=200,
    sfreq=100,
    transformer__d_model=64,
    transformer__num_encoder_layers=8,
    transformer__num_decoder_layers=4
)

# Phase 3: Fine-tuning (after Phase 1+2)
model_finetune = SignalJEPA_PreLocal(
    n_outputs=1,  # Regression
    n_chans=128,
    n_times=200,
    sfreq=100
)
# Load pretrained weights
model_finetune.load_state_dict(pretrained_weights, strict=False)
```

### Predictor Variants

- **Pre-local**: Predicts target representation from context (local temporal dependencies)
- **Post-local**: Refines predictions using additional context (longer-range dependencies)
- **Contextual**: Global context aggregation (full-sequence understanding)

**Recommendation**: Start with `SignalJEPA_PreLocal` for downstream tasks (simplest, proven effective).

