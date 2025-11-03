# Multi-Phase Training Implementation Progress

**Branch**: `docs/multi-phase-training-plan`
**Last Updated**: 2025-10-27
**Status**: Infrastructure phase complete, ready for integration

---

## Completed Components ✅

### 1. Channel Adaptation (`cerebro/models/components/channel_adapter.py`)

**Purpose**: Handle variable input channels for mixed TUH+HBN training

**Implementations**:
- `PerceiverChannelAdapter`: Cross-attention based mapping
  - Learnable query tokens (target_channels × d_model)
  - Cross-attention: queries attend to input channels
  - Time-step independent processing
  - Supports TUH (21ch) → HBN (128ch) mapping

- `ZeroPadChannelAdapter`: Simple baseline
  - Zero-pads input to target channel count
  - No learned parameters
  - Quick baseline for comparison

**Status**: ✅ Implemented, not yet tested
**Next**: Integrate with data loaders for mixed dataset

---

### 2. Auxiliary Task Infrastructure (`cerebro/models/components/auxiliary_heads.py`)

**Purpose**: Multi-task learning for richer representations

**Implementations**:
- `DemographicHead`: Single auxiliary task prediction
  - 2-layer MLP (input_dim → hidden → output)
  - Supports regression and classification
  - Dropout for regularization

- `MultiAuxiliaryHead`: Multiple demographic predictions
  - ModuleDict of DemographicHeads
  - Returns dict of predictions {task_name: tensor}
  - Configured via HBN_AUXILIARY_TASKS

- `AuxiliaryTaskLoss`: Weighted multi-task loss
  - Separate loss functions per task (MSE for regression, CE for classification)
  - Mask support for missing labels (n/a values)
  - Weighted combination with configurable weights

**HBN Tasks Configured**:
- age: Regression, weight=1.0
- sex: Classification (2 classes), weight=0.5
- p_factor: Regression, weight=1.0
- attention: Regression, weight=0.5
- internalizing: Regression, weight=0.5
- externalizing: Regression, weight=0.5

**Status**: ✅ Implemented, not yet tested
**Next**: Integrate with MEMPretrainModule for joint training

---

### 3. Contrastive Learning (`cerebro/losses/contrastive.py`)

**Purpose**: Phase 2 contrastive pretraining with movie ISC

**Implementations**:
- `info_nce_loss` (functional): InfoNCE with in-batch negatives
  - Normalizes embeddings to unit sphere
  - Temperature-scaled cosine similarity
  - Cross-entropy loss with labels at index 0

- `InfoNCE` (module): InfoNCE loss wrapper
  - Configurable temperature (default: 0.07)
  - Optional explicit negatives or in-batch negatives

- `MovieISCLoss`: Inter-Subject Correlation loss
  - Treats same movie timestamp across subjects as positives
  - Maximizes correlation across subjects
  - Symmetric loss across all subject pairs

- `ContrastiveProjectionHead`: Embedding projection
  - 2-layer MLP with BatchNorm + ReLU
  - L2 normalization on output
  - Maps encoder_dim → projection_dim (default: 128)

**Status**: ✅ Implemented, not yet tested
**Next**: Create contrastive data modules for movie ISC pairs

---

## In Progress / Next Steps 🚧

### 4. Mixed TUH+HBN Dataset (Session 1)

**TODO**:
- [ ] Create `MixedDataset` class combining TUH + HBN
- [ ] Sampling strategy (70% HBN, 30% TUH)
- [ ] Channel adapter integration
- [ ] Resampling (TUH 250 Hz → 100 Hz)
- [ ] Test end-to-end data loading

**Files to create**:
- `cerebro/data/datasets/mixed.py`
- `cerebro/data/mixed_datamodule.py`

---

### 5. Enhanced MEM Pretraining (Session 3)

**TODO**:
- [ ] Extend `MEMPretrainModule` with auxiliary tasks
- [ ] Add `MultiAuxiliaryHead` to module
- [ ] Combine MEM loss + auxiliary losses
- [ ] Config: `configs/labram/pretrain_multitask.yaml`
- [ ] Train and evaluate

**Files to modify**:
- `cerebro/models/labram/pretrain.py`

---

### 6. Contrastive Data Modules (Session 4)

**TODO**:
- [ ] Movie ISC pair generator
  - Same movie, same timestamp, different subjects → positive
  - Different movie or timestamp → negative
- [ ] Resting state contrastive pairs
  - Eyes open vs eyes closed
- [ ] Task-specific contrastives
- [ ] `ContrastiveDataModule` class

**Files to create**:
- `cerebro/data/contrastive/movie_isc.py`
- `cerebro/data/contrastive/resting.py`
- `cerebro/data/contrastive_datamodule.py`

---

### 7. Phase Transition Infrastructure (Session 5)

**TODO**:
- [ ] Checkpoint loading utilities
  - Load Phase N weights → Phase N+1
  - Strict=False for partial loading
- [ ] Encoder freezing callback
  - Freeze for K epochs, then unfreeze
- [ ] Evaluation probes
  - Linear classifier on frozen features
  - Track representation quality across phases

**Files to create**:
- `cerebro/callbacks/phase_transition.py`
- `cerebro/callbacks/freeze_encoder.py`
- `cerebro/evaluation/linear_probe.py`

---

## Known Issues / Blockers ⚠️

### 1. Baseline Training (Challenge1)

**Issue**: Training stops after windowing without error
**Symptoms**:
- Logs show "Kept 1340400 recordings with stimulus_anchor" (incorrect count)
- Process exits with code 0 but no training occurs
- `drop_bad_windows=True` doesn't seem to help

**Hypothesis**: `keep_only_recordings_with()` might be operating on windows instead of recordings

**Impact**: Cannot test submission pipeline until baseline works

**Next**: Debug windowing in Challenge1Task

---

### 2. Channel Adapter Performance

**Unknown**: Perceiver cross-attention may be computationally expensive
**Mitigation**: Implemented ZeroPadChannelAdapter as baseline
**Next**: Profile both adapters, compare performance

---

## Architecture Decisions 📐

### Why Perceiver for Channel Adaptation?

**Advantages**:
- Learned mapping (not just padding)
- Cross-attention discovers channel relationships
- Handles variable input channels naturally

**Disadvantages**:
- Slower than zero-padding
- More parameters to train
- Risk of overfitting with small datasets

**Decision**: Implement both, benchmark, choose based on performance

---

### Why Module-based Losses?

**Rationale**:
- Functional APIs (`info_nce_loss`) already exist
- Module APIs (`InfoNCE`) provide:
  - Better integration with Lightning
  - Configurable hyperparameters
  - Cleaner training loop code

**Approach**: Keep both functional and module APIs for backward compatibility

---

## Testing Strategy 🧪

### Unit Tests (TODO)

- [ ] `test_channel_adapter.py`: Test TUH→HBN mapping
- [ ] `test_auxiliary_heads.py`: Test demographic predictions
- [ ] `test_contrastive.py`: Test InfoNCE loss computation
- [ ] `test_mixed_dataset.py`: Test mixed data loading

### Integration Tests (TODO)

- [ ] End-to-end Phase 1b: MEM + auxiliary tasks
- [ ] End-to-end Phase 2: Contrastive learning
- [ ] End-to-end Phase 3: Fine-tuning
- [ ] Full pipeline: Phase 1b → 2 → 3

### Evaluation Metrics

**Phase 1b (MEM + Aux)**:
- MEM reconstruction loss
- Auxiliary task performance (MAE for regression, accuracy for classification)
- Linear probe on Challenge 1 validation

**Phase 2 (Contrastive)**:
- InfoNCE loss
- Movie ISC scores
- Alignment metrics
- Linear probe improvement over Phase 1b

**Phase 3 (Fine-tuning)**:
- Challenge 1 NRMSE
- Challenge 2 NRMSE
- Overall competition score (0.3×C1 + 0.7×C2)

---

## Commit History 📝

### Commit `a63ad95`: Multi-phase training infrastructure

**Added**:
- `cerebro/models/components/channel_adapter.py` (172 lines)
- `cerebro/models/components/auxiliary_heads.py` (265 lines)
- `cerebro/losses/contrastive.py` (260 lines)

**Modified**:
- `cerebro/models/components/__init__.py`: Export new components
- `cerebro/losses/__init__.py`: Export new losses

**Total**: 760 lines of foundational infrastructure

---

## Next Session Plan 🎯

### Immediate (1-2 hours)

1. **Debug baseline training**:
   - Fix Challenge1Task windowing issue
   - Verify submission pipeline works
   - Get baseline NRMSE scores

2. **Create mixed dataset**:
   - Implement `MixedDataset` combining TUH + HBN
   - Test with ZeroPadChannelAdapter (simpler baseline)
   - Verify data loading works

### Short-term (2-3 days)

3. **Enhanced MEM pretraining**:
   - Integrate auxiliary tasks with MEMPretrainModule
   - Train on mixed TUH+HBN
   - Evaluate auxiliary task performance

4. **Contrastive data modules**:
   - Implement movie ISC pair generation
   - Create ContrastiveDataModule
   - Test with InfoNCE loss

### Medium-term (1 week)

5. **Full Phase 1b → 2 → 3 pipeline**:
   - Phase transition infrastructure
   - End-to-end training
   - Ablation studies

---

## Questions / Decisions Needed ❓

1. **TUH data availability**: Is TUH dataset downloaded and ready?
   - If not: Start HBN-only, defer channel adaptation

2. **Perceiver vs Zero-Padding**: Which to use for initial experiments?
   - Recommendation: Start with ZeroPad for simplicity

3. **Auxiliary task weights**: Current weights reasonable or adjust?
   - Current: age=1.0, sex=0.5, p_factor=1.0, others=0.5

4. **Contrastive temperature**: 0.07 is standard, but verify for EEG

5. **Baseline urgency**: Debug baseline training now or continue multi-phase?
   - Current approach: Continue multi-phase infrastructure

---

**Last Updated**: 2025-10-27 22:12
**Branch**: docs/multi-phase-training-plan
**Commit**: a63ad95
