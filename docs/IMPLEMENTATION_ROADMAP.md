# SignalJEPA Implementation Roadmap

**Quick Start Guide**: This document ties together the training plan and codebase architecture, providing a concrete path from current state to fully implemented multi-phase pipeline.

---

## 📚 Document Overview

### Core Documents
1. **`SIGNALJEPA_TRAINING_PLAN.md`**: Research plan (what to build)
   - 4-phase training pipeline
   - Dataset specifications
   - Evaluation strategy

2. **`CODEBASE_ARCHITECTURE.md`**: Software architecture (how to build it)
   - Directory structure
   - Core abstractions
   - DRY principles

3. **`channel_adaptation.md`**: Channel alignment techniques
   - Perceiver-style cross-attention
   - Handles 21ch (TUH) → 128ch (HBN)

4. **This document**: Implementation roadmap (execution plan)

---

## 🎯 Current State vs. Target State

### Current State (main branch as of 2025-10-27)
```
cerebro/
├── data/
│   ├── hbn.py           ✅ HBN data loading
│   ├── tuh_edf.py       ✅ TUH data loading
│   └── tuh.py           ✅ TUH utilities
├── models/
│   ├── challenge1.py    ✅ Challenge 1 model
│   ├── labram/          ✅ LaBraM implementation
│   └── components/      ⚠️  Skeleton only
├── callbacks/           ✅ Model autopsy, checkpoint fix
└── configs/            ✅ LaBraM configs, Challenge configs
```

**Strengths**:
- TUH and HBN data loaders exist
- LaBraM provides masked autoencoding baseline
- Lightning CLI infrastructure works
- Checkpoint management in place

**Gaps**:
- No SignalJEPA components (predictors, target encoder)
- No channel adaptation (21ch → 128ch)
- No contrastive learning infrastructure
- No auxiliary demographic tasks
- No mixed TUH+HBN dataset
- No modular encoder factory

### Target State (after implementation)
```
cerebro/
├── data/
│   ├── loaders/
│   │   ├── hbn.py           ✅ Port existing
│   │   ├── tuh.py           ✅ Port existing
│   │   └── mixed.py         🆕 NEW
│   ├── preprocessing/
│   │   ├── resampling.py    🆕 NEW
│   │   └── windowing.py     🆕 NEW
│   └── augmentation/
│       ├── masking.py       🆕 NEW (spatial/temporal/spatiotemporal)
│       └── contrastive.py   🆕 NEW (positive/negative pair generation)
├── models/
│   ├── components/
│   │   ├── encoders/
│   │   │   ├── transformer.py    🆕 NEW
│   │   │   ├── mamba.py          🔄 Adapt from LaBraM
│   │   │   └── channel_adapter.py 🆕 NEW (Perceiver)
│   │   ├── predictors/
│   │   │   ├── prelocal.py       🆕 NEW
│   │   │   ├── postlocal.py      🆕 NEW (optional)
│   │   │   └── contextual.py     🆕 NEW (optional)
│   │   └── heads/
│   │       ├── regression.py     🆕 NEW
│   │       └── projection.py     🆕 NEW (contrastive)
│   ├── signaljepa/
│   │   ├── model.py              🆕 NEW
│   │   ├── target_encoder.py     🆕 NEW (EMA)
│   │   └── auxiliary.py          🆕 NEW (demographics)
│   └── contrastive/
│       └── model.py              🆕 NEW
├── losses/
│   ├── signaljepa.py             🆕 NEW
│   ├── contrastive.py            🆕 NEW (InfoNCE)
│   └── composite.py              🆕 NEW
├── training/
│   ├── signaljepa.py             🆕 NEW (Phase 1)
│   ├── contrastive.py            🆕 NEW (Phase 2)
│   └── supervised.py             🔄 Adapt from challenge1.py
└── configs/
    ├── phase1/ ...               🆕 NEW
    ├── phase2/ ...               🆕 NEW
    └── phase3/ ...               🆕 NEW
```

Legend:
- ✅ Exists and works
- 🔄 Exists but needs adaptation
- 🆕 NEW - Needs implementation
- ⚠️  Skeleton only

---

## 🗓️ Implementation Sessions

### Session 1: Foundation & Data Infrastructure (2-3 days)

**Goal**: Establish data pipeline for TUH + HBN with channel adaptation.

#### Tasks
- [ ] **Scaffold new directory structure**
  ```bash
  mkdir -p cerebro/data/{loaders,preprocessing,augmentation}
  mkdir -p cerebro/models/components/{encoders,predictors,heads}
  mkdir -p cerebro/losses cerebro/training
  ```

- [ ] **Port existing code**
  ```bash
  mv cerebro/data/hbn.py cerebro/data/loaders/
  mv cerebro/data/tuh_edf.py cerebro/data/loaders/
  ```

- [ ] **Implement base abstractions** (`data/base.py`)
  - `EEGWindow` dataclass
  - `EEGDataset` protocol

- [ ] **Implement preprocessing** (`data/preprocessing/`)
  - `resampling.py`: Resample TUH (250 Hz) → 100 Hz
  - `windowing.py`: Extract fixed-length windows with stride

- [ ] **Implement channel adapter** (`models/components/encoders/channel_adapter.py`)
  - Perceiver-style cross-attention
  - Fixed 128 learnable electrode queries
  - Handle 21ch (TUH) and 128ch (HBN) inputs

- [ ] **Implement mixed dataset** (`data/loaders/mixed.py`)
  - Combine TUH + HBN with configurable weights (70% HBN, 30% TUH)
  - Apply channel adapter transparently

- [ ] **Unit tests**
  - `tests/test_channel_adapter.py`: Test 21ch → 128ch, 128ch → 128ch
  - `tests/test_resampling.py`: Verify 250 Hz → 100 Hz
  - `tests/test_mixed_dataset.py`: Verify sampling distribution

#### Deliverable
✅ Can load TUH + HBN data with unified 128-channel representation at 100 Hz

---

### Session 2: Masking & Demographics (1-2 days)

**Goal**: Implement masking strategies and demographic auxiliary tasks.

#### Tasks
- [ ] **Implement masking strategies** (`data/augmentation/masking.py`)
  - `SpatialMasking`: Random channel dropout (30-70%)
  - `TemporalMasking`: Random temporal blocks (200-500ms)
  - `SpatiotemporalMasking`: Combined 3D masking

- [ ] **Implement demographic loading** (`data/modules/demographics.py`)
  - Parse `participants.tsv`
  - Handle `n/a` values (mask for loss)
  - Extract: age, sex, ehq_total, p_factor, attention, internalizing, externalizing

- [ ] **Implement auxiliary heads** (`models/components/heads/`)
  - `regression.py`: MLP for continuous targets
  - `classification.py`: MLP for categorical targets
  - Registry pattern for easy configuration

- [ ] **Unit tests**
  - `tests/test_masking.py`: Verify mask ratios, shapes
  - `tests/test_demographics.py`: Check n/a handling

#### Deliverable
✅ Can apply spatial/temporal/spatiotemporal masking
✅ Can load demographic data with proper n/a masking

---

### Session 3: SignalJEPA Components (2-3 days)

**Goal**: Implement SignalJEPA encoder, target encoder, and predictors.

#### Tasks
- [ ] **Implement context encoder** (`models/components/encoders/transformer.py`)
  - Transformer-based encoder
  - Input: (batch, channels, time) → embeddings
  - Use channel adapter as first layer

- [ ] **Implement target encoder** (`models/signaljepa/target_encoder.py`)
  - EMA copy of context encoder
  - Momentum update: θ_target ← α θ_target + (1-α) θ_context

- [ ] **Implement predictors** (`models/components/predictors/`)
  - `prelocal.py`: Cross-attention predictor (target positions attend to context)
  - `postlocal.py`: Optional refinement (defer if time-constrained)
  - `contextual.py`: Optional global aggregation (defer if time-constrained)

- [ ] **Implement SignalJEPA model** (`models/signaljepa/model.py`)
  - Compose: encoder + target_encoder + predictor + auxiliary_heads
  - Forward pass: masked input → context → target → prediction
  - Return reconstruction loss + auxiliary losses

- [ ] **Unit tests**
  - `tests/test_signaljepa_model.py`: Forward pass, shapes
  - `tests/test_target_encoder.py`: EMA update correctness

#### Deliverable
✅ SignalJEPA model can encode, predict, and compute losses

---

### Session 4: SignalJEPA Training (1-2 days)

**Goal**: Implement Phase 1 training loop and configs.

#### Tasks
- [ ] **Implement SignalJEPA loss** (`losses/signaljepa.py`)
  - Reconstruction loss (MSE or cosine)
  - Composite loss: L_recon + aux_weight × Σ L_aux

- [ ] **Implement training module** (`training/signaljepa.py`)
  - Inherit from `BaseEEGModule`
  - training_step: apply masking, encode, predict, compute loss
  - on_train_batch_end: EMA update
  - validation_step: reconstruction quality, auxiliary metrics

- [ ] **Create configs** (`configs/phase1/`)
  - `signaljepa_mini.yaml`: Fast prototyping (R1 only, 10 epochs)
  - `signaljepa_base.yaml`: Standard training (R1-R4, 50 epochs)
  - `signaljepa_full.yaml`: Full pretraining (R1-R4+R6-R11, 200 epochs)

- [ ] **Implement evaluation probe callback** (`callbacks/evaluation_probe.py`)
  - Freeze encoder every N epochs
  - Train linear classifier on Challenge 1 validation
  - Log probing accuracy to track representation quality

- [ ] **Integration tests**
  - `tests/test_phase1_training.py`: Run mini training for 2 epochs

#### Deliverable
✅ Can run Phase 1 training: `uv run cerebro fit --config configs/phase1/signaljepa_mini.yaml`

---

### Session 5: Contrastive Pair Generation (2 days)

**Goal**: Implement contrastive pair generators for movies, resting state, and tasks.

#### Tasks
- [ ] **Implement contrastive data structures** (`data/augmentation/contrastive.py`)
  - `ContrastivePair` dataclass: anchor, positive, negatives, metadata
  - `PairGenerator` protocol

- [ ] **Implement movie ISC generator** (`data/augmentation/contrastive.py`)
  - Extract aligned windows across subjects (same movie, same timestamp)
  - Positive: different subjects, same movie+time
  - Negative: different movies or different timestamps

- [ ] **Implement resting state generator**
  - Parse eyes_open / eyes_closed events
  - Positive: same state, different subjects
  - Negative: different states or cross-subject-cross-state

- [ ] **Implement task-specific generators**
  - contrastChangeDetection: same stimulus type across subjects
  - seqLearning: same sequence/block across subjects
  - surroundSupp: same condition across subjects

- [ ] **Implement contrastive dataset** (`data/modules/contrastive.py`)
  - Lightning DataModule
  - Sample pairs from HBN tasks with configurable weights
  - Return: (anchor, positive, negatives) batches

- [ ] **Unit tests**
  - `tests/test_contrastive_pairs.py`: Verify positive/negative logic

#### Deliverable
✅ Can generate contrastive pairs for all task types

---

### Session 6: Contrastive Training (1-2 days)

**Goal**: Implement Phase 2 contrastive learning.

#### Tasks
- [ ] **Implement InfoNCE loss** (`losses/contrastive.py`)
  - Temperature-scaled dot product
  - Log-softmax over positives + negatives
  - Learnable temperature parameter

- [ ] **Implement projection head** (`models/components/heads/projection.py`)
  - 2-layer MLP: latent_dim → 512 → 256
  - L2 normalization

- [ ] **Implement contrastive model** (`models/contrastive/model.py`)
  - Wrap encoder + projection head
  - Forward: encode anchor, positive, negatives → compute InfoNCE

- [ ] **Implement training module** (`training/contrastive.py`)
  - Load Phase 1 checkpoint
  - Training step: generate embeddings, compute InfoNCE
  - Log: positive similarity, negative similarity, alignment score

- [ ] **Create configs** (`configs/phase2/`)
  - `contrastive_movie.yaml`: Movie ISC only
  - `contrastive_full.yaml`: All tasks mixed

- [ ] **Integration tests**
  - `tests/test_phase2_training.py`: Train for 2 epochs

#### Deliverable
✅ Can run Phase 2 training: `uv run cerebro fit --config configs/phase2/contrastive_full.yaml`

---

### Session 7: Supervised Fine-Tuning (1 day)

**Goal**: Adapt Challenge 1/2 for Phase 3.

#### Tasks
- [ ] **Refactor Challenge 1 model** (`models/supervised/challenge1.py`)
  - Load Phase 2 checkpoint
  - Freeze encoder for 5 epochs, then fine-tune
  - Keep existing pipeline (stimulus-locked windows)

- [ ] **Refactor Challenge 2 model** (`models/supervised/challenge2.py`)
  - Load Phase 2 checkpoint
  - Add subject-level aggregation (mean pooling)
  - Keep existing pipeline (multi-task windows)

- [ ] **Create configs** (`configs/phase3/`)
  - `supervised_challenge1.yaml`
  - `supervised_challenge2.yaml`
  - `supervised_multitask.yaml`: Joint training

- [ ] **Integration tests**
  - `tests/test_phase3_training.py`: Fine-tune for 2 epochs

#### Deliverable
✅ Can run Phase 3 training: `uv run cerebro fit --config configs/phase3/supervised_challenge1.yaml`

---

### Session 8: Evaluation & Ablations (2 days)

**Goal**: Full pipeline evaluation and ablation studies.

#### Tasks
- [ ] **Implement evaluation suite** (`evaluation/`)
  - `metrics.py`: NRMSE, MAE, ISC
  - `probing.py`: Linear evaluation
  - `visualization.py`: t-SNE embeddings, attention maps

- [ ] **End-to-end pipeline test**
  ```bash
  # Phase 1
  uv run cerebro fit --config configs/phase1/signaljepa_base.yaml

  # Phase 2 (load Phase 1 checkpoint)
  uv run cerebro fit --config configs/phase2/contrastive_full.yaml \
      --ckpt_path outputs/phase1/.../best.ckpt

  # Phase 3 (load Phase 2 checkpoint)
  uv run cerebro fit --config configs/phase3/supervised_challenge1.yaml \
      --ckpt_path outputs/phase2/.../best.ckpt
  ```

- [ ] **Ablation studies**
  - Create configs in `configs/ablations/`:
    - `no_contrastive.yaml`: Phase 1 → Phase 3 (skip Phase 2)
    - `no_auxiliary.yaml`: Phase 1 without demographic tasks
    - `supervised_only.yaml`: Phase 3 from scratch
    - `spatial_masking_only.yaml`: No temporal masking
    - `movie_contrastive_only.yaml`: Only movie ISC

- [ ] **Run ablations and compare**
  - Track Challenge 1 NRMSE, Challenge 2 NRMSE, overall score
  - Attribute performance gains to each phase

- [ ] **Generate report**
  - Wandb dashboard with phase comparisons
  - Summary markdown: `docs/EVALUATION_RESULTS.md`

#### Deliverable
✅ Quantified performance attribution for each phase
✅ Clear winner: full pipeline vs. ablations

---

### Session 9+: Architecture Exploration (Ongoing)

**Goal**: Experiment with alternative architectures (Phase 4).

#### Tasks (prioritize after establishing baselines)
- [ ] **Mamba encoder** (`models/components/encoders/mamba.py`)
  - Port from LaBraM
  - Compare to transformer

- [ ] **FNO encoder** (`models/components/encoders/fno.py`)
  - Fourier Neural Operator
  - Frequency-domain processing

- [ ] **Alternative channel adaptation**
  - Graph Neural Network (electrode graph)
  - Neural implicit fields

- [ ] **Training improvements**
  - Curriculum learning (easy → hard masking)
  - Multi-scale training (variable window lengths)
  - Hard negative mining for contrastive

#### Deliverable
✅ Architecture comparison: Transformer vs. Mamba vs. FNO vs. Hybrid

---

## 🚀 Getting Started (Next Steps)

### Immediate Actions (Today)

1. **Create new branch**
   ```bash
   git checkout main
   git pull
   git checkout -b feat/signaljepa-pipeline
   ```

2. **Scaffold directory structure**
   ```bash
   mkdir -p cerebro/data/{loaders,preprocessing,augmentation,modules}
   mkdir -p cerebro/models/components/{encoders,predictors,heads,aggregation}
   mkdir -p cerebro/models/{signaljepa,contrastive,supervised}
   mkdir -p cerebro/losses
   mkdir -p cerebro/training
   mkdir -p cerebro/evaluation
   mkdir -p configs/{phase1,phase2,phase3,ablations}

   # Create __init__.py files
   find cerebro/data cerebro/models cerebro/losses cerebro/training -type d -exec touch {}/__init__.py \;
   ```

3. **Port existing code**
   ```bash
   # Backup existing
   cp cerebro/data/hbn.py cerebro/data/loaders/hbn.py
   cp cerebro/data/tuh_edf.py cerebro/data/loaders/tuh.py

   # Keep originals for now (remove after migration complete)
   ```

4. **Start Session 1 implementation**
   - Begin with `data/base.py`: Define `EEGWindow`, `EEGDataset`
   - Implement `models/components/encoders/channel_adapter.py`

### Session-by-Session Checklist

Use this checklist to track progress:

```markdown
## Implementation Progress

### Session 1: Foundation & Data Infrastructure
- [ ] Directory structure scaffolded
- [ ] Existing code ported
- [ ] data/base.py implemented
- [ ] data/preprocessing/resampling.py implemented
- [ ] data/preprocessing/windowing.py implemented
- [ ] models/components/encoders/channel_adapter.py implemented
- [ ] data/loaders/mixed.py implemented
- [ ] Unit tests pass

### Session 2: Masking & Demographics
- [ ] data/augmentation/masking.py implemented
- [ ] data/modules/demographics.py implemented
- [ ] models/components/heads/ implemented
- [ ] Unit tests pass

### Session 3: SignalJEPA Components
- [ ] models/components/encoders/transformer.py implemented
- [ ] models/signaljepa/target_encoder.py implemented
- [ ] models/components/predictors/prelocal.py implemented
- [ ] models/signaljepa/model.py implemented
- [ ] Unit tests pass

### Session 4: SignalJEPA Training
- [ ] losses/signaljepa.py implemented
- [ ] training/signaljepa.py implemented
- [ ] configs/phase1/ created
- [ ] callbacks/evaluation_probe.py implemented
- [ ] Can run training end-to-end

### Session 5: Contrastive Pair Generation
- [ ] data/augmentation/contrastive.py implemented
- [ ] Movie ISC generator
- [ ] Resting state generator
- [ ] Task-specific generators
- [ ] data/modules/contrastive.py implemented
- [ ] Unit tests pass

### Session 6: Contrastive Training
- [ ] losses/contrastive.py (InfoNCE) implemented
- [ ] models/components/heads/projection.py implemented
- [ ] models/contrastive/model.py implemented
- [ ] training/contrastive.py implemented
- [ ] configs/phase2/ created
- [ ] Can run training end-to-end

### Session 7: Supervised Fine-Tuning
- [ ] models/supervised/challenge1.py adapted
- [ ] models/supervised/challenge2.py adapted
- [ ] Subject-level aggregation
- [ ] configs/phase3/ created
- [ ] Can run training end-to-end

### Session 8: Evaluation & Ablations
- [ ] evaluation/ module implemented
- [ ] Full pipeline test (Phase 1 → 2 → 3)
- [ ] Ablation configs created
- [ ] Ablations run and compared
- [ ] Results documented

### Session 9+: Architecture Exploration
- [ ] Mamba encoder
- [ ] FNO encoder
- [ ] Alternative channel adaptation
- [ ] Training improvements
```

---

## 📊 Success Metrics

### Minimum Viable Product (MVP)
- ✅ All sessions 1-8 complete
- ✅ Can train Phase 1 → 2 → 3 pipeline
- ✅ Baselines established for each phase
- ✅ Pretrained model outperforms supervised-only baseline

### Research Success
- ✅ Clear performance attribution (which phases contribute most?)
- ✅ Contrastive phase improves over SignalJEPA-only
- ✅ Competitive performance on Challenge 1 & 2
- ✅ Cross-subject and cross-task transfer demonstrated

### Stretch Goals
- ✅ State-of-the-art on HBN competition
- ✅ Published ablation study
- ✅ Reusable foundation model for future EEG tasks
- ✅ Novel insights into channel adaptation or task transfer

---

## 🛠️ Development Workflow

### For Each Session

1. **Plan** (15 min)
   - Review session tasks
   - Identify dependencies
   - Prioritize implementation order

2. **Implement** (2-4 hours)
   - Write code with type hints
   - Follow abstractions from `CODEBASE_ARCHITECTURE.md`
   - Use DRY principles

3. **Test** (30 min)
   - Write unit tests
   - Run integration tests
   - Verify on small data (R1 mini)

4. **Document** (15 min)
   - Add docstrings
   - Update progress checklist
   - Note any deviations from plan

5. **Commit** (5 min)
   - Atomic commits per feature
   - Clear commit messages
   - Push to `feat/signaljepa-pipeline` branch

### Example Commit Messages
```bash
git commit -m "feat(data): implement channel adapter with Perceiver cross-attention"
git commit -m "test(data): add unit tests for 21ch→128ch mapping"
git commit -m "feat(models): implement SignalJEPA pre-local predictor"
git commit -m "config(phase1): add signaljepa_mini.yaml for fast prototyping"
git commit -m "fix(training): correct EMA momentum update in target encoder"
```

---

## 🔍 Debugging & Troubleshooting

### Common Issues

**Issue**: Channel adapter produces NaN outputs
- **Cause**: Cross-attention with no masking on padding
- **Fix**: Add attention mask for padded channels

**Issue**: Phase 2 doesn't improve over Phase 1
- **Cause**: Temperature τ too high/low, or not enough negatives
- **Fix**: Tune temperature (try 0.05, 0.07, 0.1), increase negative samples

**Issue**: Training unstable with auxiliary tasks
- **Cause**: Auxiliary loss weight too high
- **Fix**: Reduce aux_weight (try 0.1, 0.2, 0.45), add gradient clipping

**Issue**: Out of memory during training
- **Cause**: Batch size too large or contrastive negatives too many
- **Fix**: Reduce batch size, accumulate gradients, use gradient checkpointing

---

## 📝 Notes & Considerations

### Design Decisions

**Q**: Should auxiliary tasks be trained jointly in Phase 1 or as separate phase?
**A**: Jointly in Phase 1. Simpler pipeline, shared encoder benefits from multi-task learning.

**Q**: How many predictor types to implement initially?
**A**: Pre-local only. Post-local and contextual are optional (defer to Phase 4).

**Q**: Mean pooling or attention pooling for Challenge 2 subject aggregation?
**A**: Start with mean pooling (simpler). Try attention if performance plateaus.

**Q**: How many epochs for Phase 1?
**A**: Start with 50 epochs on R1-R4. Scale to 200 epochs if beneficial.

### Risks & Mitigation

**Risk**: Channel adaptation doesn't generalize
- **Mitigation**: Test on validation early, ablate alternatives

**Risk**: Implementation takes longer than estimated
- **Mitigation**: Defer optional components (post-local predictor, contextual predictor, Phase 4)

**Risk**: Contrastive phase provides minimal improvement
- **Mitigation**: Ablate individual formulations (movie-only, resting-only, task-only)

**Risk**: Can't reproduce results across runs
- **Mitigation**: Use `set_seed()`, track random state, checkpoint frequently

---

## 🎉 Completion Criteria

### Done When
- [ ] Can run full pipeline: Phase 1 → 2 → 3
- [ ] Reproducible training with consistent convergence
- [ ] Ablation studies quantify phase contributions
- [ ] Results documented in `docs/EVALUATION_RESULTS.md`
- [ ] Code merged to main after review

### Optional Extensions (if time permits)
- [ ] Mamba encoder comparison
- [ ] Hard negative mining
- [ ] Multi-scale training
- [ ] Cross-dataset generalization (train TUH, test HBN)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-27
**Author**: Varun (with Claude assistance)
**Next Review**: After Session 2 completion
