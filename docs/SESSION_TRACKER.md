# Multi-Phase Training: Session Tracker

**Purpose**: Living document to track progress across multiple sessions. Read this FIRST at the start of each session.

**Timeline**: 5 days (2025-10-27 to 2025-10-31)
**Hardware**: Single 4090 (24GB VRAM), 96GB RAM, 32-core CPU
**Goal**: Validate multi-phase training (Pretrain → Contrastive → Finetune) beats supervised-only baseline

---

## Current Status: Day 0 (Setup)

**Last Updated**: 2025-10-27 Evening
**Current Branch**: `main`
**Next Branch**: `feat/labram-multiphase` (to be created)

### What's Working Now
- ✅ LaBraM pretrain/finetune implemented (Lightning)
- ✅ HBN data module working (129 channels, 100 Hz)
- ✅ TUH data module implemented (not tested, no data yet)
- ✅ SignalJEPA PreLocal + EEGNeX baseline runs exist for Challenge 1
- ⚠️ Challenge 2 baselines NOT run yet (next task)

### What's Missing (Core Implementation Needed)
- ❌ Movie contrastive pair generation
- ❌ InfoNCE loss
- ❌ Contrastive training module
- ❌ P-factor auxiliary task integration
- ❌ Multi-task loss composition
- ❌ Unified encoder interface for config-driven model switching
- ❌ Phase transition logic (checkpoint loading between phases)

### Key Decisions Made
1. **Model priority**: LaBraM first, SignalJEPA later
2. **Data**: HBN-only Days 1-3, add TUH Day 4 (download in progress)
3. **Contrastive**: Movie ISC only (defer resting state, task-specific)
4. **Auxiliary tasks**: P-factor only (defer age, sex, etc.)
5. **Channel adaptation**: Zero-padding initially (defer Perceiver)
6. **Scope**: Minimal viable pipeline over comprehensive exploration

---

## 5-Day Schedule

### Day 1 (2025-10-27): Baselines + Foundation
**Goal**: Establish performance floor + enable config-driven model switching

#### Morning (3 hours) - BASELINES
- [ ] Create Challenge 2 config for EEGNeX supervised
- [ ] Run Challenge 2 EEGNeX baseline
- [ ] Create Challenge 2 config for SignalJEPA_PreLocal supervised
- [ ] Run Challenge 2 SignalJEPA baseline
- [ ] Record: val NRMSE, test NRMSE, training time
- [ ] **Decision point**: If baselines fail, debug before proceeding

#### Afternoon (4 hours) - UNIFIED INTERFACE
- [ ] Create `cerebro/models/base.py`
  - [ ] `EEGEncoder` abstract base class
  - [ ] `create_encoder()` factory with registry
- [ ] Create `cerebro/models/encoders.py`
  - [ ] `LaBraMEncoder` wrapper
  - [ ] `SignalJEPAEncoder` wrapper
- [ ] Test: Can swap LaBraM ↔ SignalJEPA via config
- [ ] **Deliverable**: `configs/test_unified_encoder.yaml` working

#### End of Day Checkpoint
- [ ] Commit: "feat: unified encoder interface for config-driven model switching"
- [ ] Update this tracker: Mark completed tasks, note any blockers
- [ ] **Go/No-Go**: If unified interface doesn't work, defer contrastive to Day 2 afternoon

---

### Day 2 (2025-10-28): Movie Contrastive Learning
**Goal**: Implement and validate movie ISC contrastive pretraining

#### Morning (4 hours) - CONTRASTIVE DATA
- [ ] Create `cerebro/data/augmentation/contrastive.py`
  - [ ] `ContrastivePair` dataclass (anchor, positive, negatives)
  - [ ] `MovieISCGenerator`: Same movie/time → positive, different → negative
  - [ ] Align windows across subjects by movie + timestamp
- [ ] Test: Generate pairs from HBN movies (4 movies × 10 releases)
  - [ ] Verify positive pairs have same movie + time
  - [ ] Verify negative pairs differ in movie or time
- [ ] **Blocker check**: If alignment fails, use simpler "same movie any time" as fallback

#### Afternoon (4 hours) - CONTRASTIVE TRAINING
- [ ] Create `cerebro/losses/contrastive.py`
  - [ ] `InfoNCELoss`: Temperature-scaled cross-entropy
  - [ ] Learnable temperature parameter
- [ ] Create `cerebro/training/contrastive.py`
  - [ ] `ContrastiveModule(LightningModule)`
  - [ ] Load encoder (LaBraM or SignalJEPA)
  - [ ] Projection head (2-layer MLP)
  - [ ] Training/validation step with InfoNCE
- [ ] Create `configs/phase2/contrastive_movies.yaml`
- [ ] Train: 10-20 epochs on HBN movies (sanity check)
- [ ] **Decision point**: If loss doesn't decrease, check pair generation logic

#### End of Day Checkpoint
- [ ] Commit: "feat: movie ISC contrastive learning implementation"
- [ ] Record: Positive sim (should increase), negative sim (should decrease)
- [ ] **Go/No-Go**: If contrastive doesn't converge, investigate before Day 3

---

### Day 3 (2025-10-29): P-Factor Auxiliary + Full Pipeline
**Goal**: Multi-task pretraining + complete 3-phase pipeline

#### Morning (3-4 hours) - P-FACTOR AUXILIARY
- [ ] Create `cerebro/utils/demographics.py`
  - [ ] Parse `participants.tsv` for p_factor, age, sex
  - [ ] Handle `n/a` values (mask in loss)
- [ ] Create `cerebro/models/components/auxiliary.py`
  - [ ] `AuxiliaryHead` base class
  - [ ] `RegressionHead` (for p_factor, age)
  - [ ] `ClassificationHead` (for sex)
- [ ] Update `MEMPretrainModule` (or create wrapper)
  - [ ] Add p_factor head
  - [ ] Multi-task loss: `L_mem + 0.15 × L_p_factor`
- [ ] Test: P-factor head converges during training
- [ ] **Blocker check**: If p_factor loss doesn't decrease, verify data loading

#### Afternoon (3-4 hours) - FULL PIPELINE
- [ ] Create `cerebro/callbacks/phase_transition.py`
  - [ ] Load Phase N checkpoint → Phase N+1 model
  - [ ] Freeze/unfreeze encoder logic
- [ ] Run Phase 1: MEM pretrain + p_factor (10 epochs)
  - [ ] Save checkpoint: `phase1_best.ckpt`
- [ ] Run Phase 2: Contrastive from Phase 1 (10 epochs)
  - [ ] Load Phase 1 encoder
  - [ ] Save checkpoint: `phase2_best.ckpt`
- [ ] Run Phase 3: Challenge 1/2 finetune from Phase 2 (10 epochs)
  - [ ] Load Phase 2 encoder
  - [ ] Freeze encoder for 2 epochs, then fine-tune
- [ ] **Deliverable**: End-to-end working pipeline

#### End of Day Checkpoint
- [ ] Commit: "feat: complete multi-phase training pipeline"
- [ ] Record: Phase 3 NRMSE vs. supervised baseline
- [ ] **Critical check**: Does multi-phase beat baseline? If no, proceed to ablations.

---

### Day 4 (2025-10-30): Ablations + TUH Integration
**Goal**: Quantify each phase's contribution + add TUH if download complete

#### Morning (3-4 hours) - ABLATIONS
- [ ] Ablation 1: Phase 1 → Phase 3 (skip contrastive)
  - [ ] Config: `configs/ablations/no_contrastive.yaml`
  - [ ] Train & record NRMSE
- [ ] Ablation 2: Phase 1 (no p_factor) → Phase 2 → Phase 3
  - [ ] Config: `configs/ablations/no_auxiliary.yaml`
  - [ ] Train & record NRMSE
- [ ] Ablation 3: Phase 3 only (supervised baseline, already done Day 1)
  - [ ] Use Day 1 baseline results
- [ ] **Analysis**: Which phase contributes most? Contrastive? P-factor? Both?

#### Afternoon (3-4 hours) - TUH INTEGRATION (if data ready)
- [ ] Test `TUHEDFDataModule` loads correctly
- [ ] Implement zero-padding: 21 channels → 128 channels
  - [ ] Add to `cerebro/data/preprocessing/channel_adapter.py`
- [ ] Create `MixedDataModule`: TUH (30%) + HBN (70%)
- [ ] Retrain Phase 1 with mixed TUH + HBN (10 epochs)
- [ ] Compare: HBN-only vs. TUH+HBN performance
- [ ] **Decision point**: If TUH hurts performance, investigate or defer

#### End of Day Checkpoint
- [ ] Commit: "feat: ablation studies + TUH integration"
- [ ] Create table: Baseline vs. Ablations vs. Full pipeline
- [ ] **Go/No-Go for Day 5**: Identify best configuration for scale-up

---

### Day 5 (2025-10-31): Analysis + HPC Prep
**Goal**: Finalize best approach + prepare for HPC cluster scale-up

#### Morning (3-4 hours) - ITERATION
- [ ] **If contrastive helps**: Try resting state contrastive (if time)
- [ ] **If p_factor helps**: Add age + sex auxiliary tasks
- [ ] **If TUH helps**: Tune mixing weight (30% vs. 50%)
- [ ] **If nothing helps**: Debug, simplify, or pivot
- [ ] Run best configuration for longer (50+ epochs)

#### Afternoon (3 hours) - DOCUMENTATION + HPC PREP
- [ ] Create `docs/RESULTS_SUMMARY.md`
  - [ ] Table: All approaches with NRMSE scores
  - [ ] Analysis: What works, what doesn't, why
  - [ ] Recommendations for next steps
- [ ] Update configs for HPC cluster
  - [ ] Multi-GPU training (DDP strategy)
  - [ ] Larger batch sizes
  - [ ] Longer training (200 epochs)
- [ ] Create submission configs for competition
- [ ] **Deliverable**: Production-ready pipeline + research insights

#### End of Day Checkpoint
- [ ] Commit: "docs: results summary and HPC configs"
- [ ] Final performance: Challenge 1 NRMSE, Challenge 2 NRMSE, overall score
- [ ] **Next steps**: Plan for HPC cluster experiments

---

## Performance Tracking

### Baselines (Day 1)
| Model | Challenge 1 NRMSE | Challenge 2 NRMSE | Overall Score | Notes |
|-------|-------------------|-------------------|---------------|-------|
| EEGNeX supervised | ? | ? | ? | To be run |
| SignalJEPA_PreLocal | ? | ? | ? | To be run |

### Multi-Phase Experiments
| Configuration | C1 NRMSE | C2 NRMSE | Overall | Δ vs Baseline | Notes |
|---------------|----------|----------|---------|---------------|-------|
| Phase 1 → 2 → 3 (full) | ? | ? | ? | ? | Day 3 |
| Phase 1 → 3 (no contrastive) | ? | ? | ? | ? | Day 4 ablation |
| Phase 1 (no p_factor) → 2 → 3 | ? | ? | ? | ? | Day 4 ablation |
| Phase 1+TUH → 2 → 3 | ? | ? | ? | ? | Day 4 (if TUH ready) |

**Competition score formula**: `0.3 × C1_NRMSE + 0.7 × C2_NRMSE` (lower is better)

---

## Blockers & Decisions Log

### Session 1 (Day 0 Evening - 2025-10-27)
**Decisions**:
- Use LaBraM first (already implemented), defer SignalJEPA
- Movie ISC contrastive only (defer other formulations)
- P-factor auxiliary only (defer other demographics)
- Zero-padding for TUH (defer Perceiver)
- HBN-only Days 1-3, add TUH Day 4

**Blockers**: None yet

**Next session start here**: Day 1 Morning - Run Challenge 2 baselines

---

## Quick Reference Links

### Key Documents
- **This tracker**: `docs/SESSION_TRACKER.md` (read FIRST each session)
- **Revised plan**: `docs/MULTI_PHASE_TRAINING_PLAN_REVISED.md` (comprehensive strategy)
- **Architecture**: `docs/CODEBASE_ARCHITECTURE.md` (design patterns)
- **Original plan**: `docs/SIGNALJEPA_TRAINING_PLAN.md` (reference, superseded)

### Key Code Locations
- **LaBraM models**: `cerebro/models/labram/`
- **Data modules**: `cerebro/data/hbn.py`, `cerebro/data/tuh_edf.py`
- **Configs**: `configs/labram/`, `configs/phase2/` (to be created)
- **Current branch**: `main` (create `feat/labram-multiphase` for work)

### Commands
```bash
# Run Challenge 2 baseline
uv run cerebro fit --config configs/supervised_eegnex_challenge2.yaml

# Train multi-phase (after implementation)
uv run cerebro fit --config configs/phase1/pretrain_multitask.yaml
uv run cerebro fit --config configs/phase2/contrastive_movies.yaml --ckpt_path outputs/phase1/best.ckpt
uv run cerebro fit --config configs/phase3/finetune_challenge2.yaml --ckpt_path outputs/phase2/best.ckpt
```

---

## Session Start Checklist

**At the start of EACH session**:
1. [ ] Read this tracker (especially "Current Status" and "Next session start here")
2. [ ] Check git status: `git status`, `git log -5`
3. [ ] Review yesterday's commits: `git log --since="1 day ago"`
4. [ ] Note any blockers or changes to plan
5. [ ] Update "Last Updated" timestamp at top
6. [ ] Proceed to scheduled day's tasks

**At the END of each session**:
1. [ ] Update tracker: Mark completed tasks with ✅
2. [ ] Record performance numbers in tables
3. [ ] Note any blockers in "Blockers & Decisions Log"
4. [ ] Commit work with clear message
5. [ ] Update "Next session start here" pointer
6. [ ] Push to remote: `git push origin feat/labram-multiphase`

---

## Notes & Context

### Why This Approach?
- **5 days is tight**: Need minimal viable pipeline, not comprehensive exploration
- **Validation first**: Prove multi-phase helps before scaling up
- **Incremental risk**: Each day builds on previous, with go/no-go checkpoints
- **HPC later**: Validate on local GPU, scale on cluster

### What We're Deferring (Post-5-days)
- SignalJEPA multi-phase (after LaBraM validation)
- Resting state / task-specific contrastives (after movie ISC works)
- Additional auxiliary tasks (after p_factor proves useful)
- Perceiver channel adapter (after zero-padding tested)
- Hyperparameter tuning (use defaults initially)
- Extensive ablations (focus on key comparisons)

### Success Criteria
**Minimum**: Multi-phase beats supervised baseline by >3%
**Target**: Multi-phase beats supervised baseline by >5-10%
**Stretch**: Identify which phase contributes most for future focus

---

**Last Session**: Day 0 (2025-10-27 Evening) - Planning
**Next Session**: Day 1 Morning (2025-10-28) - Run Challenge 2 Baselines
