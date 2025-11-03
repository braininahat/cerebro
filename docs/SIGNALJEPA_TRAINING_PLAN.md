# SignalJEPA Multi-Phase Training Plan

**Goal**: Train a foundation EEG model using SignalJEPA with multi-phase pretraining (self-supervised → contrastive → supervised) on TUH + HBN datasets for superior cross-subject and cross-task transfer learning.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset Specifications](#dataset-specifications)
3. [Phase 1: Self-Supervised SignalJEPA Pretraining](#phase-1-self-supervised-signaljepa-pretraining)
4. [Phase 2: Contrastive Learning](#phase-2-contrastive-learning)
5. [Phase 3: Supervised Fine-Tuning](#phase-3-supervised-fine-tuning)
6. [Phase 4: Architecture Exploration](#phase-4-architecture-exploration)
7. [Evaluation Strategy](#evaluation-strategy)
8. [Implementation Timeline](#implementation-timeline)

---

## Overview

### Training Philosophy
**Bottom-up representation learning**: Start with general unsupervised patterns, add task structure via contrastive learning, then specialize via supervised fine-tuning.

### Why Multi-Phase?
1. **Phase 1 (SignalJEPA)**: Learn general temporal-spatial EEG dynamics
2. **Phase 2 (Contrastive)**: Inject task-specific structure and inter-subject alignment
3. **Phase 3 (Supervised)**: Specialize for competition metrics
4. **Phase 4 (Architecture)**: Once training framework is validated, explore architectural improvements

### Key Technical Challenges
- **Channel count mismatch**: TUH (21 channels) vs HBN (128 channels)
- **Sampling rate mismatch**: TUH (250 Hz) vs HBN (100 Hz)
- **Cross-dataset transfer**: Different subject populations, electrode montages
- **Cross-task transfer**: Diverse cognitive tasks require generalizable representations

---

## Dataset Specifications

### HBN Dataset (Healthy Brain Network)
- **Channels**: 128 (full 10-20 system + extras)
- **Sampling rate**: 100 Hz
- **Subjects**: ~1000+ across R1-R4, R6-R11 (R5 held out for competition)
- **Tasks**: 10 cognitive paradigms
- **Demographics**: age, sex, ehq_total, p_factor, attention, internalizing, externalizing

#### Task Breakdown
| Task | Type | Duration | Events | Notes |
|------|------|----------|--------|-------|
| **RestingState** | Baseline | 3-6 min | eyes_open/eyes_closed | Spontaneous activity |
| **DespicableMe** | Movie | ~5 min | video_start/stop | Naturalistic stimuli |
| **ThePresent** | Movie | ~5 min | video_start/stop | Naturalistic stimuli |
| **DiaryOfAWimpyKid** | Movie | ~5 min | video_start/stop | Naturalistic stimuli |
| **FunwithFractals** | Movie | ~5 min | video_start/stop | Naturalistic stimuli |
| **contrastChangeDetection** | Cognitive | ~10 min | stimulus/response | Challenge 1 target task |
| **surroundSupp** | Perceptual | Variable | fixation/stim with contrast | Visual perception |
| **seqLearning6target** | Memory | Variable | dot sequences | Sequence learning |
| **seqLearning8target** | Memory | Variable | dot sequences | Sequence learning |
| **symbolSearch** | Cognitive | Variable | search events | Visual search |

### TUH Dataset (Temple University Hospital)
- **Channels**: 21 (standard 10-20 subset)
- **Sampling rate**: 250 Hz (typically)
- **Subjects**: Large clinical population
- **Data**: Continuous resting/clinical EEG
- **Use case**: Large-scale unsupervised pretraining

### Channel/Sampling Alignment Strategy
**Approach**: Perceiver-style cross-attention + random masking (see `docs/channel_adaptation.md`)

**Key idea**:
- Fixed 128 learnable "electrode queries"
- TUH's 21 channels → sparse observation of 128-d manifold
- Random masking during pretraining (50-90% channels masked)
- Model learns electrode placement implicitly through spatial attention

**Sampling rate**: Resample TUH to 100 Hz during preprocessing for consistency

---

## Phase 1: Self-Supervised SignalJEPA Pretraining

### Objective
Learn general temporal-spatial EEG representations without task labels using masked predictive coding.

### SignalJEPA Architecture Components

#### Full Architecture (Not Just Pre-Local)
1. **Context Encoder** (E_context)
   - Encodes unmasked regions of EEG
   - Input: (B, C, T) where C=128 (with masking), T=time steps
   - Output: Contextual embeddings

2. **Target Encoder** (E_target) - EMA of context encoder
   - Encodes masked target regions
   - Momentum update: θ_target ← α θ_target + (1-α) θ_context
   - Provides stable targets for predictor

3. **Pre-Local Predictor** (P_prelocal)
   - Predicts target representations from context
   - Cross-attention: target positions attend to context
   - Learns local temporal dependencies

4. **Post-Local Predictor** (P_postlocal) - Optional
   - Refines predictions using additional context
   - Helps with long-range dependencies

5. **Contextual Predictor** (P_contextual) - Optional
   - Global context aggregation
   - Useful for tasks requiring full-sequence understanding

### Masking Strategies

#### Spatial Masking (Channels)
- **Random channel masking**: Drop 30-70% of channels randomly
- **Block channel masking**: Drop contiguous electrode groups (e.g., all frontal)
- **Structured masking**: Mask by anatomical regions (frontal/parietal/occipital/temporal)

#### Temporal Masking
- **Random temporal blocks**: Mask 200-500ms segments
- **Periodic masking**: Mask every Nth block (simulates missing data)
- **Span masking**: Variable-length spans (50ms to 2s)

#### Spatiotemporal Masking (Combined)
- **3D masking**: Mask channel × time blocks simultaneously
- **Checkerboard**: Alternate spatial and temporal masks
- **Extreme masking**: 80-90% joint masking (forces strong representations)

### Auxiliary Demographic Tasks

**Motivation**: Organize latent space by demographic structure to improve transfer learning.

#### Auxiliary Heads (Trained Jointly)
1. **Age Regression** (continuous)
   - Head: MLP(latent) → age
   - Loss: MSE(pred_age, true_age)
   - Weight: 0.1 × main loss

2. **Sex Classification** (binary)
   - Head: MLP(latent) → [M, F]
   - Loss: CrossEntropy
   - Weight: 0.05 × main loss

3. **EHQ Total Regression** (continuous, Edinburgh Handedness)
   - Head: MLP(latent) → ehq_total
   - Loss: MSE
   - Weight: 0.05 × main loss

4. **P-Factor Regression** (continuous, externalizing psychopathology)
   - Head: MLP(latent) → p_factor
   - Loss: MSE (with mask for n/a values)
   - Weight: 0.15 × main loss (important for Challenge 2)

5. **Attention/Internalizing/Externalizing Factors** (continuous)
   - Heads: 3 × MLP(latent) → factor
   - Loss: MSE per factor
   - Weight: 0.05 × main loss each

**Total auxiliary weight**: ~0.45 × main SignalJEPA loss

**Benefits**:
- Latent space organized by demographics
- Better cross-subject generalization
- Relevant for Challenge 2 (p_factor prediction)
- Regularization effect

### Training Configuration

#### Data
- **TUH**: All available subjects, continuous EEG segments
- **HBN**: R1-R4, R6-R11, all tasks combined (movies, resting, cognitive)
- **Mixing strategy**:
  - 70% HBN, 30% TUH (weighted by sample count)
  - Alternate between datasets every batch

#### Window Parameters
- **Window length**: 4 seconds (400 time steps at 100 Hz)
- **Stride**: 2 seconds (50% overlap)
- **Context ratio**: 50-70% unmasked
- **Target ratio**: 20-40% to predict

#### Optimization
- **Optimizer**: AdamW(lr=1e-4, weight_decay=0.05)
- **Scheduler**: CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
- **Batch size**: 256 (128 per GPU on 2×A100)
- **Epochs**: 200
- **Warmup**: 5 epochs
- **Gradient clipping**: 1.0

#### Regularization
- **Dropout**: 0.1 in predictors
- **Layer dropout**: 0.05 (drop entire transformer layers randomly)
- **EMA momentum**: 0.996 for target encoder
- **Weight decay**: 0.05

### Evaluation Metrics (Phase 1)

**Reconstruction quality**:
- MSE between predicted and target representations
- Cosine similarity between predicted and target

**Auxiliary task performance**:
- Age MAE
- Sex accuracy
- P-factor correlation
- EHQ correlation

**Downstream probe** (linear evaluation):
- Freeze encoder, train linear classifier on Challenge 1 validation
- Track as "probing accuracy" to monitor representation quality

---

## Phase 2: Contrastive Learning

### Objective
Inject task-specific structure into representations learned in Phase 1. Use inter-subject synchrony and task structure to align embeddings.

### 2.1 Movie Contrastive Learning

#### Formulation: Inter-Subject Correlation (ISC)

**Positive pairs**:
- Different subjects, **same movie, same timestamp**
- Example: (subject_i[DespicableMe, t=10s], subject_j[DespicableMe, t=10s])
- Rationale: Stimulus-locked neural synchrony across subjects

**Negative pairs**:
- Different movies (any subjects/timestamps)
- Same subject, different movies
- Same movie, different timestamps (>5s apart)

#### Implementation

**Data preparation**:
```python
# For each movie
for movie in [DespicableMe, ThePresent, DiaryOfAWimpyKid, FunwithFractals]:
    # Extract 2s windows every 1s (sliding window)
    windows = extract_windows(movie_eeg, window=2s, stride=1s)

    # Align windows across subjects by timestamp
    for t in timestamps:
        anchors = [subject_i[movie, t] for i in subjects]
        positives = all_pairs(anchors)  # Same movie, same time, different subject
        negatives = other_movies_or_times  # Different context
```

**Loss**: InfoNCE with temperature τ=0.07
```
L_movie = -log( exp(sim(anchor, pos)/τ) /
                (exp(sim(anchor, pos)/τ) + Σ exp(sim(anchor, neg)/τ)) )
```

**Why this works**:
- Movies induce stimulus-locked responses across subjects
- ISC is a known phenomenon in neuroscience
- Forces model to extract stimulus-driven patterns vs. subject-specific noise

### 2.2 Resting State Contrastive Learning

#### Formulation: State-Based Contrastives

**Eyes Open Contrastives**:
- **Positive**: Different subjects, both eyes open
- **Negative**: Same subject eyes open vs. eyes closed
- **Negative**: Different subjects, one eyes open, one eyes closed
- **Rationale**: Eyes-open state has shared neural signatures across subjects

**Eyes Closed Contrastives**:
- **Positive**: Different subjects, both eyes closed
- **Negative**: Same subject across states
- **Negative**: Cross-subject cross-state
- **Rationale**: Alpha rhythm enhancement during eyes closed

#### Implementation
```python
eyes_open_windows = extract_state_windows(resting_state, event="instructed_toOpenEyes")
eyes_closed_windows = extract_state_windows(resting_state, event="instructed_toCloseEyes")

# Contrastive across subjects within state
L_open = InfoNCE(eyes_open_windows, positives=same_state_diff_subject)
L_closed = InfoNCE(eyes_closed_windows, positives=same_state_diff_subject)

# Contrastive across states (negatives)
L_state = InfoNCE(all_windows, negatives=same_subject_diff_state)
```

### 2.3 Task-Specific Contrastive Formulations

#### Contrast Change Detection (Challenge 1 Task)
**Positive pairs**:
- Same stimulus type (left_target or right_target), different subjects
- Same trial structure (target → response), different subjects

**Negative pairs**:
- Different stimulus types
- Correct vs. incorrect trials (smiley vs. sad feedback)

**Rationale**: Stimulus-locked visual ERP components (P300, N200) should align across subjects.

#### Sequence Learning Tasks
**Positive pairs**:
- Same dot sequence, different subjects
- Same learning block (e.g., block_3), different subjects

**Negative pairs**:
- Different sequences
- Early vs. late blocks (learning progression)

**Rationale**: Learning-related EEG changes (theta, alpha) synchronize across subjects.

#### Surround Suppression Task
**Positive pairs**:
- Same stimulus condition (parallel/orthogonal), different subjects
- Same contrast level, different subjects

**Negative pairs**:
- Different stimulus conditions
- Different contrast levels

**Rationale**: Visual cortex responses to surround suppression align across subjects.

#### Symbol Search Task
**Positive pairs**:
- Search events across subjects (attentional engagement)

**Negative pairs**:
- Pre-search vs. during-search
- Symbol search vs. passive viewing

### Contrastive Training Configuration

#### Data
- **HBN only**: R1-R4, R6-R11 (TUH doesn't have task labels)
- **Task mixing**:
  - 40% movie tasks
  - 20% resting state
  - 40% cognitive tasks (weighted by Challenge 1/2 relevance)

#### Architecture
- **Encoder**: Load from Phase 1 checkpoint
- **Projection head**: 2-layer MLP (latent_dim → 512 → 256)
- **Temperature**: τ=0.07 (learnable)

#### Optimization
- **Optimizer**: AdamW(lr=5e-5, weight_decay=0.01)
- **Scheduler**: CosineAnnealingLR(T_max=50)
- **Batch size**: 512 (256 anchors + 256 positives/negatives)
- **Epochs**: 50
- **Warmup**: 2 epochs

#### Loss Weighting
```python
L_total = 0.4 × L_movie +
          0.2 × L_resting +
          0.2 × L_challenge1 +
          0.1 × L_seqlearning +
          0.05 × L_surroundsupp +
          0.05 × L_symbolsearch
```

### Evaluation Metrics (Phase 2)

**Contrastive quality**:
- Positive pair similarity (higher is better)
- Negative pair similarity (lower is better)
- Alignment score: mean(pos_sim) - mean(neg_sim)

**Inter-subject correlation**:
- ISC score per movie (correlation of embeddings across subjects)

**Downstream probe**:
- Linear classifier on Challenge 1/2 validation
- Compare to Phase 1 baseline

---

## Phase 3: Supervised Fine-Tuning

### Objective
Specialize the pretrained encoder for Challenge 1 (response time) and Challenge 2 (p-factor) tasks.

### 3.1 Challenge 1: Response Time Prediction

#### Task
Predict reaction time from stimulus-locked EEG windows during contrastChangeDetection task.

#### Pipeline
1. **Load encoder**: Phase 2 checkpoint
2. **Preprocessing**:
   - Annotate trials with `rt_from_stimulus`
   - Create 2.5s windows [stim+0.5s, stim+2.5s]
   - Window size: (129, 250) → (128, 250) after dropping reference
3. **Architecture**:
   - Encoder (frozen for 5 epochs, then fine-tuned)
   - Regression head: MLP(latent → 256 → 128 → 1)
4. **Loss**: MSE(pred_rt, true_rt)

#### Training Config
- **Data**: R1-R4, R6-R11, split at subject level (80/10/10 train/val/test)
- **Optimizer**: AdamW(lr=1e-4, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau(patience=10)
- **Epochs**: 100
- **Early stopping**: Patience 15 on val_loss
- **Batch size**: 128

#### Evaluation
- **Metric**: NRMSE (Normalized Root Mean Squared Error)
- **Competition weight**: 30% of overall score

### 3.2 Challenge 2: P-Factor Prediction

#### Task
Predict externalizing psychopathology factor from multi-task EEG across all tasks.

#### Pipeline
1. **Load encoder**: Phase 2 checkpoint
2. **Preprocessing**:
   - Use **all tasks** (movies, resting, cognitive)
   - Fixed 4s windows, 2s stride
   - Random 2s crops via DatasetWrapper (augmentation)
   - Per-subject aggregation: mean pool embeddings across all windows
3. **Architecture**:
   - Encoder (frozen for 5 epochs, then fine-tuned)
   - Subject-level aggregation (mean or attention pooling)
   - Regression head: MLP(latent → 256 → 128 → 1)
4. **Loss**: MAE(pred_p_factor, true_p_factor) with n/a masking

#### Training Config
- **Data**: R1-R4, R6-R11, split at subject level (80/10/10)
- **Optimizer**: AdamW(lr=5e-5, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau(patience=15)
- **Epochs**: 150
- **Early stopping**: Patience 20 on val_loss
- **Batch size**: 64 (subject-level batches)

#### Evaluation
- **Metric**: NRMSE
- **Competition weight**: 70% of overall score

### 3.3 Multitask Fine-Tuning (Optional)

Train Challenge 1 and Challenge 2 jointly with shared encoder.

**Loss**:
```python
L_total = 0.3 × L_challenge1 + 0.7 × L_challenge2
```

**Architecture**:
- Shared encoder
- Dual heads: C1_head, C2_head
- Task-specific data loaders

---

## Phase 4: Architecture Exploration

### Objective
Once training framework is validated and baselines established, explore architectural improvements.

### When to Start Phase 4
**Only after** completing Phases 1-3 and having:
1. Reproducible training pipeline
2. Baseline performance on Challenge 1 & 2
3. Understanding of which phases contribute most to performance

### Exploration Directions

#### 4.1 Encoder Architectures
- **Mamba (State Space Models)**: Replace transformers with Mamba blocks
- **FNO (Fourier Neural Operators)**: Frequency-domain processing
- **Hybrid**: Temporal Mamba + Spatial Transformer
- **Hierarchical**: SlowFast-style dual-rate processing

#### 4.2 Channel Adaptation Methods
- Test alternatives from `docs/channel_adaptation.md`:
  - Graph Neural Networks (electrode graph)
  - Neural Implicit Fields (continuous spatial representation)
  - PointNet-style set networks

#### 4.3 Attention Mechanisms
- **Sparse attention**: Longformer, BigBird patterns
- **Linear attention**: Performer, FLASH attention
- **Cross-channel attention**: Separate temporal and spatial attention

#### 4.4 Training Strategies
- **Curriculum learning**: Easy → hard masking ratios
- **Multi-scale training**: Variable window lengths
- **Data augmentation**: Time warping, frequency shifting

---

## Evaluation Strategy

### Continuous Monitoring (All Phases)

#### Phase-Specific Metrics
| Phase | Primary Metric | Secondary Metrics |
|-------|---------------|-------------------|
| Phase 1 | Reconstruction MSE | Auxiliary task MAE, Linear probe |
| Phase 2 | ISC alignment | Contrastive accuracy, Linear probe |
| Phase 3 | Challenge NRMSE | Val loss, Convergence speed |

#### Checkpoint Strategy
- **Phase 1**: Save every 10 epochs + best validation
- **Phase 2**: Save every 5 epochs + best ISC
- **Phase 3**: Save best validation NRMSE

### Ablation Studies (After Full Pipeline)

1. **Pretraining ablations**:
   - SignalJEPA only (no contrastive)
   - Contrastive only (no SignalJEPA)
   - Supervised from scratch (no pretraining)

2. **Masking ablations**:
   - Spatial only
   - Temporal only
   - Spatiotemporal

3. **Auxiliary task ablations**:
   - No auxiliary tasks
   - Demographics only
   - P-factor only

4. **Contrastive formulation ablations**:
   - Movie only
   - Resting state only
   - Task-specific only

### Final Comparison

**Baselines**:
1. Supervised-only EEGNeX
2. SignalJEPA → Supervised (no contrastive)
3. Full pipeline (SignalJEPA → Contrastive → Supervised)

**Evaluation**:
- Challenge 1 NRMSE
- Challenge 2 NRMSE
- Overall competition score: 0.3 × C1 + 0.7 × C2
- Cross-subject generalization (train on R1-R3, test on R4+R6)
- Cross-task transfer (train on movies, test on cognitive tasks)

---

## Implementation Timeline

### Session 1-2: Foundation & Data Infrastructure
**Duration**: 2-3 days
- [ ] Unified data pipeline for TUH + HBN
- [ ] Channel adaptation (Perceiver-style cross-attention)
- [ ] Sampling rate harmonization (resample to 100 Hz)
- [ ] Demographic data loading and masking (for n/a values)
- [ ] Window extraction with configurable masking strategies

### Session 3-4: SignalJEPA Implementation (Phase 1)
**Duration**: 3-4 days
- [ ] Context encoder (transformer-based)
- [ ] Target encoder (EMA)
- [ ] Pre-local predictor
- [ ] Post-local predictor (optional, can defer)
- [ ] Contextual predictor (optional, can defer)
- [ ] Spatial/temporal/spatiotemporal masking
- [ ] Auxiliary demographic heads
- [ ] Joint loss with configurable weights
- [ ] Training loop with TUH + HBN mixing
- [ ] Evaluation: reconstruction quality, auxiliary task performance

### Session 5-6: Contrastive Learning Implementation (Phase 2)
**Duration**: 2-3 days
- [ ] Movie ISC contrastive formulation
- [ ] Resting state contrastive formulation
- [ ] Task-specific contrastive formulations
- [ ] Projection head architecture
- [ ] InfoNCE loss with temperature
- [ ] Data loader for contrastive pairs
- [ ] Training loop with multi-task mixing
- [ ] Evaluation: ISC, alignment scores

### Session 7: Supervised Fine-Tuning (Phase 3)
**Duration**: 1-2 days
- [ ] Challenge 1 fine-tuning pipeline
- [ ] Challenge 2 fine-tuning pipeline
- [ ] Subject-level aggregation for Challenge 2
- [ ] Training loops with early stopping
- [ ] Evaluation: NRMSE, competition score

### Session 8: Evaluation & Analysis
**Duration**: 1-2 days
- [ ] Full pipeline evaluation (Phase 1 → 2 → 3)
- [ ] Ablation studies
- [ ] Checkpoint comparisons
- [ ] Cross-subject/cross-task analysis
- [ ] Performance attribution (which phase helps most?)

### Session 9+: Architecture Exploration (Phase 4)
**Duration**: Ongoing
- [ ] Mamba encoder experiments
- [ ] FNO experiments
- [ ] Alternative channel adaptation
- [ ] Training strategy improvements

---

## Success Criteria

### Minimum Viable Pipeline (MVP)
- [ ] Phases 1-3 implemented and training successfully
- [ ] Reproducible training runs with consistent convergence
- [ ] Baselines established for each phase
- [ ] Evaluation metrics tracked at each phase

### Research Success
- [ ] Pretrained model outperforms supervised-only baseline
- [ ] Contrastive phase improves over SignalJEPA-only
- [ ] Clear attribution of performance gains to each phase
- [ ] Competitive performance on Challenge 1 & 2

### Stretch Goals
- [ ] State-of-the-art performance on HBN competition
- [ ] Published ablation study quantifying phase contributions
- [ ] Reusable foundation model for future EEG tasks
- [ ] Novel insights into channel adaptation or task transfer

---

## Notes & Open Questions

### Questions to Resolve During Implementation
1. **Sampling rate**: Resample TUH to 100 Hz, or upsample HBN to 250 Hz?
   - **Decision**: Resample TUH to 100 Hz (less data to process, HBN is target)

2. **Auxiliary task timing**: Joint with Phase 1, or separate phase?
   - **Decision**: Joint with Phase 1 (single training loop)

3. **Encoder freezing**: How many epochs frozen during fine-tuning?
   - **Decision**: 5 epochs frozen, then full fine-tuning

4. **Contrastive negatives**: Hard negatives vs. random sampling?
   - **Decision**: Start with random, add hard negative mining if needed

5. **Subject-level aggregation**: Mean pooling vs. attention pooling?
   - **Decision**: Start with mean, try attention if performance plateaus

### Risks & Mitigation
- **Risk**: Channel adaptation doesn't transfer well
  - **Mitigation**: Test on validation set early, try alternatives from `docs/channel_adaptation.md`

- **Risk**: Contrastive phase doesn't improve over SignalJEPA
  - **Mitigation**: Ablate formulations individually, tune temperature and loss weights

- **Risk**: Training unstable with multi-phase pipeline
  - **Mitigation**: Careful learning rate tuning, gradient clipping, checkpoint frequently

- **Risk**: Phase 1 takes too long (200 epochs × large dataset)
  - **Mitigation**: Start with fewer epochs (50-100), scale up if beneficial

---

## References

### Key Papers
1. **SignalJEPA**: [link to paper]
2. **Perceiver**: Jaegle et al., "Perceiver: General Perception with Iterative Attention"
3. **InfoNCE**: Oord et al., "Representation Learning with Contrastive Predictive Coding"
4. **ISC in Neuroscience**: Hasson et al., "Intersubject synchronization of cortical activity during natural vision"

### Code References
- `docs/channel_adaptation.md`: Detailed channel adaptation strategies
- `cerebro/data/hbn.py`: Current HBN data module
- `cerebro/data/tuh_edf.py`: TUH data loading
- Task annotations: `data/ds005505-bdf/task-*_events.json`
- Demographics: `data/ds005505-bdf/participants.tsv`

---

**Document Version**: 1.0
**Last Updated**: 2025-10-27
**Author**: Varun (with Claude assistance)
