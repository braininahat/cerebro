# Final Sprint: Mamba-based EEG Competition Approach

## Overview
Direct supervision using Mamba 2 SSM (State Space Model) for both challenges, avoiding attention mechanisms while capturing spatial-temporal dynamics through implicit channel ordering and local spatial processing.

## Timeline & Priorities

### Phase 1: Direct Supervision (Days 1-2)
**Goal**: Achieve sub-1.0 NRMSE on Challenge 1 validation

#### Challenge 1 Implementation
- **Model**: Mamba2 with spatial pre-processing
- **Architecture**:
  ```
  EEG Input (batch, 129 channels, 200 timesteps)
  → Channel Embedding (learnable channel positions)
  → Depthwise Conv1D (local spatial relationships)
  → Mamba2 Block (temporal dynamics)
  → Global pooling
  → MLP head → Response time
  ```
- **Windowing**: 2-second windows from contrast change detection
- **Training**: MSE loss, AdamW optimizer
- **Validation**: Subject-level splits (80/20)

#### Challenge 2 Implementation
- **Model**: Same architecture, different head
- **Windowing**: 2-second windows, task-agnostic
- **Aggregation**: Mean pooling across windows per subject
- **Target**: P_factor (externalizing score)

### Phase 2: Contrastive Pretraining (Days 3-4)
**Trigger**: Once Challenge 1 achieves < 0.95 NRMSE

#### Contrastive Learning Strategy
- **Positive pairs**:
  - Same subject, same task, nearby time windows
  - Leverage task event annotations for semantic similarity
- **Negative pairs**:
  - Different subjects
  - Different tasks (using task JSON descriptions)
- **Demographic conditioning**: Age, sex as auxiliary inputs
- **Loss**: InfoNCE with temperature scaling

#### Task Annotations Utilization
From HBN JSON files:
- `contrastChangeDetection`: Visual attention task
- `DespicableMe`, `DiaryOfAWimpyKid`, `ThePresent`: Movie watching (natural stimuli)
- `FunwithFractals`: Abstract visual processing
- `seqLearning6/8target`: Sequential learning
- `surroundSupp`: Visual suppression
- `symbolSearch`: Cognitive search task
- `RestingState`: Baseline activity

### Phase 3: Fine-tuning (Day 5)
- Freeze Mamba backbone
- Fine-tune task heads
- Multi-task learning if time permits

## Technical Specifications

### Mamba2 Integration
```python
# Key hyperparameters
mamba_config = {
    'd_model': 256,        # Hidden dimension
    'd_state': 16,         # SSM state dimension
    'n_layers': 4,         # Mamba blocks
    'expand': 2,           # Expansion factor
    'dt_rank': 'auto',     # Timestep rank
    'conv_size': 4,        # Conv kernel size
}
```

### Spatial Modeling Without Coordinates
1. **Implicit ordering**: Channels maintain consistent topographic order
2. **Local spatial conv**: 1D depthwise convolutions capture neighbor relationships
3. **Channel embeddings**: Learnable position encodings per channel
4. **Cross-channel mixing**: Through projection layers between Mamba blocks

### Data Pipeline
```python
# Reuse from cerebro:
- EEGChallengeDataset wrapper
- Subject-level splitting logic
- Caching mechanisms

# New implementations:
- Mamba-compatible batch collation
- Window-level augmentations
- Task-aware sampling for contrastive learning
```

## Success Metrics

### Minimum Viable Product (2 days)
- [ ] Challenge 1 NRMSE < 1.0 on validation
- [ ] Challenge 2 NRMSE < 1.2 on validation
- [ ] Both models train without OOM on 24GB GPU

### Target Performance (4 days)
- [ ] Challenge 1 NRMSE < 0.95
- [ ] Challenge 2 NRMSE < 1.0
- [ ] Contrastive pretraining improves both tasks

### Stretch Goals (if time)
- [ ] Multi-task learning
- [ ] Ensemble with existing models
- [ ] TUH dataset integration

## Risk Mitigation

1. **Mamba installation issues**: Pre-built wheels, fallback to S4
2. **Memory constraints**: Gradient checkpointing, smaller batch sizes
3. **Convergence problems**: Careful initialization, warmup scheduling
4. **Spatial modeling inadequate**: Add graph convolution layer if needed

## Implementation Notes

### Notebook Structure
Each notebook will be self-contained with:
1. Imports and setup
2. Data loading (direct from HBN)
3. Model definition
4. Training loop
5. Evaluation and visualization
6. Checkpoint saving

### Key Differences from Transformers
- **Linear complexity**: O(n) vs O(n²) for sequences
- **Recurrent structure**: Natural for streaming inference
- **No attention maps**: Less interpretable but more efficient
- **Continuous-time modeling**: Better for irregular sampling

## Validation Strategy

### Challenge 1 Checkpoints
- Log response time predictions vs ground truth
- Plot residuals by trial difficulty
- Analyze errors by subject demographics

### Challenge 2 Checkpoints
- Correlation with p_factor subscales
- Cross-task consistency checks
- Subject-level aggregation stability

## Contrastive Learning Details

### Event-based Augmentations
Using task JSON annotations:
- Stim onset/offset boundaries
- Response periods
- Inter-trial intervals
- Task-specific markers

### Demographic Incorporation
```python
# Auxiliary network
demo_features = [age, sex_encoded, handedness]
demo_embedding = MLP(demo_features)
# Concatenate with Mamba output before projection head
```

## Final Submission Checklist

- [ ] Remove all validation data leaks
- [ ] Test with startkit local_scoring.py
- [ ] Verify R5 is never touched during training
- [ ] Package model weights correctly
- [ ] Document preprocessing exactly
- [ ] Submission notebook runs end-to-end

## Fallback Plans

If Mamba doesn't converge:
1. **Plan B**: S4 model (simpler SSM)
2. **Plan C**: Temporal CNN + spatial attention-free transformer
3. **Plan D**: Ensemble existing EEGNeX models

## Notes on Avoiding Common Pitfalls

1. **No coordinate regression**: Channels as categorical entities
2. **No attention layers**: Pure SSM + convolutions
3. **No complex augmentations**: Focus on task structure
4. **No excessive hyperparameter tuning**: Fixed schedule, few key params
5. **No R5 contamination**: Strict validation protocol

---

**Timeline Summary**:
- Hour 0-12: Challenge 1 Mamba notebook
- Hour 12-24: Challenge 2 Mamba notebook
- Hour 24-36: Debug and optimize for < 1.0 NRMSE
- Hour 36-48: Contrastive pretraining
- Hour 48-60: Fine-tuning and final runs
- Hour 60-72: Submission preparation

**Critical Path**: Challenge 1 supervised → Validate < 1.0 → Contrastive design → Challenge 2 integration