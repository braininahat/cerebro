# CLAUDE.md - Cerebro Project

This file provides project-specific guidance for the Cerebro EEG Competition project.
See user-level CLAUDE.md for general development patterns and memory-first workflow.

## Project Overview

Competition submission for NeurIPS 2025 EEG Foundation Challenge with a 10-day development timeline. Two regression tasks: Challenge 1 (30% weight) predicts response time from Contrast Change Detection task EEG; Challenge 2 (70% weight) predicts p_factor (externalizing psychopathology) from multi-task EEG. Overall score: 0.3 × NRMSE_C1 + 0.7 × NRMSE_C2.

**Phased approach**: Days 1-4 supervised baselines (safety net) → Days 5-7 movie contrastive pretraining + multitask finetuning → Days 8-9 architecture iteration (only if contrastive beats baseline) → Day 10 submission.

## Project-Specific Memory Patterns

### Cerebro-Specific Entity Examples

```python
# Competition-specific queries
search_nodes("Challenge 1 windowing")  # EEG windowing strategies
search_nodes("movie_pretrain optimizer")  # Contrastive learning config
search_nodes("NRMSE calculation")  # Competition metric
search_nodes("R5 protection")  # Validation data safeguards

# Entity creation for EEG components
create_entities([{
  name: "Challenge1DataModule",
  entityType: "DataModule",
  observations: [
    "Two modes: dev (with local validation) and submission (no validation)",
    "Implementation: cerebro/data/challenge1.py",
    "R5 protection: ValueError if 'R5' in releases list",
    "Subject-level splits to prevent data leakage",
    "Windowing: 2.5s windows with 0.5s stimulus anchor"
  ]
}])

# Competition-specific relationships
create_relations([
  {from: "Challenge1DataModule", to: "cerebro", relationType: "part_of"},
  {from: "Challenge1DataModule", to: "EEGNeX", relationType: "uses"},
  {from: "movie_pretrain", to: "InfoNCE", relationType: "uses_loss"}
])
```

## Competition-Specific Architecture

### Supervised Training Configuration
```python
# Challenge 1: Response time prediction
supervised_c1:
  model: EEGNeX
  loss: MSE
  optimizer: AdamW(lr=0.001, weight_decay=0.00001)
  scheduler: CosineAnnealingLR
  epochs: 100
  early_stopping: patience=10 on val_loss

# Challenge 2: P_factor prediction  
supervised_c2:
  model: EEGNeX
  loss: MSE
  optimizer: AdamW(lr=0.0005, weight_decay=0.00001)
  scheduler: ReduceLROnPlateau
  epochs: 150
  early_stopping: patience=15 on val_loss
```

### Movie Contrastive Pretraining (Days 5-7)
```python
movie_pretrain:
  model: EEGNeX with projection head
  loss: InfoNCE(temperature=0.07)
  optimizer: AdamW(lr=0.0001)
  batch_size: 256
  epochs: 200
  data: All movie tasks (R1-R4, R6-R11)
  
multitask_finetune:
  pretrained: movie_pretrain checkpoint
  freeze_encoder_epochs: 5  # Freeze for 5 epochs, then fine-tune end-to-end
  task_weights: {challenge1: 0.3, challenge2: 0.7}
```

## Competition Data

### HBN Dataset Structure
- **Releases**: 
  - R1: ds005504-bdf through R11: ds005516-bdf
  - Note: ds005513 (R5 candidate) does not exist
- **Training data**: R1-R4, R6-R11 (10 releases)
- **Competition validation data**: R5 only (held out, provides leaderboard feedback)
- 129 channels (including reference Cz), 100 Hz sampling
- 9 excluded subjects (listed in configs/data/hbn.yaml under challenge1.excluded_subjects)

### Challenge-Specific Preprocessing

#### Challenge 1 (Response Time)
- **Windowing**: 2.5s windows with 0.5s pre-stimulus anchor
- **Events**: "Stim" markers for stimulus onset
- **Target**: Response time (log-transformed in some experiments)
- **Implementation**: `annotate_trials_with_target()` adds RT to epochs

#### Challenge 2 (P_factor)
- **Windowing**: Task-dependent (varies by cognitive task)
- **Aggregation**: Subject-level features across all tasks
- **Target**: P_factor from participants.tsv
- **Note**: Some subjects have `n/a` for factor scores

## Competition Data Strategy

### Prototyping Phase (Days 1-9)

**Data splits**:
- **Train**: Subset of {R1-R4, R6-R11} split at **subject level** (e.g., 80% of subjects)
- **Val**: Held-out subjects from {R1-R4, R6-R11} (e.g., 20% of subjects)
- **Test**: R5 (treated as external test, checked sparingly)

**Why subject-level splits?**
- Same subject may have multiple recordings across different tasks
- Window-level or recording-level splits risk data leakage
- Model could memorize subject-specific EEG patterns rather than task patterns

### Final Runs Phase (Day 10)

**Data splits**:
- **Train**: ALL subjects from {R1-R4, R6-R11} (no validation split)
- **Val/Test**: R5 (final submission to competition)

**Critical**: R5 is a competition validation set that provides leaderboard feedback, NOT a traditional test set. Never train on R5 under any circumstances.

### DataModule Modes

#### Dev Mode (`mode="dev"`)
**Purpose**: Development and hyperparameter tuning with local validation

**Configs**:
- `configs/challenge1_base.yaml` - Standard dev mode with local test
- `configs/challenge1_mini.yaml` - Fast prototyping with R1 mini dataset
- `configs/challenge1_r5test.yaml` - Dev mode with R5 test evaluation

**Usage**:
```bash
# Dev mode with local test split (fastest iteration)
uv run cerebro fit --config configs/challenge1_base.yaml

# Dev mode with R5 test (validate against competition distribution)
uv run cerebro fit --config configs/challenge1_r5test.yaml
```

#### Submission Mode (`mode="submission"`)
**Purpose**: Final submission with maximum training data

**Usage**:
```bash
# Day 10: Final submission training
uv run cerebro fit --config configs/challenge1_submission.yaml
```

### R5 Protection

**Multiple layers of protection prevent R5 contamination**:
1. **ValueError guard**: `Challenge1DataModule.__init__()` raises error if "R5" in releases list
2. **Mode validation**: Submission mode requires `test_on_r5=True` (raises error otherwise)
3. **Separate caching**: R5 cached in `cache/challenge1/r5/` to avoid mixing with training data
4. **Validation notebook**: `notebooks/08_validate_data_quality.py` checks for train/R5 subject overlap

## Priority Guidelines (10-Day Timeline)

**Must have**:
- Working supervised baselines (C1 + C2)
- Local scoring integration
- Submission packaging

**Should have**:
- Movie contrastive pretraining
- Multitask fine-tuning
- Wandb tracking

**Nice to have** (only if time permits):
- SignalJEPA baseline
- Spatial/temporal hierarchies
- Hyperparameter sweeps
- MOABB data integration

## Key Design Decisions

1. **Installable package**: Editable install (`uv pip install -e .`) for clean imports and IDE support
2. **CLI entry point**: `uv run cerebro fit` command via package entry point (backward compatible with direct python)
3. **Config-driven experiments**: Change hyperparameters via Lightning CLI, not code
4. **Aggressive caching**: EEGChallengeDataset caches downloads, preprocessed data cached via pickle
5. **Baselines first**: Supervised training is safety net before trying contrastive learning
6. **Local scoring critical**: Test with `uv run python startkit/local_scoring.py` before submitting to competition
