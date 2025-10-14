# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Competition submission for NeurIPS 2025 EEG Foundation Challenge with a 10-day development timeline. Two regression tasks: Challenge 1 (30% weight) predicts response time from Contrast Change Detection task EEG; Challenge 2 (70% weight) predicts p_factor (externalizing psychopathology) from multi-task EEG. Overall score: 0.3 × NRMSE_C1 + 0.7 × NRMSE_C2.

**Phased approach**: Days 1-4 supervised baselines (safety net) → Days 5-7 movie contrastive pretraining + multitask finetuning → Days 8-9 architecture iteration (only if contrastive beats baseline) → Day 10 submission.

## Memory-First Workflow

**CRITICAL**: Always consult the memory graph before investigating anything. Memory persists across sessions and accumulates project knowledge over time.

### Workflow

```
┌─────────────────────┐
│  Question/Task      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Search Memory      │  ← mcp__memory__search_nodes
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
  Found?      Not Found?
     │           │
     ▼           ▼
 Use Info    Investigate
     │        (Read/Grep/
     │         Notebooks)
     │           │
     └─────┬─────┘
           │
           ▼
┌─────────────────────┐
│  Capture Findings   │  ← create_entities, add_observations, create_relations
└─────────────────────┘
```

### When to Consult Memory

**Always start by searching memory for**:
- "How does [component] work?"
- "Where is [function/class] defined?"
- "What are the differences between Challenge 1 and Challenge 2?"
- "What optimizer/loss/scheduler does [experiment] use?"
- "Why was [decision] made?"
- "What are the known issues with [component]?"

**Examples**:
```python
# Question: "How does Challenge 1 windowing work?"
search_nodes(query="Challenge 1 windowing pipeline stimulus")

# Question: "What optimizer does movie_pretrain use?"
search_nodes(query="movie_pretrain optimizer AdamW")

# Question: "Where is NRMSE calculated?"
search_nodes(query="NRMSE metric calculation local_scoring")

# Question: "What are the movie tasks?"
search_nodes(query="movie tasks DespicableMe")
```

### Investigation Flow

1. **Search memory first**: Use `search_nodes` with relevant keywords
2. **If found**: Use the information, cross-reference with code if needed
3. **If not found**: Investigate using Read/Grep/Glob/notebooks
4. **Capture findings**: ALWAYS update memory with what you learned

### What to Capture

**Always capture**:
- **Bug root causes + fixes**: "Bug in DatasetWrapper when recordings <4s: filters out after windowing instead of before"
- **Design decisions**: "Use AdamW over Adamax for supervised training because of better weight decay handling"
- **Implementation details**: "Challenge 1 uses stimulus-locked windows via create_windows_from_events with offset=0.5s"
- **Configuration meanings**: "freeze_encoder_epochs=5 means encoder weights frozen for first 5 epochs, then fine-tuned end-to-end"
- **Gotchas**: "DatasetWrapper random crop needs seed parameter for reproducibility across workers"
- **Relationships**: "movie_pretrain uses InfoNCE loss" / "MultitaskModel requires pretrained_checkpoint parameter"
- **Timeline insights**: "Days 5-7 conditional on movie contrastive beating baseline"

**Don't capture**:
- Vague descriptions: "model uses some optimizer" (too vague)
- Obvious facts: "Python uses indentation" (not project-specific)
- Temporary notes: "TODO: check this later" (not persistent knowledge)

### Entity Types

Use these `entityType` values for consistency:

- **project** - Overall codebase (cerebro)
- **competition** - External challenges (NeurIPS 2025 EEG Foundation Challenge)
- **task** - Challenge tasks (Challenge 1, Challenge 2)
- **model** - Neural architectures (EEGNeX, SignalJEPA, MultitaskModel)
- **architecture** - Design patterns (SlowFast, Mamba)
- **function** - Specific functions (annotate_trials_with_target, create_windows_from_events)
- **class** - Python classes (DatasetWrapper, Submission)
- **experiment** - Configured experiments (baseline_eegnex_c1, movie_pretrain, multitask_finetune)
- **training_strategy** - Training approaches (Movie Contrastive Pretraining, supervised learning)
- **dataset** - Data sources (HBN-EEG Dataset)
- **eeg_task** - Recording paradigms (contrastChangeDetection, Movie Tasks)
- **api** - External interfaces (EEGChallengeDataset)
- **library** - External dependencies (braindecode, eegdash, PyTorch, MNE-Python)
- **tool** - Development tools (uv, wandb, Hydra)
- **loss_function** - Loss functions (InfoNCE, MSE, MAE)
- **metric** - Evaluation metrics (NRMSE)
- **optimizer** - Optimizers (AdamW, Adamax)
- **scheduler** - LR schedulers (CosineAnnealingLR, CosineAnnealingWarmRestarts)
- **code_module** - Source directories (src/data, src/models, src/training, src/evaluation, notebooks, configs)
- **configuration** - Config files (configs/*)
- **reference_code** - Starter code (startkit)
- **script** - Executable scripts (local_scoring, train.py)
- **directory** - File system directories (weights directory)
- **timeline_phase** - Development phases (Days 1-4, Days 5-7, Days 8-9, Day 10)
- **person** - Project contributors (Varun)
- **framework** - Conceptual frameworks (SimCLR, JEPA)
- **infrastructure** - System components (Hydra Configuration System)

### Relation Types

Use these `relationType` values for clarity:

- **uses** / **uses_tool** / **uses_model** / **uses_optimizer** / **uses_scheduler** / **uses_loss** / **uses_dataset**
- **implements** / **implemented_in**
- **depends_on** / **required_for**
- **part_of** / **part_of_phase** / **phase_of**
- **from_library** / **provides**
- **targets** / **solves** / **evaluates**
- **loads** / **loaded_by**
- **adapts_from** / **replicates**
- **prototypes_for**
- **configures** / **manages**
- **submits_to** / **provided_by**
- **developed_by**
- **alternative_to**
- **pretrains**
- **follows**
- **inspired** / **based_on**
- **should_be_adapted_to**
- **wraps_output_of**
- **loss_function_for**
- **calculates**
- **framework_for**
- **baseline_for**
- **explored_in_phase**

### Memory Operations (MCP Tools)

```python
# Search for existing knowledge
mcp__memory__search_nodes(query="relevant keywords")
mcp__memory__open_nodes(names=["entity_name"])

# Create new knowledge
mcp__memory__create_entities(entities=[{
    "name": "Entity Name",
    "entityType": "appropriate_type",
    "observations": [
        "Specific observation 1 with details",
        "Specific observation 2 with measurements/parameters",
        "Gotcha: Known issue and workaround"
    ]
}])

# Add to existing knowledge
mcp__memory__add_observations(observations=[{
    "entityName": "Existing Entity",
    "contents": [
        "New insight discovered during debugging",
        "Configuration change: what + why"
    ]
}])

# Connect knowledge
mcp__memory__create_relations(relations=[{
    "from": "Source Entity",
    "to": "Target Entity",
    "relationType": "appropriate_relation"
}])
```

### Mandatory Capture Scenarios

**ALWAYS update memory after**:

1. **Resolving bugs**: Capture root cause, symptoms, and fix as observations
2. **Making design decisions**: Capture what was chosen, why, and alternatives considered
3. **Discovering gotchas**: Capture the issue and workaround
4. **Implementing features**: Capture relationships between new and existing components
5. **Changing configs**: Capture what changed, why, and expected impact
6. **Learning from startkit**: Capture pipeline differences, parameter meanings, function purposes
7. **Debugging failed experiments**: Capture what failed, why, and resolution

### Why This Matters (10-Day Timeline)

- **No repeated work**: Past investigations immediately accessible
- **Faster debugging**: Known issues and fixes in memory
- **Design consistency**: Previous decisions inform current choices
- **Session continuity**: Pick up exactly where you left off
- **Living documentation**: Memory graph becomes paper/future work foundation

**Memory is not optional—it's a productivity multiplier for time-constrained work.**

## Documentation Lookup with Context7

When memory doesn't have the answer and you need library documentation, use **Context7 MCP tools** before diving into source code.

### When to Use Context7

**Use after checking memory for**:
- API documentation (function signatures, parameters, return types)
- Usage examples and best practices
- Configuration options for models/optimizers/schedulers
- Understanding library-specific concepts

**Priority**: Memory (fastest) → Context7 Docs (authoritative) → Source Code (detailed but slow)

### Libraries in This Project

**Core dependencies with Context7 support**:
- **braindecode** - EEG models (EEGNeX, SignalJEPA), preprocessing utilities
- **PyTorch** - torch.optim (AdamW, Adamax), torch.nn (losses, layers), DataLoader
- **MNE-Python** - Raw data handling, BIDS format, preprocessing pipelines

### Two-Step Workflow

```python
# Step 1: Resolve library name to Context7 ID
mcp__context7__resolve_library_id(libraryName="package_name")
# Returns: {library_id: "/org/project", ...}

# Step 2: Get focused documentation
mcp__context7__get_library_docs(
    context7CompatibleLibraryID="/org/project",
    topic="specific topic or function",  # Narrows results
    tokens=3000  # Amount of context (default: 5000)
)
```

### Examples

**Example 1: Understanding EEGNeX parameters**
```python
# Question: "What parameters does EEGNeX.forward() expect?"
mcp__context7__resolve_library_id(libraryName="braindecode")
# → {library_id: "/braindecode/braindecode"}

mcp__context7__get_library_docs(
    context7CompatibleLibraryID="/braindecode/braindecode",
    topic="EEGNeX model initialization parameters",
    tokens=3000
)

# After reading docs, capture in memory:
mcp__memory__add_observations(observations=[{
    "entityName": "EEGNeX",
    "contents": [
        "Input shape: (batch_size, n_chans, n_times) - e.g. (128, 129, 200)",
        "n_chans parameter: number of EEG channels (129 for HBN)",
        "n_times parameter: window length in samples (200 = 2s at 100Hz)",
        "sfreq parameter: sampling frequency for temporal convolutions (100 Hz)",
        "n_outputs parameter: 1 for regression (RT or p_factor prediction)",
        "Returns: (batch_size, n_outputs) tensor"
    ]
}])
```

**Example 2: Configuring AdamW optimizer**
```python
# Question: "How does weight_decay work in AdamW vs Adam?"
mcp__context7__resolve_library_id(libraryName="torch")

mcp__context7__get_library_docs(
    context7CompatibleLibraryID="/pytorch/pytorch",
    topic="AdamW optimizer weight decay decoupled",
    tokens=2000
)

# Capture insight:
mcp__memory__add_observations(observations=[{
    "entityName": "AdamW",
    "contents": [
        "Decoupled weight decay: applied directly to weights, not gradients (unlike Adam)",
        "Better for fine-tuning: weight_decay=0.01 typical for transfer learning",
        "Supervised training uses weight_decay=0.00001, contrastive uses 0.0001"
    ]
}])
```

**Example 3: Loading BIDS data with MNE**
```python
# Question: "How to read BIDS EEG data with MNE?"
mcp__context7__resolve_library_id(libraryName="mne")

mcp__context7__get_library_docs(
    context7CompatibleLibraryID="/mne-tools/mne-python",
    topic="read_raw_bids BIDS dataset",
    tokens=2500
)
```

### Parameters

- **topic** (optional): Focus documentation on specific area
  - More specific = more relevant results
  - Examples: "EEGNeX forward pass", "AdamW weight decay", "BIDS read_raw"

- **tokens** (optional, default: 5000): Amount of documentation to retrieve
  - More tokens = more context but slower
  - Typical range: 2000-5000 for focused queries

### After Getting Documentation

**ALWAYS capture relevant findings in memory**:

```python
# Pattern: Create entities for new concepts
mcp__memory__create_entities(entities=[{
    "name": "discovered_concept",
    "entityType": "appropriate_type",
    "observations": ["Key insight from docs"]
}])

# Pattern: Add to existing entities
mcp__memory__add_observations(observations=[{
    "entityName": "existing_entity",
    "contents": [
        "Parameter meaning from docs",
        "Gotcha discovered: specific edge case behavior"
    ]
}])
```

### Tips

- **Start specific**: Use focused topics instead of broad queries
- **Verify with code**: Cross-reference documentation with actual usage in codebase
- **Adjust tokens**: Start with 3000, increase if incomplete, decrease if too verbose
- **Don't repeat**: Once captured in memory, never look up again
- **Prefer Context7 over web search**: More structured, contains code examples

### Integration with Memory Workflow

```
Question
    ↓
Search Memory (mcp__memory__search_nodes)
    ↓ not found?
Context7 Docs (mcp__context7__*)
    ↓ still unclear?
Read Source Code (Read/Grep/Glob)
    ↓ always
Capture in Memory (mcp__memory__add_observations)
```

**The goal**: Build memory graph so comprehensive that Context7 lookups become rare.

## Running Code

Cerebro is an installable package. **One-time setup** (after cloning):

```bash
cd cerebro
uv pip install -e .  # Install in editable mode
```

### CLI Commands (Lightning CLI)

```bash
# Training with Lightning CLI
uv run cerebro fit --config configs/challenge1_base.yaml
uv run cerebro fit --config configs/challenge1_mini.yaml

# With LR finder
uv run cerebro fit --config configs/challenge1_base.yaml --run_lr_finder true

# Override config values
uv run cerebro fit --config configs/challenge1_base.yaml \
    --model.init_args.lr 0.0001 \
    --data.init_args.batch_size 256

# Backward compatible (direct python)
uv run python cerebro/cli/train.py fit --config configs/challenge1_base.yaml

# Other Lightning CLI subcommands
uv run cerebro validate --config configs/challenge1_base.yaml
uv run cerebro test --config configs/challenge1_base.yaml --ckpt_path outputs/.../best.ckpt
```

### Scripts (future work)

```bash
# Local evaluation (critical for iteration)
uv run python scripts/evaluate.py checkpoint=outputs/.../best.pt
uv run python scripts/evaluate.py checkpoint=outputs/.../best.pt --fast-dev-run

# Package submission
uv run python scripts/package_submission.py checkpoint=outputs/.../best.pt output=submission.zip

# Test submission locally before uploading
uv run python startkit/local_scoring.py --submission-zip submission.zip --data-dir data/full --output-dir outputs/test --fast-dev-run
```

## Critical Data Pipeline Differences

### Challenge 1 (Response Time Prediction)
**Input**: Contrast Change Detection task only
**Windowing**: Stimulus-locked windows [stim + 0.5s, stim + 2.5s] → (129, 200)
**Label**: `rt_from_stimulus` from `annotate_trials_with_target` (per-trial)
**Loss**: MSE
**Key functions**: `annotate_trials_with_target`, `add_aux_anchors`, `create_windows_from_events`, `add_extras_columns`

### Challenge 2 (P-Factor Prediction)
**Input**: Any task (CCD, RestingState, movies, surroundSupp, etc.)
**Windowing**: Fixed 4s windows, 2s stride → random 2s crops via `DatasetWrapper` → (129, 200)
**Label**: `externalizing` field from participants.tsv (per-subject, constant across windows)
**Loss**: MAE/L1
**Key difference**: DatasetWrapper provides data augmentation through random cropping

### Movie Contrastive Pretraining
**Input**: 4 movie tasks (DespicableMe, ThePresent, DiaryOfAWimpyKid, FunwithFractals)
- **Note**: All 4 movies available in ALL releases including R5 (verified empirically)
**Windowing**: 2s sliding windows (configurable stride: 1s for overlap or 2s for non-overlapping)
**Positive pairs**: (subject_i[movie_A, t], subject_j[movie_A, t]) - exact same movie and timestamp, different subjects
**Negative pairs**: Different movies (any subjects/timestamps)
**Loss**: InfoNCE with temperature scaling
**Alignment strategy**: Perfect temporal alignment (no time binning) to preserve inter-subject correlation (ISC) precision

## Architecture Patterns

### Config Composition (Hydra)
Configs are composed from base modules:
- `configs/config.yaml`: Global settings (seed, device, paths, wandb)
- `configs/data/hbn.yaml`: Dataset parameters, windowing specs, excluded subjects
- `configs/model/*.yaml`: Model architectures (eegnex, jepa, contrastive)
- `configs/training/*.yaml`: Training loops (supervised, contrastive, multitask)
- `configs/experiment/*.yaml`: Complete experiments composing above configs

Experiment configs use `defaults` to inherit and `@package _global_` to override at root level.

### Module Organization
**cerebro/data/**: Dataset classes replicating startkit pipelines exactly
- `challenge1.py`: Implements annotate→filter→window→label pipeline (Lightning DataModule)
- `challenge2.py`: Implements filter→window→wrap pipeline with DatasetWrapper
- `movies.py`: Implements positive/negative pair creation for contrastive learning

**cerebro/models/**: Model wrappers around braindecode implementations
- `challenge1.py`: Challenge1Module (Lightning Module with EEGNeX)
- `encoders.py`: Wraps braindecode.models (EEGNeX, SignalJEPA)
- `projector.py`: MLP projection head for contrastive learning
- `heads.py`: Regression heads for C1/C2
- `multitask.py`: Shared encoder + dual heads architecture

**cerebro/cli/**: Command-line interface
- `train.py`: CerebroCLI (extends Lightning CLI with logging, LR finder, batch size finder)

**cerebro/utils/**: Utilities
- `logging.py`: Rich logging setup (console + file)
- `tuning.py`: LR finder and batch size scaler wrappers

**cerebro/training/**: Training loops with different objectives (future work)
- `supervised.py`: Single-task MSE/MAE training
- `contrastive.py`: InfoNCE loss with pair sampling
- `multitask.py`: Joint C1+C2 optimization with frozen encoder phase

**cerebro/evaluation/**: Local scoring before submission (future work)
- `local_scoring.py`: Adapted from startkit, runs full evaluation pipeline
- `submission_wrapper.py`: Converts checkpoint to Submission class format expected by competition

## Startkit Integration

The `startkit/` directory contains reference implementations that MUST be replicated exactly:
- `challenge_1.py`: Shows exact preprocessing for Challenge 1 (lines 144-193: annotate→windows→labels)
- `challenge_2.py`: Shows exact preprocessing for Challenge 2 (lines 231-259: filter→windows→wrap)
- `local_scoring.py`: Defines evaluation protocol (lines 76-114: NRMSE calculation, 107-115: overall score)
- `submission.py`: Shows required Submission class interface with `get_model_challenge_1()` and `get_model_challenge_2()`

**Critical**: Submission must be single-level zip (NO folder) containing submission.py + weights files.

## Data Specifications

**HBN-EEG Dataset**:
- **11 releases total**: ds005505-bdf (R1) through ds005516-bdf (R11)
  - R1: ds005505-bdf
  - R2: ds005506-bdf
  - R3: ds005507-bdf
  - R4: ds005508-bdf
  - **R5: ds005509-bdf (COMPETITION VALIDATION SET - NEVER TRAIN ON THIS)**
  - R6: ds005510-bdf
  - R7: ds005511-bdf
  - R8: ds005512-bdf
  - R9: ds005514-bdf (note: ds005513 does not exist)
  - R10: ds005515-bdf
  - R11: ds005516-bdf
- **Training data**: R1-R4, R6-R11 (10 releases)
- **Competition validation data**: R5 only (held out, provides leaderboard feedback)
- 129 channels (including reference Cz), 100 Hz sampling
- 9 excluded subjects (listed in configs/data/hbn.yaml under challenge1.excluded_subjects)

**Download**: Use `EEGChallengeDataset` from eegdash library (handles caching automatically)

**participants.tsv**: Contains p_factor, age, sex, task availability flags. Note: some subjects have `n/a` for all factor scores.

**Important**: The startkit uses `mini=True` with `release="R5"` for quick demos/tutorials only. For actual training, use `mini=False` (or omit mini parameter) with `release` in ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"].

## Competition Data Strategy

### Prototyping Phase (Days 1-9)

**Goal**: Architecture selection, hyperparameter tuning, feature engineering

**Data splits**:
- **Train**: Subset of {R1-R4, R6-R11} split at **subject level** (e.g., 80% of subjects)
- **Val**: Held-out subjects from {R1-R4, R6-R11} (e.g., 20% of subjects)
- **Test**: R5 (treated as external test, checked sparingly)

**Workflow**:
1. Split training releases at subject level to prevent data leakage
2. Use local validation for rapid iteration and hyperparameter tuning
3. Minimize R5 checks to avoid overfitting to leaderboard signal
4. Treat R5 as external "test" for architecture/hyperparameter selection

**Why subject-level splits?**
- Same subject may have multiple recordings across different tasks
- Window-level or recording-level splits risk data leakage
- Model could memorize subject-specific EEG patterns rather than task patterns

### Final Runs Phase (Day 10)

**Goal**: Maximum performance for competition leaderboard

**Data splits**:
- **Train**: ALL subjects from {R1-R4, R6-R11} (no validation split)
- **Val/Test**: R5 (final submission to competition)

**Workflow**:
1. Architecture and hyperparameters are locked (no more tuning)
2. Train on 100% of available training data to maximize model capacity
3. Submit final predictions on R5 for leaderboard evaluation
4. No local validation (all data used for training)

**Why this approach?**
- Standard competition pattern (Kaggle, NeurIPS, etc.)
- Maximizes training data when decisions are finalized
- Prevents leaderboard overfitting during development
- Local validation eliminates need for frequent R5 checks

**Critical**: R5 is a competition validation set that provides leaderboard feedback, NOT a traditional test set. Never train on R5 under any circumstances.

### Challenge1DataModule Modes

The `Challenge1DataModule` supports two modes via the `mode` parameter:

#### Dev Mode (`mode="dev"`)
**Purpose**: Development and hyperparameter tuning with local validation

**Data splits**:
- Splits training releases (R1-R4, R6-R11) into train/val/test at subject level
- `test_on_r5=false` (default): Use local test split from training releases
- `test_on_r5=true`: Use R5 as test set (matches competition evaluation)

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

# Mini mode for prototyping
uv run cerebro fit --config configs/challenge1_mini.yaml
```

**When to use**:
- Days 1-9: Architecture selection, hyperparameter tuning
- Use `test_on_r5=false` for fast iteration
- Use `test_on_r5=true` to validate major decisions against R5 distribution

#### Submission Mode (`mode="submission"`)
**Purpose**: Final submission with maximum training data

**Data splits**:
- Train on ALL subjects from training releases (R1-R4, R6-R11)
- No val split (uses all data for training)
- `test_on_r5=true` (required): Test on R5 for competition submission

**Config**:
- `configs/challenge1_submission.yaml` - Final submission mode

**Usage**:
```bash
# Day 10: Final submission training
uv run cerebro fit --config configs/challenge1_submission.yaml
```

**When to use**:
- Day 10 only, after architecture and hyperparameters are locked
- Maximizes training data for final submission
- No validation monitoring (save checkpoints periodically)

#### Mode Comparison Table

| Feature | Dev Mode (local test) | Dev Mode (R5 test) | Submission Mode |
|---------|----------------------|-------------------|-----------------|
| Train data | 80% of training releases | 80% of training releases | 100% of training releases |
| Val data | 10% of training releases | 10% of training releases | None |
| Test data | 10% of training releases | R5 | R5 |
| Validation monitoring | ✓ | ✓ | ✗ |
| Early stopping | ✓ | ✓ | ✗ |
| When to use | Fast iteration | Validate decisions | Final submission |
| Config | `challenge1_base.yaml` | `challenge1_r5test.yaml` | `challenge1_submission.yaml` |

#### R5 Protection

**Multiple layers of protection prevent R5 contamination**:
1. **ValueError guard**: `Challenge1DataModule.__init__()` raises error if "R5" in releases list
2. **Mode validation**: Submission mode requires `test_on_r5=True` (raises error otherwise)
3. **Separate caching**: R5 cached in `cache/challenge1/r5/` to avoid mixing with training data
4. **Validation notebook**: `notebooks/08_validate_data_quality.py` checks for train/R5 subject overlap

#### Verification

**Before training**:
```bash
# Validate data quality and R5 separation
uv run python notebooks/08_validate_data_quality.py

# Test R5 evaluation matches local_scoring.py
uv run python test_r5_evaluation.py --use-mini
```

**Expected outputs**:
- No subject overlap between training releases and R5
- R5 distribution may differ from training (expected - different subject pool)
- R5 preprocessing matches `startkit/local_scoring.py` exactly

## Development Workflow

1. **Start with mini=True**: Fast prototyping on small dataset
2. **Test local scoring early**: `uv run python scripts/evaluate.py --fast-dev-run` after every change
3. **Use notebooks for exploration**: Jupytext .py format (# %%) in notebooks/
4. **Hydra outputs**: Auto-generates timestamped directories in outputs/
5. **Wandb integration**: All experiments logged (set WANDB_API_KEY in .env)
6. **Git commits**: After each working milestone

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

## User Preferences (from ~/.claude/CLAUDE.md)

- Prefer editing existing files over creating new ones
- Use `uv run python ...` instead of `python ...`
- Prefer github MCP to git commands (but avoid in this workflow due to speed)
- Never add or remove packages
- Always use `uv run ...`
- Functional style over classes
- Implement prototypes as notebooks (.py with #%% for VSCode)
- Ask when faced with design choices (user has PhD)
- Verify assumptions before internalizing
- Don't pollute repo with throwaway code
- Questions are just questions - don't infer intent to act
