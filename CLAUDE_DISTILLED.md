# CLAUDE_DISTILLED.md

Optimized guidance for Claude Code. Original: 1150 lines → Distilled: ~300 lines.

## Core Workflow

```
Question → search_nodes(1-3 keywords) → Found? Use it : Investigate → capture findings
```

**Memory first, always.** Project filter: `relation.to == "cerebro"` (excludes UltraSpeech-Qt).

## Competition Overview

NeurIPS 2025 EEG Challenge. Score: 0.3×NRMSE_C1 + 0.7×NRMSE_C2
- **C1 (30%)**: RT prediction from CCD task
- **C2 (70%)**: p_factor from multi-task EEG
- **Timeline**: Days 1-4 baselines → 5-7 contrastive → 8-9 iterate → 10 submit
- **Critical**: NEVER train on R5 (competition validation)

## Memory Operations

### Search
```python
search_nodes("1-3 keywords")  # Broad→specific iteration
# Examples: "Challenge 1", "movie contrastive", "NRMSE calculation"
```

### Create/Update
```python
create_entities([{
    "name": "2-4 word phrase",  # Searchable, consistent
    "entityType": "type",        # See types below
    "observations": ["One fact per observation with details"]
}])

add_observations([{
    "entityName": "Existing Entity",
    "contents": ["New findings", "Config: what+why", "Bug: cause+fix"]
}])

create_relations([{
    "from": "Source", "to": "Target",
    "relationType": "uses"  # Prefer generic "uses" over specific
}])
```

### Entity Types (use these 10, others rare)
`project, task, model, function, class, experiment, dataset, optimizer, loss_function, metric`

### Relation Types (use these 8, others rare)
`uses, implements, part_of, solves, requires, provides, configures, from_library`

### Capture These
- Bug root causes + fixes
- Design decisions + alternatives
- Implementation details + parameters
- Config changes + rationale
- Gotchas + workarounds

## Data Specifications

### Releases
- **Train**: R1-R4, R6-R11 (10 releases, ~1000 subjects)
- **Val/Test**: R5 only (competition leaderboard, held out)
- **Excluded**: 9 subjects in configs/data/hbn.yaml

### Challenge Pipelines

| Challenge | Input | Window | Label | Loss |
|-----------|-------|--------|-------|------|
| C1 | CCD only | Stimulus+[0.5s,2.5s]→200 samples | rt_from_stimulus | MSE |
| C2 | All tasks | 4s/2s stride→random 2s crop | externalizing | MAE |
| Movies | 4 movies | 2s sliding | Same movie+time=positive | InfoNCE |

### DataModule Modes

| Mode | Train | Val | Test | Use Case |
|------|-------|-----|------|----------|
| dev | 80% | 10% | 10% or R5 | Prototyping |
| submission | 100% | - | R5 | Final run |

## Commands

### Training
```bash
# Development
uv run cerebro fit --config configs/challenge1_base.yaml
uv run cerebro fit --config configs/challenge1_mini.yaml  # Fast prototyping

# With tuning
uv run cerebro fit --config configs/challenge1_base.yaml --run_lr_finder true

# Submission mode (Day 10)
uv run cerebro fit --config configs/challenge1_submission.yaml
```

### Submission Prep
```bash
# 1. Train → 2. Convert to TorchScript → 3. Package
uv run python -m cerebro.utils.checkpoint_to_torchscript \
    --ckpt outputs/*/best.ckpt --output model_challenge_1.pt \
    --input-shape 1 129 200

zip -j submission.zip submission.py model_*.pt  # NO folders

# Test locally
uv run python startkit/local_scoring.py \
    --submission-zip submission.zip --data-dir data
```

## Architecture

### Modules
- `cerebro/data/`: challenge1.py, challenge2.py, movies.py (DataModules)
- `cerebro/models/`: challenge1.py (Lightning), encoders.py (EEGNeX/JEPA)
- `cerebro/cli/train.py`: Lightning CLI with LR finder
- `startkit/`: Reference implementation (replicate exactly)

### Config Structure (Hydra)
```yaml
defaults:
  - config.yaml         # Global settings
  - data/hbn.yaml      # Dataset specs
  - model/*.yaml       # Architectures
  - experiment/*.yaml  # Complete configs
```

## Critical Implementation Notes

1. **Startkit alignment**: Lines 144-193 (C1), 231-259 (C2) are canonical
2. **Input shape**: Always (batch, 129, 200) after preprocessing
3. **Submission.py**: Must handle Codabench mount points (/app/input/res/, etc.)
4. **R5 protection**: ValueError if "R5" in train releases
5. **Cache paths**: `cache/challenge1/r5/` separate from training
6. **TorchScript**: Required for mamba-ssm/neuralop models (not in Codabench)

## User Preferences

- `uv run` always (never raw python)
- Functional > classes
- Notebooks first (.py with #%%)
- Ask before design decisions
- Don't add/remove packages
- Edit existing files > create new

## Context7 Documentation

```python
# When memory lacks library details:
mcp__context7__resolve_library_id(libraryName="braindecode")
mcp__context7__get_library_docs(
    context7CompatibleLibraryID="/braindecode/braindecode",
    topic="EEGNeX parameters",
    tokens=3000
)
# Then capture findings in memory
```

Libraries: braindecode (models), PyTorch (optim/nn), MNE-Python (BIDS/preprocessing)

## Quick Reference

### Common Patterns
```python
# Parallel tools for speed
Read multiple files simultaneously
Grep/Glob in parallel when independent

# Memory lifecycle
"IMPLEMENTED 2025-10-15: Feature X"  # Not "currently" or "planned"
"Status as of 2025-10-15: ..."       # Snapshot with date

# Observation atomicity
✓ "Optimizer: AdamW(lr=0.001, weight_decay=0.00001)"  # One fact
✗ "Uses AdamW and CosineAnnealingLR"                  # Two facts
```

### Performance Rules
- AI operates 50-100x faster than humans for parallel tasks
- Memory search: O(1), Context7: O(seconds), Source diving: O(minutes)
- Batch edits in single message for parallelization

---
**File size: ~300 lines (75% reduction). All actionable information preserved.**