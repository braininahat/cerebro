# Lightning CLI Configurations

This directory contains Lightning CLI configurations for training Challenge 1 models.

## Available Configs

### challenge1_base.yaml
Production configuration for full dataset training:
- **Releases**: R1-R4, R6-R11 (10 releases, ~10K+ windows)
- **Epochs**: 100 with early stopping (patience=10)
- **Batch size**: 512
- **Precision**: bf16-mixed (requires Ampere+ GPUs like RTX 30/40 series)
- **Features**:
  - Wandb logging with artifact upload
  - ModelCheckpoint (save top 3 by val_nrmse)
  - EarlyStopping on val_nrmse
  - Optional LR finder (set `run_lr_finder: true`)
  - Optional batch size finder (set `run_batch_size_finder: true`)

**Usage:**
```bash
uv run python src/cli/train.py fit --config configs/challenge1_base.yaml
```

### challenge1_mini.yaml
Fast prototyping configuration for mini dataset:
- **Releases**: R5 only (mini dataset, ~1200 windows)
- **Epochs**: 2 (quick iteration)
- **Batch size**: 64
- **Precision**: bf16-mixed
- **Features**: Same as base, but optimized for speed

**Usage:**
```bash
uv run python src/cli/train.py fit --config configs/challenge1_mini.yaml

# With fast_dev_run (1 batch only)
uv run python src/cli/train.py fit --config configs/challenge1_mini.yaml --trainer.fast_dev_run true
```

## Overriding Parameters

You can override any parameter via CLI:

```bash
# Change learning rate
uv run python src/cli/train.py fit \
    --config configs/challenge1_base.yaml \
    --model.lr 0.0001

# Change batch size and num_workers
uv run python src/cli/train.py fit \
    --config configs/challenge1_base.yaml \
    --data.batch_size 256 \
    --data.num_workers 16

# Enable LR finder
uv run python src/cli/train.py fit \
    --config configs/challenge1_base.yaml \
    --run_lr_finder true
```

## Wandb Sweeps

For hyperparameter search, use the sweep config in `sweeps/challenge1_sweep.yaml`:

```bash
# Initialize sweep
wandb sweep sweeps/challenge1_sweep.yaml
# Output: wandb: Created sweep with ID: abc123xyz

# Run agent (can run multiple in parallel)
wandb agent <your-entity>/eeg2025/abc123xyz
```

The sweep searches over:
- Learning rate: log-uniform (1e-5 to 1e-2)
- Weight decay: categorical [1e-5, 1e-4, 1e-3]
- Batch size: categorical [128, 256, 512, 1024]

## Config Structure

Lightning CLI configs follow this structure:

```yaml
seed_everything: 42  # Reproducibility

# Tuning flags (optional)
run_lr_finder: false
run_batch_size_finder: false

# Trainer configuration
trainer:
  max_epochs: 100
  accelerator: auto  # Auto-detect GPU/CPU
  precision: bf16-mixed
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: eeg2025
      entity: ubcse-eeg2025
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: val_nrmse

# Model configuration (class registered in CLI)
model:
  n_chans: 129
  lr: 0.001
  weight_decay: 0.00001
  epochs: 100

# Data configuration (class registered in CLI)
data:
  data_dir: ${oc.env:EEG2025_DATA_ROOT,data/full}
  releases: [R1, R2, ...]
  batch_size: 512
```

Note: Model and data classes are registered in `src/cli/train.py`, so you only specify `__init__` parameters here (no `class_path` needed).

## Environment Variables

- `EEG2025_DATA_ROOT`: Override data directory (default: `data/full` or `data/mini`)
- `WANDB_MODE`: Set to `offline` to disable wandb syncing
- `WANDB_ENTITY`: Override wandb team/username
- `WANDB_PROJECT`: Override wandb project name

Example:
```bash
WANDB_MODE=offline uv run python src/cli/train.py fit --config configs/challenge1_mini.yaml
```
