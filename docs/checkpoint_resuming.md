# Checkpoint Resuming Guide

## Overview

The Cerebro CLI now has **automatic checkpoint resuming** with wandb run association. No manual intervention required!

## Features

### 1. Automatic Checkpoint Fixing
When you pass `--ckpt_path`, the CLI automatically:
- Removes `hyper_parameters` that cause validation conflicts
- Extracts wandb run metadata for automatic resuming
- Saves the fixed checkpoint back to the same path

### 2. Automatic Wandb Resuming
If the checkpoint contains wandb metadata, the CLI automatically:
- Configures wandb logger with the original run ID
- Sets `resume="must"` to continue the same wandb run
- Preserves run name, project, and entity

### 3. Future-Proof Checkpoints
The `CheckpointCompatibilityCallback` ensures all new checkpoints:
- Are saved without `_class_path` conflicts
- Include wandb metadata for easy resuming

## Usage

### Basic Resuming (Simplest)

```bash
python -m cerebro.cli.train fit \
    --config configs/codebook.yaml \
    --ckpt_path outputs/challenge1/20251021_205540/checkpoints/last.ckpt
```

**What happens:**
1. CLI detects `--ckpt_path`
2. Automatically fixes checkpoint (removes hyper_parameters)
3. Extracts wandb run ID (if available)
4. Resumes training from epoch 82
5. Continues the same wandb run (if metadata exists)

**Output:**
```
✓ Auto-fixed checkpoint (removed hyper_parameters): last.ckpt
✓ Found wandb run ID in checkpoint: u74th6hp
✓ Auto-configured wandb to resume run: u74th6hp
```

### What Gets Restored

✅ **Model weights** (`state_dict`)
✅ **Optimizer state** (momentum buffers, etc.)
✅ **LR scheduler state** (current LR, step count)
✅ **Epoch number** (continues from where it stopped)
✅ **Global step counter** (batch counter)
✅ **Callback states** (early stopping patience, etc.)
✅ **Training loops state**
✅ **Wandb run** (same run ID, continuous metrics)

### Old Checkpoints (Without Wandb Metadata)

If you have checkpoints from before this feature was added:

#### Option 1: Let it create a new wandb run
```bash
# Just resume - will create a new wandb run
python -m cerebro.cli.train fit \
    --config configs/codebook.yaml \
    --ckpt_path path/to/old/checkpoint.ckpt
```

#### Option 2: Add wandb metadata manually
```bash
# Add wandb metadata to checkpoint
python scripts/add_wandb_to_checkpoint.py \
    --checkpoint outputs/challenge1/20251021_205540/checkpoints/last.ckpt \
    --run-id u74th6hp \
    --project eeg-challenge-2025 \
    --entity ubcse-eeg2025 \
    --name tokenizer

# Then resume - will automatically use the run ID
python -m cerebro.cli.train fit \
    --config configs/codebook.yaml \
    --ckpt_path outputs/challenge1/20251021_205540/checkpoints/last.ckpt
```

#### Option 3: Override wandb config on command line
```bash
python -m cerebro.cli.train fit \
    --config configs/codebook.yaml \
    --ckpt_path path/to/checkpoint.ckpt \
    --trainer.logger.init_args.id "your-wandb-run-id" \
    --trainer.logger.init_args.resume "must"
```

## How It Works

### Saving (New Checkpoints)

When training with `CheckpointCompatibilityCallback` enabled:

```yaml
# configs/codebook.yaml (line 74)
callbacks:
  - class_path: cerebro.callbacks.CheckpointCompatibilityCallback
```

**On each checkpoint save:**
1. Removes `_class_path` from `hyper_parameters`
2. Extracts wandb run metadata:
   - `wandb_run_id`
   - `wandb_run_name`
   - `wandb_project`
   - `wandb_entity`
3. Adds metadata to checkpoint as `checkpoint["wandb_metadata"]`

**Result:** All future checkpoints are compatible and include wandb info.

### Loading (Resuming)

When you pass `--ckpt_path`:

1. **Before CLI parsing** (`_auto_fix_checkpoint_in_args()`):
   - Loads checkpoint
   - Extracts `wandb_metadata` if available
   - Removes `hyper_parameters`, `datamodule_hyper_parameters`, etc.
   - Saves fixed checkpoint

2. **Before instantiation** (`before_instantiate_classes()`):
   - If `wandb_metadata` exists:
     - Injects `id`, `resume`, `name`, `project`, `entity` into logger config
     - Prints confirmation message

3. **Training starts:**
   - Wandb connects to existing run
   - Metrics continue seamlessly

## Troubleshooting

### "Validation failed: Key '_class_path' is not expected"

**Cause:** Old checkpoint with `_class_path` in hyper_parameters

**Fix:** Automatic! Just run the command again - the CLI now auto-fixes on first load.

### Checkpoint epoch doesn't match max_epochs

```
Error: You restored a checkpoint with current_epoch=71, but you have set Trainer(max_epochs=1).
```

**Cause:** Using `--trainer.fast_dev_run` with a checkpoint from epoch 71

**Fix:** Either:
- Remove `--trainer.fast_dev_run` for normal training
- Use `--trainer.max_epochs 1000` to allow continuation

### Wandb creates new run instead of resuming

**Cause:** Checkpoint doesn't have `wandb_metadata`

**Options:**
1. Let it create a new run (simplest)
2. Add metadata using `scripts/add_wandb_to_checkpoint.py`
3. Override on command line with `--trainer.logger.init_args.id`

### Can't find wandb run ID

**Find it in:**
1. Wandb dashboard URL: `https://wandb.ai/entity/project/runs/RUN_ID`
2. Original training logs: "Syncing run tokenizer", "View run at .../runs/RUN_ID"
3. Checkpoint (if saved with new callback): `checkpoint["wandb_metadata"]["wandb_run_id"]`

**Extract from checkpoint:**
```python
import torch
ckpt = torch.load("checkpoint.ckpt")
print(ckpt.get("wandb_metadata", {}).get("wandb_run_id", "Not found"))
```

## Examples

### Resume from last checkpoint
```bash
python -m cerebro.cli.train fit \
    --config configs/codebook.yaml \
    --ckpt_path outputs/challenge1/20251021_205540/checkpoints/last.ckpt
```

### Resume from best checkpoint
```bash
python -m cerebro.cli.train fit \
    --config configs/codebook.yaml \
    --ckpt_path outputs/challenge1/20251021_205540/checkpoints/challenge1-epoch=71-val_tok_loss=1.6088.ckpt
```

### Resume and change learning rate
```bash
python -m cerebro.cli.train fit \
    --config configs/codebook.yaml \
    --ckpt_path outputs/challenge1/20251021_205540/checkpoints/last.ckpt \
    --model.init_args.learning_rate 0.00005
```

### Resume and train longer
```bash
python -m cerebro.cli.train fit \
    --config configs/codebook.yaml \
    --ckpt_path outputs/challenge1/20251021_205540/checkpoints/last.ckpt \
    --trainer.max_epochs 2000
```

## Summary

**Just use:**
```bash
python -m cerebro.cli.train fit \
    --config configs/codebook.yaml \
    --ckpt_path path/to/checkpoint.ckpt
```

Everything else is automatic:
- ✅ Checkpoint fixing
- ✅ Wandb resuming
- ✅ Training state restoration

No manual intervention needed!
