# Challenge 1 CLI Refactoring Progress

## Goal
Extract notebook 04_train_challenge1.py functionality into Lightning CLI with full feature parity.

**DO NOT MODIFY notebooks/04_train_challenge1.py** - keep as reference implementation.

---

## Completed ✅

### 1. Utilities Extracted
- ✅ `cerebro/utils/logging.py` - Rich logging (console + file, PlainFormatter)
- ✅ `cerebro/utils/tuning.py` - LR finder + batch size scaler wrappers

### 2. Data Module Extracted
- ✅ `cerebro/data/challenge1.py` - Challenge1DataModule with:
  - Cache key generation from windowing params
  - Pickle-based caching (skip preprocessing on repeated runs)
  - EEGChallengeDataset loading (lines 220-246 from notebook 04)
  - Preprocessing: annotate_trials_with_target + add_aux_anchors (lines 254-271)
  - Stimulus-locked windowing with create_windows_from_events (lines 282-290)
  - add_extras_columns for metadata injection (lines 295-302)
  - Subject-level splits (train/val/test)
  - DataLoaders with persistence and pin_memory

### 3. Model Module Extracted
- ✅ `cerebro/models/challenge1.py` - Challenge1Module with:
  - EEGNeX wrapper (n_chans=129, n_outputs=1, n_times=200, sfreq=100)
  - MSE loss function
  - NRMSE metric calculation (validation + test)
  - Prediction storage for epoch-level NRMSE
  - AdamW optimizer with configurable lr/weight_decay
  - CosineAnnealingLR scheduler

### 4. Custom CLI Entry Point
- ✅ `cerebro/cli/train.py` - CerebroCLI with:
  - Rich logging setup (console + file with PlainFormatter)
  - Optional LR finder with plot upload to wandb
  - Optional batch size finder
  - Comprehensive wandb config logging (mimics notebook 04 lines 599-642)
  - before_fit() hook for setup before training

### 5. Configuration Files
- ✅ `configs/challenge1_base.yaml` - Full feature config:
  - Trainer (100 epochs, bf16-mixed, gradient clipping)
  - WandbLogger with artifact upload
  - Callbacks (ModelCheckpoint, EarlyStopping, TQDMProgressBar, RichModelSummary)
  - Model hyperparameters (lr, weight_decay)
  - Data parameters (10 releases, batch_size=512, windowing)
  - Tuning flags (run_lr_finder, run_batch_size_finder)

- ✅ `configs/challenge1_mini.yaml` - Fast prototyping variant:
  - Inherits from base
  - Overrides: use_mini=true, releases=[R5], max_epochs=2

- ✅ `sweeps/challenge1_sweep.yaml` - Wandb hyperparameter search:
  - Bayes optimization
  - Flattened parameter names (model.init_args.lr, etc.)
  - Uses ${args_json_file} for Lightning CLI compatibility
  - Early termination with Hyperband

---

## In Progress 🚧

### 6. Testing
**Status**: ✅ CLI validated end-to-end

- [x] Test basic training: `uv run cerebro fit --config configs/challenge1_mini.yaml --trainer.fast_dev_run true`
- [x] Test backward compatibility: `uv run python cerebro/cli/train.py fit --config configs/challenge1_mini.yaml --trainer.fast_dev_run true`
- [ ] Test mini config full run: `uv run cerebro fit --config configs/challenge1_mini.yaml`
- [ ] Test LR finder: `uv run cerebro fit --config configs/challenge1_mini.yaml --run_lr_finder true`
- [ ] Test parameter override: `uv run cerebro fit --config configs/challenge1_mini.yaml --model.init_args.lr 0.0001`
- [ ] Test sweep (after basic test passes): `wandb sweep sweeps/challenge1_sweep.yaml && wandb agent <sweep-id>`

---

## Remaining Tasks 📋

None! All components extracted. Just need to test and fix any bugs.

---

## Key Design Decisions

1. **Caching**: Pickle-based with descriptive keys (releases + windowing params + mini flag)
2. **Logging**: Rich console + file with PlainFormatter (strips markup from file logs)
3. **Splits**: Subject-level to prevent data leakage
4. **Tuning**: Optional via config flags (run_lr_finder, run_batch_size_finder)
5. **Config system**: Lightning CLI with jsonargparse + OmegaConf interpolation
6. **Wandb sweeps**: Flattened parameter names with dot notation (e.g., model.init_args.lr)

---

## Files Created/Modified

**Created:**
- ✅ cerebro/utils/logging.py
- ✅ cerebro/utils/tuning.py
- ✅ cerebro/data/challenge1.py
- ✅ cerebro/models/challenge1.py
- ✅ cerebro/cli/__init__.py
- ✅ cerebro/cli/train.py
- ✅ configs/challenge1_base.yaml (Lightning CLI format)
- ✅ configs/challenge1_mini.yaml (Lightning CLI format)
- ✅ sweeps/challenge1_sweep.yaml
- ✅ REFACTORING_PROGRESS.md

**Refactored (Package Structure):**
- ✅ Renamed: src/ → cerebro/ (git mv preserves history)
- ✅ Updated: pyproject.toml (packages=['cerebro'], CLI entry point)
- ✅ Updated: All imports (src. → cerebro.)
- ✅ Removed: sys.path manipulation hacks
- ✅ Installed: Package in editable mode (uv pip install -e .)

**Removed:**
- ❌ configs/config.yaml (old Hydra)
- ❌ configs/data/ (old Hydra)
- ❌ configs/model/ (old Hydra)
- ❌ configs/training/ (old Hydra)
- ❌ configs/experiment/ (old Hydra)

**Unchanged:**
- ✅ notebooks/04_train_challenge1.py (kept as reference)

---

## Usage Examples (After Completion)

### Interactive development (notebook)
```python
from cerebro.data.challenge1 import Challenge1DataModule
from cerebro.models.challenge1 import Challenge1Module
from lightning import Trainer

datamodule = Challenge1DataModule(data_dir="data/mini", releases=["R5"], use_mini=True)
model = Challenge1Module(lr=0.001)
trainer = Trainer(max_epochs=2, accelerator="auto")
trainer.fit(model, datamodule)
```

### CLI training
```bash
# Basic training (new CLI entry point)
uv run cerebro fit --config configs/challenge1_base.yaml

# Mini dataset (fast prototyping)
uv run cerebro fit --config configs/challenge1_mini.yaml

# With LR finder
uv run cerebro fit --config configs/challenge1_base.yaml --run_lr_finder true

# Override parameters
uv run cerebro fit --config configs/challenge1_base.yaml \
    --model.init_args.lr 0.0001 \
    --data.init_args.batch_size 256

# Backward compatible (direct python)
uv run python cerebro/cli/train.py fit --config configs/challenge1_base.yaml
```

### Wandb sweeps
```bash
wandb sweep sweeps/challenge1_sweep.yaml  # Returns sweep ID
wandb agent <entity>/<project>/<sweep-id>
```

---

## Notes for Fresh Sessions

- notebook 04 is the **source of truth** - don't modify it
- All extracted code should preserve exact behavior from notebook 04
- Test with `--fast_dev_run true` after each component
- Memory graph has context under "notebook refactoring 2025-01"
