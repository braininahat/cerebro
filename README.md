# cerebro

Competition submission for the NeurIPS 2025 EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding.

## Competition Overview

The challenge addresses fundamental problems in EEG decoding across tasks and subjects using the Healthy Brain Network (HBN) dataset with 3,000+ participants and 129-channel recordings at 100 Hz.

**Challenge 1 - Response Time Prediction (30% weight)**: Predict response time (RT) from EEG recordings in the Contrast Change Detection (CCD) task. Models receive 2-second EEG windows (0.5-2.5s after stimulus onset, 129 channels, 100Hz) and must predict how quickly subjects responded to contrast changes.

**Challenge 2 - P-Factor Prediction (70% weight)**: Predict the externalizing psychopathology factor (p_factor) from EEG recordings. Models can use data from any task (CCD, movies, resting state, etc.) to predict this trait-level score reflecting behavioral and emotional dysregulation.

Final score combines both challenges: **S_overall = 0.3 × NRMSE_C1 + 0.7 × NRMSE_C2**

Evaluation metric: Normalized Root Mean Squared Error (NRMSE) normalized by standard deviation of targets.

## Approach

### Phase 1: Supervised Baselines (Days 1-4)

Establish baseline performance using standard EEG models from braindecode:
- **EEGNeX**: Efficient ConvNet architecture for EEG
- **SignalJEPA**: Joint-embedding predictive architecture baseline

Direct supervised training on each challenge independently provides lower bounds and validates data pipeline correctness.

### Phase 2: Movie Contrastive Pretraining (Days 5-7)

**Key insight**: Multiple subjects view identical video clips (DespicableMe, ThePresent, etc.), producing temporally aligned neural responses. We leverage this natural synchronization for cross-subject representation learning.

**Contrastive learning strategy**:
- **Positive pairs**: Same movie clip, same timestamp, different subjects → should produce similar representations
- **Negative pairs**: Different movie clips (any subjects/timestamps) → should produce dissimilar representations
- **InfoNCE loss**: Learns encoder that makes positive pairs similar while pushing apart negatives

This pretraining addresses the core challenge: learning subject-invariant, stimulus-sensitive representations without task labels. The encoder captures shared neural patterns across individuals responding to identical stimuli.

**Multitask fine-tuning**: After pretraining, we add task-specific heads for both challenges (RT prediction + p_factor regression) and fine-tune jointly. Shared encoder benefits from both tasks' training signals.

### Phase 3: Architecture Exploration (Days 8-9, if time permits)

If movie contrastive pretraining beats baselines:
- **Multi-scale temporal processing**: SlowFast architecture with dual pathways for different frequency bands (fast: beta/gamma 100ms windows, slow: delta/theta/alpha 5s windows)
- **Hierarchical spatial aggregation**: Regional tokens (frontal/parietal/temporal/occipital) capturing anatomical organization
- **Efficient sequence modeling**: Mamba state-space models for long-range dependencies in continuous recordings

Priority: working submission over architectural complexity. Advanced architectures only if core approach succeeds.

## Repository Structure

**Design philosophy**: Installable package (`uv pip install -e .`) for clean imports and IDE support. Lightning CLI for experiment management with rich diagnostics.

```
cerebro/
├── pyproject.toml                # uv dependency management + CLI entry point
├── README.md
├── .env                          # Paths, API keys (gitignored)
│
├── notebooks/                    # Jupytext .py format (# %%), executable in VSCode
│   ├── 001_train_challenge1_eegnex.py   # ✅ EEGNeX baseline for RT prediction
│   ├── 002_train_challenge1_jepa.py     # ✅ SignalJEPA baseline for RT prediction
│   ├── 003_validate_data_quality.py     # ✅ Data pipeline validation, R5 separation checks
│   ├── 015_train_challenge2.py          # 🚧 p_factor prediction (planned)
│   ├── 020_train_multitask.py           # 🚧 Joint C1+C2 training (planned)
│   ├── 023_movie_contrastive_windowing.py  # ✅ Movie windowing exploration
│   └── 024_test_contrastive_dataset.py  # ✅ Contrastive pair validation
│
├── cerebro/                      # Main package (installed via `uv pip install -e .`)
│   ├── __init__.py
│   │
│   ├── cli/
│   │   └── train.py              # ✅ CerebroCLI (Lightning CLI with logging, tuning)
│   │
│   ├── data/
│   │   └── challenge1.py         # ✅ Challenge1DataModule: CCD windows + RT labels (636 lines)
│   │
│   ├── models/
│   │   └── challenge1.py         # ✅ Challenge1Module: EEGNeX/SignalJEPA for RT regression
│   │
│   ├── callbacks/
│   │   └── model_autopsy.py      # ✅ ModelAutopsyCallback: comprehensive diagnostics
│   │
│   ├── diagnostics/              # ✅ Diagnostic modules (9 modules implemented)
│   │   ├── predictions.py        # Prediction analysis, baseline comparisons
│   │   ├── gradients.py          # Gradient flow analysis
│   │   ├── activations.py        # Dead neurons, layer statistics
│   │   ├── captum_attributions.py   # Integrated Gradients (Captum)
│   │   ├── captum_layers.py      # Layer GradCAM (Captum)
│   │   ├── failure_modes.py      # Top-K worst predictions
│   │   ├── ablation.py           # Channel/temporal ablation studies
│   │   └── visualizations.py     # Plot generation for all diagnostics
│   │
│   ├── utils/
│   │   ├── logging.py            # ✅ Rich logging (console + file)
│   │   ├── tuning.py             # ✅ LR finder, batch size finder wrappers
│   │   ├── movie_windows.py      # ✅ Movie task windowing (fixed-length)
│   │   └── contrastive_dataset.py   # ✅ ContrastivePairDataset (pos/neg pairs)
│   │
│   ├── training/                 # ❌ Planned (empty directory)
│   │   └── __init__.py
│   │
│   └── evaluation/               # ❌ Planned (empty directory)
│       └── __init__.py
│
├── configs/                      # Lightning CLI configuration files
│   ├── challenge1_eegnex.yaml          # ✅ EEGNeX for Challenge 1
│   ├── challenge1_eegnex_mini.yaml     # ✅ Fast prototyping (R1 mini)
│   ├── challenge1_jepa.yaml            # ✅ SignalJEPA for Challenge 1
│   ├── challenge1_jepa_mini.yaml       # ✅ Fast prototyping (R1 mini)
│   ├── challenge1_submission.yaml      # ✅ Final submission (all training data)
│   └── README.md                       # Config documentation
│
├── startkit/                    # Original competition startkit (reference)
│   ├── challenge_1.py           # Reference preprocessing for C1
│   ├── challenge_2.py           # Reference preprocessing for C2
│   └── local_scoring.py         # Local evaluation (NRMSE calculation)
│
├── cache/                       # Preprocessed data cache (gitignored)
├── data/                        # HBN BIDS data (gitignored)
└── outputs/                     # Checkpoints, logs, wandb (gitignored)
```

**Status Legend**: ✅ Implemented | 🚧 Partially implemented | ❌ Planned

## Timeline & Milestones (10 Days)

**Days 1-2: Foundation & Data Understanding** ✅
- ✅ Downloaded HBN releases using `EEGChallengeDataset` API
- ✅ Explored BIDS structure, participants.tsv, event annotations
- ✅ Worked through startkit code (challenge_1.py, challenge_2.py)
- ✅ Set up local scoring pipeline via R5 test evaluation

**Days 3-4: Supervised Baselines** ✅
- ✅ Implemented Challenge1DataModule (CCD windows + RT labels, 636 lines)
- ✅ Implemented Challenge1Module (EEGNeX + SignalJEPA support)
- ✅ Integrated Lightning CLI for experiment management
- ✅ Added comprehensive diagnostics (ModelAutopsyCallback + 9 modules)
- ✅ Integrated wandb logging with artifact management
- 🚧 Challenge2Dataset (planned, not yet implemented)

**Days 5-7: Movie Contrastive Pretraining** 🚧
- ✅ Implemented movie windowing utilities (movie_windows.py)
- ✅ Implemented ContrastivePairDataset (contrastive_dataset.py)
- ✅ Validated infrastructure with notebooks (023, 024)
- ❌ Training loop not yet implemented (cerebro/training/ empty)
- ❌ Multitask fine-tuning not yet implemented

**Days 8-9: Iteration & Architecture Exploration** ⏳
- Hyperparameter tuning via LR finder / batch size finder
- Try different architectures (SignalJEPA validated)
- Experiment with training strategies

**Day 10: Final Submission** ⏳
- Select best checkpoint via local scoring
- Package submission.zip (TorchScript conversion)
- Test with `startkit/local_scoring.py`
- Submit to competition platform

**Status Legend**: ✅ Complete | 🚧 In progress | ❌ Not started | ⏳ Upcoming

## Setup

**1. Install dependencies and package:**

```bash
cd cerebro
uv sync                   # Install dependencies from pyproject.toml
uv pip install -e .       # Install cerebro package in editable mode
```

This registers the `cerebro` CLI command and enables clean imports throughout the codebase.

**2. Create `.env` file for paths:**

```bash
echo "HBN_ROOT=/path/to/your/data" > .env
echo "WANDB_API_KEY=your_key_here" >> .env
```

**Environment variables:**
- `HBN_ROOT`: Parent directory containing HBN releases (e.g., `/home/user/data`)
- `WANDB_API_KEY`: Weights & Biases API key for experiment tracking

## Data Download

Use EEGChallengeDataset API (handles caching automatically):

```python
# In notebooks/00_download_all_data.py
from eegdash import EEGChallengeDataset

for release in ["R1", "R2", "R3", "R4", "R5"]:
    dataset = EEGChallengeDataset(
        release=release,
        task="contrastChangeDetection",  # Or any task
        cache_dir="data/full",
        mini=False  # Download full dataset
    )
```

## Running Experiments

All experiments use **Lightning CLI** for configuration management. The `cerebro` command is registered via `pyproject.toml` during installation.

### Basic Training

**Train Challenge 1 baseline (EEGNeX):**
```bash
uv run cerebro fit --config configs/challenge1_eegnex.yaml
```

**Train Challenge 1 baseline (SignalJEPA):**
```bash
uv run cerebro fit --config configs/challenge1_jepa.yaml
```

**Fast prototyping with mini dataset:**
```bash
uv run cerebro fit --config configs/challenge1_eegnex_mini.yaml
```

### With Automatic Tuning

**Learning rate finder:**
```bash
uv run cerebro fit --config configs/challenge1_eegnex.yaml --run_lr_finder true
```

The LR finder runs before training, plots the loss curve, and uploads it to wandb.

**Batch size finder:**
```bash
uv run cerebro fit --config configs/challenge1_eegnex.yaml --run_batch_size_finder true
```

Automatically finds the largest batch size that fits in GPU memory.

### Overriding Config Values

**Override model hyperparameters:**
```bash
uv run cerebro fit --config configs/challenge1_eegnex.yaml \
  --model.lr 0.0001 \
  --model.weight_decay 0.0001
```

**Override data parameters:**
```bash
uv run cerebro fit --config configs/challenge1_eegnex.yaml \
  --data.batch_size 256 \
  --data.num_workers 16
```

**Override trainer settings:**
```bash
uv run cerebro fit --config configs/challenge1_eegnex.yaml \
  --trainer.max_epochs 50 \
  --trainer.precision "16-mixed"
```

### Final Submission Training

**Train on all available data (no validation split):**
```bash
uv run cerebro fit --config configs/challenge1_submission.yaml
```

This uses `mode="submission"` to train on 100% of R1-R4, R6-R11 for maximum performance.

### Backward Compatibility

You can also run the CLI directly via Python:
```bash
uv run python cerebro/cli/train.py fit --config configs/challenge1_eegnex.yaml
```

## Local Evaluation

**Current status**: R5 evaluation is integrated into training via `test_on_r5=true` in configs. Dedicated evaluation script planned but not yet implemented.

### During Training

Configs with `test_on_r5: true` automatically evaluate on R5 after training:

```bash
uv run cerebro fit --config configs/challenge1_eegnex.yaml
# Training completes → automatic R5 test evaluation → test_nrmse logged
```

### Manual Evaluation (Planned)

```bash
# Planned: scripts/evaluate.py
uv run python startkit/local_scoring.py \
  --submission-zip submission.zip \
  --data-dir $HBN_ROOT \
  --output-dir outputs/test_submission
```

This uses the competition's official scoring script to compute:
- Challenge 1 NRMSE
- Challenge 2 NRMSE
- Overall score: 0.3 × C1_NRMSE + 0.7 × C2_NRMSE

## Creating Submission

**Current status**: Submission packaging planned but not yet implemented. Use manual TorchScript conversion workflow.

### TorchScript Conversion Workflow (Recommended)

**Why TorchScript?** Competition environment lacks custom dependencies (mamba-ssm, neuralop). TorchScript bundles model architecture + weights in a single `.pt` file that only requires PyTorch.

**1. Convert Lightning checkpoint to TorchScript:**

```bash
uv run python -m cerebro.utils.checkpoint_to_torchscript \
  --ckpt outputs/challenge1/TIMESTAMP/checkpoints/best.ckpt \
  --output model_challenge_1.pt \
  --input-shape 1 129 200
```

**2. Create submission.py:**

See `cerebro/submission/submission.py` template (planned). Minimal example:

```python
import torch
from pathlib import Path

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        return torch.jit.load("model_challenge_1.pt", map_location=self.device)

    def get_model_challenge_2(self):
        return torch.jit.load("model_challenge_2.pt", map_location=self.device)
```

**3. Package submission (single-level zip):**

```bash
cd cerebro/submission
zip -j ../../submission.zip submission.py model_challenge_1.pt model_challenge_2.pt
```

**Critical**: Use `zip -j` to create single-level zip (no folders).

**4. Test locally:**

```bash
uv run python startkit/local_scoring.py \
  --submission-zip submission.zip \
  --data-dir $HBN_ROOT \
  --output-dir outputs/local_scoring
```

## Key Dependencies

Core libraries (managed via `pyproject.toml`):

- **eegdash** (0.3.8+): Competition-specific HBN data loader with `EEGChallengeDataset`
- **braindecode** (1.2.0+): EEG models (EEGNeX, SignalJEPA) and preprocessing
- **MNE-Python**: Signal processing, BIDS support, Raw data handling
- **PyTorch** (2.8.0+): Deep learning framework, automatic differentiation
- **Lightning** (2.5.5+): Training framework with CLI, callbacks, loggers
- **Captum** (0.8.0+): Model interpretability (Integrated Gradients, GradCAM)
- **wandb** (0.21.4+): Experiment tracking, artifact management
- **Rich** (13.9.0+): Beautiful terminal logging

Install all with: `uv sync && uv pip install -e .`

## Configuration Management with Lightning CLI

**Design**: Self-contained YAML configs with all parameters. Each config defines a complete experiment.

**Config structure** (e.g., `configs/challenge1_eegnex.yaml`):

```yaml
seed_everything: 42  # Reproducibility

# Tuning flags (optional)
run_lr_finder: false
run_batch_size_finder: false

# Trainer configuration
trainer:
  max_epochs: 1000
  accelerator: auto
  precision: "bf16-mixed"
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: eeg2025
      name: challenge1_baseline
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: val_nrmse
        mode: min
    - class_path: cerebro.callbacks.ModelAutopsyCallback
      init_args:
        diagnostics: ["predictions", "gradients", "activations"]

# Model configuration
model:
  n_chans: 129
  n_outputs: 1
  model_class: EEGNeX
  lr: 0.001

# Data configuration
data:
  data_dir: ${oc.env:HBN_ROOT,data}
  releases: [R1, R2, R3, R4, R6, R7, R8, R9, R10, R11]
  batch_size: 512
```

**Benefits**:
- Complete experiment in one file
- Lightning ecosystem integration (callbacks, loggers, profilers)
- Built-in LR finder and batch size finder
- Config saved with each checkpoint
- Override any parameter via CLI: `--model.lr 0.0001`

**Available configs**:
- `challenge1_eegnex.yaml` - EEGNeX for RT prediction
- `challenge1_jepa.yaml` - SignalJEPA for RT prediction
- `challenge1_*_mini.yaml` - Fast prototyping with R1 mini
- `challenge1_submission.yaml` - Final submission (all training data)

## Model Autopsy & Diagnostics

**Automatic comprehensive diagnostics** when early stopping fires or training completes.

### ModelAutopsyCallback

Configured in YAML via `trainer.callbacks`:

```yaml
callbacks:
  - class_path: cerebro.callbacks.ModelAutopsyCallback
    init_args:
      run_on_training_end: true
      run_on_early_stop: true
      diagnostics:
        - predictions          # Distribution, residuals, baseline comparisons
        - gradients            # Per-layer gradient flow
        - activations          # Dead neurons, layer statistics
        - integrated_gradients # Captum IG (memory optimized)
        - layer_gradcam        # Captum Layer GradCAM (memory optimized)
        - failure_modes        # Top-K worst predictions
      save_plots: true
      log_to_wandb: true
      generate_report: true
      num_samples: 500         # Analyze 500 samples (not full val set)
```

### Diagnostic Modules

**Tier 1 - Basic Diagnostics** (always enabled):
- **predictions.py**: NRMSE, baseline comparisons, prediction distribution
- **gradients.py**: Gradient flow, dead layers, grad/param ratios
- **activations.py**: Dead neurons, layer statistics

**Tier 2 - Attribution Analysis** (Captum):
- **captum_attributions.py**: Integrated Gradients (IG) for input attribution
  - Temporal profiles (when model attends)
  - Spatial profiles (which channels important)
  - Memory optimized (batched computation)
- **captum_layers.py**: Layer GradCAM for layer-wise importance
  - Auto-detects convolutional layers
  - Layer hierarchy analysis

**Tier 3 - Advanced Analysis** (opt-in):
- **ablation.py**: Channel/temporal ablation studies
- **failure_modes.py**: Top-K worst predictions with metadata analysis

**Tier 4 - Visualization**:
- **visualizations.py**: Plot generation for all diagnostics
- Automatic wandb upload of plots

### Outputs

**1. Diagnostic plots** (saved to `outputs/TIMESTAMP/autopsy/`):
- `prediction_distribution.png` - Predicted vs actual, residuals
- `gradient_flow.png` - Per-layer gradient magnitudes
- `activation_stats.png` - Dead neuron percentages
- `integrated_gradients.png` - Temporal/spatial attribution
- `layer_gradcam.png` - Layer importance hierarchy
- `failure_modes.png` - Worst predictions analysis

**2. Wandb artifacts**:
- Plots uploaded to `autopsy/*` namespace
- Markdown report uploaded as artifact
- Summary metrics table for cross-run comparison
- (Optional) Raw attribution data as compressed `.npz`

**3. Autopsy report** (`autopsy_report.md`):
- Prediction analysis (NRMSE, baseline comparisons)
- Gradient health (dead layers, magnitude issues)
- Activation health (dead neurons)
- Captum insights (temporal/spatial patterns)
- **Actionable recommendations** (increase LR, reduce weight decay, etc.)

### Example Workflow

```bash
# Training with autopsy enabled (default in configs)
uv run cerebro fit --config configs/challenge1_eegnex.yaml

# After early stopping or training end:
# → Autopsy runs automatically
# → Plots saved to outputs/TIMESTAMP/autopsy/
# → Report generated
# → Artifacts uploaded to wandb
```

Check wandb for:
- `autopsy/prediction_distribution` - Visual diagnostics
- `autopsy/summary` - Table of metrics
- `autopsy_report` artifact - Markdown report with recommendations

## Design Principles (10-Day Constraints)

**Priority 1: Working submission**
- Supervised baselines first (safety net)
- Test local scoring early and often
- Checkpoint after every milestone

**Priority 2: Fast iteration**
- src/ modules but no packaging overhead
- Aggressive caching of preprocessed data
- Start with mini=True, scale to full dataset once working

**Priority 3: Reproducibility**
- Git commit after each working state
- wandb logs everything (loss, NRMSE, hyperparameters)
- Hydra saves full config with checkpoints

**Non-priorities** (unless time permits):
- Extensive unit tests (focus on integration tests)
- Distributed training (single GPU sufficient)
- Complex preprocessing (use braindecode/eegdash defaults)
- Custom architectures (use braindecode models first)

**Code style**:
- Type hints for public functions
- Docstrings for non-obvious logic
- Notebooks for exploration, src/ for reusable code
- Configuration as documentation

## Future Improvements (If Time Permits)

### Challenge 1: Multi-Task Learning with Feedback

**Current limitation**: Model fails catastrophically on fast RT trials (< 0.5s) with errors > 1.5s. The competition's fixed evaluation window [0.5s, 2.5s] post-stimulus means these trials have minimal pre-response signal.

**Proposed solution**: Add auxiliary classification task to predict response correctness (smiley vs sad face feedback).

**Architecture**:
```
Input (129, 200) → EEGNeX Encoder → [RT Head, Correctness Head]
                                       ↓            ↓
                                   RT prediction  [Correct/Incorrect]
```

**Rationale**:
- Error trials likely have different neural signatures in available signal (0.5-2.5s window)
- Multi-task learning may improve shared representations
- Feedback annotations available in BIDS data (`feedback` field: smiley_face/sad_face)
- Expected benefit: 5-15% NRMSE reduction (speculative)

**Implementation notes**:
- Extract `feedback` field via `add_extras_columns` in preprocessing
- Add binary classification head to encoder
- Loss: `L_total = L_rt + 0.1 * L_correctness` (MSE + weighted CrossEntropy)
- At inference, only use RT head (auxiliary task is training-only)

**Why not pursued initially**: Challenge 2 (70% of score) higher priority; contrastive pretraining more promising.

### Challenge 1: Understanding the Task Constraint

The Contrast Change Detection task has a 1.6s stimulus ramping period (contrast gradually changes 50%→100% over 1.6 seconds). The [0.5s, 2.5s] evaluation window captures:
- 0.5-1.6s: Mid-to-late stimulus ramping
- 1.6-2.4s: Stimulus return to baseline
- 2.4-2.5s: First 100ms of feedback

For RTs < 0.5s, the window captures partial stimulus ramping, which may still contain speed-discriminative features. The competition's window choice appears intentional - focusing on decision/motor processes rather than early visual encoding.

**Why temporal jittering won't work**: Shifting the window breaks the stimulus-response time relationship (the label we're predicting). Unlike image augmentation, time is causally linked to RT.

## References

**Dataset**:
- HBN-EEG Dataset: Shirazi et al., bioRxiv 2024. DOI: 10.1101/2024.10.03.615261
- Healthy Brain Network: Alexander et al., Scientific Data 2017. DOI: 10.1038/sdata.2017.181

**Competition**:
- EEG Foundation Challenge, NeurIPS 2025: https://eeg2025.github.io

**Architecture foundations**:
- EEGNeX: Chen et al., 2024
- SignalJEPA: (Foundation model for EEG from braindecode)
- Mamba: Gu & Dao, 2024. State-space models for sequence modeling
- SlowFast: Feichtenhofer et al., ICCV 2019. Multi-scale temporal modeling

**Self-supervised learning**:
- SimCLR: Chen et al., ICML 2020. Contrastive learning framework
- InfoNCE: Oord et al., 2018. Noise-contrastive estimation
- JEPA: LeCun, 2022. Joint-embedding predictive architectures

**Tools**:
- braindecode: Schirrmeister et al., 2017. Deep learning for EEG
- MNE-Python: Gramfort et al., 2013. MEG/EEG analysis in Python
- eegdash: HBN competition data loader
- Hydra: Facebook Research. Configuration management