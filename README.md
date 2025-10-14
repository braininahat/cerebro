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

**Design philosophy**: Modular src/ directory for organization, but NOT packaged as installable library. Run scripts from root with `PYTHONPATH=. python scripts/train.py` for fast iteration.

```
cerebro/
├── pyproject.toml                # uv dependency management
├── README.md
├── .env                          # Paths, API keys (gitignored)
│
├── notebooks/                    # Jupytext .py format (# %%), executable in VSCode
│   ├── 00_download_all_data.py       # Download R1-R5 via EEGChallengeDataset
│   ├── 01_explore_hbn_structure.py   # BIDS structure, events, participants.tsv
│   ├── 02_understand_startkit.py     # Work through challenge_1.py, challenge_2.py
│   ├── 03_challenge1_baseline.py     # EEGNeX for RT prediction
│   ├── 04_challenge2_baseline.py     # EEGNeX for p_factor prediction
│   ├── 05_movie_contrastive_data.py  # Explore movies, create positive/negative pairs
│   ├── 06_movie_pretrain.py          # Contrastive pretraining loop
│   └── 07_multitask_finetune.py      # Joint C1+C2 fine-tuning
│
├── src/
│   ├── data/
│   │   ├── challenge1.py         # Challenge1Dataset: CCD windows + RT labels
│   │   ├── challenge2.py         # Challenge2Dataset: multi-task windows + p_factor
│   │   ├── movies.py             # MoviePairDataset: contrastive pairs from videos
│   │   ├── preprocessing.py      # annotate_trials_with_target, windowing utils
│   │   └── augmentation.py       # Time jitter, amplitude scaling
│   │
│   ├── models/
│   │   ├── encoders.py           # Wrap braindecode models (EEGNeX, SignalJEPA)
│   │   ├── projector.py          # MLP projection head for contrastive learning
│   │   ├── heads.py              # RegressionHead for C1/C2 predictions
│   │   └── multitask.py          # MultitaskModel: shared encoder + dual heads
│   │
│   ├── training/
│   │   ├── supervised.py         # SupervisedTrainer: single task training loop
│   │   ├── contrastive.py        # ContrastiveTrainer: InfoNCE loss, pair sampling
│   │   ├── multitask.py          # MultitaskTrainer: joint C1+C2 optimization
│   │   └── metrics.py            # NRMSE calculation, local scoring wrapper
│   │
│   ├── evaluation/
│   │   ├── local_scoring.py      # Adapted from startkit/local_scoring.py
│   │   └── submission_wrapper.py # Convert checkpoint → Submission class format
│   │
│   └── utils/
│       ├── checkpoint.py         # save_checkpoint(), load_checkpoint()
│       └── config.py             # Config dataclasses (if needed beyond Hydra)
│
├── configs/                      # Hydra configuration files
│   ├── config.yaml              # Base: seed, device, paths, wandb
│   ├── data/
│   │   └── hbn.yaml             # cache_dir, releases, batch_size, num_workers
│   ├── model/
│   │   ├── eegnex.yaml          # n_chans, n_times, sfreq, n_outputs
│   │   ├── jepa.yaml            # SignalJEPA configuration
│   │   └── contrastive.yaml     # encoder config + projection_dim, temperature
│   ├── training/
│   │   ├── supervised.yaml      # lr, weight_decay, epochs, early_stopping
│   │   ├── contrastive.yaml     # lr, temperature, epochs
│   │   └── multitask.yaml       # loss_weights, lr, freeze_encoder_epochs
│   └── experiment/              # Composed experiment configs
│       ├── baseline_eegnex_c1.yaml
│       ├── baseline_eegnex_c2.yaml
│       ├── movie_pretrain.yaml
│       └── multitask_finetune.yaml
│
├── scripts/
│   ├── train.py                 # Main training entry (Hydra-decorated)
│   ├── evaluate.py              # Run local_scoring on checkpoint
│   └── package_submission.py    # Create submission.zip
│
├── startkit/                    # Original competition startkit (reference)
├── data/full/ds005505-bdf/     # HBN BIDS data (gitignored)
└── outputs/                     # Checkpoints, logs, wandb (gitignored)
```

## Timeline & Milestones (10 Days)

**Days 1-2: Foundation & Data Understanding**
- Download all HBN releases (R1-R5) using `EEGChallengeDataset` API
- Explore BIDS structure, participants.tsv, event annotations
- Work through startkit code cell-by-cell in notebooks
- Set up local scoring pipeline

**Days 3-4: Supervised Baselines**
- Implement Challenge1Dataset (CCD windows + RT labels)
- Implement Challenge2Dataset (multi-task windows + p_factor labels)
- Train EEGNeX baselines for both challenges
- Establish baseline NRMSE scores via local evaluation
- Integrate wandb logging

**Days 5-7: Movie Contrastive Pretraining**
- Implement MoviePairDataset (positive/negative pairs)
- Train contrastive encoder with InfoNCE loss
- Implement multitask fine-tuning (shared encoder, dual heads)
- Evaluate: does it beat supervised baselines?

**Days 8-9: Iteration & Architecture Exploration** (if movie approach works)
- Hyperparameter tuning via wandb sweeps
- Try SignalJEPA, spatial hierarchies, SlowFast (if time permits)
- Experiment with different loss weights, temperatures

**Day 10: Final Submission**
- Select best checkpoint via local scoring
- Package submission.zip following competition format
- Test with `local_scoring.py --fast-dev-run`
- Submit to competition platform

## Setup

Install dependencies using uv:

```bash
cd cerebro
uv sync
```

Create `.env` file for paths:

```bash
echo "DATA_DIR=/home/varun/repos/cerebro/data/full" > .env
echo "WANDB_API_KEY=your_key_here" >> .env
```

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

All experiments use Hydra for configuration management. Run from repository root:

**Train supervised baseline:**
```bash
uv run python scripts/train.py experiment=baseline_eegnex_c1
uv run python scripts/train.py experiment=baseline_eegnex_c2
```

**Train contrastive pretraining:**
```bash
uv run python scripts/train.py experiment=movie_pretrain
```

**Fine-tune multitask:**
```bash
uv run python scripts/train.py experiment=multitask_finetune \
  training.pretrained_checkpoint=outputs/movie_pretrain/best.pt
```

**Override config values:**
```bash
uv run python scripts/train.py experiment=baseline_eegnex_c1 \
  training.lr=0.0001 data.batch_size=256
```

## Local Evaluation

Critical for rapid iteration before submitting:

```bash
# Evaluate checkpoint on R5 dataset (same as competition)
uv run python scripts/evaluate.py \
  checkpoint=outputs/baseline_eegnex_c1/best.pt

# Fast dev run (single subject, quick validation)
uv run python scripts/evaluate.py \
  checkpoint=outputs/baseline_eegnex_c1/best.pt \
  --fast-dev-run
```

Output:
```
Challenge 1 NRMSE: 0.8234
Challenge 2 NRMSE: 0.9112
Overall Score: 0.8850
```

## Creating Submission

Package checkpoint for competition submission:

```bash
uv run python scripts/package_submission.py \
  checkpoint=outputs/multitask_finetune/best.pt \
  output=submission.zip
```

Test locally before submitting:
```bash
uv run python startkit/local_scoring.py \
  --submission-zip submission.zip \
  --data-dir data/full \
  --output-dir outputs/test_submission \
  --fast-dev-run
```

## Key Dependencies

Core libraries (managed via `pyproject.toml`):

- **eegdash** (0.3.8+): Competition-specific HBN data loader with `EEGChallengeDataset`
- **braindecode** (1.2.0+): EEG models (EEGNeX, SignalJEPA) and preprocessing
- **MNE-Python**: Signal processing, BIDS support, Raw data handling
- **PyTorch** (2.2.2+): Deep learning framework, automatic differentiation
- **Hydra**: Hierarchical configuration composition and command-line overrides
- **wandb**: Experiment tracking, hyperparameter sweeps, model checkpointing
- **jupytext**: Notebook-as-code (.py format with # %% cells)

Install all with: `uv sync`

## Configuration Management with Hydra

Hydra enables clean experiment management through composition:

**Base configs** (`configs/config.yaml`, `data/hbn.yaml`, `model/eegnex.yaml`):
- Define reusable building blocks
- Version-controlled defaults
- Override via command line

**Experiment configs** (`configs/experiment/`):
- Compose base configs
- Specify complete experimental setups
- Example: `baseline_eegnex_c1.yaml` combines data/hbn + model/eegnex + training/supervised

**Benefits**:
- No code changes to try new architectures/hyperparameters
- Auto-generated output directories with timestamps
- Config saved with each checkpoint for reproducibility
- Easy sweeps: `hydra --multirun training.lr=0.001,0.0001,0.00001`

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