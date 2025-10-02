# Cerebro: EEG2025 Challenge Toolkit

Cerebro packages the official EEG Foundation start kits plus local tooling for the NeurIPS 2025 EEG2025 competition. The start kits are the primary source of truth for the two tasks:

- **Challenge 1 – Cross-Task Transfer Learning**: predict response time (seconds) from 2 s stimulus-locked segments of the Contrast Change Detection (CCD) task.
- **Challenge 2 – Psychopathology Prediction**: predict the subject-level p-factor score from EEG recordings (example pipeline illustrated on CCD data).

All baseline code, datasets, and evaluation conventions originate from the start kit repository (`startkit/README.md`, `startkit/challenge_1.py`, `startkit/challenge_2.py`). The additional modules under `src/cerebro/` and the benchmark notebooks extend those references with reusable helpers and experiment scripts.

## Repository Structure

```
startkit/              # Official challenge notebooks & scripts (authoritative reference)
src/cerebro/           # Shared data, preprocessing, model, metric, and training utilities
notebooks/             # Local experiments (EDA, benchmark sweeps, data inspection)
weights/               # Saved checkpoints (git-ignored)
wandb/                 # Local W&B run artifacts (git-ignored)
AGENTS.md              # Contributor guide for coding agents
CLAUDE.md              # Claude-specific workflow tips (mirrors AGENTS)
README.md              # This document
gpt_roadmap.md         # High-level roadmap for custom model development
```

## Data Access & Preparation

The examples follow the start kit convention of caching mini releases (100 Hz, 129 channels) with `EEGChallengeDataset`. Update the cache path to suit your environment and keep it configurable (e.g., via `DATA_DIR` or an env var).

### Configuring the Data Cache

Set `EEG2025_DATA_DIR` to the directory where releases should be cached. If unset, Cerebro defaults to `<repo>/data`.

```bash
export EEG2025_DATA_DIR=~/mne_data/eeg2025_competition
```

The start kit scripts accept the same directory when run from this repository (they create it if missing).

### Downloading Mini Releases

```bash
pip install -r startkit/requirements.txt
uv run python scripts/download_all_releases.py  # caches all mini releases and full releases
```

All mini releases are mutually exclusive sets of 20 subjects. Use the following exclusion list before splitting by subject:

```
NDARWV769JM7, NDARME789TD2, NDARUA442ZVF, NDARJP304NK1,
NDARTY128YLU, NDARDW550GU6, NDARLD243KRE, NDARUJ292JXV, NDARBA381JGH
```

### Challenge Windowing Rules

- **Challenge 1**: 2 s windows, sampled 0.5 s after the stimulus anchor; stride = 1 s; target = `rt_from_stimulus` (response time).
- **Challenge 2**: 4 s windows with a 2 s stride, followed by random 2 s crops during training; target = subject-level `p_factor`.

Both scripts rely on `EEGDash` metadata to inject annotations (`annotate_trials_with_target`, `add_aux_anchors`, `add_extras_columns`). Refer to `startkit/challenge_1.py` and `startkit/challenge_2.py` for end-to-end examples.

## Baseline Workflows from the Start Kits

| Challenge | Entry Point | Model | Loss | Notes |
|-----------|-------------|-------|------|-------|
| 1 | `startkit/challenge_1.py` | Braindecode `EEGNeX` | MSE | Subject-wise train/val/test split, early stopping on validation RMSE |
| 2 | `startkit/challenge_2.py` | Braindecode `EEGNeX` | L1 | Random crops per window; trains for one epoch as illustration |

The `startkit/submission.py` template shows how organisers expect weights and inference hooks to be packaged for final submission.

## Local Experimentation

### Benchmark Sweep (`notebooks/002_benchmark.py`)

This script orchestrates Weights & Biases sweeps over the Braindecode model zoo using model-specific defaults. It mirrors the Challenge 1 baseline pipeline while logging additional metrics (per-epoch timing, parameter counts, artifact uploads).

```bash
uv run python notebooks/002_benchmark.py                 # run comparison sweep (default)
uv run python notebooks/002_benchmark.py --show-configs  # list available sweep presets
uv run python notebooks/002_benchmark.py --test --model EEGNetv4
```

Metrics logged per run include train/val/test loss, RMSE, MAE, learning rate, timing, and best-epoch checkpoints stored in `weights/`.

### Pretraining Utilities

Use `src/cerebro/pretraining.py:create_pretraining_dataloader` to sample random crops across passive tasks for self-supervised objectives. The helper expects releases cached via `scripts/download_all_releases.py`.

Foundation code lives under `fm/`: datamodules for pretraining, simple JEPA-style encoder (`fm/models`), and loss helpers (`fm/tasks`).

### Data Exploration (`notebooks/001_eda.py`, `notebooks/003_dataset_dataloader.py`)

- `001_eda.py` reproduces the start kit Challenge 1 workflow with additional visualisation hooks.
- `003_dataset_dataloader.py` inspects the structure of full CCD releases (R1–R11) and documents open questions about pooling across tasks.

These notebooks currently target Challenge 1; extending them to load p-factor labels will be the first step toward richer Challenge 2 experiments.

## Development Workflow

1. **Environment setup**
   - `pip install -e .` to install Cerebro utilities in editable mode.
   - `pip install -r startkit/requirements.txt` for the official dependencies.
   - Use `uv run python <script>` when executing notebooks or scripts to ensure a consistent environment (see `CLAUDE.md`).
2. **Formatting & linting**
   - `make format` or `black --check . && isort --check .` before committing.
3. **Testing**
   - `pytest -q tests/` (add tests alongside new utilities under `src/cerebro/`).

## Challenge Metrics & Evaluation

- **Challenge 1**: MAE (40 %), R² (20 %), AUC-ROC (30 %), Balanced Accuracy (10 %). Baseline pipelines use MAE/RMSE for monitoring; add classification heads if you extend to success-rate prediction.
- **Challenge 2**: Concordance Correlation Coefficient (50 %), RMSE (30 %), Spearman correlation (20 %). The start kit example focuses on p-factor only; extend it to additional CBCL scores as needed.

When reporting results, keep Release 5 as a validation/test hold-out and avoid data leakage across subjects.

## Roadmap

`gpt_roadmap.md` outlines the next phase: move from the provided baselines toward a foundation model that pre-trains on passive tasks, then fine-tunes for Challenge 1. Review and update that file as plans evolve; once a solid Challenge 1 pipeline is in place, expand to Challenge 2 and optional reinforcement-learning enhancements.

## Additional Resources

- [EEG Foundation Challenge website](https://eeg2025.github.io)
- [EEGDash documentation](https://eeglab.org/EEGDash/overview.html)
- [Braindecode model catalogue](https://braindecode.org/stable/models/models_table.html)
- [Dataset download guide](https://eeg2025.github.io/data/#downloading-the-data)

For submission packaging requirements, consult `startkit/submission.py` and follow the organisers’ instructions for checkpoint naming and zipped deliverables.
