# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains code for the EEG2025 NeurIPS Challenge: "From Cross-Task to Cross-Subject EEG Decoding". The project focuses on developing EEG decoders that can transfer knowledge across cognitive tasks and generalize across subjects using the HBN-EEG dataset (3000+ participants, 6 cognitive tasks).

## Common Commands

### Running Python Scripts
```bash
# Always use uv to run Python scripts
uv run python script_name.py

# Download mini datasets (already completed - 34.71 GB in eeg_mini_datasets/)
uv run python download_mini_parallel.py

# Analyze subject distribution across datasets
uv run python analyze_subject_distribution.py

# Analyze task completion per subject
uv run python analyze_tasks_per_subject.py
```

### Data Access
```bash
# List mini dataset contents
ls eeg_mini_datasets/R*_mini_L100/

# Check AWS S3 bucket contents (public, no credentials needed)
aws s3 ls s3://nmdatasets/NeurIPS25/ --no-sign-request

# Count .set files in a dataset
find eeg_mini_datasets/R1_mini_L100 -name "*.set" | wc -l
```

## Architecture Overview

### Data Pipeline
1. **Data Download**: Scripts fetch EEG data from AWS S3 (`s3://nmdatasets/NeurIPS25/`)
2. **Data Analysis**: Scripts analyze subject distribution and task completion
3. **Model Development**: (To be implemented) Using braindecode/EEGDash for EEG processing

### Directory Structure
- `eeg_mini_datasets/`: Downloaded mini datasets (R1-R11, 100Hz, ~3-4GB each)
- `eeg2025.github.io/`: Challenge website documentation (git submodule)
- `downsample-datasets/`: MATLAB/Python scripts for downsampling to 100Hz (git submodule)
- `*.csv`: Analysis outputs (subject distributions, task completions)
- `*.json`: Metadata and analysis results

### Key Data Formats
- **EEG Data**: EEGLAB .set files (without .fdt for mini datasets)
- **Events**: .tsv files with onset, duration, value, event_code, feedback columns
- **Metadata**: .json files with task descriptions and EEG parameters
- **Participants**: .tsv with demographics and psychopathology scores

## Challenge Requirements

### Challenge 1: Cross-Task Transfer Learning (40% final score)
- **Input**: 2-second pre-trial EEG epochs from Contrast Change Detection (CCD) + full Surround Suppression (SuS) data
- **Targets**: Response time (regression) + Success rate (classification)
- **Metrics**: MAE (40%), R² (20%), AUC-ROC (30%), Balanced Accuracy (10%)

### Challenge 2: Psychopathology Prediction (60% final score)
- **Input**: All available EEG tasks (minimum 15 minutes per subject)
- **Targets**: 4 continuous scores (p-factor, internalizing, externalizing, attention)
- **Metrics**: CCC (50%), RMSE (30%), Spearman correlation (20%)

### Constraints
- Models must run on single GPU with 20GB memory at inference
- Data downsampled to 100Hz, filtered 0.5-50Hz for evaluation
- Code submission competition (not results submission)

## Data Specifications

### EEG Recording
- 128-channel Magstim EGI system
- Reference: Cz electrode
- Sampling: 100Hz (downsampled from 500Hz)
- Filtering: 0.5-50Hz bandpass

### Tasks
**Passive**: Resting State, Surround Suppression, Movie Watching (4 films)
**Active**: Contrast Change Detection (3 runs), Sequence Learning (6/8 target), Symbol Search

### Available Features
- Demographics: age, sex, handedness (EHQ score -100 to +100)
- Psychopathology: p_factor, attention, internalizing, externalizing (from CBCL)
- Task availability: Boolean flags for each task/run

## Key Scripts Purpose

- `find_complete_subjects.py`: Identifies subjects present across multiple datasets (found: none overlap)
- `analyze_subject_distribution.py`: Creates distribution matrix of subjects across datasets
- `analyze_tasks_per_subject.py`: Analyzes task completion rates per subject
- `download_mini_parallel.py`: Parallel download of mini datasets from S3 (5 workers, ~280 Mbps)
- `download_eeg_data.py`: Downloads specific subjects using EEGDash API

## Important Libraries

- **braindecode**: Deep learning for EEG data (main framework)
- **EEGDash**: EEG data loading and API access
- **MNE-Python**: EEG/MEG analysis (via braindecode)
- **MOABB**: Mother of All BCI Benchmarks (benchmarking tools)

## Dataset Facts

- 11 mini datasets (R1-R11) with 20 subjects each
- No subjects appear in multiple datasets (completely disjoint)
- Each subject has 220-240 .set files across all tasks
- Total downloaded: 34.71 GB for all mini datasets
- Participants.tsv contains target labels for both challenges
- Before web search consult @knowledge_base.md, only searching the web if not already documented and document findings in that case. Also document whatever we learn experimentally along the way.