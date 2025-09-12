# EEG2025 Challenge Knowledge Base

> Living document for the NeurIPS 2025 EEG Challenge: "From Cross-Task to Cross-Subject EEG Decoding"
> Last Updated: 2025-01-11

## Dataset Format Decision

### Chosen Format: Plain Mini (R*_mini_L100)
- **File Format**: EEGLAB `.set` files (~9MB per file)
- **Location**: `eeg_mini_datasets/R*_mini_L100/`
- **Total Size**: 34.71 GB (already downloaded)

### Why Not BDF Mini (R*_mini_L100_bdf)?
- BDF contains `.bdf` files (BioSemi Data Format, ~6.7MB per file)
- Requires additional conversion steps with `mne.io.read_raw_bdf()`
- Less commonly used in EEG deep learning pipelines
- Startkit examples exclusively use plain mini format
- EEGDash natively expects EEGLAB `.set` files

### Data Access Pattern
```python
from eegdash import EEGChallengeDataset
dataset = EEGChallengeDataset(
    task="contrastChangeDetection",
    release="R5", 
    cache_dir=DATA_DIR,
    mini=True  # Uses plain mini datasets
)
raw = dataset.datasets[0].raw  # MNE Raw object
```

## Technical Specifications

### EEG Recording Parameters
- **Channels**: 129 (128-channel Magstim EGI system + Cz reference)
- **Sampling Rate**: 100 Hz (downsampled from 500 Hz)
- **Filtering**: 0.5-50 Hz bandpass
- **Reference**: Cz electrode
- **Input Shape**: `(batch_size, 129, n_times)`
  - Challenge 1: `(batch_size, 129, 200)` for 2-second windows
  - Challenge 2: `(batch_size, 129, 200)` for 2-second crops

### Available Tasks
**Passive Tasks**:
- Resting State
- Surround Suppression (SuS)
- Movie Watching (4 films: Despicable Me, Diary of a Wimpy Kid, Fun with Fractals, The Present)

**Active Tasks**:
- Contrast Change Detection (CCD) - 3 runs
- Sequence Learning - 6/8 target
- Symbol Search

### Window Creation Strategies
**Challenge 1 (CCD Task)**:
- 2-second epochs locked to stimulus onset
- Shift: +0.5s after stimulus
- Window: 2.0s duration
- Stride: 100 samples (1 second)

**Challenge 2 (All Tasks)**:
- 4-second windows with 2-second stride
- Random 2-second crops during training
- Minimum recording length: 4 seconds

## Challenge Requirements

### Challenge 1: Cross-Task Transfer Learning (40% final score)
**Objective**: Transfer knowledge from passive to active EEG tasks

**Training Data**:
- Source: Full Surround Suppression (SuS) data
- Target: 2-second pre-trial epochs from Contrast Change Detection (CCD)

**Targets**:
- Response time (regression)
- Success rate (classification)

**Evaluation Metrics**:
- MAE: 40%
- R²: 20%
- AUC-ROC: 30%
- Balanced Accuracy: 10%

### Challenge 2: Psychopathology Prediction (60% final score)
**Objective**: Predict psychopathology scores from multi-task EEG

**Input**: All available EEG tasks (minimum 15 minutes per subject)

**Targets** (continuous scores from CBCL):
- p_factor (general psychopathology)
- internalizing
- externalizing
- attention

**Evaluation Metrics**:
- CCC (Concordance Correlation Coefficient): 50%
- RMSE: 30%
- Spearman correlation: 20%

### Model Constraints
- Must run on single GPU with 20GB memory at inference
- Data evaluation at 100Hz, filtered 0.5-50Hz
- Code submission competition (not results submission)

## API and Library Details

### EEGDash Dataset Loading
```python
from eegdash import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with
)
```

### Braindecode Integration
```python
from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.preprocessing import (
    preprocess, 
    Preprocessor, 
    create_windows_from_events,
    create_fixed_length_windows
)
from braindecode.models import EEGNeX  # or other models
```

### MNE Compatibility
- Data internally stored as `mne.io.Raw` objects
- Can use `mne.io.read_raw_eeglab()` for direct loading
- Convert to Braindecode: `create_from_mne_raw()`

## Submission Requirements

### File Structure
```
submission.zip (single-level, no folders)
├── submission.py
├── weights_challenge_1.pt
└── weights_challenge_2.pt
```

### Submission Class Template
```python
class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ  # 100 Hz
        self.device = DEVICE
    
    def get_model_challenge_1(self):
        # Return Challenge 1 model
        
    def get_model_challenge_2(self):
        # Return Challenge 2 model
```

### Model Loading Pattern
```python
model.load_state_dict(
    torch.load("/app/output/weights.pt", 
               map_location=self.device)
)
```

## Dataset Statistics

### Mini Datasets Overview
- **Total**: 11 mini datasets (R1-R11)
- **Subjects per dataset**: 20
- **Files per subject**: 220-240 .set files
- **No overlap**: Subjects are completely disjoint across datasets

### Subject Distribution Findings
- Complete subjects analysis: No subjects appear in multiple datasets
- Each dataset contains unique participants
- Participants.tsv contains target labels for both challenges

### Problematic Subjects (to exclude)
```python
sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", 
          "NDARJP304NK1", "NDARTY128YLU", "NDARDW550GU6", 
          "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]
```

## Data Access Paths

### AWS S3 Bucket
- Public bucket: `s3://nmdatasets/NeurIPS25/`
- No credentials needed: `--no-sign-request`
- Available formats:
  - `R*_mini_L100/` - Plain mini (EEGLAB .set)
  - `R*_mini_L100_bdf/` - BDF mini
  - `R*_L100_bdf/` - Full BDF datasets

### Local Directory Structure
```
eeg_mini_datasets/
├── R1_mini_L100/
│   ├── sub-NDAR*/
│   │   ├── eeg/
│   │   │   ├── *_eeg.set
│   │   │   ├── *_eeg.json
│   │   │   ├── *_channels.tsv
│   │   │   └── *_events.tsv
│   │   └── phenotype/
│   └── participants.tsv
└── ... (R2-R11 similar structure)
```

## Important Libraries

### Core Dependencies
- **braindecode**: Deep learning for EEG (main framework)
- **eegdash** (>=0.3.8): EEG data loading and challenge API
- **MNE-Python**: EEG/MEG analysis (via braindecode)
- **PyTorch**: Deep learning backend
- **Lightning**: Training framework (optional)

### Additional Tools
- **MOABB**: Mother of All BCI Benchmarks
- **scikit-learn**: Model evaluation and splitting
- **joblib**: Parallel processing

## Key Findings and Notes

### Data Quality Considerations
- Filter recordings with <4 seconds duration
- Ensure 129 channels present
- Check for NaN p_factor values (Challenge 2)
- Some subjects have incomplete task sets

### Performance Baselines
- Challenge 1: EEGNeX model baseline provided
- Challenge 2: Single epoch training example
- Both use MSE/L1 loss for regression

### Training Recommendations
- Batch size: 128 (adjustable based on GPU memory)
- Learning rate: 1e-3 to 2e-3
- Optimizer: AdamW or Adamax
- Scheduler: CosineAnnealingLR
- Early stopping patience: 5-50 epochs

## References and Resources

### Official Documentation
- [Competition Website](https://eeg2025.github.io)
- [EEGDash Documentation](https://eeglab.org/EEGDash/overview.html)
- [Braindecode Models](https://braindecode.org/stable/models/models_table.html)
- [Dataset Download Guide](https://eeg2025.github.io/data/#downloading-the-data)

### Key Papers
- [Aristimunha et al., 2023](https://arxiv.org/abs/2308.02408) - Transfer learning in EEG
- [Wimpff et al., 2025](https://arxiv.org/abs/2502.06828) - Cross-task applications
- [Wu et al., 2025](https://arxiv.org/abs/2507.09882) - Transfer learning methods
- [McElroy et al., 2017](https://doi.org/10.1111/jcpp.12849) - P-factor assessment with CBCL

## TODO: Information to Research
- [ ] Optimal preprocessing pipelines for each task
- [ ] Best practices for cross-task transfer in EEG
- [ ] Self-supervised learning approaches for EEG
- [ ] Detailed CBCL score interpretation
- [ ] Hardware optimization for 20GB GPU constraint