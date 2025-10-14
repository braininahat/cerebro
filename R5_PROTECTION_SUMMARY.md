# R5 Protection and Data Validation Summary

**Date**: October 13, 2025
**Context**: Comprehensive codebase audit to prevent training on R5 (competition validation set) and ensure proper subject-level data splitting.

## Critical Issues Found and Fixed

### Issue 1: `challenge1_mini.yaml` Used R5

**Problem**:
- `configs/challenge1_mini.yaml` line 81 specified `releases: [R5]`
- Anyone using mini config would contaminate competition validation set
- **Severity**: HIGH - Direct violation of competition rules

**Fix**:
```yaml
# Before
releases:
  - R5

# After
releases:
  - R1  # Mini prototyping with R1 (NOT R5!)
```

Also changed `data_dir` from `data/mini` to `data/full` since we're now using R1.

**Impact**: Mini config now safe for prototyping without R5 contamination.

---

### Issue 2: Missing R5 Guard in DataModule

**Problem**:
- No enforcement mechanism to prevent R5 usage
- Relied on trust-based system (config discipline only)
- **Severity**: MEDIUM - Could catch accidental mistakes

**Fix** (`cerebro/data/challenge1.py` lines 95-100):
```python
def __init__(self, data_dir: str, releases: List[str], ...):
    super().__init__()
    self.save_hyperparameters()

    # R5 guard: Prevent training on competition validation set
    if "R5" in releases:
        raise ValueError(
            "R5 is the COMPETITION VALIDATION SET and must NEVER be used for training! "
            f"Got releases={releases}. Use releases from [R1, R2, R3, R4, R6, R7, R8, R9, R10, R11] only."
        )
```

**Impact**: Runtime error if R5 accidentally included in config.

---

### Issue 3: Validation Notebook Had Buggy Subject Extraction

**Problem**:
- `notebooks/08_validate_data_quality.py` function `extract_subjects()` was accessing `ds.description["subject"]`
- This returned dataset indices (0, 1, 2...) instead of HBN subject IDs (NDARXX...)
- Caused false positive leakage detection (33 overlapping subjects)
- **Severity**: HIGH - Misleading validation results

**Fix**:
```python
# Before (BUGGY)
def extract_subjects(dataset):
    subjects = []
    if hasattr(dataset, 'datasets'):
        for ds in dataset.datasets:
            if hasattr(ds, 'description') and 'subject' in ds.description:
                subjects.extend(ds.description["subject"])  # Returns indices!
    return subjects

# After (CORRECT)
def extract_subjects(dataset):
    """Extract subject IDs from BaseConcatDataset.

    Uses get_metadata() to properly extract HBN subject IDs (e.g., "NDARWV769JM7")
    from the metadata DataFrame, not dataset indices.
    """
    if hasattr(dataset, 'get_metadata'):
        metadata = dataset.get_metadata()
        if 'subject' in metadata.columns:
            return metadata["subject"].unique().tolist()
    return []
```

**Impact**: Now correctly extracts subject IDs and validates no leakage (1360/170/170 split verified).

---

### Issue 4: Validation Notebook Had Buggy Label Extraction

**Problem**:
- `notebooks/08_validate_data_quality.py` function `extract_labels()` was trying to access `ds.description["target"]`
- Challenge 1 data stores labels in metadata (via `get_metadata()`), not description
- Resulted in 0 labels extracted (empty arrays)
- **Severity**: MEDIUM - Validation incomplete

**Fix**:
```python
# Before (BUGGY)
def extract_labels(dataset):
    """Extract RT labels from WindowsDataset or ConcatDataset"""
    labels = []
    if hasattr(dataset, 'datasets'):
        for ds in dataset.datasets:
            if hasattr(ds, 'description') and 'target' in ds.description:
                labels.extend(ds.description["target"])  # Doesn't exist!
    return np.array(labels)

# After (CORRECT)
def extract_labels(dataset):
    """Extract RT labels from BaseConcatDataset.

    Uses get_metadata() to extract 'target' column (response time labels)
    from the metadata DataFrame.
    """
    if hasattr(dataset, 'get_metadata'):
        metadata = dataset.get_metadata()
        if 'target' in metadata.columns:
            return metadata["target"].values
    return np.array([])
```

**Impact**: Now correctly extracts 80,889 train and 10,193 val labels, validates distribution similarity (KS p=0.21).

---

### Issue 5: Insufficient R5 Warnings in Configs

**Problem**:
- Config files had release lists without prominent warnings
- Easy to accidentally include R5
- **Severity**: LOW - Documentation gap

**Fix** (both `challenge1_base.yaml` and `challenge1_mini.yaml`):
```yaml
data:
  data_dir: ${oc.env:EEG2025_DATA_ROOT,data/full}
  # WARNING: R5 is the COMPETITION VALIDATION SET - NEVER include in training!
  # Training releases: R1-R4, R6-R11 only (10 releases total)
  # R5 is held out for competition leaderboard evaluation
  releases:
    - R1
    - R2
    ...
```

**Impact**: Clear documentation prevents accidental R5 inclusion.

---

## Verification Tests

### Test 1: R5 Guard Enforcement

**Test Script**: `test_r5_guard.py`

**Results**:
```
✓ Test 1: Creating datamodule with R1, R2, R3 (should work)
  ✓ PASS: Datamodule created successfully

✓ Test 2: Creating datamodule with R5 (should fail)
  ✓ PASS: Correctly raised ValueError

✓ Test 3: Creating datamodule with only R5 (should fail)
  ✓ PASS: Correctly raised ValueError

✅ ALL TESTS PASSED
```

**Conclusion**: R5 guard working correctly at DataModule level.

---

### Test 2: Data Quality Validation

**Test Script**: `notebooks/08_validate_data_quality.py`

**Results**:
```
============================================================
DATA QUALITY VALIDATION
============================================================

📊 Loading data...
✓ Train windows: 80889
✓ Val windows: 10193
✓ Test windows: 9690

============================================================
1. SUBJECT LEAKAGE DETECTION
============================================================
Train subjects: 1360
Val subjects: 170
Test subjects: 170

Checking for overlap...
✓ No train-val overlap detected
✓ No train-test or val-test overlap detected

✅ Subject leakage check PASSED

============================================================
2. LABEL DISTRIBUTION COMPARISON
============================================================
Train labels: 80889 samples
Val labels: 10193 samples

Train label stats: mean=1.597s, std=0.410s
Val label stats:   mean=1.587s, std=0.425s
KS test statistic: 0.0111
KS test p-value: 0.2143
✓ Train and val distributions are similar (p ≥ 0.05)

============================================================
VALIDATION SUMMARY
============================================================

Checklist:
  ✓ Subject Leakage: PASS
  ✓ Label Distribution: PASS
  ✓ No Nan Inf: PASS
  ✓ Normalization Mean: PASS
  ⚠️  Normalization Std: WARN (expected for raw EEG)
  ⚠️  Flat Channels: WARN (expected for raw EEG)
  ✓ Extreme Outliers: PASS

⚠️  SOME WARNINGS DETECTED (but critical checks passed)
```

**Conclusion**:
- ✅ No subject leakage (proper subject-level splitting verified)
- ✅ Label distributions similar (no distribution shift)
- ⚠️ Raw EEG data warnings expected (not standardized, microvolts range)

---

### Test 3: Mini Config Training

**Command**: `WANDB_MODE=offline uv run cerebro fit --config configs/challenge1_mini.yaml --trainer.fast_dev_run 1`

**Results**:
- ✅ Loaded R1 (not R5)
- ✅ Loaded 60 recordings from R1
- ✅ Created 1513 windows
- ✅ Split correctly: 16 train / 2 val / 2 test subjects
- ✅ Training completed successfully
- ✅ Cache file: `windows_R1_shift5_len20_miniTrue.pkl`

**Conclusion**: Mini config safe for prototyping with R1.

---

## Key Validation Findings

### ✅ Challenge1DataModule Splits Correctly

**Lines 264-310** in `cerebro/data/challenge1.py`:
```python
def _create_splits(self, single_windows, metadata):
    """Create subject-level train/val/test splits."""
    subjects = metadata["subject"].unique()
    subjects = [s for s in subjects if s not in self.hparams.excluded_subjects]

    # Split: train / (val + test)
    train_subj, valid_test_subj = train_test_split(
        subjects,  # ← Subject IDs, not windows!
        test_size=(self.hparams.val_frac + self.hparams.test_frac),
        random_state=check_random_state(self.hparams.seed),
        shuffle=True
    )

    # Split: val / test
    valid_subj, test_subj = train_test_split(
        valid_test_subj,
        test_size=self.hparams.test_frac / (self.hparams.val_frac + self.hparams.test_frac),
        random_state=check_random_state(self.hparams.seed + 1),
        shuffle=True
    )

    # Create splits from subject-level splits
    subject_split = single_windows.split("subject")
    self.train_set = BaseConcatDataset(
        [subject_split[s] for s in train_subj if s in subject_split]
    )
    self.val_set = BaseConcatDataset(
        [subject_split[s] for s in valid_subj if s in subject_split]
    )
    self.test_set = BaseConcatDataset(
        [subject_split[s] for s in test_subj if s in subject_split]
    )
```

**Confirmation**: Uses sklearn's `train_test_split` on **unique subject IDs**, not windows. No window-level or recording-level leakage possible.

---

## Files Modified

1. ✅ `configs/challenge1_mini.yaml` - Changed R5 → R1, added warnings
2. ✅ `configs/challenge1_base.yaml` - Added R5 warnings
3. ✅ `cerebro/data/challenge1.py` - Added R5 guard assertion
4. ✅ `notebooks/08_validate_data_quality.py` - Fixed subject and label extraction

## Files Created

1. ✅ `test_r5_guard.py` - R5 guard test suite
2. ✅ `R5_PROTECTION_SUMMARY.md` - This document

---

## Recommendations

### For Development (Days 1-9)

**Use subject-level train/val splits from R1-R4, R6-R11**:
- Train: 80% of subjects
- Val: 10% of subjects
- Test (local): 10% of subjects

**Check R5 sparingly**:
- Only for major architecture decisions
- Avoid overfitting to leaderboard signal

### For Final Submission (Day 10)

**Train on ALL subjects from R1-R4, R6-R11**:
- No validation split (all data for training)
- Architecture and hyperparameters locked
- Submit predictions on R5 for leaderboard

### Pre-Training Checklist

Before every training run:
1. ✅ Run `notebooks/08_validate_data_quality.py` to verify no leakage
2. ✅ Check config file: `releases` should NOT include R5
3. ✅ Verify subject-level splits: ~1360/170/170 for full dataset
4. ✅ Confirm label distributions similar (KS p > 0.05)

---

## Competition Data Hygiene

### R5 (ds005509-bdf) Status

- **Purpose**: Competition validation set (provides leaderboard feedback)
- **Usage**: NEVER for training
- **Protection**: ValueError in Challenge1DataModule.__init__()
- **Documentation**: Warnings in all config files, CLAUDE.md

### Training Releases

**Available for training**: R1, R2, R3, R4, R6, R7, R8, R9, R10, R11 (10 releases)

**Release mapping**:
- R1: ds005505-bdf
- R2: ds005506-bdf
- R3: ds005507-bdf
- R4: ds005508-bdf
- **R5: ds005509-bdf ← COMPETITION VALIDATION SET (DO NOT USE)**
- R6: ds005510-bdf
- R7: ds005511-bdf
- R8: ds005512-bdf
- R9: ds005514-bdf (note: ds005513 does not exist)
- R10: ds005515-bdf
- R11: ds005516-bdf

---

## Summary

**All critical R5 protection and data validation issues resolved**:

1. ✅ R5 removed from mini config (now uses R1)
2. ✅ R5 guard added to Challenge1DataModule (raises ValueError)
3. ✅ Validation notebook fixed (correct subject and label extraction)
4. ✅ Subject leakage verified: 1360/170/170 split, no overlap
5. ✅ Label distributions verified: KS p=0.21 (no distribution shift)
6. ✅ R5 warnings added to all config files

**Codebase is now protected against R5 contamination** at multiple levels:
- **Config level**: Warnings in all training configs
- **Runtime level**: ValueError guard in DataModule
- **Validation level**: Pre-training data quality checks

**Safe to proceed with training** using R1-R4, R6-R11 with subject-level splits.
