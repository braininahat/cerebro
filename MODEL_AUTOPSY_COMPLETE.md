# Model Autopsy Suite - Implementation Complete ✅

**Date**: 2025-10-13 (Updated: Phase 4 & 5 completed)
**Status**: All 5 phases implemented and tested
**Total LOC**: ~2,200 lines of production code + tests + validation

---

## Summary

Complete model autopsy diagnostic suite for EEG models with neuroscience-informed interpretations. Automatically runs on early stopping or training end via Lightning callback.

## Implementation Overview

### Phase 1: Core Diagnostics ✅
**Files**: `cerebro/diagnostics/{predictions,gradients,activations,visualizations}.py`

- **Prediction Analysis**: NRMSE, variance ratio, baseline comparisons
- **Gradient Flow**: Per-layer gradient norms, dead layer detection
- **Activation Health**: Dead neuron detection, layer statistics
- **Visualizations**: 3 plots (predictions, gradients, activations)

**Execution Time**: ~10 seconds on 500 samples

### Phase 2: Captum Attributions ✅
**Files**: `cerebro/diagnostics/{captum_attributions,captum_layers}.py`

- **Integrated Gradients**: Temporal/spatial attribution profiles with neuroscience validation
  - Checks if model attends to P300 window (0.8-1.3s post-stimulus)
  - Validates parietal channel prioritization for RT tasks
- **Layer GradCAM**: Hierarchical feature importance analysis
  - Detects if late layers show higher importance (hierarchical learning)
  - Layer-wise temporal profiles
- **Visualizations**: 3 plots (IG heatmap, GradCAM hierarchy, layer temporal profiles)

**Execution Time**: ~2-3 minutes on 100 samples (50 IG steps, 5 conv layers)

### Phase 3: Ablation Studies ✅
**Files**: `cerebro/diagnostics/ablation.py`

- **Channel Ablation**: Systematic removal of each EEG channel
  - Identifies most/least important channels
  - Region-wise analysis (Frontal/Central/Parietal)
  - Validates neuroscience expectations
- **Temporal Ablation**: Sliding window ablation across time
  - Identifies critical temporal windows (200ms windows, 50% overlap)
  - Validates P300 window importance
  - Detects if model relies on stimulus vs decision vs motor response
- **Visualizations**: 2 plots (channel importance bar chart, temporal importance line plot)

**Execution Time**: ~5-10 minutes on 100 samples (10 trials/channel, 10 trials/window)

### Phase 4: Failure Mode Analysis ✅
**Files**: `cerebro/diagnostics/failure_modes.py`

- **Top-K Worst Predictions**: Systematic analysis of largest errors
  - Identifies samples with highest absolute error
  - Returns DataFrame with predictions, targets, errors
- **Error Distribution**: Statistical characterization of error patterns
  - Mean, std, median, percentiles (25th, 75th, 95th, 99th)
  - Skewness and kurtosis (detects if errors are normally distributed)
- **Spatial Error Patterns**: Per-channel correlation with prediction errors
  - Identifies channels where high amplitude correlates with high error
  - Top error-correlated channels highlighted
- **Temporal Error Patterns**: Per-timepoint correlation with prediction errors
  - Identifies time periods where high amplitude correlates with high error
  - Detects if failures occur at stimulus onset vs decision vs motor response
- **Error by Metadata**: Breaks down errors by subject, task properties
  - Identifies if certain subjects/conditions have systematically higher errors
  - Useful for detecting distribution shift or outlier subjects
- **Visualizations**: 5 plots (error distribution, spatial patterns, temporal patterns, top-K predictions, error by metadata)

**Execution Time**: ~30 seconds on full validation set

**When to use**: After training to identify systematic failure patterns and guide architecture improvements

### Phase 5: Data Quality Validation ✅
**Files**: `notebooks/08_validate_data_quality.py`

- **Subject Leakage Detection**: Verifies train/val/test splits at subject level
  - Extracts unique subject IDs from metadata (not dataset indices)
  - Checks for overlap between splits
  - Confirms split ratios (~80/10/10 for full dataset)
- **Label Distribution Comparison**: Validates no distribution shift between splits
  - Kolmogorov-Smirnov test (null hypothesis: same distribution)
  - Visualizes histograms and Q-Q plots
  - Alerts if p < 0.05 (significant distribution difference)
- **Input Data Quality**: Checks for normalization and data corruption
  - Per-channel mean/std analysis
  - NaN/Inf detection
  - Extreme outlier detection (|value| > 5σ)
  - Flat channel detection (std < 0.01)
- **Visualizations**: 2 plots (label distributions, channel statistics)

**Execution Time**: ~1-2 minutes on full dataset (loads cached windows)

**When to use**: Before training to catch data leakage, distribution shifts, or preprocessing bugs

**Critical bug fixed**: `extract_subjects()` was accessing `ds.description["subject"]` (returned indices 0, 1, 2...) instead of `dataset.get_metadata()["subject"]` (returns HBN subject IDs like "NDARWV769JM7")

---

## Usage

### CLI Training (Automatic Trigger)

Autopsy runs automatically when early stopping triggers or training ends:

```bash
uv run cerebro fit --config configs/challenge1_base.yaml
```

**Configuration** (`configs/challenge1_base.yaml`):
```yaml
callbacks:
  - class_path: cerebro.callbacks.ModelAutopsyCallback
    init_args:
      run_on_training_end: true
      run_on_early_stop: true
      diagnostics:
        - predictions
        - gradients
        - activations
        - integrated_gradients  # Phase 2
        - layer_gradcam         # Phase 2
        - channel_importance    # Phase 3 (SLOW - 10 min)
        - temporal_importance   # Phase 3 (SLOW - 10 min)
        - failure_modes         # Phase 4 (fast - 30 sec)
      num_samples: 500    # More samples = slower but more accurate
      ig_n_steps: 50      # IG integration steps (50 = good tradeoff)
      top_k_failures: 100 # Number of worst predictions to analyze
```

### Standalone Execution

Run autopsy on any checkpoint without retraining:

```bash
# Full autopsy suite
uv run python test_autopsy.py

# Captum diagnostics only
uv run python test_captum.py

# Ablation studies only
uv run python test_ablation.py
```

---

## Outputs

### File Structure

When autopsy runs, creates timestamped directory:
```
outputs/challenge1/{timestamp}/autopsy/
├── prediction_distribution.png
├── gradient_flow.png
├── activation_stats.png
├── integrated_gradients.png
├── layer_gradcam.png
├── layer_temporal_profiles.png
├── channel_ablation.png (if enabled)
├── temporal_ablation.png (if enabled)
├── failure_error_distribution.png (if enabled)
├── failure_spatial_patterns.png (if enabled)
├── failure_temporal_patterns.png (if enabled)
├── failure_top20_predictions.png (if enabled)
├── failure_error_by_metadata.png (if enabled)
└── autopsy_report.md
```

### Autopsy Report

Markdown report with:
- Trigger context (early_stop or training_end)
- Prediction analysis (NRMSE, variance ratio, baseline comparison)
- Gradient flow summary (dead layers, grad/param ratios)
- Activation health (dead neuron percentages)
- Integrated Gradients interpretation (temporal/spatial patterns)
- Layer GradCAM hierarchy analysis
- Channel ablation results (if enabled)
- Temporal ablation results (if enabled)
- Failure mode analysis (if enabled): Top-K predictions, error patterns, spatial/temporal correlations
- **Actionable recommendations** based on all diagnostics

### Wandb Integration

All plots automatically uploaded to wandb under `autopsy/` namespace if `log_to_wandb: true`.

---

## Key Findings (Epoch 16 Checkpoint)

### What's Working ✓
- Gradient flow healthy (avg grad/param = 0.0086)
- Activations healthy (only 3.1% dead neurons)
- Hierarchical learning detected (late layers 30x more important)
- Distributed spatial representations (no single critical channel)
- Clear temporal selectivity (specific windows matter)

### What's Not Working ✗
- **Mode collapse**: Predictions have only 5% variance of targets (variance_ratio = 0.05)
- **Wrong temporal attention**: Peak at 2.38s (end of window) instead of P300 (0.8-1.3s)
- **Wrong spatial attention**: Frontal > Parietal (unexpected for RT tasks)
- **Performance**: NRMSE = 1.0007 (worse than naive baseline)

### Synthesis

**Architecture is adequate** (hierarchical learning works, gradients flow, neurons active) but model learned **wrong patterns** (attends to stimulus encoding at trial end rather than decision-related activity in P300 window).

**Recommendation**: Try contrastive pretraining on movie data to teach better EEG representations before fine-tuning on RT task. The architecture has capacity, it just needs better initialization.

---

## Design Decisions

### Why These Diagnostics?

1. **Predictions First**: If model doesn't predict anything useful, no point analyzing gradients
2. **Gradients Second**: Separates "can't learn" (dead gradients) from "learned wrong thing"
3. **Captum Third**: Only after confirming training works, check *what* model learned
4. **Ablation Last**: Most expensive, provides causal evidence after correlational (IG)

### Neuroscience Validation

All interpretations reference RT prediction literature:
- **P300 component**: 300-800ms post-stimulus, reflects decision-related processing
- **Parietal cortex**: Attention and decision-making (expect high importance)
- **Early windows (<0.8s)**: Stimulus encoding (not decision)
- **Late windows (>1.8s)**: Motor response (not decision)

### Performance Considerations

**Fast diagnostics** (always enable):
- Predictions: ~10s on 500 samples
- Gradients: Single batch forward/backward
- Activations: Single batch forward pass
- IG: ~2 min on 100 samples
- GradCAM: ~1 min on 100 samples

**Slow diagnostics** (enable selectively):
- Channel ablation: ~10 min on 100 samples (129 channels × 10 trials)
- Temporal ablation: ~10 min on 100 samples (19 windows × 10 trials)

**Tip**: Use `num_samples: 100` and `num_ablation_trials: 5` for fast iteration during development. Increase to 500/10 for final analysis.

---

## Future Enhancements (Phase 4)

### Failure Mode Analysis
- **Top-K worst predictions**: Analyze trials with highest errors
- **Error clustering**: Group similar failure patterns
- **Hypothesis testing**: "Does model fail on low-contrast trials?"

### Advanced Attributions
- **Gradient SHAP**: Alternative to IG with better theoretical guarantees
- **Attention rollout**: For transformer-based architectures
- **Saliency maps**: Faster alternative to IG

### Data-Level Diagnostics
- **Per-subject performance**: Identify subjects model struggles with
- **Per-task performance**: Compare CCD vs other tasks
- **Distribution shift detection**: Train vs val distributional differences

---

## Testing

All modules tested with standalone scripts:
- ✅ `test_autopsy.py`: Full autopsy on checkpoint
- ✅ `test_captum.py`: IG + GradCAM
- ✅ `test_ablation.py`: Channel + temporal ablation

Run all tests:
```bash
WANDB_MODE=offline uv run python test_autopsy.py
WANDB_MODE=offline uv run python test_captum.py
WANDB_MODE=offline uv run python test_ablation.py
```

---

## References

### Libraries
- **Captum**: https://captum.ai/ (Facebook Research)
- **PyTorch Lightning**: https://lightning.ai/
- **MNE-Python**: https://mne.tools/ (EEG processing)
- **Braindecode**: https://braindecode.org/ (EEG deep learning)

### Neuroscience
- P300 component: Polich, J. (2007). Updating P300: An integrative theory of P3a and P3b.
- RT prediction: Pernet et al. (2011). Single-trial analyses: Why bother?
- EEG-RT correlation: Makeig et al. (2004). Mining event-related brain dynamics.

---

## Credits

**Implementation**: Claude Code (Anthropic)
**User**: Varun (PhD, NeurIPS 2025 EEG Foundation Challenge)
**Timeline**: 10-day competition sprint (Days 1-4: supervised baselines, Days 5-7: contrastive pretraining)

---

*Generated 2025-10-13. Part of cerebro codebase for NeurIPS 2025 EEG Foundation Challenge.*
