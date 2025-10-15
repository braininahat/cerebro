# CLAUDE_REFERENCE.md

Detailed examples and explanations extracted from original CLAUDE.md.
Load this file only when you need verbose examples or detailed explanations.

## Memory Search Examples

### Progressive Refinement Pattern
```python
# Instead of one complex query:
search_nodes("Challenge 1 windowing stimulus locked preprocessing")  # ❌ Returns nothing

# Use progressive refinement:
# Step 1: Start broad
search_nodes("Challenge 1 windowing")
# → Find: Challenge 1 entity, windowing strategies, mention of stimulus_anchor

# Step 2: Drill down on specific detail
search_nodes("stimulus anchor")
# → Find: add_aux_anchors function, create_windows_from_events usage

# Step 3: Look up exact function
search_nodes("annotate_trials_with_target")
# → Find: Function signature, parameters, pipeline position
```

### Good vs Bad Observation Examples

**Good (atomic facts with details):**
```
✅ "Loss function: MSE (Mean Squared Error)"
✅ "Optimizer: AdamW(lr=0.001, weight_decay=0.00001, betas=[0.9,0.999])"
✅ "Input shape: (batch_size, n_chans, n_times) where n_chans=129, n_times=200"
✅ "Pipeline: annotate → filter → window → label (from startkit lines 144-193)"
```

**Bad (compound facts - should split):**
```
❌ "Uses AdamW optimizer and CosineAnnealingLR scheduler with T_max=epochs"
❌ "Data scale: startkit uses R5 mini, production uses 10 releases"
❌ "Production uses FP16 mixed precision which is faster with negligible accuracy impact"
```

## Submission.py Full Template

```python
from pathlib import Path
import torch

def resolve_path(name="model_file_name"):
    """Handle Codabench mount points."""
    paths = [
        f"/app/input/res/{name}",
        f"/app/input/{name}",
        f"{name}",
        str(Path(__file__).parent.joinpath(f"{name}"))
    ]
    for path in paths:
        if Path(path).exists():
            return path
    raise FileNotFoundError(f"Could not find {name}")

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        return torch.jit.load(
            resolve_path("model_challenge_1.pt"),
            map_location=self.device
        )

    def get_model_challenge_2(self):
        return torch.jit.load(
            resolve_path("model_challenge_2.pt"),
            map_location=self.device
        )
```

## TorchScript Conversion Details

```python
# 1. Load Lightning module
from cerebro.models.challenge1 import Challenge1Module
pl_module = Challenge1Module.load_from_checkpoint("best.ckpt")
model = pl_module.model  # Extract raw PyTorch model

# 2. Convert to TorchScript
model.eval()  # CRITICAL: Must be in eval mode
dummy_input = torch.zeros(1, 129, 200)  # (batch, channels, time)
scripted_model = torch.jit.trace(model, dummy_input)

# 3. Optimize for inference
scripted_model = torch.jit.optimize_for_inference(scripted_model)

# 4. Save
scripted_model.save("model_challenge_1.pt")

# 5. Test before submission
model = torch.jit.load("model_challenge_1.pt")
model.eval()
dummy_input = torch.zeros(16, 129, 200)
with torch.inference_mode():
    output = model(dummy_input)
assert output.shape == (16, 1), f"Expected (16, 1), got {output.shape}"
```

## Complete Entity Types List

### Common (use these)
- project, competition, task, model, function, class, experiment
- dataset, optimizer, loss_function, metric, scheduler

### Specialized (rarely needed)
- architecture, eeg_task, api, library, tool, code_module
- configuration, reference_code, script, directory, timeline_phase
- person, framework, infrastructure, training_strategy

## Complete Relation Types List

### Common (use these)
- uses, implements, implemented_in, part_of, contains
- solves, requires, required_for, provides, configures, from_library

### Specialized (rarely needed)
- pretrains, adapts_from, replicates, alternative_to, analyzes
- targets, evaluates, loads, loaded_by, prototypes_for, manages
- submits_to, provided_by, developed_by, follows, inspired
- based_on, loss_function_for, calculates, framework_for
- baseline_for, phase_of, part_of_phase, explored_in_phase

## AI Time Estimation Examples

| Task | Human Time | AI Time | Speedup |
|------|------------|---------|---------|
| Memory audit (62 entities) | 2-3 hours | 5-6 min | 30x |
| Refactor 10 configs | 30-45 min | 3-4 min | 10x |
| Search + fix 5 bugs | 2-4 hours | 8-10 min | 20x |
| Read 20 files + summarize | 1-2 hours | 2-3 min | 40x |

## Context7 Detailed Examples

### EEGNeX Parameters
```python
mcp__context7__resolve_library_id(libraryName="braindecode")
# Returns: {"library_id": "/braindecode/braindecode", ...}

mcp__context7__get_library_docs(
    context7CompatibleLibraryID="/braindecode/braindecode",
    topic="EEGNeX model initialization forward parameters",
    tokens=3000
)

# Capture findings:
mcp__memory__add_observations(observations=[{
    "entityName": "EEGNeX",
    "contents": [
        "Input shape: (batch_size, n_chans, n_times) - e.g. (128, 129, 200)",
        "n_chans: number of EEG channels (129 for HBN)",
        "n_times: window length in samples (200 = 2s at 100Hz)",
        "sfreq: sampling frequency for temporal convolutions (100 Hz)",
        "n_outputs: 1 for regression (RT or p_factor)",
        "Returns: (batch_size, n_outputs) tensor"
    ]
}])
```

### AdamW vs Adam
```python
mcp__context7__get_library_docs(
    context7CompatibleLibraryID="/pytorch/pytorch",
    topic="AdamW optimizer weight decay decoupled difference from Adam",
    tokens=2000
)
# Key insight: AdamW applies weight decay directly to weights, not gradients
```

## Startkit Pipeline References

### Challenge 1 (lines 144-193)
```python
# 1. Annotate trials with RT
annotate_trials_with_target(
    target_field='rt_from_stimulus',
    epoch_length=2.0
)

# 2. Add auxiliary anchors
add_aux_anchors(
    raw,
    annotations,
    targets,
    stimulus_anchor='stimulus',
    aux_anchors={'response': 'response'}
)

# 3. Create windows
create_windows_from_events(
    raw,
    events=annotations_df,
    event_id={'Stimulus/Correct': 0},
    preload=True,
    window_size=(0, int(2 * raw.info['sfreq'])),
    window_stride=(int(2 * raw.info['sfreq']), 0),
    baseline=None
)
```

### Challenge 2 (lines 231-259)
```python
# 1. Filter by minimum length
raw.n_times / raw.info['sfreq'] >= 4

# 2. Create fixed windows
create_fixed_length_windows(
    raw,
    start_offset_samples=0,
    stop_offset_samples=0,
    window_size_samples=int(4 * raw.info['sfreq']),
    window_stride_samples=int(2 * raw.info['sfreq']),
    drop_last_window=False,
    preload=True
)

# 3. Wrap with DatasetWrapper for cropping
DatasetWrapper(
    dataset=dataset,
    n_predictions_per_input=1,
    crop_size=int(2 * raw.info['sfreq'])  # 2s crops from 4s windows
)
```

## Why Patterns (removed from main doc)

These explanations were removed as non-actionable, but preserved here for reference:

- **Memory-first matters**: No repeated work, faster debugging, design consistency
- **10-day timeline pressure**: Every minute counts, can't afford to re-investigate
- **AI speed advantage**: Parallel operations, no context switching, instant recall
- **Submission complexity**: TorchScript needed for custom dependencies not in Codabench
- **Subject-level splits**: Prevent memorization of subject-specific patterns

---
**This reference file contains verbose examples and explanations. Load only when needed.**