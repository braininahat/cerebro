# Captum Integration Issue

**Status**: Unresolved
**Date**: 2025-10-13
**Components**: `cerebro/diagnostics/captum_attributions.py`, `cerebro/diagnostics/captum_layers.py`

## Problem Summary

Captum's `IntegratedGradients` and `LayerGradCam` fail when used with regression models that output shape `(batch, 1)`. The error occurs during Captum's internal gradient computation:

```python
IndexError: index 0 is out of bounds for dimension 0 with size 0
  File "captum/_utils/gradient.py", line 133, in compute_gradients
    assert outputs[0].numel() == 1, (
           ~~~~~~~^^^
```

## Confirmed Working

- **Model forward pass**: Confirmed working independently - outputs correct shape `(batch, 1)`
- **Test case**: Direct model call produces valid outputs:
  ```python
  test_output = model(test_input)  # Shape: torch.Size([2, 1])
  ```
- **Other diagnostics**: predictions, gradients, and activations diagnostics all work correctly

## Root Cause

Captum expects scalar outputs (shape `[batch]`) for regression tasks, but our models output `(batch, 1)`. During Captum's internal gradient computation, something causes the output tensor to become empty `(size 0)`, triggering the IndexError.

## Attempted Solutions

### 1. ✗ Using `target` parameter
**Tried**: Adding `target=0` to `ig.attribute()` calls
```python
batch_attr = ig.attribute(
    batch_samples,
    baselines=batch_baseline,
    target=0,  # Specify output dimension 0
    n_steps=n_steps,
)
```
**Result**: Same error - outputs still become empty

### 2. ✗ Lambda forward function
**Tried**: Wrapping model with lambda function that squeezes output
```python
def forward_func(x):
    out = model(x)
    return out.squeeze(-1) if out.dim() == 2 and out.shape[-1] == 1 else out

ig = IntegratedGradients(forward_func)
```
**Result**: Captum requires proper `nn.Module`, not callable

### 3. ✗ Model wrapper class (current approach)
**Tried**: Creating proper `nn.Module` wrapper class
```python
class SqueezeOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if out.dim() == 2 and out.shape[-1] == 1:
            return out.squeeze(-1)
        return out

wrapped_model = SqueezeOutputWrapper(model).to(device)
ig = IntegratedGradients(wrapped_model)
```
**Result**: Same error - something in Captum's gradient computation still produces empty outputs

### 4. ✗ Removing target parameter entirely
**Tried**: Let Captum auto-detect target
```python
batch_attr = ig.attribute(
    batch_samples,
    baselines=batch_baseline,
    n_steps=n_steps,
    # No target specified
)
```
**Result**: Same error

## Debugging Evidence

```python
# This works fine:
DEBUG: Model test output shape: torch.Size([2, 1])
DEBUG: Model test output: tensor([[1.1086], [1.1072]], device='cuda:0')

# But Captum's gradient computation fails:
# outputs[0] is somehow empty inside Captum
```

## Code Locations

**Affected files**:
- `cerebro/diagnostics/captum_attributions.py` - IntegratedGradients implementation
- `cerebro/diagnostics/captum_layers.py` - LayerGradCam implementation
- `cerebro/callbacks/model_autopsy.py:240` - Where IG is called
- `cerebro/callbacks/model_autopsy.py:281` - Where GradCAM is called

**Error originates from**:
- `captum/_utils/gradient.py:133` - Internal Captum gradient computation

## Hypotheses (Untested)

1. **Lightning Module interaction**: The model might be wrapped in Lightning's module structure in a way that confuses Captum's internal hooks
2. **Checkpoint loading issue**: When reloading from checkpoint, some state might not be properly restored for Captum
3. **Autograd graph issue**: The wrapper might be breaking Captum's autograd graph traversal
4. **Device mismatch**: Despite `.to(device)` calls, there might be a subtle device mismatch in Captum's internal operations

## Potential Solutions to Try

### Option A: Use Captum with Lightning Module directly
Try passing `pl_module.model` instead of reloading from checkpoint:
```python
# In model_autopsy.py, instead of:
pl_module = pl_module.__class__.load_from_checkpoint(best_ckpt)

# Try using the existing Lightning module's inner model:
model = pl_module.model  # Get the raw torch model
```

### Option B: Manual attribution implementation
Implement custom Integrated Gradients without Captum:
- Manually interpolate between baseline and input
- Compute gradients at each interpolation step
- Aggregate gradients along path

### Option C: Different Captum attribution method
Try simpler attribution methods that might not have this issue:
- `Saliency` - Just input gradients (no path integration)
- `DeepLift` - Reference-based attribution
- `GradientShap` - SHAP-based attribution

### Option D: Reshape model output
Modify the Challenge1Module itself to output shape `(batch,)` instead of `(batch, 1)` for regression

### Option E: Debug Captum internals
Add hooks to understand where outputs become empty:
```python
def hook_fn(module, input, output):
    print(f"Layer {module.__class__.__name__}: output shape {output.shape}")

for name, layer in wrapped_model.named_modules():
    layer.register_forward_hook(hook_fn)
```

## Workaround (Current)

Captum diagnostics (`integrated_gradients`, `layer_gradcam`, and related) are **disabled** in all configs. Working diagnostics (predictions, gradients, activations, failure_modes) remain enabled.

## Configuration Changes

All config files updated to remove Captum diagnostics from the diagnostics list:
- `configs/challenge1_base.yaml`
- `configs/challenge1_mini.yaml`
- `configs/challenge1_jepa.yaml`
- `configs/challenge1_jepa_mini.yaml`

## Next Steps

1. Try Option A first (use existing pl_module.model instead of checkpoint reload)
2. If that fails, add debugging hooks (Option E) to understand where outputs disappear
3. Consider implementing manual IG (Option B) as fallback
4. Document findings in this file

## Related Issues

- Captum GitHub issues to check:
  - Search for "regression" + "empty output"
  - Search for "Lightning" + "IntegratedGradients"
  - Check if there are known issues with checkpoint-loaded models

## References

- Captum docs: https://captum.ai/api/integrated_gradients.html
- Captum source: https://github.com/pytorch/captum
- Issue discussion: This issue affects both IG and GradCAM, suggesting it's fundamental to how Captum interacts with our model structure
