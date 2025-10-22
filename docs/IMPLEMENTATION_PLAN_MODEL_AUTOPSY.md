# Model Autopsy Suite - Implementation Plan

**Created**: 2025-10-13
**Status**: Ready to implement
**Estimated Time**: 8-10 hours total
**Goal**: Build comprehensive diagnostic system for debugging EEG deep learning models

---

## Overview

We're building a two-pronged diagnostic strategy:

1. **Data Quality Notebook** (`notebooks/08_validate_data_quality.py`)
   - One-time validation before training
   - Catches preprocessing bugs, leakage, distribution shifts
   - Fast (~2-5 min), interactive, exploratory

2. **Model Autopsy Callback** (`cerebro/callbacks/model_autopsy.py`)
   - Automatic diagnostics when training completes or early stopping triggers
   - Analyzes model behavior: predictions, gradients, attributions
   - Moderate runtime (5-15 min), automated, fire-and-forget

---

## Context: Current State

### What We Have
- Working Challenge 1 training pipeline (Lightning CLI)
- EEGNeX model training with early stopping
- Current best val NRMSE: ~1.0007 (barely better than naive baseline of 1.0)
- Checkpoint: `outputs/challenge1/20251013_181025/checkpoints/challenge1-epoch=16-val_nrmse=1.0007.ckpt`

### The Problem
Model predictions appear to collapse to mean (variance ratio = 0.20, meaning predictions have 20% variance of targets). We need to diagnose whether this is:
- A training issue (gradients, learning rate, optimization)
- An architecture issue (model capacity, hierarchical learning)
- A data issue (preprocessing, leakage, quality)
- A task difficulty ceiling (RT prediction from EEG is genuinely hard)

### The Solution
Build diagnostic tools to empirically determine root cause and guide next steps.

---

## Architecture Design

### File Structure
```
cerebro/
├── callbacks/
│   ├── __init__.py                         # NEW
│   └── model_autopsy.py                    # NEW - Main callback (300 lines)
│
├── diagnostics/                            # NEW DIRECTORY
│   ├── __init__.py                         # NEW
│   ├── predictions.py                      # NEW - Tier 1 (100 lines)
│   ├── gradients.py                        # NEW - Tier 1 (150 lines)
│   ├── activations.py                      # NEW - Tier 1 (100 lines)
│   ├── captum_attributions.py              # NEW - Tier 2A (300 lines)
│   ├── captum_layers.py                    # NEW - Tier 2B (200 lines)
│   ├── eeg_spatial.py                      # NEW - Tier 2C (150 lines)
│   ├── eeg_temporal.py                     # NEW - Tier 2C (150 lines)
│   ├── failure_modes.py                    # NEW - Tier 2D (100 lines)
│   └── visualizations.py                   # NEW - All plots (300 lines)
│
notebooks/
└── 08_validate_data_quality.py             # NEW - Data diagnostics (400 lines)

configs/
├── challenge1_base.yaml                     # MODIFY - Add ModelAutopsyCallback
└── challenge1_mini.yaml                     # MODIFY - Disable autopsy for mini
```

### Dependencies to Add
```toml
# Add to pyproject.toml [project.dependencies]
captum = ">=0.7.0"  # For interpretability (Integrated Gradients, GradCAM, etc.)
scipy = ">=1.10.0"  # For KS test, statistical tests
```

---

## Phase 1: Core Infrastructure (2-3 hours)

### Task 1.1: Create Callback Skeleton
**File**: `cerebro/callbacks/model_autopsy.py`

**Key Components**:
```python
class ModelAutopsyCallback(Callback):
    def __init__(
        self,
        run_on_training_end: bool = True,
        run_on_early_stop: bool = True,
        diagnostics: List[str] = ["predictions", "gradients", "activations"],
        output_dir: Optional[Path] = None,
        save_plots: bool = True,
        log_to_wandb: bool = True,
        generate_report: bool = True,
        num_samples: Optional[int] = None,  # null = full val set
    ):
        self.autopsy_triggered = False
        self.early_stop_detected = False
        # ... store params

    def on_validation_end(self, trainer, pl_module):
        """Detect early stopping by checking trainer.should_stop"""
        if trainer.should_stop and not self.early_stop_detected:
            self.early_stop_detected = True
            if self.run_on_early_stop:
                self._run_autopsy(trainer, pl_module, trigger="early_stop")

    def on_train_end(self, trainer, pl_module):
        """Run autopsy at training end if not already run"""
        if self.run_on_training_end and not self.autopsy_triggered:
            self._run_autopsy(trainer, pl_module, trigger="training_end")

    def _run_autopsy(self, trainer, pl_module, trigger: str):
        """Execute comprehensive diagnostics"""
        if self.autopsy_triggered:
            return  # Only run once

        self.autopsy_triggered = True

        # 1. Load best checkpoint
        # 2. Get validation dataloader
        # 3. Run diagnostics (dispatch to modules)
        # 4. Generate plots
        # 5. Log to wandb
        # 6. Generate markdown report
```

**Success Criteria**:
- Callback triggers on early stopping (detect `trainer.should_stop`)
- Callback triggers on training end (`on_train_end`)
- Callback runs only once (even if both conditions met)
- Loads best checkpoint path from `trainer.checkpoint_callback.best_model_path`
- Creates output directory at `trainer.log_dir/autopsy`

---

### Task 1.2: Prediction Diagnostics (Tier 1)
**File**: `cerebro/diagnostics/predictions.py`

**Functions**:
```python
def analyze_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: Optional[int] = None
) -> dict:
    """
    Computes predictions and analyzes distribution vs ground truth.

    Returns:
        {
            "predictions": np.array,      # (N,) predictions
            "targets": np.array,          # (N,) ground truth
            "residuals": np.array,        # (N,) errors
            "pred_mean": float,
            "pred_std": float,
            "target_mean": float,
            "target_std": float,
            "variance_ratio": float,      # pred_std / target_std
            "nrmse": float,               # Model NRMSE
            "rmse": float,                # Model RMSE
        }
    """
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[0].to(device), batch[1].to(device)
            y_pred = model(X)
            preds.append(y_pred.cpu().numpy())
            targets.append(y.cpu().numpy())

            if num_samples and len(preds) * len(preds[0]) >= num_samples:
                break

    preds = np.concatenate(preds).squeeze()
    targets = np.concatenate(targets).squeeze()

    # Compute statistics
    residuals = preds - targets
    rmse = np.sqrt(np.mean(residuals**2))
    nrmse = rmse / np.std(targets)

    return {
        "predictions": preds,
        "targets": targets,
        "residuals": residuals,
        "pred_mean": np.mean(preds),
        "pred_std": np.std(preds),
        "target_mean": np.mean(targets),
        "target_std": np.std(targets),
        "variance_ratio": np.std(preds) / np.std(targets),
        "nrmse": nrmse,
        "rmse": rmse,
    }


def compute_baseline_scores(predictions: np.array, targets: np.array) -> dict:
    """
    Compare model vs naive baselines.

    Returns:
        {
            "naive_mean_rmse": float,
            "naive_mean_nrmse": float,
            "naive_median_rmse": float,
            "naive_median_nrmse": float,
            "model_nrmse": float,
            "improvement_over_mean": float,  # (baseline - model) / baseline
        }
    """
    target_mean = np.mean(targets)
    target_median = np.median(targets)
    target_std = np.std(targets)

    # Naive baselines
    naive_mean_rmse = np.sqrt(np.mean((targets - target_mean)**2))
    naive_mean_nrmse = naive_mean_rmse / target_std

    naive_median_rmse = np.sqrt(np.mean((targets - target_median)**2))
    naive_median_nrmse = naive_median_rmse / target_std

    # Model performance
    model_rmse = np.sqrt(np.mean((predictions - targets)**2))
    model_nrmse = model_rmse / target_std

    improvement = (naive_mean_nrmse - model_nrmse) / naive_mean_nrmse

    return {
        "naive_mean_rmse": naive_mean_rmse,
        "naive_mean_nrmse": naive_mean_nrmse,
        "naive_median_rmse": naive_median_rmse,
        "naive_median_nrmse": naive_median_nrmse,
        "model_nrmse": model_nrmse,
        "improvement_over_mean": improvement,
    }
```

**Success Criteria**:
- Can run inference on validation set
- Computes prediction statistics (mean, std, NRMSE)
- Compares to naive baselines (always predict mean)
- Detects mode collapse (variance_ratio < 0.5 = warning)

---

### Task 1.3: Gradient Flow Diagnostics (Tier 1)
**File**: `cerebro/diagnostics/gradients.py`

**Functions**:
```python
def analyze_gradient_flow(
    model: nn.Module,
    batch: tuple,
    device: torch.device
) -> dict:
    """
    Analyzes gradient flow through model layers.

    Args:
        model: PyTorch model
        batch: Single batch (X, y) from dataloader
        device: Device to run on

    Returns:
        {
            "layer_names": List[str],
            "grad_norms": List[float],        # L2 norm of gradients per layer
            "param_norms": List[float],       # L2 norm of parameters per layer
            "grad_to_param_ratio": List[float], # grad_norm / param_norm
            "dead_layers": List[str],         # Layers with grad_norm < 1e-7
        }
    """
    model.train()
    model.zero_grad()

    X, y = batch[0].to(device), batch[1].to(device)
    y_pred = model(X)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    loss.backward()

    # Collect gradient statistics per layer
    layer_names = []
    grad_norms = []
    param_norms = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_names.append(name)
            grad_norms.append(param.grad.norm().item())
            param_norms.append(param.data.norm().item())

    grad_to_param_ratio = [g / p if p > 0 else 0 for g, p in zip(grad_norms, param_norms)]
    dead_layers = [name for name, gn in zip(layer_names, grad_norms) if gn < 1e-7]

    return {
        "layer_names": layer_names,
        "grad_norms": grad_norms,
        "param_norms": param_norms,
        "grad_to_param_ratio": grad_to_param_ratio,
        "dead_layers": dead_layers,
    }
```

**Success Criteria**:
- Runs backward pass on single batch
- Extracts gradient norms per layer
- Identifies dead layers (grad_norm ≈ 0)
- Computes grad/param ratio (should be ~0.001-0.01)

---

### Task 1.4: Activation Diagnostics (Tier 1)
**File**: `cerebro/diagnostics/activations.py`

**Functions**:
```python
def analyze_activations(
    model: nn.Module,
    batch: tuple,
    device: torch.device
) -> dict:
    """
    Analyzes activation statistics through model layers.

    Returns:
        {
            "layer_names": List[str],
            "activation_means": List[float],
            "activation_stds": List[float],
            "dead_neuron_pcts": List[float],  # % of neurons with output=0
            "sparsity": List[float],          # % of activations near zero
        }
    """
    model.eval()
    activations = {}

    # Register forward hooks to capture activations
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            hooks.append(module.register_forward_hook(get_activation(name)))

    # Forward pass
    X, y = batch[0].to(device), batch[1].to(device)
    with torch.no_grad():
        _ = model(X)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute statistics
    layer_names = []
    activation_means = []
    activation_stds = []
    dead_neuron_pcts = []
    sparsity = []

    for name, act in activations.items():
        layer_names.append(name)
        activation_means.append(act.mean().item())
        activation_stds.append(act.std().item())
        dead_neuron_pcts.append((act.abs() < 1e-6).float().mean().item() * 100)
        sparsity.append((act.abs() < 0.01).float().mean().item() * 100)

    return {
        "layer_names": layer_names,
        "activation_means": activation_means,
        "activation_stds": activation_stds,
        "dead_neuron_pcts": dead_neuron_pcts,
        "sparsity": sparsity,
    }
```

**Success Criteria**:
- Captures activations via forward hooks
- Computes per-layer statistics (mean, std, sparsity)
- Identifies dead neurons (activation ≈ 0)

---

### Task 1.5: Basic Visualizations
**File**: `cerebro/diagnostics/visualizations.py`

**Functions**:
```python
def plot_prediction_distribution(diagnostics: dict, output_path: Path):
    """Overlayed histograms of predictions vs targets"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist([diagnostics["targets"], diagnostics["predictions"]],
                 bins=50, label=['Targets', 'Predictions'], alpha=0.7)
    axes[0].set_xlabel('Response Time (s)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution Comparison')
    axes[0].legend()

    # Residual plot
    axes[1].scatter(diagnostics["predictions"], diagnostics["residuals"], alpha=0.5)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted RT')
    axes[1].set_ylabel('Residual (Pred - True)')
    axes[1].set_title('Residual Analysis')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_gradient_flow(diagnostics: dict, output_path: Path):
    """Bar chart of gradient norms per layer"""
    fig, ax = plt.subplots(figsize=(14, 6))

    layer_names = diagnostics["layer_names"]
    grad_norms = diagnostics["grad_norms"]

    # Shorten layer names for display
    short_names = [name.split('.')[-1] for name in layer_names]

    ax.bar(range(len(grad_norms)), grad_norms)
    ax.set_xticks(range(len(grad_norms)))
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_ylabel('Gradient L2 Norm')
    ax.set_title('Gradient Flow Through Layers')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_activation_stats(diagnostics: dict, output_path: Path):
    """Dead neuron percentages per layer"""
    fig, ax = plt.subplots(figsize=(14, 6))

    layer_names = diagnostics["layer_names"]
    dead_pcts = diagnostics["dead_neuron_pcts"]

    short_names = [name.split('.')[-1] for name in layer_names]

    ax.bar(range(len(dead_pcts)), dead_pcts)
    ax.set_xticks(range(len(dead_pcts)))
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_ylabel('Dead Neurons (%)')
    ax.set_title('Dead Neuron Detection')
    ax.axhline(10, color='r', linestyle='--', label='10% threshold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

**Success Criteria**:
- Generates 3 core plots (distribution, gradients, activations)
- Saves to disk as PNG files
- Returns list of file paths for wandb upload

---

### Task 1.6: Report Generation
**File**: `cerebro/callbacks/model_autopsy.py` (extend `_run_autopsy`)

**Function**:
```python
def _generate_report(self, diagnostics: dict, trigger: str, checkpoint_path: str) -> str:
    """
    Generate markdown autopsy report.

    Returns markdown string with:
    - Header (trigger, checkpoint, timestamp)
    - Prediction analysis (NRMSE, variance ratio, baseline comparison)
    - Gradient flow summary (dead layers, grad/param ratio)
    - Activation health (dead neurons, sparsity)
    - Diagnosis (detected issues)
    - Recommendations (actionable next steps)
    """
    report = f"""# 🔬 MODEL AUTOPSY REPORT

**Trigger**: {trigger}
**Checkpoint**: {checkpoint_path}
**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 PREDICTION ANALYSIS

- Model NRMSE: **{diagnostics['predictions']['nrmse']:.4f}**
- Naive Mean NRMSE: **{diagnostics['baselines']['naive_mean_nrmse']:.4f}** (baseline)
- Improvement: **{diagnostics['baselines']['improvement_over_mean']*100:.2f}%**
- Prediction Std: **{diagnostics['predictions']['pred_std']:.4f}**
- Target Std: **{diagnostics['predictions']['target_std']:.4f}**
- Variance Ratio: **{diagnostics['predictions']['variance_ratio']:.2f}**

"""

    # Add diagnosis section
    if diagnostics['predictions']['variance_ratio'] < 0.5:
        report += "🚨 **WARNING**: Predictions collapsed to narrow range!\n\n"

    if diagnostics['baselines']['improvement_over_mean'] < 0:
        report += "🚨 **WARNING**: Model worse than naive baseline!\n\n"

    # Add gradient flow section
    report += f"""## 🌊 GRADIENT FLOW

- Layers with gradients: **{len(diagnostics['gradients']['layer_names']) - len(diagnostics['gradients']['dead_layers'])}/{len(diagnostics['gradients']['layer_names'])}**
- Dead layers: **{len(diagnostics['gradients']['dead_layers'])}**
- Avg grad/param ratio: **{np.mean(diagnostics['gradients']['grad_to_param_ratio']):.4f}**

"""

    # Add recommendations
    report += """## 💡 RECOMMENDATIONS

Based on the diagnostics:

"""

    if diagnostics['predictions']['variance_ratio'] < 0.5:
        report += "1. **Increase learning rate** (predictions collapsed)\n"
        report += "2. **Reduce weight decay** (may be over-regularized)\n"

    if len(diagnostics['gradients']['dead_layers']) > 0:
        report += f"3. **Investigate dead layers**: {diagnostics['gradients']['dead_layers']}\n"

    return report
```

**Success Criteria**:
- Generates markdown report with all diagnostic sections
- Includes diagnosis (detected issues)
- Provides actionable recommendations
- Saves to `output_dir/autopsy_report.md`

---

### Task 1.7: Wire Everything Together
**File**: `cerebro/callbacks/model_autopsy.py` (complete `_run_autopsy`)

```python
def _run_autopsy(self, trainer, pl_module, trigger: str):
    """Execute comprehensive diagnostics"""
    if self.autopsy_triggered:
        return

    self.autopsy_triggered = True
    logger.info(f"🔬 Running model autopsy (trigger: {trigger})...")

    # 1. Setup
    output_dir = Path(trainer.log_dir) / "autopsy"
    output_dir.mkdir(exist_ok=True)

    # 2. Load best checkpoint
    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt:
        pl_module = pl_module.__class__.load_from_checkpoint(best_ckpt)
        logger.info(f"Loaded best checkpoint: {best_ckpt}")

    pl_module = pl_module.to(trainer.device)

    # 3. Get validation dataloader
    val_loader = trainer.datamodule.val_dataloader()

    # 4. Run diagnostics
    results = {}

    if "predictions" in self.diagnostics:
        logger.info("  Analyzing predictions...")
        results["predictions"] = analyze_predictions(
            pl_module.model, val_loader, trainer.device, self.num_samples
        )

    if "baselines" in self.diagnostics or "predictions" in self.diagnostics:
        logger.info("  Computing baseline comparisons...")
        results["baselines"] = compute_baseline_scores(
            results["predictions"]["predictions"],
            results["predictions"]["targets"]
        )

    if "gradients" in self.diagnostics:
        logger.info("  Analyzing gradient flow...")
        batch = next(iter(val_loader))
        results["gradients"] = analyze_gradient_flow(pl_module.model, batch, trainer.device)

    if "activations" in self.diagnostics:
        logger.info("  Analyzing activations...")
        batch = next(iter(val_loader))
        results["activations"] = analyze_activations(pl_module.model, batch, trainer.device)

    # 5. Generate plots
    plot_paths = []
    if self.save_plots:
        logger.info("  Generating diagnostic plots...")

        if "predictions" in results:
            plot_paths.append(output_dir / "prediction_distribution.png")
            plot_prediction_distribution(results["predictions"], plot_paths[-1])

        if "gradients" in results:
            plot_paths.append(output_dir / "gradient_flow.png")
            plot_gradient_flow(results["gradients"], plot_paths[-1])

        if "activations" in results:
            plot_paths.append(output_dir / "activation_stats.png")
            plot_activation_stats(results["activations"], plot_paths[-1])

    # 6. Log to wandb
    if self.log_to_wandb and trainer.logger:
        logger.info("  Uploading plots to wandb...")
        import wandb
        for plot_path in plot_paths:
            trainer.logger.experiment.log({
                f"autopsy/{plot_path.stem}": wandb.Image(str(plot_path))
            })

    # 7. Generate report
    if self.generate_report:
        logger.info("  Generating autopsy report...")
        report = self._generate_report(results, trigger, best_ckpt)
        report_path = output_dir / "autopsy_report.md"
        report_path.write_text(report)
        logger.info(f"📋 Autopsy report saved: {report_path}")

    logger.info("✓ Model autopsy complete!")
```

**Success Criteria**:
- Orchestrates all diagnostic modules
- Generates plots and saves to disk
- Uploads to wandb if available
- Generates and saves markdown report
- Logs progress with rich formatting

---

### Task 1.8: Config Integration
**File**: `configs/challenge1_base.yaml`

**Add to callbacks section**:
```yaml
trainer:
  callbacks:
    # ... existing callbacks (ModelCheckpoint, EarlyStopping, etc.) ...

    # Model Autopsy
    - class_path: cerebro.callbacks.ModelAutopsyCallback
      init_args:
        run_on_training_end: true
        run_on_early_stop: true
        diagnostics:
          - predictions
          - gradients
          - activations
        save_plots: true
        log_to_wandb: true
        generate_report: true
        num_samples: 500  # Analyze 500 samples (not full val set)
```

**File**: `configs/challenge1_mini.yaml`

**Disable for mini**:
```yaml
trainer:
  callbacks:
    - class_path: cerebro.callbacks.ModelAutopsyCallback
      init_args:
        run_on_training_end: false
        run_on_early_stop: false
        diagnostics: []
```

**Success Criteria**:
- Callback included in base config
- Disabled for mini config
- Can override via CLI: `--trainer.callbacks[4].init_args.run_on_training_end=true`

---

## Phase 2: Captum Integration (2-3 hours)

### Task 2.1: Install Captum
```bash
uv pip install captum
```

---

### Task 2.2: Integrated Gradients Analysis
**File**: `cerebro/diagnostics/captum_attributions.py`

**Functions**:
```python
from captum.attr import IntegratedGradients
import torch
import numpy as np

def compute_integrated_gradients(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
    n_steps: int = 50,
    baseline_type: str = "zero"
) -> dict:
    """
    Computes Integrated Gradients attributions for EEG input.

    Args:
        model: PyTorch model (wrapped, not LightningModule)
        dataloader: Validation dataloader
        device: Device to run on
        num_samples: Number of samples to analyze
        n_steps: IG steps (50 = good tradeoff)
        baseline_type: "zero", "mean", or "random"

    Returns:
        {
            "attributions": np.array,        # (num_samples, 129, 200)
            "temporal_profile": np.array,    # (200,) - sum over channels
            "spatial_profile": np.array,     # (129,) - sum over time
            "predictions": np.array,         # (num_samples,)
            "targets": np.array,             # (num_samples,)
        }
    """
    model.eval()
    ig = IntegratedGradients(model)

    # Collect samples
    samples = []
    targets = []
    for batch in dataloader:
        X, y = batch[0], batch[1]
        samples.append(X)
        targets.append(y)
        if len(samples) * X.shape[0] >= num_samples:
            break

    samples = torch.cat(samples, dim=0)[:num_samples].to(device)
    targets = torch.cat(targets, dim=0)[:num_samples].to(device)

    # Compute baseline
    if baseline_type == "zero":
        baseline = torch.zeros_like(samples)
    elif baseline_type == "mean":
        baseline = samples.mean(dim=0, keepdim=True).expand_as(samples)
    else:  # random
        baseline = torch.randn_like(samples) * 0.001

    # Compute attributions (requires_grad=True)
    samples.requires_grad = True

    attributions = ig.attribute(
        samples,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=32  # Process in batches to save memory
    )

    # Convert to numpy
    attributions_np = attributions.detach().cpu().numpy()  # (num_samples, 129, 200)
    predictions = model(samples).detach().cpu().numpy().squeeze()
    targets_np = targets.cpu().numpy().squeeze()

    # Aggregate attributions
    temporal_profile = np.abs(attributions_np).sum(axis=(0, 1))  # Sum over samples and channels
    spatial_profile = np.abs(attributions_np).sum(axis=(0, 2))   # Sum over samples and time

    return {
        "attributions": attributions_np,
        "temporal_profile": temporal_profile,
        "spatial_profile": spatial_profile,
        "predictions": predictions,
        "targets": targets_np,
    }
```

**Success Criteria**:
- Computes IG attributions for 100 samples
- Returns (129, 200) attribution map per sample
- Aggregates to temporal profile (200,) and spatial profile (129,)
- Runs in <2 minutes on GPU

---

### Task 2.3: Layer GradCAM Analysis
**File**: `cerebro/diagnostics/captum_layers.py`

**Functions**:
```python
from captum.attr import LayerGradCam

def detect_conv_layers(model: nn.Module) -> List[str]:
    """Auto-detect convolutional layers in EEGNeX"""
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            conv_layers.append(name)
    return conv_layers


def compute_layer_gradcam(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_layers: List[str] = None,
    num_samples: int = 100
) -> dict:
    """
    Computes GradCAM for specified convolutional layers.

    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        device: Device
        target_layers: Layer names (or None = auto-detect)
        num_samples: Number of samples to analyze

    Returns:
        {
            "layer_name": {
                "gradcam": np.array,  # (num_samples, H, W) or (num_samples, C, T)
                "predictions": np.array,
                "targets": np.array,
            }
        }
    """
    if target_layers is None or target_layers == ["auto"]:
        target_layers = detect_conv_layers(model)
        # Select early, mid, late layers
        if len(target_layers) > 3:
            target_layers = [
                target_layers[0],                  # Early
                target_layers[len(target_layers)//2],  # Mid
                target_layers[-1]                  # Late
            ]

    # Collect samples
    samples = []
    targets = []
    for batch in dataloader:
        X, y = batch[0], batch[1]
        samples.append(X)
        targets.append(y)
        if len(samples) * X.shape[0] >= num_samples:
            break

    samples = torch.cat(samples, dim=0)[:num_samples].to(device)
    targets = torch.cat(targets, dim=0)[:num_samples].to(device)

    results = {}

    for layer_name in target_layers:
        # Get layer module
        layer = dict(model.named_modules())[layer_name]

        # Compute GradCAM
        layer_gc = LayerGradCam(model, layer)

        attributions = layer_gc.attribute(samples, target=None)  # Regression: target=None

        # Convert to numpy and aggregate
        attributions_np = attributions.detach().cpu().numpy()
        predictions = model(samples).detach().cpu().numpy().squeeze()
        targets_np = targets.cpu().numpy().squeeze()

        results[layer_name] = {
            "gradcam": attributions_np,
            "predictions": predictions,
            "targets": targets_np,
        }

    return results
```

**Success Criteria**:
- Auto-detects convolutional layers in EEGNeX
- Computes GradCAM for early, mid, late layers
- Returns attention maps per layer
- Visualizes which regions activate each layer

---

### Task 2.4: Attribution Visualizations
**File**: `cerebro/diagnostics/visualizations.py` (extend)

**Add functions**:
```python
def plot_ig_heatmap(
    attributions: np.array,
    sample_idx: int = 0,
    output_path: Path = None
):
    """
    Plots attribution heatmap for single sample.

    Args:
        attributions: (num_samples, 129, 200)
        sample_idx: Which sample to visualize
        output_path: Save path
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    attr = attributions[sample_idx]  # (129, 200)

    im = ax.imshow(attr, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(attr).max(), vmax=np.abs(attr).max())
    ax.set_xlabel('Time (samples @ 100Hz)')
    ax.set_ylabel('Channel')
    ax.set_title(f'Integrated Gradients Attribution (Sample {sample_idx})')
    plt.colorbar(im, ax=ax, label='Attribution')

    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close()


def plot_temporal_importance(
    temporal_profile: np.array,
    sfreq: int = 100,
    window_start: float = 0.5,
    output_path: Path = None
):
    """
    Plots temporal importance profile.

    Args:
        temporal_profile: (200,) - importance per time sample
        sfreq: Sampling frequency (100 Hz)
        window_start: Window start time relative to stimulus (0.5s)
        output_path: Save path
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Convert samples to time (relative to stimulus)
    time = np.arange(len(temporal_profile)) / sfreq + window_start

    ax.plot(time, temporal_profile, linewidth=2)
    ax.fill_between(time, 0, temporal_profile, alpha=0.3)
    ax.axvline(1.0, color='r', linestyle='--', label='P300 expected (~1.0s)')
    ax.set_xlabel('Time post-stimulus (s)')
    ax.set_ylabel('Attribution (sum over channels)')
    ax.set_title('Temporal Importance Profile')
    ax.legend()
    ax.grid(alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close()


def plot_spatial_importance(
    spatial_profile: np.array,
    output_path: Path = None
):
    """
    Plots spatial importance as bar chart.

    TODO: Replace with brain topography plot (requires MNE channel positions)

    Args:
        spatial_profile: (129,) - importance per channel
        output_path: Save path
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get top 20 channels
    top_indices = np.argsort(spatial_profile)[-20:][::-1]
    top_values = spatial_profile[top_indices]

    ax.bar(range(len(top_values)), top_values)
    ax.set_xticks(range(len(top_values)))
    ax.set_xticklabels([f"Ch{i}" for i in top_indices], rotation=45)
    ax.set_ylabel('Attribution (sum over time)')
    ax.set_title('Top 20 Channels by Importance')

    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close()


def plot_gradcam_comparison(
    gradcam_results: dict,
    sample_idx: int = 0,
    output_path: Path = None
):
    """
    Plots GradCAM for multiple layers side-by-side.

    Args:
        gradcam_results: Dict from compute_layer_gradcam()
        sample_idx: Which sample to visualize
        output_path: Save path
    """
    num_layers = len(gradcam_results)
    fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 6))

    if num_layers == 1:
        axes = [axes]

    for ax, (layer_name, data) in zip(axes, gradcam_results.items()):
        gc = data["gradcam"][sample_idx]  # Shape depends on layer

        # Upsample to input resolution if needed
        # (This is a simplification; real upsampling may need interpolation)

        im = ax.imshow(gc, aspect='auto', cmap='hot')
        ax.set_title(f'{layer_name.split(".")[-1]}')
        ax.set_xlabel('Time dimension')
        ax.set_ylabel('Spatial dimension')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close()
```

**Success Criteria**:
- Plots IG heatmap (channels × time)
- Plots temporal importance profile (line plot)
- Plots spatial importance (bar chart or brain map)
- Plots GradCAM layer comparison

---

### Task 2.5: Integrate Captum into Callback
**File**: `cerebro/callbacks/model_autopsy.py` (extend `_run_autopsy`)

**Add to diagnostic execution**:
```python
# Inside _run_autopsy(), after existing diagnostics:

if "integrated_gradients" in self.diagnostics:
    logger.info("  Computing Integrated Gradients...")
    results["ig"] = compute_integrated_gradients(
        pl_module.model,
        val_loader,
        trainer.device,
        num_samples=min(100, self.num_samples or 100),
        n_steps=self.ig_n_steps,
        baseline_type=self.ig_baseline
    )

if "layer_gradcam" in self.diagnostics:
    logger.info("  Computing Layer GradCAM...")
    results["gradcam"] = compute_layer_gradcam(
        pl_module.model,
        val_loader,
        trainer.device,
        target_layers=self.target_layers,
        num_samples=min(100, self.num_samples or 100)
    )

# Add plots
if self.save_plots:
    if "ig" in results:
        plot_paths.append(output_dir / "ig_heatmap.png")
        plot_ig_heatmap(results["ig"]["attributions"], 0, plot_paths[-1])

        plot_paths.append(output_dir / "temporal_importance.png")
        plot_temporal_importance(results["ig"]["temporal_profile"], plot_paths[-1])

        plot_paths.append(output_dir / "spatial_importance.png")
        plot_spatial_importance(results["ig"]["spatial_profile"], plot_paths[-1])

    if "gradcam" in results:
        plot_paths.append(output_dir / "gradcam_comparison.png")
        plot_gradcam_comparison(results["gradcam"], 0, plot_paths[-1])
```

**Success Criteria**:
- Captum diagnostics run when enabled in config
- IG and GradCAM plots generated
- Plots uploaded to wandb

---

## Phase 3: EEG-Specific Diagnostics (1-2 hours)

### Task 3.1: Channel Importance (Ablation)
**File**: `cerebro/diagnostics/eeg_spatial.py`

**Functions**:
```python
def analyze_channel_importance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 500,
    num_ablation_trials: int = 10
) -> dict:
    """
    Measures channel importance via ablation.

    Process:
    1. Baseline: Compute NRMSE on unmodified data
    2. For each channel: Zero out channel, compute NRMSE
    3. Importance = NRMSE_ablated - NRMSE_baseline

    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        device: Device
        num_samples: Number of samples to test
        num_ablation_trials: Number of channels to ablate (129 = all, 10 = top-10 only)

    Returns:
        {
            "baseline_nrmse": float,
            "channel_importance": np.array,  # (129,) - NRMSE increase per channel
            "channel_ranking": List[int],    # Indices sorted by importance
        }
    """
    model.eval()

    # Collect samples
    samples = []
    targets = []
    for batch in dataloader:
        X, y = batch[0], batch[1]
        samples.append(X)
        targets.append(y)
        if len(samples) * X.shape[0] >= num_samples:
            break

    samples = torch.cat(samples, dim=0)[:num_samples].to(device)
    targets = torch.cat(targets, dim=0)[:num_samples].to(device)

    # Baseline NRMSE
    with torch.no_grad():
        preds = model(samples).squeeze()
        baseline_nrmse = torch.sqrt(torch.mean((preds - targets)**2)) / torch.std(targets)

    # Ablate each channel
    n_chans = samples.shape[1]  # 129
    channel_importance = np.zeros(n_chans)

    # If num_ablation_trials < n_chans, randomly sample channels
    if num_ablation_trials < n_chans:
        channels_to_test = np.random.choice(n_chans, num_ablation_trials, replace=False)
    else:
        channels_to_test = range(n_chans)

    for ch_idx in tqdm(channels_to_test, desc="Ablating channels"):
        # Zero out channel
        samples_ablated = samples.clone()
        samples_ablated[:, ch_idx, :] = 0

        # Compute NRMSE
        with torch.no_grad():
            preds = model(samples_ablated).squeeze()
            nrmse = torch.sqrt(torch.mean((preds - targets)**2)) / torch.std(targets)

        # Importance = performance degradation
        channel_importance[ch_idx] = (nrmse - baseline_nrmse).item()

    # Ranking
    channel_ranking = np.argsort(channel_importance)[::-1].tolist()

    return {
        "baseline_nrmse": baseline_nrmse.item(),
        "channel_importance": channel_importance,
        "channel_ranking": channel_ranking,
    }
```

**Success Criteria**:
- Ablates channels one-by-one
- Measures NRMSE increase per channel
- Returns ranked list (most important first)
- Validates against IG spatial profile (should correlate)

---

### Task 3.2: Temporal Importance (Ablation)
**File**: `cerebro/diagnostics/eeg_temporal.py`

**Functions**:
```python
def analyze_temporal_importance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 500,
    time_bin_size: int = 20  # 20 samples = 200ms @ 100Hz
) -> dict:
    """
    Measures temporal importance via ablation.

    Process:
    1. Baseline: Compute NRMSE
    2. For each time bin: Zero out time bin, compute NRMSE
    3. Importance = NRMSE_ablated - NRMSE_baseline

    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        device: Device
        num_samples: Number of samples
        time_bin_size: Size of time bins (20 = 200ms)

    Returns:
        {
            "baseline_nrmse": float,
            "temporal_importance": np.array,  # (num_bins,) - NRMSE increase per bin
            "time_bins": List[tuple],         # [(start, end), ...] in samples
        }
    """
    model.eval()

    # Collect samples
    samples = []
    targets = []
    for batch in dataloader:
        X, y = batch[0], batch[1]
        samples.append(X)
        targets.append(y)
        if len(samples) * X.shape[0] >= num_samples:
            break

    samples = torch.cat(samples, dim=0)[:num_samples].to(device)
    targets = torch.cat(targets, dim=0)[:num_samples].to(device)

    # Baseline NRMSE
    with torch.no_grad():
        preds = model(samples).squeeze()
        baseline_nrmse = torch.sqrt(torch.mean((preds - targets)**2)) / torch.std(targets)

    # Create time bins
    n_times = samples.shape[2]  # 200
    time_bins = [(i, min(i + time_bin_size, n_times)) for i in range(0, n_times, time_bin_size)]

    temporal_importance = np.zeros(len(time_bins))

    for bin_idx, (start, end) in enumerate(tqdm(time_bins, desc="Ablating time bins")):
        # Zero out time bin
        samples_ablated = samples.clone()
        samples_ablated[:, :, start:end] = 0

        # Compute NRMSE
        with torch.no_grad():
            preds = model(samples_ablated).squeeze()
            nrmse = torch.sqrt(torch.mean((preds - targets)**2)) / torch.std(targets)

        temporal_importance[bin_idx] = (nrmse - baseline_nrmse).item()

    return {
        "baseline_nrmse": baseline_nrmse.item(),
        "temporal_importance": temporal_importance,
        "time_bins": time_bins,
    }
```

**Success Criteria**:
- Ablates time bins (200ms windows)
- Measures NRMSE increase per bin
- Identifies peak importance window (P300?)
- Validates against IG temporal profile

---

### Task 3.3: Visualizations for EEG Diagnostics
**File**: `cerebro/diagnostics/visualizations.py` (extend)

**Add**:
```python
def plot_channel_importance(
    channel_importance: np.array,
    channel_ranking: List[int],
    output_path: Path = None,
    top_k: int = 20
):
    """Bar chart of top-K channel importance"""
    fig, ax = plt.subplots(figsize=(14, 6))

    top_channels = channel_ranking[:top_k]
    top_values = channel_importance[top_channels]

    ax.bar(range(len(top_values)), top_values)
    ax.set_xticks(range(len(top_values)))
    ax.set_xticklabels([f"Ch{i}" for i in top_channels], rotation=45)
    ax.set_ylabel('NRMSE Increase (ablation)')
    ax.set_title(f'Top {top_k} Channels by Importance (Ablation Study)')

    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close()


def plot_temporal_importance(
    temporal_importance: np.array,
    time_bins: List[tuple],
    sfreq: int = 100,
    window_start: float = 0.5,
    output_path: Path = None
):
    """Line plot of temporal importance"""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Convert bins to time
    bin_centers = [(start + end) / 2 / sfreq + window_start for start, end in time_bins]

    ax.plot(bin_centers, temporal_importance, 'o-', linewidth=2, markersize=8)
    ax.fill_between(bin_centers, 0, temporal_importance, alpha=0.3)
    ax.axvline(1.0, color='r', linestyle='--', label='P300 expected (~1.0s)')
    ax.set_xlabel('Time post-stimulus (s)')
    ax.set_ylabel('NRMSE Increase (ablation)')
    ax.set_title('Temporal Importance Profile (Ablation Study)')
    ax.legend()
    ax.grid(alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close()
```

**Success Criteria**:
- Visualizes channel importance rankings
- Visualizes temporal importance over time
- Compares ablation vs IG (should correlate)

---

## Phase 4: Failure Analysis (1 hour)

### Task 4.1: Failure Mode Detection
**File**: `cerebro/diagnostics/failure_modes.py`

**Functions**:
```python
def analyze_failure_modes(
    predictions: np.array,
    targets: np.array,
    metadata: dict = None,
    top_k: int = 50
) -> dict:
    """
    Identifies worst predictions and error patterns.

    Args:
        predictions: (N,) model predictions
        targets: (N,) ground truth
        metadata: Optional dict with sample metadata (subject_id, correct, trial_num, etc.)
        top_k: Number of worst predictions to return

    Returns:
        {
            "worst_indices": List[int],        # Indices of top-K worst predictions
            "worst_errors": np.array,          # Errors for worst predictions
            "worst_predictions": np.array,     # Predicted values
            "worst_targets": np.array,         # True values
            "error_stats": dict,               # Overall error statistics
            "error_correlations": dict,        # Error correlation with metadata
        }
    """
    residuals = predictions - targets
    abs_errors = np.abs(residuals)

    # Top-K worst
    worst_indices = np.argsort(abs_errors)[-top_k:][::-1]

    # Error statistics
    error_stats = {
        "mean_error": np.mean(residuals),
        "std_error": np.std(residuals),
        "mae": np.mean(abs_errors),
        "rmse": np.sqrt(np.mean(residuals**2)),
        "max_error": np.max(abs_errors),
    }

    # Error correlations with metadata (if available)
    error_correlations = {}
    if metadata:
        # Example: Error vs RT range
        low_rt_mask = targets < np.median(targets)
        high_rt_mask = targets >= np.median(targets)

        error_correlations["low_rt_mae"] = np.mean(abs_errors[low_rt_mask])
        error_correlations["high_rt_mae"] = np.mean(abs_errors[high_rt_mask])

        # Systematic bias detection
        error_correlations["bias_low_rt"] = np.mean(residuals[low_rt_mask])
        error_correlations["bias_high_rt"] = np.mean(residuals[high_rt_mask])

    return {
        "worst_indices": worst_indices.tolist(),
        "worst_errors": abs_errors[worst_indices],
        "worst_predictions": predictions[worst_indices],
        "worst_targets": targets[worst_indices],
        "error_stats": error_stats,
        "error_correlations": error_correlations,
    }
```

**Success Criteria**:
- Identifies top-K worst predictions
- Computes error statistics
- Detects systematic biases (e.g., overpredicts low RTs)

---

### Task 4.2: Failure Visualizations
**File**: `cerebro/diagnostics/visualizations.py` (extend)

**Add**:
```python
def plot_failure_modes(
    failure_results: dict,
    output_path: Path = None
):
    """Visualizes worst predictions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top-K worst predictions table (as scatter)
    ax = axes[0]
    ax.scatter(failure_results["worst_targets"],
               failure_results["worst_predictions"],
               s=100, alpha=0.6, c='red')
    ax.plot([0, 2.5], [0, 2.5], 'k--', label='Perfect prediction')
    ax.set_xlabel('True RT (s)')
    ax.set_ylabel('Predicted RT (s)')
    ax.set_title(f'Top {len(failure_results["worst_indices"])} Worst Predictions')
    ax.legend()
    ax.grid(alpha=0.3)

    # Error vs target value
    ax = axes[1]
    # (Need full predictions/targets, not just worst)
    # This requires passing full arrays to this function
    # For now, skip or simplify

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close()
```

**Success Criteria**:
- Plots worst predictions (scatter)
- Plots error vs RT (detect bias)

---

## Phase 5: Data Quality Notebook (1 hour)

### Task 5.1: Create Notebook
**File**: `notebooks/08_validate_data_quality.py`

**Structure**:
```python
# %% [markdown]
# # Data Quality Validation
#
# Run this notebook once before training to validate:
# 1. No subject leakage (train/val/test)
# 2. Label distributions (train vs val)
# 3. Input data quality (normalization, outliers, artifacts)
# 4. Temporal autocorrelation (overlapping windows)
# 5. Distribution shift (R5 vs R1-R4)

# %% Setup
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Add cerebro to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cerebro.data.challenge1 import Challenge1DataModule

# %% Load data
datamodule = Challenge1DataModule(
    data_dir=REPO_ROOT / "data" / "full",
    releases=["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"],
    batch_size=512,
    num_workers=8,
)

datamodule.setup()

print(f"Train windows: {len(datamodule.train_set)}")
print(f"Val windows: {len(datamodule.val_set)}")
print(f"Test windows: {len(datamodule.test_set)}")

# %% Subject leakage detection
def extract_subjects(dataset):
    """Extract subject IDs from dataset"""
    subjects = []
    for ds in dataset.datasets:
        subjects.extend(ds.description["subject"])
    return subjects

train_subjects = set(extract_subjects(datamodule.train_set))
val_subjects = set(extract_subjects(datamodule.val_set))
test_subjects = set(extract_subjects(datamodule.test_set))

print(f"Train subjects: {len(train_subjects)}")
print(f"Val subjects: {len(val_subjects)}")
print(f"Test subjects: {len(test_subjects)}")

# Critical: Check for overlap
train_val_overlap = train_subjects & val_subjects
train_test_overlap = train_subjects & test_subjects
val_test_overlap = val_subjects & test_subjects

assert len(train_val_overlap) == 0, f"LEAKAGE: {len(train_val_overlap)} subjects in train+val!"
assert len(train_test_overlap) == 0, f"LEAKAGE: {len(train_test_overlap)} subjects in train+test!"
assert len(val_test_overlap) == 0, f"LEAKAGE: {len(val_test_overlap)} subjects in val+test!"

print("✓ No subject leakage detected!")

# %% Label distribution comparison
def extract_labels(dataset):
    """Extract labels from dataset"""
    labels = []
    for ds in dataset.datasets:
        labels.extend(ds.description["target"])
    return np.array(labels)

train_labels = extract_labels(datamodule.train_set)
val_labels = extract_labels(datamodule.val_set)

stat, pval = ks_2samp(train_labels, val_labels)

print(f"Train label stats: mean={train_labels.mean():.3f}, std={train_labels.std():.3f}")
print(f"Val label stats:   mean={val_labels.mean():.3f}, std={val_labels.std():.3f}")
print(f"KS test p-value: {pval:.4f}")

if pval < 0.05:
    print("⚠️ WARNING: Train and val distributions differ significantly!")

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist([train_labels, val_labels], bins=50, label=['Train', 'Val'], alpha=0.7)
plt.xlabel('Response Time (s)')
plt.ylabel('Count')
plt.title(f'Label Distribution (KS p={pval:.3f})')
plt.legend()

plt.subplot(1, 2, 2)
from scipy.stats import probplot
probplot(val_labels, dist="norm", plot=plt)
plt.title('Q-Q Plot (Val labels)')
plt.tight_layout()
plt.savefig(REPO_ROOT / "outputs" / "data_validation_labels.png")
print("Saved: outputs/data_validation_labels.png")

# %% Input data quality
train_loader = datamodule.train_dataloader()
sample_batch = next(iter(train_loader))
X, y = sample_batch[0], sample_batch[1]  # (batch, 129, 200)

print(f"Batch shape: {X.shape}")

# Per-channel statistics
channel_means = X.mean(dim=(0, 2))  # (129,)
channel_stds = X.std(dim=(0, 2))

# Detect issues
flat_channels = (channel_stds < 0.01).sum().item()
extreme_outliers = (X.abs() > 5).any(dim=(0, 2)).sum().item()

print(f"Flat channels (<0.01 std): {flat_channels}")
print(f"Channels with extreme outliers (>5σ): {extreme_outliers}")

# Check overall normalization
print(f"Overall mean: {X.mean():.6f} (should be ≈0)")
print(f"Overall std: {X.std():.6f} (should be ≈1)")

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.bar(range(129), channel_means)
plt.xlabel('Channel')
plt.ylabel('Mean')
plt.title('Per-Channel Mean (should be ≈0)')

plt.subplot(1, 2, 2)
plt.bar(range(129), channel_stds)
plt.xlabel('Channel')
plt.ylabel('Std')
plt.title('Per-Channel Std (should be ≈1)')
plt.tight_layout()
plt.savefig(REPO_ROOT / "outputs" / "data_validation_channels.png")
print("Saved: outputs/data_validation_channels.png")

# %% NaN/Inf detection
nan_count = torch.isnan(X).sum().item()
inf_count = torch.isinf(X).sum().item()

assert nan_count == 0, f"Found {nan_count} NaN values!"
assert inf_count == 0, f"Found {inf_count} Inf values!"

print("✓ No NaN/Inf values detected")

# %% Summary
print("\n" + "="*60)
print("DATA VALIDATION SUMMARY")
print("="*60)
print(f"✓ No subject leakage")
print(f"✓ Label distributions similar (p={pval:.3f})")
print(f"✓ No NaN/Inf values")
print(f"✓ Input normalization: mean={X.mean():.6f}, std={X.std():.6f}")
if flat_channels > 0:
    print(f"⚠️ {flat_channels} flat channels detected")
if extreme_outliers > 0:
    print(f"⚠️ {extreme_outliers} channels with extreme outliers")
print("="*60)
```

**Success Criteria**:
- Detects subject leakage (critical!)
- Compares train/val label distributions
- Checks input normalization
- Detects NaN/Inf values
- Generates validation plots

---

## Testing & Validation

### Test 1: Run on Existing Checkpoint
```bash
# Load existing checkpoint and run autopsy manually
uv run python -c "
from cerebro.callbacks.model_autopsy import ModelAutopsyCallback
from cerebro.models.challenge1 import Challenge1Module
from cerebro.data.challenge1 import Challenge1DataModule
import torch

# Load checkpoint
model = Challenge1Module.load_from_checkpoint(
    'outputs/challenge1/20251013_181025/checkpoints/challenge1-epoch=16-val_nrmse=1.0007.ckpt'
)

# Setup data
datamodule = Challenge1DataModule(
    data_dir='data/full',
    releases=['R1', 'R2', 'R3', 'R4'],
    batch_size=512
)
datamodule.setup()

# Mock trainer
class MockTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = 'outputs/test_autopsy'
        self.datamodule = datamodule
        self.logger = None

        class MockCheckpointCallback:
            best_model_path = 'outputs/challenge1/20251013_181025/checkpoints/challenge1-epoch=16-val_nrmse=1.0007.ckpt'

        self.checkpoint_callback = MockCheckpointCallback()

trainer = MockTrainer()

# Run autopsy
callback = ModelAutopsyCallback(
    diagnostics=['predictions', 'gradients', 'activations']
)
callback._run_autopsy(trainer, model, 'manual_test')
"
```

**Expected Output**:
- `outputs/test_autopsy/autopsy/` directory created
- 3 plots generated (prediction_distribution, gradient_flow, activation_stats)
- `autopsy_report.md` generated
- Terminal output shows diagnostic progress

---

### Test 2: Run Full Training with Autopsy
```bash
# Train for 2 epochs with autopsy enabled
uv run cerebro fit --config configs/challenge1_mini.yaml \
    --trainer.callbacks[4].init_args.run_on_training_end=true \
    --trainer.callbacks[4].init_args.diagnostics='["predictions","gradients"]'
```

**Expected Output**:
- Training completes (2 epochs for mini)
- Autopsy triggers at end
- Plots and report generated
- Wandb shows uploaded plots

---

## Completion Checklist

### Phase 1: Core Infrastructure ✅
- [ ] Task 1.1: Callback skeleton
- [ ] Task 1.2: Prediction diagnostics
- [ ] Task 1.3: Gradient flow diagnostics
- [ ] Task 1.4: Activation diagnostics
- [ ] Task 1.5: Basic visualizations
- [ ] Task 1.6: Report generation
- [ ] Task 1.7: Wire everything together
- [ ] Task 1.8: Config integration

### Phase 2: Captum Integration ✅
- [ ] Task 2.1: Install Captum
- [ ] Task 2.2: Integrated Gradients
- [ ] Task 2.3: Layer GradCAM
- [ ] Task 2.4: Attribution visualizations
- [ ] Task 2.5: Integrate into callback

### Phase 3: EEG-Specific ✅
- [ ] Task 3.1: Channel importance
- [ ] Task 3.2: Temporal importance
- [ ] Task 3.3: Visualizations

### Phase 4: Failure Analysis ✅
- [ ] Task 4.1: Failure mode detection
- [ ] Task 4.2: Failure visualizations

### Phase 5: Data Quality ✅
- [ ] Task 5.1: Create notebook

### Testing ✅
- [ ] Test 1: Manual autopsy on existing checkpoint
- [ ] Test 2: Full training with autopsy

---

## Quick Start (Resume Session)

```bash
# 1. Check current status
cat /home/varun/repos/cerebro/IMPLEMENTATION_PLAN_MODEL_AUTOPSY.md

# 2. Check which phase to start
# Look at "Completion Checklist" section above

# 3. Start implementing
# Example: Start Phase 1, Task 1.1
cd /home/varun/repos/cerebro

# Create directories
mkdir -p cerebro/callbacks cerebro/diagnostics

# Start coding
# vim cerebro/callbacks/model_autopsy.py
```

---

## Notes & Tips

1. **Incremental Testing**: After each task, write a small test script to validate the function works before moving on.

2. **Memory Management**: Captum can be memory-intensive. Use `internal_batch_size` parameter for IG and process in chunks.

3. **Debugging**: If callback doesn't trigger, check:
   - Is `trainer.should_stop` being set by EarlyStopping?
   - Is `on_train_end` being called?
   - Check Lightning logs for callback execution order

4. **Wandb Offline Mode**: If wandb is offline, plots won't upload but everything else works.

5. **EEGNeX Layers**: Use `model.named_modules()` to inspect actual layer names. They may differ from expected.

6. **Checkpoint Loading**: If checkpoint fails to load, check:
   - Path exists
   - Lightning version compatibility
   - Model architecture matches

7. **Config Override**: To test without modifying configs:
   ```bash
   uv run cerebro fit --config configs/challenge1_base.yaml \
       --trainer.callbacks='[{"class_path": "cerebro.callbacks.ModelAutopsyCallback", "init_args": {"diagnostics": ["predictions"]}}]'
   ```

---

## Future Extensions (Post-Initial Implementation)

1. **Layer Conductance** (Tier 2B, Phase 2)
2. **GradientSHAP** (Tier 2A, Phase 2)
3. **Noise Tunnel / SmoothGrad** (Tier 3, optional)
4. **Per-Subject Consistency** (Challenge 2 specific)
5. **Frequency Band Analysis** (EEG-specific, advanced)
6. **Uncertainty Estimation** (MC Dropout)
7. **Brain Topography Plots** (requires MNE channel positions)
8. **Interactive Captum Insights Dashboard** (web UI)

---

## Contact & Resources

- **Captum Docs**: https://captum.ai/
- **Lightning Callbacks**: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
- **MNE-Python** (for brain topography): https://mne.tools/stable/index.html
- **Competition Startkit**: `startkit/` directory

---

**Last Updated**: 2025-10-13
**Status**: Ready to implement
**Next Session**: Start Phase 1, Task 1.1 (Callback Skeleton)
