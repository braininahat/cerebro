# Performance Comparison Sweep Guide

## 🎯 Quick Start

The benchmark notebook is now configured for **pure performance comparison** using model-specific defaults:

```bash
# Run the performance comparison sweep (RECOMMENDED)
uv run python notebooks/002_benchmark.py

# See configuration options
uv run python notebooks/002_benchmark.py --show-configs

# Test single model with defaults
uv run python notebooks/002_benchmark.py --test --model EEGNetv4
```

## 🔧 What's Changed

### 1. **New Default Configuration: `sweep_config_comparison`**
- **One run per model** - no hyperparameter exploration
- Uses **model-specific optimized defaults** automatically
- Focuses on **test/rmse** as the primary metric
- Comprehensive **timing and performance logging**

### 2. **Enhanced Metrics Logging**
Now tracks:
- **Performance**: RMSE, MAE for train/val/test
- **Timing**: Per-epoch time, total training time, epochs to convergence
- **Efficiency**: Parameters count, time to best validation
- **Configuration**: Batch size, learning rate, optimizer used per model

### 3. **Model-Specific Defaults Applied**
Examples of optimized settings:
- **EEGNetv4**: batch_size=256, lr=1e-3 (lightweight, can use large batches)  
- **Deep4Net**: batch_size=64, lr=5e-4 (deeper, needs smaller batches)
- **BIOT/EEGConformer**: batch_size=32, lr=1e-4 (transformers, memory intensive)

### 4. **Results Analysis Tools**
After running a sweep:
```python
# Get sweep ID from output, then analyze results
create_results_summary("your_sweep_id_here")
```

This creates a ranked comparison table with:
- Model performance (RMSE/MAE)
- Training efficiency (time, parameters)
- Convergence speed (epochs to best)

## 📊 What You'll Get

### During Training
- Real-time W&B dashboard with all metrics
- Console output showing progress per model
- Automatic model checkpointing to `weights/` directory
- Model weights uploaded to W&B as artifacts for cloud storage

### After Completion
- **Performance ranking** by test RMSE
- **Efficiency analysis** (performance per parameter)
- **Speed comparison** (training time per model)
- **Convergence insights** (which models learn fastest)

## 🎛️ Available Configurations

1. **`sweep_config_comparison`** ⭐ **CURRENT DEFAULT**
   - Pure performance comparison with model defaults
   - ~2-4 hours for all models

2. **`sweep_config_focused`** 
   - Top 4 models only (EEGNetv4, Deep4Net, EEGNeX, ShallowFBCSPNet)
   - ~30-60 minutes

3. **`sweep_config_search`**
   - Hyperparameter optimization if you want to tune further
   - ~8-12 hours

4. **`sweep_config_default`**
   - Original configuration (same as comparison but different metric focus)

## 🔄 Switching Configurations

In the notebook, change this line:
```python
sweep_config = sweep_config_comparison  # Current default
# sweep_config = sweep_config_focused   # For quick testing
# sweep_config = sweep_config_search    # For hyperparameter tuning
```

## 🎯 Perfect for Your Use Case

This setup gives you exactly what you want:
- ✅ **Uses model defaults** (no hyperparameter search)
- ✅ **Logs everything** for easy comparison  
- ✅ **One run per model** (efficient benchmarking)
- ✅ **Comprehensive metrics** (performance + efficiency)
- ✅ **Easy results analysis** with built-in tools

Just run `uv run python notebooks/002_benchmark.py` and you'll get a complete performance comparison of all models with their optimal settings!