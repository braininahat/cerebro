# %% [markdown]
# # Braindecode Model Benchmarking with W&B Sweeps

# %% [markdown]
# ## Imports

import sys
import time
from pathlib import Path

import torch
import wandb

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro.constants import *
from cerebro.data import prepare_data_pipeline
from cerebro.metrics import calculate_mae, calculate_rmse
from cerebro.models import create_model, get_all_models, get_model_config
from cerebro.training import get_optimizer, get_scheduler, train_one_epoch, validate

# %% [markdown]
# ## Training Function


def train_model(config=None):
    with wandb.init(config=config, project=WANDB_PROJECT, entity=WANDB_ENTITY):
        config = wandb.config

        # Get model-specific configuration
        use_model_defaults = getattr(config, 'use_model_defaults', True)
        model_config = get_model_config(
            config.model_name, 
            use_defaults=use_model_defaults
        )
        
        # Override with sweep parameters if they exist
        final_config = {}
        for param in ['batch_size', 'lr', 'weight_decay', 'optimizer']:
            # Use sweep value if available, otherwise use model-specific default
            if hasattr(config, param):
                final_config[param] = getattr(config, param)
            else:
                final_config[param] = model_config[param]
        
        print(f"Training model: {config.model_name}")
        print(f"Model-specific config: {model_config}")
        print(f"Final config: LR={final_config['lr']}, BS={final_config['batch_size']}, WD={final_config['weight_decay']}, Opt={final_config['optimizer']}")
        
        # Update wandb config with final values
        wandb.config.update(final_config)

        # Create weights directory for model checkpoints
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)

        # Load and prepare data using final batch size
        train_loader, valid_loader, test_loader = prepare_data_pipeline(
            task="contrastChangeDetection", 
            release="R1", 
            remove_bad_subjects=True,
            batch_size=final_config['batch_size']
        )

        print(
            f"Data loaded: {len(train_loader)} train batches, {len(valid_loader)} valid batches"
        )

        # Create model
        model = create_model(config.model_name)
        model.to(config.device)

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        wandb.log(
            {
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
            }
        )

        print(
            f"Model created: {total_params:,} total params, {trainable_params:,} trainable"
        )

        # Setup training with final config
        optimizer = get_optimizer(
            model, final_config['optimizer'], final_config['lr'], final_config['weight_decay']
        )
        scheduler = get_scheduler(optimizer, config.scheduler, T_max=config.n_epochs)
        loss_fn = torch.nn.MSELoss()

        # Training loop
        best_val_rmse = float("inf")
        patience_counter = 0
        training_start_time = time.time()
        epochs_to_best = 0

        for epoch in range(config.n_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{config.n_epochs}")

            # Training
            train_loss, train_rmse = train_one_epoch(
                train_loader,
                model,
                loss_fn,
                optimizer,
                scheduler,
                config.device,
                print_batch_stats=False,  # Disable for cleaner logs
            )

            # Validation with predictions for MAE calculation
            val_loss, val_rmse, val_preds, val_targets = validate(
                valid_loader,
                model,
                loss_fn,
                config.device,
                print_batch_stats=False,
                return_predictions=True,
            )

            # Calculate additional metrics
            val_mae = calculate_mae(val_preds, val_targets)

            # Calculate timing metrics
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]["lr"]

            # Log metrics including timing
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/rmse": train_rmse,
                    "val/loss": val_loss,
                    "val/rmse": val_rmse,
                    "val/mae": val_mae,
                    "train/lr": current_lr,
                    "timing/epoch_seconds": epoch_time,
                }
            )

            print(
                f"Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}, Val MAE: {val_mae:.6f}, LR: {current_lr:.2e}"
            )

            # Early stopping
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
                epochs_to_best = epoch + 1

                # Save best model to weights directory
                model_path = weights_dir / f"best_{config.model_name}_weights.pt"
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Final test evaluation with MAE
        test_loss, test_rmse, test_preds, test_targets = validate(
            test_loader,
            model,
            loss_fn,
            config.device,
            print_batch_stats=False,
            return_predictions=True,
        )

        # Calculate final test MAE and timing metrics
        test_mae = calculate_mae(test_preds, test_targets)
        total_training_time = time.time() - training_start_time

        wandb.log(
            {
                "test/loss": test_loss,
                "test/rmse": test_rmse,
                "test/mae": test_mae,
                "best_val_rmse": best_val_rmse,
                "timing/total_training_seconds": total_training_time,
                "timing/epochs_to_best": epochs_to_best,
                "timing/avg_epoch_seconds": total_training_time / (epoch + 1),
            }
        )

        # Upload best model weights to W&B as artifact
        try:
            model_artifact = wandb.Artifact(
                f"model-{config.model_name}", 
                type="model",
                description=f"Best {config.model_name} model weights (Val RMSE: {best_val_rmse:.6f})",
                metadata={
                    "model_name": config.model_name,
                    "best_val_rmse": best_val_rmse,
                    "test_rmse": test_rmse,
                    "test_mae": test_mae,
                    "epochs_to_best": epochs_to_best,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                }
            )
            model_artifact.add_file(str(model_path))
            wandb.log_artifact(model_artifact)
            print(f"✅ Model weights uploaded to W&B as artifact: model-{config.model_name}")
        except Exception as e:
            print(f"⚠️  Failed to upload model artifact: {e}")
            print("Model weights still saved locally to weights/ directory")

        print(f"Final Test RMSE: {test_rmse:.6f}, Test MAE: {test_mae:.6f}")
        print(f"Best Val RMSE: {best_val_rmse:.6f}")


# %% [markdown]
# ## Sweep Configuration

# Default sweep config: uses model-specific defaults
sweep_config_default = {
    "method": "grid", 
    "metric": {"name": "val/rmse", "goal": "minimize"},
    "parameters": {
        "model_name": {"values": list(get_all_models().keys())},
        "use_model_defaults": {"value": True},  # Use model-specific defaults
        "scheduler": {"value": "cosine"},
        "n_epochs": {"value": DEFAULT_N_EPOCHS},
        "patience": {"value": DEFAULT_EARLY_STOPPING_PATIENCE},
        "device": {"value": DEVICE},
    },
}

# Hyperparameter search sweep config: explores ranges for all models
sweep_config_search = {
    "method": "bayes",  # More efficient for hyperparameter search
    "metric": {"name": "val/rmse", "goal": "minimize"},
    "parameters": {
        "model_name": {"values": list(get_all_models().keys())},
        "use_model_defaults": {"value": False},  # Don't use model-specific defaults
        "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "batch_size": {"values": [32, 64, 128, 256]},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-3},
        "optimizer": {"values": ["adamw", "adam", "sgd"]},
        "scheduler": {"value": "cosine"},
        "n_epochs": {"value": DEFAULT_N_EPOCHS},
        "patience": {"value": DEFAULT_EARLY_STOPPING_PATIENCE},
        "device": {"value": DEVICE},
    },
}

# Focused sweep config: test specific models with their defaults plus variations
sweep_config_focused = {
    "method": "grid",
    "metric": {"name": "val/rmse", "goal": "minimize"}, 
    "parameters": {
        "model_name": {"values": ["EEGNetv4", "Deep4Net", "EEGNeX", "ShallowFBCSPNet"]},
        "use_model_defaults": {"value": True},  # Use model-specific defaults
        # Optional overrides - only if you want to test variations
        # "lr": {"values": [None]},  # None means use model default
        # "batch_size": {"values": [None]},  # None means use model default
        "scheduler": {"value": "cosine"},
        "n_epochs": {"value": DEFAULT_N_EPOCHS},
        "patience": {"value": DEFAULT_EARLY_STOPPING_PATIENCE},
        "device": {"value": DEVICE},
    },
}

# Performance comparison sweep config: pure model comparison with defaults
sweep_config_comparison = {
    "method": "grid",
    "metric": {"name": "test/rmse", "goal": "minimize"},  # Focus on final test performance
    "parameters": {
        "model_name": {"values": list(get_all_models().keys())},
        "use_model_defaults": {"value": True},  # Always use model-specific defaults
        "scheduler": {"value": "cosine"},
        "n_epochs": {"value": DEFAULT_N_EPOCHS},
        "patience": {"value": DEFAULT_EARLY_STOPPING_PATIENCE},
        "device": {"value": DEVICE},
    },
}

# Choose which configuration to use
sweep_config = sweep_config_comparison  # Default to pure performance comparison

# %% [markdown]
# ## Configuration Selection Helper

def print_sweep_options():
    """Print available sweep configurations and their purposes."""
    print("Available Sweep Configurations:")
    print("\n1. sweep_config_comparison (CURRENTLY SELECTED)")
    print("   - Pure model performance comparison")
    print("   - Uses model-specific optimized defaults")
    print("   - One run per model - no hyperparameter search")
    print("   - Comprehensive timing and performance metrics")
    print("   - Best for benchmarking all models")
    print("   - Expected runtime: ~2-4 hours for all models")
    
    print("\n2. sweep_config_default")
    print("   - Uses model-specific optimized defaults")
    print("   - Fast grid search across all models")
    print("   - Best for initial benchmarking")
    print("   - Expected runtime: ~2-4 hours for all models")
    
    print("\n3. sweep_config_search")
    print("   - Bayesian hyperparameter optimization")
    print("   - Explores learning rates, batch sizes, optimizers")
    print("   - Ignores model-specific defaults")
    print("   - Best for finding optimal hyperparameters")
    print("   - Expected runtime: ~8-12 hours")
    
    print("\n4. sweep_config_focused") 
    print("   - Tests only top 4 performing models")
    print("   - Uses model-specific defaults")
    print("   - Fast targeted evaluation")
    print("   - Expected runtime: ~30-60 minutes")
    
    print(f"\nCurrently selected: sweep_config_comparison")
    print("To change, set: sweep_config = sweep_config_<option>")


# %% [markdown]
# ## Run Single Model Test

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run single model test")
    parser.add_argument("--model", default="EEGNetv4", help="Model to test")
    parser.add_argument("--show-configs", action="store_true", help="Show available sweep configurations")
    args = parser.parse_args()
    
    if args.show_configs:
        print_sweep_options()
        exit()

    if args.test:
        # Test run with single model using model-specific defaults
        model_config = get_model_config(args.model)
        test_config = {
            "model_name": args.model,
            "use_model_defaults": True,
            "scheduler": "cosine",
            "n_epochs": 5,  # Short test
            "patience": 10,
            "device": DEVICE,
        }
        
        # Add model-specific defaults to the config
        test_config.update(model_config)
        
        print(f"Running test with {args.model}")
        print(f"Using model config: {model_config}")
        train_model(test_config)

    else:
        # Run full sweep
        sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)

        print(f"Starting sweep: {sweep_id}")
        print(f"W&B Project: {WANDB_ENTITY}/{WANDB_PROJECT}")
        print(f"Total models to test: {len(get_all_models())}")

        # Run agent
        wandb.agent(sweep_id, train_model)

# %% [markdown]
# ## Model Information


def print_model_summary():
    """Print summary of all available models."""
    models = get_all_models()
    print(f"Total models: {len(models)}")
    print("\nAvailable models:")
    for name in sorted(models.keys()):
        print(f"  - {name}")


def print_model_configs():
    """Print model configurations for all models."""
    
    print("Model-Specific Configurations:")
    print("=" * 60)
    print(f"{'Model':<20} {'Batch Size':<12} {'Learning Rate':<15} {'Weight Decay':<12}")
    print("-" * 60)
    
    # Group models by similar configs for cleaner display
    config_groups = {}
    for model_name in sorted(get_all_models().keys()):
        config = get_model_config(model_name)
        key = (config['batch_size'], config['lr'], config['weight_decay'])
        if key not in config_groups:
            config_groups[key] = []
        config_groups[key].append(model_name)
    
    # Print grouped configs
    for (batch_size, lr, weight_decay), models in config_groups.items():
        print(f"{models[0]:<20} {batch_size:<12} {lr:<15.1e} {weight_decay:<12.1e}")
        for model in models[1:]:
            print(f"{model:<20} {'(same)':<12} {'(same)':<15} {'(same)':<12}")
        print()


# %%
print_model_summary()

# %%
# Uncomment to see all model configurations:
# print_model_configs()

# %% [markdown]
# ## Results Comparison Helper

def create_results_summary(sweep_id: str):
    """Create a summary table of sweep results for easy comparison.
    
    Args:
        sweep_id: W&B sweep ID to analyze
        
    Usage:
        # After running a sweep, get the sweep ID from the output and run:
        # create_results_summary("your_sweep_id_here")
    """
    import wandb
    import pandas as pd
    
    # Get sweep results
    api = wandb.Api()
    sweep = api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}")
    
    results = []
    for run in sweep.runs:
        if run.state == "finished":
            # Extract key metrics
            summary = run.summary
            config = run.config
            
            result = {
                "model": config.get("model_name", "unknown"),
                "test_rmse": summary.get("test/rmse", float("inf")),
                "test_mae": summary.get("test/mae", float("inf")),
                "best_val_rmse": summary.get("best_val_rmse", float("inf")),
                "total_training_time": summary.get("timing/total_training_seconds", 0),
                "epochs_to_best": summary.get("timing/epochs_to_best", 0),
                "avg_epoch_time": summary.get("timing/avg_epoch_seconds", 0),
                "total_params": summary.get("model/total_params", 0),
                "batch_size": config.get("batch_size", 0),
                "lr": config.get("lr", 0),
                "run_id": run.id
            }
            results.append(result)
    
    # Create DataFrame and sort by test RMSE
    df = pd.DataFrame(results)
    df = df.sort_values("test_rmse").reset_index(drop=True)
    
    # Format for display
    df["training_time_min"] = (df["total_training_time"] / 60).round(1)
    df["params_M"] = (df["total_params"] / 1e6).round(2)
    
    print("🏆 Model Performance Comparison")
    print("=" * 100)
    print(f"{'Rank':<4} {'Model':<20} {'Test RMSE':<10} {'Test MAE':<10} {'Val RMSE':<10} {'Time(min)':<10} {'Params(M)':<10}")
    print("-" * 100)
    
    for i, row in df.head(10).iterrows():  # Show top 10
        print(f"{i+1:<4} {row['model']:<20} {row['test_rmse']:<10.6f} {row['test_mae']:<10.6f} {row['best_val_rmse']:<10.6f} {row['training_time_min']:<10.1f} {row['params_M']:<10.2f}")
    
    # Show some statistics
    print("\n📊 Summary Statistics:")
    print(f"Best performing model: {df.iloc[0]['model']} (RMSE: {df.iloc[0]['test_rmse']:.6f})")
    print(f"Fastest training: {df.loc[df['training_time_min'].idxmin(), 'model']} ({df['training_time_min'].min():.1f} min)")
    print(f"Most efficient (lowest RMSE/param): {df.loc[(df['test_rmse'] / df['params_M']).idxmin(), 'model']}")
    
    return df


def print_wandb_instructions():
    """Print instructions for accessing W&B results."""
    print("📈 Accessing Your Results:")
    print("\n1. After running the sweep, copy the sweep ID from the output")
    print("2. Visit your W&B dashboard to view real-time results")
    print("3. For a summary table, run:")
    print("   create_results_summary('your_sweep_id_here')")
    print("\n4. Your W&B project URL:")
    print(f"   https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
    
# %%
# Uncomment to see W&B instructions:
# print_wandb_instructions()
