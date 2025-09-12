# %% [markdown]
# # Braindecode Model Benchmarking with W&B Sweeps

# %% [markdown]
# ## Imports

import sys
from pathlib import Path

import torch
import wandb

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro.constants import *
from cerebro.data import prepare_data_pipeline
from cerebro.metrics import calculate_mae, calculate_rmse
from cerebro.models import create_model, get_all_models
from cerebro.training import get_optimizer, get_scheduler, train_one_epoch, validate

# %% [markdown]
# ## Training Function


def train_model(config=None):
    with wandb.init(config=config, project=WANDB_PROJECT, entity=WANDB_ENTITY):
        config = wandb.config

        print(f"Training model: {config.model_name}")
        print(
            f"Config: LR={config.lr}, BS={config.batch_size}, WD={config.weight_decay}"
        )

        # Load and prepare data (full pipeline)
        train_loader, valid_loader, test_loader = prepare_data_pipeline(
            task="contrastChangeDetection", release="R1", remove_bad_subjects=True
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

        # Setup training
        optimizer = get_optimizer(
            model, config.optimizer, config.lr, config.weight_decay
        )
        scheduler = get_scheduler(optimizer, config.scheduler, T_max=config.n_epochs)
        loss_fn = torch.nn.MSELoss()

        # Training loop
        best_val_rmse = float("inf")
        patience_counter = 0

        for epoch in range(config.n_epochs):
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

            # Current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # Log metrics
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/rmse": train_rmse,
                    "val/loss": val_loss,
                    "val/rmse": val_rmse,
                    "val/mae": val_mae,
                    "train/lr": current_lr,
                }
            )

            print(
                f"Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}, Val MAE: {val_mae:.6f}, LR: {current_lr:.2e}"
            )

            # Early stopping
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0

                # Save best model
                torch.save(model.state_dict(), f"best_{config.model_name}_weights.pt")
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

        # Calculate final test MAE
        test_mae = calculate_mae(test_preds, test_targets)

        wandb.log(
            {
                "test/loss": test_loss,
                "test/rmse": test_rmse,
                "test/mae": test_mae,
                "best_val_rmse": best_val_rmse,
            }
        )

        print(f"Final Test RMSE: {test_rmse:.6f}, Test MAE: {test_mae:.6f}")
        print(f"Best Val RMSE: {best_val_rmse:.6f}")


# %% [markdown]
# ## Sweep Configuration

sweep_config = {
    "method": "grid",
    "metric": {"name": "val/rmse", "goal": "minimize"},
    "parameters": {
        "model_name": {"values": list(get_all_models().keys())},
        "lr": {"values": [1e-4]},
        "batch_size": {
            "values": [
                64,
            ]
        },
        "weight_decay": {"values": [1e-4]},
        "optimizer": {"values": ["adamw"]},
        "scheduler": {"value": "cosine"},
        "n_epochs": {"value": DEFAULT_N_EPOCHS},
        "patience": {"value": DEFAULT_EARLY_STOPPING_PATIENCE},
        "device": {"value": DEVICE},
    },
}

# %% [markdown]
# ## Run Single Model Test

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run single model test")
    parser.add_argument("--model", default="EEGNetv4", help="Model to test")
    args = parser.parse_args()

    if args.test:
        # Test run with single model
        test_config = {
            "model_name": args.model,
            "lr": 1e-3,
            "batch_size": 128,
            "weight_decay": 1e-5,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "n_epochs": 5,  # Short test
            "patience": 10,
            "device": DEVICE,
        }

        print(f"Running test with {args.model}")
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


# %%
print_model_summary()
