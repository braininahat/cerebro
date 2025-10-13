"""Hyperparameter tuning utilities for PyTorch Lightning.

Wrappers around Lightning's Tuner for learning rate and batch size optimization.
Extracted from notebook 04_train_challenge1.py.
"""

import logging
from pathlib import Path
from typing import Any

import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.core.datamodule import LightningDataModule

logger = logging.getLogger(__name__)


def run_lr_finder(
    trainer: Trainer,
    model: LightningModule,
    datamodule: LightningDataModule,
    min_lr: float = 1e-8,
    max_lr: float = 1e-1,
    num_training: int = 200,
    mode: str = "exponential",
    output_dir: Path | None = None,
    wandb_logger: Any | None = None,
) -> float:
    """Run learning rate finder and optionally upload plot to wandb.

    Args:
        trainer: Configured Lightning Trainer
        model: LightningModule to tune
        datamodule: DataModule for training data
        min_lr: Minimum learning rate to try
        max_lr: Maximum learning rate to try
        num_training: Number of training iterations
        mode: Search mode ('exponential' or 'linear')
        output_dir: Directory to save LR finder plot (optional)
        wandb_logger: WandbLogger instance for uploading plot (optional)

    Returns:
        Suggested learning rate

    Example:
        >>> trainer = Trainer(...)
        >>> model = Challenge1Module(lr=1e-3)
        >>> datamodule = Challenge1DataModule(...)
        >>> suggested_lr = run_lr_finder(trainer, model, datamodule)
        >>> model.hparams.lr = suggested_lr
    """
    logger.info("\n" + "="*60)
    logger.info("[bold]Running learning rate finder...[/bold]")
    logger.info("="*60)

    # Capture original LR before lr_find modifies it
    original_lr = model.hparams.lr

    tuner = Tuner(trainer)

    # Run LR finder
    lr_finder = tuner.lr_find(
        model,
        datamodule,
        min_lr=min_lr,
        max_lr=max_lr,
        num_training=num_training,
        mode=mode,
        attr_name="lr"  # Update model.lr (or model.hparams.lr)
    )

    # Get suggestion
    suggested_lr = lr_finder.suggestion()

    logger.info(f"[green]Suggested LR:[/green] {suggested_lr:.6f}")
    logger.info(f"[yellow]Original LR:[/yellow] {original_lr:.6f}")

    # Save plot if output directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        lr_plot_path = output_dir / "lr_finder_plot.png"

        fig = lr_finder.plot(suggest=True)
        fig.savefig(lr_plot_path)
        logger.info(f"[green]LR finder plot saved to:[/green] {lr_plot_path}")

        # Upload to wandb if logger provided
        if wandb_logger is not None:
            wandb_logger.experiment.log({"lr_finder_plot": wandb.Image(str(lr_plot_path))})
            logger.info("[green]LR finder plot uploaded to wandb[/green]")

    # Update model hyperparameter
    model.hparams.lr = suggested_lr
    logger.info(f"[bold]Updated model LR to:[/bold] {suggested_lr:.6f}")

    return suggested_lr


def run_batch_size_finder(
    trainer: Trainer,
    model: LightningModule,
    datamodule: LightningDataModule,
    mode: str = "power",
    steps_per_trial: int = 3,
    init_val: int = 32,
    max_trials: int = 6,
) -> int:
    """Run batch size scaler to find optimal batch size.

    Args:
        trainer: Configured Lightning Trainer
        model: LightningModule to tune
        datamodule: DataModule to update batch size
        mode: Scaling mode ('power' or 'binsearch')
        steps_per_trial: Steps per trial
        init_val: Initial batch size
        max_trials: Maximum number of trials (caps scaling)

    Returns:
        Optimal batch size

    Example:
        >>> trainer = Trainer(...)
        >>> model = Challenge1Module(...)
        >>> datamodule = Challenge1DataModule(batch_size=128)
        >>> optimal_bs = run_batch_size_finder(trainer, model, datamodule)
        >>> # datamodule.batch_size automatically updated
    """
    logger.info("\n" + "="*60)
    logger.info("[bold]Running batch size scaler...[/bold]")
    logger.info("="*60)

    original_batch_size = datamodule.batch_size
    logger.info(f"[yellow]Original batch size:[/yellow] {original_batch_size}")

    tuner = Tuner(trainer)

    # Run batch size scaler
    tuner.scale_batch_size(
        model,
        datamodule,
        mode=mode,
        steps_per_trial=steps_per_trial,
        init_val=init_val,
        max_trials=max_trials  # Caps at init_val * 2^max_trials
    )

    new_batch_size = datamodule.batch_size
    logger.info(f"[green]Optimal batch size:[/green] {new_batch_size}")
    logger.info(f"[bold]Updated datamodule batch_size to:[/bold] {new_batch_size}")

    return new_batch_size
