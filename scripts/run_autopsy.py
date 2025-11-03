#!/usr/bin/env python
"""Run ModelAutopsyCallback on a trained checkpoint without training.

Usage:
    python scripts/run_autopsy.py \\
        --checkpoint outputs/challenge1/20251021_205540/checkpoints/last.ckpt \\
        --config configs/codebook.yaml \\
        --output-dir outputs/autopsy_results

    # With specific diagnostics
    python scripts/run_autopsy.py \\
        --checkpoint last.ckpt \\
        --config configs/codebook.yaml \\
        --diagnostics predictions gradients activations

    # Analyze more samples
    python scripts/run_autopsy.py \\
        --checkpoint last.ckpt \\
        --config configs/codebook.yaml \\
        --num-samples 1000
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from cerebro.callbacks.model_autopsy import ModelAutopsyCallback
from cerebro.data.hbn import HBNDataModule

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path, config):
    """Load model from checkpoint based on config.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Config dict with model class_path

    Returns:
        Loaded Lightning module
    """
    # Get model class from config
    model_class_path = config["model"]["class_path"]
    module_path, class_name = model_class_path.rsplit(".", 1)

    # Import the class
    import importlib
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    model = model_class.load_from_checkpoint(checkpoint_path)

    return model


def run_autopsy(
    checkpoint_path,
    config_path,
    output_dir=None,
    diagnostics=None,
    num_samples=500,
    save_plots=True,
    log_to_wandb=False,
):
    """Run autopsy analysis on a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to config YAML
        output_dir: Output directory for results
        diagnostics: List of diagnostics to run (None = all)
        num_samples: Number of samples to analyze
        save_plots: Save diagnostic plots
        log_to_wandb: Upload to wandb
    """
    # Load config
    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Use OmegaConf for variable interpolation
    config = OmegaConf.to_container(OmegaConf.create(config), resolve=True)

    # Create output directory
    if output_dir is None:
        ckpt_name = Path(checkpoint_path).stem
        output_dir = Path("outputs") / "autopsy" / ckpt_name

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load model
    model = load_model_from_checkpoint(checkpoint_path, config)
    model.eval()

    # Load data
    print("Loading data...")
    data_config = config.get("data", {})
    datamodule = HBNDataModule(**data_config)
    datamodule.setup("fit")

    # Create a minimal trainer (just for callback compatibility)
    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
    )
    trainer.datamodule = datamodule

    # Create autopsy callback
    print("Creating autopsy callback...")
    autopsy_callback = ModelAutopsyCallback(
        run_on_training_end=False,  # We'll trigger manually
        run_on_early_stop=False,
        diagnostics=diagnostics or ["predictions", "gradients", "activations"],
        output_dir=output_dir,
        save_plots=save_plots,
        log_to_wandb=log_to_wandb,
        generate_report=True,
        num_samples=num_samples,
    )

    # Run autopsy manually
    print("\n" + "="*60)
    print("RUNNING MODEL AUTOPSY")
    print("="*60 + "\n")

    autopsy_callback._run_autopsy(
        trainer=trainer,
        pl_module=model,
        trigger="manual_analysis"
    )

    print("\n" + "="*60)
    print(f"✅ Autopsy complete! Results saved to: {output_dir}")
    print("="*60 + "\n")

    # Print report path
    report_path = output_dir / "autopsy_report.md"
    if report_path.exists():
        print(f"📋 Report: {report_path}")
        print("\nTo view the report:")
        print(f"  cat {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run model autopsy on a checkpoint")

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: outputs/autopsy/{checkpoint_name})"
    )
    parser.add_argument(
        "--diagnostics",
        nargs="+",
        default=None,
        help="Diagnostics to run (default: predictions gradients activations)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to analyze (default: 500)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Upload results to wandb"
    )

    args = parser.parse_args()

    run_autopsy(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        diagnostics=args.diagnostics,
        num_samples=args.num_samples,
        save_plots=not args.no_plots,
        log_to_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
