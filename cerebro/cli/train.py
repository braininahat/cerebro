"""Custom Lightning CLI for Cerebro training.

Extends LightningCLI to support:
- Rich logging setup (console + file)
- Optional LR finder with wandb plot upload
- Optional batch size finder
- Comprehensive wandb config logging

Usage:
    # Basic training
    uv run cerebro fit --config configs/challenge1_base.yaml

    # With LR finder
    uv run cerebro fit \\
        --config configs/challenge1_base.yaml \\
        --run_lr_finder true

    # Override parameters
    uv run cerebro fit \\
        --config configs/challenge1_base.yaml \\
        --model.init_args.lr 0.0001 \\
        --data.init_args.batch_size 256

    # Backward compatible (direct python)
    uv run python cerebro/cli/train.py fit --config configs/challenge1_base.yaml
"""

import logging
from datetime import datetime
from pathlib import Path

import torch
from lightning.pytorch.cli import LightningCLI

# Enable Tensor Cores on RTX 4090 for faster matmul operations
torch.set_float32_matmul_precision('high')  # Trades minimal precision for 30-50% speedup

# Import modules to register with CLI
from cerebro.data.challenge1 import Challenge1DataModule
from cerebro.models.challenge1 import Challenge1Module
from cerebro.utils.logging import setup_logging
from cerebro.utils.tuning import run_lr_finder, run_batch_size_finder

logger = logging.getLogger(__name__)


class CerebroCLI(LightningCLI):
    """Custom Lightning CLI with logging and tuning support.

    Adds support for:
    - Rich logging to console + file
    - Optional LR finder (set run_lr_finder=true in config)
    - Optional batch size finder (set run_batch_size_finder=true in config)
    - Comprehensive wandb config upload
    - Unique timestamped output directories per run
    """

    def __init__(self, *args, **kwargs):
        """Initialize CLI with unique run timestamp."""
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        super().__init__(*args, **kwargs)

    def before_instantiate_classes(self):
        """Inject unique timestamp into output paths before instantiating trainer/callbacks."""
        # Only modify fit subcommand (not validate/test/predict)
        if self.subcommand != "fit":
            return

        cfg = self.config[self.subcommand]

        if "trainer" not in cfg:
            return

        trainer_cfg = cfg["trainer"]

        # Update WandbLogger save_dir
        if "logger" in trainer_cfg:
            logger_cfg = trainer_cfg["logger"]
            if hasattr(logger_cfg, "init_args") and "save_dir" in logger_cfg.init_args:
                base_dir = logger_cfg.init_args["save_dir"]
                logger_cfg.init_args["save_dir"] = f"{base_dir}/{self.run_timestamp}"

        # Update ModelCheckpoint dirpath
        if "callbacks" in trainer_cfg:
            for callback in trainer_cfg["callbacks"]:
                class_path = callback.get("class_path", "")
                if "ModelCheckpoint" in class_path:
                    if hasattr(callback, "init_args") and "dirpath" in callback.init_args:
                        base_dir = callback.init_args["dirpath"]
                        # Replace outputs/challenge1 with outputs/challenge1/TIMESTAMP
                        callback.init_args["dirpath"] = base_dir.replace(
                            "outputs/challenge1",
                            f"outputs/challenge1/{self.run_timestamp}"
                        )

    def add_arguments_to_parser(self, parser):
        """Add custom arguments to parser.

        Args:
            parser: Lightning CLI argument parser
        """
        # Add tuning flags at root level (not under trainer)
        parser.add_argument(
            "--run_lr_finder",
            type=bool,
            default=False,
            help="Run learning rate finder before training"
        )
        parser.add_argument(
            "--run_batch_size_finder",
            type=bool,
            default=False,
            help="Run batch size finder before training"
        )
        parser.add_argument(
            "--lr_finder_min",
            type=float,
            default=1e-8,
            help="Minimum LR for finder"
        )
        parser.add_argument(
            "--lr_finder_max",
            type=float,
            default=1e-1,
            help="Maximum LR for finder"
        )
        parser.add_argument(
            "--lr_finder_num_training",
            type=int,
            default=200,
            help="Number of training steps for LR finder"
        )
        parser.add_argument(
            "--bs_finder_mode",
            type=str,
            default="power",
            help="Batch size finder mode (power or binsearch)"
        )
        parser.add_argument(
            "--bs_finder_init_val",
            type=int,
            default=32,
            help="Initial batch size for finder"
        )
        parser.add_argument(
            "--bs_finder_max_trials",
            type=int,
            default=6,
            help="Maximum trials for batch size finder"
        )

    def before_fit(self):
        """Setup logging and optionally run tuners before training."""
        # Setup Rich logging
        self._setup_logging()

        # Log configuration
        self._log_config_to_wandb()

        # Run tuners if enabled (read from root config, not trainer)
        if self.config["fit"].get("run_lr_finder", False):
            self._run_lr_finder()

        if self.config["fit"].get("run_batch_size_finder", False):
            self._run_batch_size_finder()

    def _setup_logging(self):
        """Setup Rich logging to console and file."""
        # Get output directory from trainer logger
        if hasattr(self.trainer.logger, "save_dir") and self.trainer.logger.save_dir is not None:
            log_dir = Path(self.trainer.logger.save_dir)
        else:
            # Fallback to outputs directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path("outputs") / "challenge1" / timestamp

        log_dir.mkdir(parents=True, exist_ok=True)

        # Determine mini or full from config
        use_mini = self.datamodule.hparams.get("use_mini", False)
        log_file = log_dir / f"train_{'mini' if use_mini else 'full'}.log"

        # Setup logging
        setup_logging(
            log_file=log_file,
            logger_name="cerebro",
            log_level=logging.INFO,
            rich_tracebacks=True,
            rich_markup=True,
        )

        logger.info(f"[bold green]Logging to:[/bold green] {log_file}")

    def _log_config_to_wandb(self):
        """Log comprehensive configuration to wandb (mimics notebook 04 lines 599-642)."""
        # Only log if wandb logger is configured
        wandb_logger = None
        for logger_instance in self.trainer.loggers:
            if hasattr(logger_instance, "experiment"):
                wandb_logger = logger_instance
                break

        if wandb_logger is None:
            return

        # Get datamodule and model hyperparameters
        data_hparams = self.datamodule.hparams
        model_hparams = self.model.hparams

        # Build comprehensive config dict
        config_dict = {
            # Data settings
            "data_dir": str(data_hparams.data_dir),
            "releases": data_hparams.releases,
            "num_releases": len(data_hparams.releases),
            "use_mini": data_hparams.use_mini,
            "excluded_subjects": data_hparams.excluded_subjects,

            # Windowing parameters
            "epoch_len_s": data_hparams.epoch_len_s,
            "sfreq": data_hparams.sfreq,
            "anchor": data_hparams.anchor,
            "shift_after_stim": data_hparams.shift_after_stim,
            "window_len": data_hparams.window_len,

            # Training parameters
            "batch_size": data_hparams.batch_size,
            "epochs": self.trainer.max_epochs,
            "lr": model_hparams.lr,
            "weight_decay": model_hparams.weight_decay,
            "precision": str(self.trainer.precision),

            # Hyperparameter tuning switches
            "run_lr_finder": self.config["fit"].get("run_lr_finder", False),
            "run_batch_size_finder": self.config["fit"].get("run_batch_size_finder", False),

            # Splits
            "val_frac": data_hparams.val_frac,
            "test_frac": data_hparams.test_frac,
            "seed": data_hparams.seed,

            # Dataset sizes (will be populated after setup)
            "num_train_windows": len(self.datamodule.train_set) if self.datamodule.train_set else 0,
            "num_val_windows": len(self.datamodule.val_set) if self.datamodule.val_set else 0,
            "num_test_windows": len(self.datamodule.test_set) if self.datamodule.test_set else 0,

            # Model architecture
            "model_name": self.model.model.__class__.__name__,
            "n_chans": model_hparams.n_chans,
            "n_times": model_hparams.n_times,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }

        # Upload to wandb (only if experiment is initialized)
        try:
            if hasattr(wandb_logger.experiment, 'config') and hasattr(wandb_logger.experiment.config, 'update'):
                wandb_logger.experiment.config.update(config_dict)
                logger.info("[green]Configuration uploaded to wandb[/green]")
            else:
                logger.info("[yellow]Wandb config upload skipped (offline/fast_dev_run mode)[/yellow]")
        except Exception as e:
            logger.warning(f"[yellow]Could not upload config to wandb: {e}[/yellow]")

    def _run_lr_finder(self):
        """Run learning rate finder with wandb plot upload."""
        # Get wandb logger
        wandb_logger = None
        for logger_instance in self.trainer.loggers:
            if hasattr(logger_instance, "experiment"):
                wandb_logger = logger_instance
                break

        # Get output directory
        if hasattr(self.trainer.logger, "save_dir"):
            output_dir = Path(self.trainer.logger.save_dir)
        else:
            output_dir = Path("outputs") / "challenge1"

        # Run LR finder (read params from root config)
        suggested_lr = run_lr_finder(
            trainer=self.trainer,
            model=self.model,
            datamodule=self.datamodule,
            min_lr=self.config["fit"].get("lr_finder_min", 1e-8),
            max_lr=self.config["fit"].get("lr_finder_max", 1e-1),
            num_training=self.config["fit"].get("lr_finder_num_training", 200),
            mode="exponential",
            output_dir=output_dir,
            wandb_logger=wandb_logger,
        )

        # Log to wandb
        if wandb_logger is not None:
            wandb_logger.experiment.config.update({
                "lr_finder_enabled": True,
                "lr_original": self.config["fit"]["model"]["lr"],
                "lr_suggested": suggested_lr,
            })

    def _run_batch_size_finder(self):
        """Run batch size finder."""
        original_bs = self.datamodule.hparams.batch_size

        # Read params from root config
        optimal_bs = run_batch_size_finder(
            trainer=self.trainer,
            model=self.model,
            datamodule=self.datamodule,
            mode=self.config["fit"].get("bs_finder_mode", "power"),
            init_val=self.config["fit"].get("bs_finder_init_val", 32),
            max_trials=self.config["fit"].get("bs_finder_max_trials", 6),
        )

        # Log to wandb
        wandb_logger = None
        for logger_instance in self.trainer.loggers:
            if hasattr(logger_instance, "experiment"):
                wandb_logger = logger_instance
                break

        if wandb_logger is not None:
            wandb_logger.experiment.config.update({
                "batch_size_finder_enabled": True,
                "batch_size_original": original_bs,
                "batch_size_optimal": optimal_bs,
            })


def cli_main():
    """CLI entry point."""
    CerebroCLI(
        Challenge1Module,
        Challenge1DataModule,
        save_config_callback=None,  # Wandb handles config saving
        parser_kwargs={"parser_mode": "omegaconf"},  # Use OmegaConf for interpolation
    )


if __name__ == "__main__":
    cli_main()
