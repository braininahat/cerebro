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

from cerebro.utils.tuning import run_batch_size_finder, run_lr_finder
from cerebro.utils.logging import setup_logging
from cerebro.models.challenge1 import Challenge1Module
from cerebro.data.hbn import HBNDataModule
import logging
from datetime import datetime
from pathlib import Path

import torch
from lightning.pytorch.cli import LightningCLI

# Enable Tensor Cores on RTX 4090 for faster matmul operations
torch.set_float32_matmul_precision(
    "high"
)  # Trades minimal precision for 30-50% speedup

# Import modules to register with CLI

logger = logging.getLogger(__name__)


class CerebroCLI(LightningCLI):
    """Custom Lightning CLI with logging and tuning support.

    Adds support for:
    - Rich logging to console + file
    - Optional LR finder (set run_lr_finder=true in config)
    - Optional batch size finder (set run_batch_size_finder=true in config)
    - Comprehensive wandb config upload
    - Unique timestamped output directories per run
    - Fixed checkpoint resuming (removes _class_path validation conflicts)
    """

    def __init__(self, *args, **kwargs):
        """Initialize CLI with unique run timestamp and auto-fix checkpoints."""
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Auto-fix checkpoint BEFORE parent __init__ (before parsing/validation)
        self._auto_fix_checkpoint_in_args()

        super().__init__(*args, **kwargs)

    def _auto_fix_checkpoint_in_args(self):
        """Auto-fix checkpoint files before CLI parsing.

        Removes _class_path from checkpoint hyper_parameters to avoid
        validation conflicts when resuming training with --ckpt_path.

        Also extracts wandb metadata for automatic run resuming.
        """
        import sys

        # Check if --ckpt_path is in command line arguments
        try:
            if "--ckpt_path" in sys.argv:
                idx = sys.argv.index("--ckpt_path")
                if idx + 1 < len(sys.argv):
                    ckpt_path = sys.argv[idx + 1]

                    # Load and fix checkpoint
                    ckpt = torch.load(ckpt_path, map_location="cpu")

                    # Extract wandb metadata if available
                    if "wandb_metadata" in ckpt:
                        self.wandb_metadata = ckpt["wandb_metadata"]
                        print(
                            f"✓ Found wandb run ID in checkpoint: {self.wandb_metadata.get('wandb_run_id', 'N/A')}")
                    else:
                        self.wandb_metadata = None

                    # Remove hyper_parameters entirely (config file provides these)
                    # Keep only essential training state
                    modified = False
                    if "hyper_parameters" in ckpt:
                        del ckpt["hyper_parameters"]
                        modified = True
                    if "datamodule_hyper_parameters" in ckpt:
                        del ckpt["datamodule_hyper_parameters"]
                        modified = True
                    if "hparams_name" in ckpt:
                        del ckpt["hparams_name"]
                        modified = True
                    if "datamodule_hparams_name" in ckpt:
                        del ckpt["datamodule_hparams_name"]
                        modified = True

                    if modified:
                        torch.save(ckpt, ckpt_path)
                        print(
                            f"✓ Auto-fixed checkpoint (removed hyper_parameters): {ckpt_path}")
            else:
                self.wandb_metadata = None

        except Exception as e:
            self.wandb_metadata = None
            print(f"Warning: Could not auto-fix checkpoint: {e}")
            print("You may need to manually strip hyper_parameters from checkpoint")

    def before_instantiate_classes(self):
        """Inject unique timestamp into output paths before instantiating trainer/callbacks.

        Also injects wandb metadata for automatic run resuming.
        """
        # Only modify fit subcommand (not validate/test/predict)
        if self.subcommand != "fit":
            return

        cfg = self.config[self.subcommand]

        if "trainer" not in cfg:
            return

        trainer_cfg = cfg["trainer"]

        # Inject wandb metadata if resuming from checkpoint
        if self.wandb_metadata and "logger" in trainer_cfg:
            logger_cfg = trainer_cfg["logger"]

            # Check if it's a WandbLogger
            class_path = logger_cfg.get("class_path", "")
            if "WandbLogger" in class_path:
                if hasattr(logger_cfg, "init_args"):
                    # Override wandb settings to resume the run
                    logger_cfg.init_args["id"] = self.wandb_metadata["wandb_run_id"]
                    logger_cfg.init_args["resume"] = "must"

                    # Optionally override name, project, entity if they're in metadata
                    if "wandb_run_name" in self.wandb_metadata:
                        logger_cfg.init_args["name"] = self.wandb_metadata["wandb_run_name"]
                    if "wandb_project" in self.wandb_metadata:
                        logger_cfg.init_args["project"] = self.wandb_metadata["wandb_project"]
                    if "wandb_entity" in self.wandb_metadata:
                        logger_cfg.init_args["entity"] = self.wandb_metadata["wandb_entity"]

                    print(
                        f"✓ Auto-configured wandb to resume run: {self.wandb_metadata['wandb_run_id']}")

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
                    if (
                        hasattr(callback, "init_args")
                        and "dirpath" in callback.init_args
                    ):
                        base_dir = callback.init_args["dirpath"]
                        # Replace outputs/challenge1 with outputs/challenge1/TIMESTAMP
                        callback.init_args["dirpath"] = base_dir.replace(
                            "outputs/challenge1",
                            f"outputs/challenge1/{self.run_timestamp}",
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
            help="Run learning rate finder before training",
        )
        parser.add_argument(
            "--run_batch_size_finder",
            type=bool,
            default=False,
            help="Run batch size finder before training",
        )
        parser.add_argument(
            "--lr_finder_min", type=float, default=1e-8, help="Minimum LR for finder"
        )
        parser.add_argument(
            "--lr_finder_max", type=float, default=1e-1, help="Maximum LR for finder"
        )
        parser.add_argument(
            "--lr_finder_num_training",
            type=int,
            default=200,
            help="Number of training steps for LR finder",
        )
        parser.add_argument(
            "--bs_finder_mode",
            type=str,
            default="power",
            help="Batch size finder mode (power or binsearch)",
        )
        parser.add_argument(
            "--bs_finder_init_val",
            type=int,
            default=32,
            help="Initial batch size for finder",
        )
        parser.add_argument(
            "--bs_finder_max_trials",
            type=int,
            default=6,
            help="Maximum trials for batch size finder",
        )
        parser.add_argument(
            "--bs_finder_steps_per_trial",
            type=int,
            default=3,
            help="Number of training steps per batch size trial",
        )

    def before_fit(self):
        """Setup logging and optionally run tuners before training."""
        # Setup Rich logging
        self._setup_logging()

        # Log configuration
        self._log_config_to_wandb()

        # Run tuners if enabled (read from root config, not trainer)
        # IMPORTANT: Batch size finder must run FIRST because optimal LR depends on batch size
        # (larger batches → larger LR according to linear scaling rule)
        if self.config["fit"].get("run_batch_size_finder", False):
            self._run_batch_size_finder()

        if self.config["fit"].get("run_lr_finder", False):
            self._run_lr_finder()

    def _setup_logging(self):
        """Setup Rich logging to console and file."""
        # Get output directory from trainer logger
        if (
            hasattr(self.trainer.logger, "save_dir")
            and self.trainer.logger.save_dir is not None
        ):
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

        # Build comprehensive config dict - start with trainer config
        config_dict = {
            # Training parameters
            "epochs": self.trainer.max_epochs,
            "precision": str(self.trainer.precision),
            # Hyperparameter tuning switches
            "run_lr_finder": self.config["fit"].get("run_lr_finder", False),
            "run_batch_size_finder": self.config["fit"].get(
                "run_batch_size_finder", False
            ),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }

        # Add all datamodule hyperparameters (converted to dict)
        if hasattr(data_hparams, "__dict__"):
            for key, value in data_hparams.__dict__.items():
                if not key.startswith("_"):  # Skip private attributes
                    # Convert Path objects to strings for serialization
                    if hasattr(value, "__fspath__"):
                        value = str(value)
                    config_dict[f"data_{key}"] = value

        # Add all model hyperparameters (converted to dict)
        if hasattr(model_hparams, "__dict__"):
            for key, value in model_hparams.__dict__.items():
                if not key.startswith("_"):  # Skip private attributes
                    # Convert Path objects to strings for serialization
                    if hasattr(value, "__fspath__"):
                        value = str(value)
                    config_dict[f"model_{key}"] = value

        # Add dataset sizes if available
        if hasattr(self.datamodule, "train_set") and self.datamodule.train_set:
            config_dict["num_train_windows"] = len(self.datamodule.train_set)
        if hasattr(self.datamodule, "val_set") and self.datamodule.val_set:
            config_dict["num_val_windows"] = len(self.datamodule.val_set)
        if hasattr(self.datamodule, "test_set") and self.datamodule.test_set:
            config_dict["num_test_windows"] = len(self.datamodule.test_set)

        # Add model name if available
        if hasattr(self.model, "model") and hasattr(self.model.model, "__class__"):
            config_dict["model_name"] = self.model.model.__class__.__name__
        else:
            config_dict["model_name"] = self.model.__class__.__name__

        # Upload to wandb (only if experiment is initialized)
        try:
            if hasattr(wandb_logger.experiment, "config") and hasattr(
                wandb_logger.experiment.config, "update"
            ):
                wandb_logger.experiment.config.update(config_dict)
                logger.info("[green]Configuration uploaded to wandb[/green]")
            else:
                logger.info(
                    "[yellow]Wandb config upload skipped (offline/fast_dev_run mode)[/yellow]"
                )
        except Exception as e:
            logger.warning(
                f"[yellow]Could not upload config to wandb: {e}[/yellow]")

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
            wandb_logger.experiment.config.update(
                {
                    "lr_finder_enabled": True,
                    "lr_original": self.config["fit"]["model"]["lr"],
                    "lr_suggested": suggested_lr,
                }
            )

    def _run_batch_size_finder(self):
        """Run batch size finder."""
        original_bs = self.datamodule.hparams.batch_size

        # Read params from root config
        optimal_bs = run_batch_size_finder(
            trainer=self.trainer,
            model=self.model,
            datamodule=self.datamodule,
            mode=self.config["fit"].get("bs_finder_mode", "power"),
            steps_per_trial=self.config["fit"].get(
                "bs_finder_steps_per_trial", 3),
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
            wandb_logger.experiment.config.update(
                {
                    "batch_size_finder_enabled": True,
                    "batch_size_original": original_bs,
                    "batch_size_optimal": optimal_bs,
                }
            )


def cli_main():
    """CLI entry point.

    Supports two modes:
    1. Config-driven (recommended): Use class_path in model/data sections of YAML
       Example: model: {class_path: "cerebro.models.labram.tokenizer.VQNSP", ...}

    2. Hardcoded: If no class_path, uses Challenge1Module and Challenge1DataModule
       Example: model: {n_chans: 129, lr: 0.001, ...}
    """
    CerebroCLI(
        model_class=None,
        datamodule_class=HBNDataModule,
        save_config_callback=None,  # Wandb handles config saving
        # Use OmegaConf for interpolation
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    cli_main()
