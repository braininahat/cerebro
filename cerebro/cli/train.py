"""Custom Lightning CLI for Cerebro training.

Extends LightningCLI to support:
- Rich logging setup (console + file)
- Optional LR finder with wandb plot upload
- Optional batch size finder
- Comprehensive wandb config logging
- Flexible model/trainer/data composition

The new architecture separates models (pure architectures) from trainers
(training logic), enabling mix-and-match experimentation.

Usage:
    # Supervised training
    uv run cerebro fit --config configs/supervised_eegnex_challenge1.yaml

    # Contrastive pretraining
    uv run cerebro fit --config configs/contrastive_eegnex_movies.yaml

    # With LR finder
    uv run cerebro fit \\
        --config configs/supervised_eegnex_challenge1.yaml \\
        --run_lr_finder true

    # Override parameters
    uv run cerebro fit \\
        --config configs/supervised_eegnex_challenge1.yaml \\
        --trainer.init_args.lr 0.0001 \\
        --data.init_args.batch_size 256

    # Direct python invocation
    uv run python cerebro/cli/train.py fit --config configs/supervised_eegnex_challenge1.yaml
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lightning.pytorch.cli import LightningCLI
from omegaconf import OmegaConf

# Enable Tensor Cores on RTX 4090 for faster matmul operations
torch.set_float32_matmul_precision(
    "high"
)  # Trades minimal precision for 30-50% speedup

# Import utilities
from cerebro.utils.logging import setup_logging
from cerebro.utils.tuning import run_batch_size_finder, run_lr_finder

logger = logging.getLogger(__name__)


class CerebroCLI(LightningCLI):
    """Custom Lightning CLI with logging and tuning support.

    Adds support for:
    - Rich logging to console + file
    - Optional LR finder (set run_lr_finder=true in config)
    - Optional batch size finder (set run_batch_size_finder=true in config)
    - Comprehensive wandb config upload
    - Unique timestamped output directories per run
    - Flexible model/trainer/data composition via config

    The new architecture separates:
    - Models: Pure nn.Module architectures (RegressorModel, ContrastiveModel)
    - Trainers: LightningModule with training logic (SupervisedTrainer, ContrastiveTrainer)
    - Data: Task-specific DataModules (Challenge1DataModule, MovieDataModule)

    This allows mix-and-match composition through configuration files.
    """

    def __init__(self, *args, **kwargs):
        """Initialize CLI with unique run timestamp.

        We set model_class and datamodule_class to None to allow
        configs to specify them via class_path.
        """
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.wandb_metadata = None  # For checkpoint resuming (populated by CheckpointCompatibilityCallback)

        # Override defaults to allow class_path in configs
        kwargs.setdefault('model_class', None)
        kwargs.setdefault('datamodule_class', None)
        kwargs.setdefault('subclass_mode_model', False)
        kwargs.setdefault('subclass_mode_data', False)

        super().__init__(*args, **kwargs)

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
        """Log comprehensive configuration to wandb.

        Handles both old architecture (Challenge1Module) and new architecture
        (separate model/trainer) for backward compatibility.
        """
        # Only log if wandb logger is configured
        wandb_logger = None
        for logger_instance in self.trainer.loggers:
            if hasattr(logger_instance, "experiment"):
                wandb_logger = logger_instance
                break

        if wandb_logger is None:
            return

        # Get datamodule hyperparameters
        data_hparams = self.datamodule.hparams if self.datamodule else {}

        # Build config dict - handle different architectures
        config_dict = {
            # Data settings (if available)
            "data_class": self.datamodule.__class__.__name__ if self.datamodule else None,
        }

        # Add data-specific parameters
        if hasattr(data_hparams, "data_dir"):
            config_dict["data_dir"] = str(data_hparams.data_dir)
        if hasattr(data_hparams, "releases"):
            config_dict["releases"] = data_hparams.releases
            config_dict["num_releases"] = len(data_hparams.releases)
        if hasattr(data_hparams, "use_mini"):
            config_dict["use_mini"] = data_hparams.use_mini
        if hasattr(data_hparams, "batch_size"):
            config_dict["batch_size"] = data_hparams.batch_size

        # Model and trainer info
        config_dict["model_class"] = self.model.__class__.__name__
        config_dict["epochs"] = self.trainer.max_epochs
        config_dict["precision"] = str(self.trainer.precision)

        # Get trainer hyperparameters (new architecture)
        if hasattr(self.model, "hparams"):
            trainer_hparams = self.model.hparams
            if hasattr(trainer_hparams, "lr"):
                config_dict["lr"] = trainer_hparams.lr
            if hasattr(trainer_hparams, "weight_decay"):
                config_dict["weight_decay"] = trainer_hparams.weight_decay

            # For new architecture with separate model
            if hasattr(trainer_hparams, "model"):
                actual_model = trainer_hparams.model
                config_dict["actual_model_class"] = actual_model.__class__.__name__
                config_dict["num_parameters"] = sum(p.numel() for p in actual_model.parameters())
            else:
                config_dict["num_parameters"] = sum(p.numel() for p in self.model.parameters())

        # Hyperparameter tuning switches
        config_dict["run_lr_finder"] = self.config["fit"].get("run_lr_finder", False)
        config_dict["run_batch_size_finder"] = self.config["fit"].get("run_batch_size_finder", False)

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
            # Try to get original LR from config (handle both flat and nested structures)
            try:
                # Try nested init_args first (class_path style)
                lr_original = self.config["fit"]["model"]["init_args"]["lr"]
            except (KeyError, TypeError):
                try:
                    # Fall back to flat structure
                    lr_original = self.config["fit"]["model"]["lr"]
                except (KeyError, TypeError):
                    # Fall back to model's current hparams
                    lr_original = self.model.hparams.lr if hasattr(self.model.hparams, "lr") else None

            wandb_logger.experiment.config.update(
                {
                    "lr_finder_enabled": True,
                    "lr_original": lr_original,
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

    Now uses config files to specify model and data classes via class_path,
    enabling flexible composition of models, trainers, and data modules.

    Example config structure:
        model:
            class_path: cerebro.trainers.supervised.SupervisedTrainer
            init_args:
                model:
                    class_path: cerebro.models.architectures.RegressorModel
                    init_args:
                        encoder_class: EEGNeX
                loss_fn: mse
                lr: 0.001

        data:
            class_path: cerebro.data.challenge1.Challenge1DataModule
            init_args:
                releases: [R1, R2, ...]
                batch_size: 512
    """
    # Load environment variables from .env file
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    # Register OmegaConf resolver for ${now:format} in configs
    # Caches first call to ensure single timestamp per run
    _timestamp_cache = {}

    def cached_now_resolver(format_str):
        if 'timestamp' not in _timestamp_cache:
            _timestamp_cache['timestamp'] = datetime.now().strftime(format_str)
        return _timestamp_cache['timestamp']

    OmegaConf.register_new_resolver("now", cached_now_resolver, replace=True)

    CerebroCLI(
        save_config_callback=None,  # Wandb handles config saving
        # Use OmegaConf for interpolation
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    cli_main()
