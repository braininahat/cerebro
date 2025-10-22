"""Callback to fix checkpoint compatibility issues with Lightning CLI.

When using Lightning CLI with class_path, checkpoints are saved with
_class_path in hyper_parameters, which causes validation errors when
resuming training. This callback removes _class_path after saving.
"""

import logging
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class CheckpointCompatibilityCallback(Callback):
    """Ensures checkpoints are compatible with Lightning CLI resuming.

    Automatically removes _class_path from hyper_parameters after each
    checkpoint save to prevent validation conflicts when resuming with --ckpt_path.

    Also saves wandb run ID for automatic resuming.

    This fixes the error:
        "Validation failed: Key '_class_path' is not expected"

    Usage:
        Add to your config callbacks:
        - class_path: cerebro.callbacks.CheckpointCompatibilityCallback
    """

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Remove _class_path and save wandb metadata before checkpoint is saved.

        Args:
            trainer: Lightning Trainer
            pl_module: Lightning Module
            checkpoint: Checkpoint dict being saved
        """
        # Remove _class_path from hyper_parameters
        if "hyper_parameters" in checkpoint and "_class_path" in checkpoint["hyper_parameters"]:
            del checkpoint["hyper_parameters"]["_class_path"]
            logger.debug("Removed _class_path from checkpoint for CLI compatibility")

        # Save wandb run metadata for automatic resuming
        wandb_metadata = {}
        for logger_instance in trainer.loggers:
            if hasattr(logger_instance, "experiment"):
                try:
                    # Extract wandb run info
                    if hasattr(logger_instance.experiment, "id"):
                        wandb_metadata["wandb_run_id"] = logger_instance.experiment.id
                    if hasattr(logger_instance.experiment, "name"):
                        wandb_metadata["wandb_run_name"] = logger_instance.experiment.name
                    if hasattr(logger_instance.experiment, "project"):
                        wandb_metadata["wandb_project"] = logger_instance.experiment.project
                    if hasattr(logger_instance.experiment, "entity"):
                        wandb_metadata["wandb_entity"] = logger_instance.experiment.entity
                    break
                except Exception as e:
                    logger.debug(f"Could not extract wandb metadata: {e}")

        if wandb_metadata:
            checkpoint["wandb_metadata"] = wandb_metadata
            logger.debug(f"Saved wandb metadata: run_id={wandb_metadata.get('wandb_run_id', 'N/A')}")
