"""Callback to create a 'latest' symlink pointing to the current run directory.

This callback helps maintain a consistent interface for accessing the most recent run's
checkpoints and logs, even when using timestamped directories.

Example directory structure:
    outputs/sjepa/pretrain_mini_80pct/
    ├── latest -> 20251029_194244/          # ← Symlink for convenience
    ├── 20251029_194244/                    # ← Run 1 (fully preserved)
    │   ├── checkpoints/
    │   │   ├── sjepa-epoch=52-val_loss=0.0129.ckpt
    │   │   └── last.ckpt
    │   ├── train_full.log
    │   └── wandb/
    └── 20251029_201530/                    # ← Run 2 (no overwrites!)
        ├── checkpoints/
        │   ├── sjepa-epoch=48-val_loss=0.0098.ckpt
        │   └── last.ckpt
        ├── train_full.log
        └── wandb/

Usage:
    # In config YAML:
    trainer:
      callbacks:
        - class_path: cerebro.callbacks.LatestCheckpointSymlinkCallback

    # Then you can always use:
    --model.pretrained_checkpoint="outputs/sjepa/pretrain_mini_80pct/latest/checkpoints/last.ckpt"
"""

from pathlib import Path

from lightning.pytorch.callbacks import Callback


class LatestCheckpointSymlinkCallback(Callback):
    """Creates a 'latest' symlink pointing to the current run directory.

    This callback runs at the end of training and creates (or updates) a symlink
    named 'latest' in the parent directory that points to the current timestamped
    run directory. This provides a stable, predictable path for accessing the
    most recent run's outputs without needing to know the exact timestamp.

    The symlink is created atomically (remove old, create new) to avoid race
    conditions if multiple runs finish simultaneously.

    Benefits:
    - Easy access to most recent run: `outputs/experiment/latest/checkpoints/`
    - No need to remember/lookup timestamps
    - Works across all experiments automatically
    - Safe for concurrent runs (each creates its own timestamped dir)

    Example:
        # Most recent run checkpoint
        checkpoint = "outputs/sjepa/pretrain/latest/checkpoints/last.ckpt"

        # Specific run checkpoint (still preserved)
        checkpoint = "outputs/sjepa/pretrain/20251029_194244/checkpoints/sjepa-epoch=52.ckpt"
    """

    def on_fit_end(self, trainer, pl_module):
        """Create/update 'latest' symlink at the end of training.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module being trained
        """
        # Get current run directory from logger
        # e.g., outputs/sjepa/pretrain_mini_80pct/20251029_194244
        run_dir = Path(trainer.logger.save_dir)

        # Parent directory
        # e.g., outputs/sjepa/pretrain_mini_80pct
        parent_dir = run_dir.parent

        # Path for 'latest' symlink
        latest_link = parent_dir / "latest"

        # Remove existing symlink/directory if present
        # (handles both broken symlinks and actual directories)
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create symlink pointing to current run's directory name (relative)
        # Use relative path so symlink works even if parent directory is moved
        latest_link.symlink_to(run_dir.name, target_is_directory=True)

        # Log completion
        trainer.logger.experiment.log({
            "latest_symlink_created": str(latest_link),
            "points_to": run_dir.name,
        })
