"""Test script for Model Autopsy Callback.

Loads existing checkpoint and runs autopsy manually to validate implementation.
"""

import sys
from pathlib import Path

import torch

# Add cerebro to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from cerebro.callbacks.model_autopsy import ModelAutopsyCallback
from cerebro.data.challenge1 import Challenge1DataModule
from cerebro.models.challenge1 import Challenge1Module


def main():
    print("=" * 60)
    print("MODEL AUTOPSY TEST")
    print("=" * 60)

    # Load checkpoint
    checkpoint_path = "outputs/challenge1/20251013_181025/checkpoints/challenge1-epoch=16-val_nrmse=1.0007.ckpt"

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please update checkpoint_path to a valid checkpoint.")
        return

    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    try:
        model = Challenge1Module.load_from_checkpoint(checkpoint_path)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    # Setup data
    print("\n📊 Setting up data module...")
    datamodule = Challenge1DataModule(
        data_dir=REPO_ROOT / "data" / "full",
        releases=["R1", "R2", "R3", "R4"],
        batch_size=512,
        num_workers=8,
    )
    datamodule.setup()
    print(f"✓ Data loaded: {len(datamodule.val_set)} val samples")

    # Mock trainer
    class MockTrainer:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_dir = str(REPO_ROOT / "outputs" / "test_autopsy")
            self.datamodule = datamodule
            self.logger = None  # No wandb for test

            class MockCheckpointCallback:
                best_model_path = checkpoint_path

            self.checkpoint_callback = MockCheckpointCallback()

    trainer = MockTrainer()
    print(f"✓ Mock trainer created (device: {trainer.device})")

    # Run autopsy
    print("\n🔬 Running model autopsy...")
    print("-" * 60)

    callback = ModelAutopsyCallback(
        diagnostics=["predictions", "gradients", "activations"],
        save_plots=True,
        log_to_wandb=False,  # No wandb for test
        generate_report=True,
        num_samples=500,
    )

    callback._run_autopsy(trainer, model, "manual_test")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"\nCheck outputs at: {trainer.log_dir}/autopsy/")
    print("Expected files:")
    print("  - prediction_distribution.png")
    print("  - gradient_flow.png")
    print("  - activation_stats.png")
    print("  - autopsy_report.md")


if __name__ == "__main__":
    main()
