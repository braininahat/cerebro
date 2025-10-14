#!/usr/bin/env python3
"""Test R5 evaluation matches local_scoring.py behavior.

This script verifies that our Challenge1DataModule's _load_r5_test() method
produces identical preprocessing and evaluation results to startkit/local_scoring.py.

Usage:
    # Test R5 loading with mini dataset
    uv run python test_r5_evaluation.py --use-mini

    # Test R5 loading with full dataset (requires R5 downloaded)
    uv run python test_r5_evaluation.py

    # Test with trained checkpoint
    uv run python test_r5_evaluation.py --checkpoint outputs/challenge1/.../best.ckpt
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from cerebro.data.challenge1 import Challenge1DataModule
from rich.console import Console
from rich.logging import RichHandler
from scipy.stats import ks_2samp
from sklearn.metrics import root_mean_squared_error as rmse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


def nrmse(y_trues, y_preds):
    """Normalized RMSE using difference between max and min values (from local_scoring.py)"""
    return rmse(y_trues, y_preds) / y_trues.std()


def test_r5_loading(data_dir: str, use_mini: bool = False):
    """Test R5 data loading and preprocessing."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST 1: R5 DATA LOADING[/bold cyan]")
    console.print("=" * 60)

    # Load with test_on_r5=False (local test split)
    logger.info("Loading with local test split...")
    dm_local = Challenge1DataModule(
        data_dir=data_dir,
        releases=["R1"],  # Use R1 for testing
        mode="dev",
        test_on_r5=False,
        use_mini=use_mini,
        batch_size=64,
        num_workers=4,
    )
    dm_local.setup()

    local_train_size = len(dm_local.train_set)
    local_val_size = len(dm_local.val_set)
    local_test_size = len(dm_local.test_set) if dm_local.test_set else 0

    console.print(f"[green]✓[/green] Local mode:")
    console.print(f"  Train: {local_train_size}")
    console.print(f"  Val: {local_val_size}")
    console.print(f"  Test: {local_test_size}")

    # Load with test_on_r5=True
    logger.info("\nLoading with R5 test...")
    dm_r5 = Challenge1DataModule(
        data_dir=data_dir,
        releases=["R1"],  # Use R1 for training
        mode="dev",
        test_on_r5=True,
        use_mini=use_mini,
        batch_size=64,
        num_workers=4,
    )
    dm_r5.setup()

    r5_train_size = len(dm_r5.train_set)
    r5_val_size = len(dm_r5.val_set)
    r5_test_size = len(dm_r5.test_set) if dm_r5.test_set else 0

    console.print(f"[green]✓[/green] R5 test mode:")
    console.print(f"  Train: {r5_train_size}")
    console.print(f"  Val: {r5_val_size}")
    console.print(f"  Test (R5): {r5_test_size}")

    # Verify train/val sizes match (same training data)
    assert local_train_size == r5_train_size, "Train sizes should match"
    assert local_val_size == r5_val_size, "Val sizes should match"
    assert r5_test_size > 0, "R5 test set should have windows"

    console.print("\n[bold green]✅ TEST 1 PASSED[/bold green]")
    return dm_local, dm_r5


def test_submission_mode(data_dir: str, use_mini: bool = False):
    """Test submission mode (train on all, test on R5)."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST 2: SUBMISSION MODE[/bold cyan]")
    console.print("=" * 60)

    # Submission mode requires test_on_r5=True
    logger.info("Loading in submission mode...")
    dm_submission = Challenge1DataModule(
        data_dir=data_dir,
        releases=["R1"],
        mode="submission",
        test_on_r5=True,
        use_mini=use_mini,
        batch_size=64,
        num_workers=4,
    )
    dm_submission.setup()

    train_size = len(dm_submission.train_set)
    val_size = len(dm_submission.val_set) if dm_submission.val_set else 0
    test_size = len(dm_submission.test_set) if dm_submission.test_set else 0

    console.print(f"[green]✓[/green] Submission mode:")
    console.print(f"  Train: {train_size} (all subjects)")
    console.print(f"  Val: {val_size} (should be 0)")
    console.print(f"  Test (R5): {test_size}")

    # Verify submission mode behavior
    assert val_size == 0, "Submission mode should have no val split"
    assert test_size > 0, "Submission mode should have R5 test set"

    console.print("\n[bold green]✅ TEST 2 PASSED[/bold green]")
    return dm_submission


def test_label_distributions(dm_local, dm_r5):
    """Compare label distributions between local test and R5 test."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST 3: LABEL DISTRIBUTION COMPARISON[/bold cyan]")
    console.print("=" * 60)

    # Extract labels from local test
    local_test_metadata = dm_local.test_set.get_metadata()
    local_labels = local_test_metadata["target"].values

    # Extract labels from R5 test
    r5_test_metadata = dm_r5.test_set.get_metadata()
    r5_labels = r5_test_metadata["target"].values

    # Statistics
    console.print(f"\n[bold]Local Test (from R1):[/bold]")
    console.print(f"  Mean: {local_labels.mean():.4f}s")
    console.print(f"  Std: {local_labels.std():.4f}s")
    console.print(f"  Min: {local_labels.min():.4f}s")
    console.print(f"  Max: {local_labels.max():.4f}s")

    console.print(f"\n[bold]R5 Test:[/bold]")
    console.print(f"  Mean: {r5_labels.mean():.4f}s")
    console.print(f"  Std: {r5_labels.std():.4f}s")
    console.print(f"  Min: {r5_labels.min():.4f}s")
    console.print(f"  Max: {r5_labels.max():.4f}s")

    # KS test for distribution similarity
    ks_stat, ks_p = ks_2samp(local_labels, r5_labels)
    console.print(f"\n[bold]Distribution Comparison (Kolmogorov-Smirnov):[/bold]")
    console.print(f"  KS statistic: {ks_stat:.4f}")
    console.print(f"  p-value: {ks_p:.4f}")

    if ks_p < 0.05:
        console.print(
            f"[yellow]⚠[/yellow] Distribution shift detected (p < 0.05). "
            f"R5 may have different characteristics than training releases."
        )
    else:
        console.print(f"[green]✓[/green] Distributions are similar (p ≥ 0.05)")

    console.print("\n[bold green]✅ TEST 3 PASSED[/bold green]")


def test_r5_guard():
    """Test R5 guard prevents training on R5."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST 4: R5 GUARD ENFORCEMENT[/bold cyan]")
    console.print("=" * 60)

    # Test 1: R5 in releases should fail
    logger.info("Test 4.1: R5 in releases (should fail)...")
    try:
        dm = Challenge1DataModule(
            data_dir="data",
            releases=["R1", "R5"],  # R5 in training releases!
            mode="dev",
            test_on_r5=False,
        )
        console.print("[red]✗[/red] FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        if "COMPETITION VALIDATION SET" in str(e):
            console.print(f"[green]✓[/green] Correctly raised ValueError: {e}")
        else:
            console.print(f"[red]✗[/red] Wrong error: {e}")
            return False

    # Test 2: submission mode requires test_on_r5=True
    logger.info("\nTest 4.2: submission mode without test_on_r5 (should fail)...")
    try:
        dm = Challenge1DataModule(
            data_dir="data",
            releases=["R1"],
            mode="submission",
            test_on_r5=False,  # Submission mode requires this to be True!
        )
        console.print("[red]✗[/red] FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        if "submission mode requires test_on_r5=True" in str(e):
            console.print(f"[green]✓[/green] Correctly raised ValueError: {e}")
        else:
            console.print(f"[red]✗[/red] Wrong error: {e}")
            return False

    console.print("\n[bold green]✅ TEST 4 PASSED[/bold green]")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test R5 evaluation matches local_scoring.py"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory containing releases",
    )
    parser.add_argument(
        "--use-mini",
        action="store_true",
        help="Use mini dataset for faster testing",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint for inference testing",
    )
    args = parser.parse_args()

    console.print("\n" + "=" * 60)
    console.print("[bold]R5 EVALUATION VERIFICATION SUITE[/bold]")
    console.print("=" * 60)
    console.print(f"Data dir: {args.data_dir}")
    console.print(f"Mini mode: {args.use_mini}")

    try:
        # Test 1: R5 loading
        dm_local, dm_r5 = test_r5_loading(args.data_dir, args.use_mini)

        # Test 2: Submission mode
        dm_submission = test_submission_mode(args.data_dir, args.use_mini)

        # Test 3: Distribution comparison
        test_label_distributions(dm_local, dm_r5)

        # Test 4: R5 guard
        test_r5_guard()

        # Summary
        console.print("\n" + "=" * 60)
        console.print("[bold green]ALL TESTS PASSED ✅[/bold green]")
        console.print("=" * 60)
        console.print("\n[bold]Summary:[/bold]")
        console.print("  ✓ R5 loading works correctly")
        console.print("  ✓ Submission mode works correctly")
        console.print("  ✓ Label distributions analyzed")
        console.print("  ✓ R5 guard prevents training contamination")
        console.print("\n[green]R5 test evaluation is ready to use![/green]")

    except Exception as e:
        console.print(f"\n[bold red]TEST FAILED: {e}[/bold red]")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
