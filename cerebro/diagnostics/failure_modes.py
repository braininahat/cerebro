"""Failure Mode Analysis for Model Autopsy.

Analyzes patterns in model failures to identify systematic issues:
- Top-K worst predictions (largest errors)
- Error distribution by metadata (subject, task, trial properties)
- Correlation between errors and input features
- Temporal/spatial patterns in failures

Usage:
    from cerebro.diagnostics.failure_modes import analyze_failure_modes

    failure_analysis = analyze_failure_modes(
        model=model,
        dataloader=val_loader,
        device=device,
        top_k=100,
        metadata_keys=["subject", "correct", "rt_from_stimulus"]
    )
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def analyze_failure_modes(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    top_k: int = 100,
    metadata_keys: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Analyze systematic patterns in model failures.

    Args:
        model: Trained model to analyze
        dataloader: DataLoader with validation/test data
        device: Device to run inference on
        top_k: Number of worst predictions to analyze in detail
        metadata_keys: List of metadata columns to correlate with errors
        output_dir: Directory to save plots (if None, plots not saved)

    Returns:
        Dictionary with failure analysis results:
            - top_k_failures: DataFrame with worst predictions
            - error_by_metadata: Dict mapping metadata keys to error statistics
            - error_distribution: Dict with error distribution stats
            - spatial_error_patterns: Per-channel error correlation
            - temporal_error_patterns: Per-timepoint error correlation
    """
    model.eval()
    if metadata_keys is None:
        metadata_keys = ["subject", "correct", "rt_from_stimulus"]

    logger.info("[bold cyan]FAILURE MODE ANALYSIS[/bold cyan]")
    logger.info(f"Analyzing top {top_k} worst predictions...")

    # Collect predictions and metadata
    all_predictions = []
    all_targets = []
    all_inputs = []
    all_metadata = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions"):
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

            # Extract metadata if available
            if len(batch) > 2:
                all_metadata.append(batch[2])

    # Concatenate results
    predictions = np.concatenate(all_predictions, axis=0).squeeze()
    targets = np.concatenate(all_targets, axis=0).squeeze()
    inputs = np.concatenate(all_inputs, axis=0)

    # Calculate errors
    errors = np.abs(predictions - targets)
    squared_errors = (predictions - targets) ** 2

    logger.info(f"\nError statistics:")
    logger.info(f"  MAE: {errors.mean():.4f}")
    logger.info(f"  RMSE: {np.sqrt(squared_errors.mean()):.4f}")
    logger.info(f"  Max error: {errors.max():.4f}")
    logger.info(f"  Min error: {errors.min():.4f}")

    # 1. Top-K worst predictions
    worst_indices = np.argsort(errors)[-top_k:][::-1]
    top_k_df = pd.DataFrame(
        {
            "sample_idx": worst_indices,
            "prediction": predictions[worst_indices],
            "target": targets[worst_indices],
            "error": errors[worst_indices],
            "squared_error": squared_errors[worst_indices],
        }
    )

    logger.info(f"\n[bold]Top 10 worst predictions:[/bold]")
    logger.info(top_k_df.head(10).to_string())

    # 2. Error distribution analysis
    error_distribution = {
        "mean": errors.mean(),
        "std": errors.std(),
        "median": np.median(errors),
        "q25": np.percentile(errors, 25),
        "q75": np.percentile(errors, 75),
        "q95": np.percentile(errors, 95),
        "q99": np.percentile(errors, 99),
        "skewness": stats.skew(errors),
        "kurtosis": stats.kurtosis(errors),
    }

    logger.info(f"\n[bold]Error distribution:[/bold]")
    logger.info(f"  Mean: {error_distribution['mean']:.4f}")
    logger.info(f"  Std: {error_distribution['std']:.4f}")
    logger.info(f"  Median: {error_distribution['median']:.4f}")
    logger.info(f"  95th percentile: {error_distribution['q95']:.4f}")
    logger.info(f"  99th percentile: {error_distribution['q99']:.4f}")
    logger.info(f"  Skewness: {error_distribution['skewness']:.4f}")

    # 3. Spatial error patterns (per-channel correlation with error)
    spatial_patterns = analyze_spatial_error_patterns(inputs, errors)

    # 4. Temporal error patterns (per-timepoint correlation with error)
    temporal_patterns = analyze_temporal_error_patterns(inputs, errors)

    # 5. Error by metadata (if available)
    error_by_metadata = {}
    if all_metadata and len(all_metadata) > 0:
        try:
            # Concatenate metadata from all batches
            metadata_list = []
            for m in all_metadata:
                if isinstance(m, dict):
                    metadata_list.append(pd.DataFrame([m]))
                elif hasattr(m, "__iter__"):
                    metadata_list.append(pd.DataFrame(m))

            if metadata_list:
                metadata_df = pd.concat(metadata_list, ignore_index=True)

                # Verify length matches
                if len(metadata_df) == len(errors):
                    metadata_df["error"] = errors

                    for key in metadata_keys:
                        if key in metadata_df.columns:
                            error_by_metadata[key] = (
                                metadata_df.groupby(key)["error"]
                                .agg(["mean", "std", "count"])
                                .sort_values("mean", ascending=False)
                            )

                            logger.info(f"\n[bold]Error by {key}:[/bold]")
                            logger.info(error_by_metadata[key].head(10).to_string())
                else:
                    logger.warning(
                        f"⚠️  Metadata length ({len(metadata_df)}) doesn't match errors ({len(errors)}). Skipping metadata analysis."
                    )
        except Exception as e:
            logger.warning(f"⚠️  Could not process metadata: {e}")

    # Generate plots
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_failure_modes(
            predictions=predictions,
            targets=targets,
            errors=errors,
            top_k_df=top_k_df,
            spatial_patterns=spatial_patterns,
            temporal_patterns=temporal_patterns,
            error_by_metadata=error_by_metadata,
            output_dir=output_dir,
        )
        logger.info(f"\n[green]Plots saved to:[/green] {output_dir}")

    return {
        "top_k_failures": top_k_df,
        "error_by_metadata": error_by_metadata,
        "error_distribution": error_distribution,
        "spatial_error_patterns": spatial_patterns,
        "temporal_error_patterns": temporal_patterns,
    }


def analyze_spatial_error_patterns(
    inputs: np.ndarray,
    errors: np.ndarray,
) -> Dict:
    """Analyze per-channel correlation with prediction errors.

    Args:
        inputs: Input data (N, C, T)
        errors: Prediction errors (N,)

    Returns:
        Dictionary with spatial pattern analysis
    """
    n_channels = inputs.shape[1]

    # Per-channel mean absolute amplitude
    channel_amps = np.abs(inputs).mean(axis=(0, 2))  # (C,)

    # Per-channel correlation with error
    channel_error_corr = np.zeros(n_channels)
    for ch in range(n_channels):
        ch_features = np.abs(inputs[:, ch, :]).mean(
            axis=1
        )  # (N,) - mean amplitude per sample
        channel_error_corr[ch], _ = stats.pearsonr(ch_features, errors)

    return {
        "channel_amplitudes": channel_amps,
        "channel_error_correlation": channel_error_corr,
        "top_error_correlated_channels": np.argsort(np.abs(channel_error_corr))[-10:][
            ::-1
        ],
    }


def analyze_temporal_error_patterns(
    inputs: np.ndarray,
    errors: np.ndarray,
) -> Dict:
    """Analyze per-timepoint correlation with prediction errors.

    Args:
        inputs: Input data (N, C, T)
        errors: Prediction errors (N,)

    Returns:
        Dictionary with temporal pattern analysis
    """
    n_times = inputs.shape[2]

    # Per-timepoint mean absolute amplitude
    time_amps = np.abs(inputs).mean(axis=(0, 1))  # (T,)

    # Per-timepoint correlation with error
    time_error_corr = np.zeros(n_times)
    for t in range(n_times):
        t_features = np.abs(inputs[:, :, t]).mean(
            axis=1
        )  # (N,) - mean amplitude per sample
        time_error_corr[t], _ = stats.pearsonr(t_features, errors)

    return {
        "time_amplitudes": time_amps,
        "time_error_correlation": time_error_corr,
        "high_error_timepoints": np.where(np.abs(time_error_corr) > 0.1)[0],
    }


def plot_failure_modes(
    predictions: np.ndarray,
    targets: np.ndarray,
    errors: np.ndarray,
    top_k_df: pd.DataFrame,
    spatial_patterns: Dict,
    temporal_patterns: Dict,
    error_by_metadata: Dict,
    output_dir: Path,
):
    """Generate comprehensive failure mode visualizations.

    Args:
        predictions: Model predictions (N,)
        targets: Ground truth targets (N,)
        errors: Absolute errors (N,)
        top_k_df: DataFrame with top-K worst predictions
        spatial_patterns: Spatial error pattern analysis
        temporal_patterns: Temporal error pattern analysis
        error_by_metadata: Error statistics by metadata
        output_dir: Directory to save plots
    """
    sns.set_style("whitegrid")

    # Plot 1: Error distribution and Q-Q plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram
    ax = axes[0]
    ax.hist(errors, bins=50, color="#2E86AB", alpha=0.7, edgecolor="black")
    ax.axvline(
        errors.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {errors.mean():.3f}",
    )
    ax.axvline(
        np.median(errors),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(errors):.3f}",
    )
    ax.set_xlabel("Absolute Error", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Error Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Q-Q plot
    ax = axes[1]
    stats.probplot(errors, dist="norm", plot=ax)
    ax.set_title(
        "Q-Q Plot: Errors vs Normal Distribution", fontsize=14, fontweight="bold"
    )
    ax.grid(alpha=0.3)

    # Prediction vs Target scatter
    ax = axes[2]
    ax.scatter(targets, predictions, alpha=0.3, s=10, color="#2E86AB")
    ax.plot(
        [targets.min(), targets.max()],
        [targets.min(), targets.max()],
        "r--",
        linewidth=2,
        label="Perfect prediction",
    )
    ax.set_xlabel("Target", fontsize=12)
    ax.set_ylabel("Prediction", fontsize=12)
    ax.set_title("Prediction vs Target", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "failure_error_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Plot 2: Spatial error patterns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Channel amplitudes
    ax = axes[0]
    ax.bar(
        range(len(spatial_patterns["channel_amplitudes"])),
        spatial_patterns["channel_amplitudes"],
        color="#2E86AB",
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xlabel("Channel Index", fontsize=12)
    ax.set_ylabel("Mean Amplitude", fontsize=12)
    ax.set_title("Per-Channel Mean Amplitude", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    # Channel error correlation
    ax = axes[1]
    corr = spatial_patterns["channel_error_correlation"]
    colors = ["red" if abs(c) > 0.1 else "#2E86AB" for c in corr]
    ax.bar(range(len(corr)), corr, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(
        0.1, color="red", linestyle="--", linewidth=1, label="|r| = 0.1 threshold"
    )
    ax.axhline(-0.1, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Channel Index", fontsize=12)
    ax.set_ylabel("Correlation with Error", fontsize=12)
    ax.set_title("Channel-Error Correlation", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "failure_spatial_patterns.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Plot 3: Temporal error patterns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Timepoint amplitudes
    ax = axes[0]
    ax.plot(temporal_patterns["time_amplitudes"], color="#2E86AB", linewidth=2)
    ax.set_xlabel("Time Sample", fontsize=12)
    ax.set_ylabel("Mean Amplitude", fontsize=12)
    ax.set_title("Temporal Mean Amplitude", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    # Timepoint error correlation
    ax = axes[1]
    corr = temporal_patterns["time_error_correlation"]
    ax.plot(corr, color="#2E86AB", linewidth=2)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(
        0.1, color="red", linestyle="--", linewidth=1, label="|r| = 0.1 threshold"
    )
    ax.axhline(-0.1, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Time Sample", fontsize=12)
    ax.set_ylabel("Correlation with Error", fontsize=12)
    ax.set_title("Temporal-Error Correlation", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "failure_temporal_patterns.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Plot 4: Top-K worst predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    top_20 = top_k_df.head(20)
    x = np.arange(len(top_20))
    width = 0.35

    ax.bar(
        x - width / 2,
        top_20["prediction"],
        width,
        label="Prediction",
        color="#2E86AB",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        top_20["target"],
        width,
        label="Target",
        color="#A23B72",
        alpha=0.7,
    )
    ax.set_xlabel("Sample Rank (worst → best)", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Top 20 Worst Predictions", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, 21))
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        output_dir / "failure_top20_predictions.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Plot 5: Error by metadata (if available)
    if error_by_metadata:
        n_metadata = len(error_by_metadata)
        fig, axes = plt.subplots(1, n_metadata, figsize=(6 * n_metadata, 5))
        if n_metadata == 1:
            axes = [axes]

        for ax, (key, stats_df) in zip(axes, error_by_metadata.items()):
            top_10 = stats_df.head(10)
            ax.barh(
                range(len(top_10)),
                top_10["mean"],
                xerr=top_10["std"],
                color="#2E86AB",
                alpha=0.7,
                edgecolor="black",
            )
            ax.set_yticks(range(len(top_10)))
            ax.set_yticklabels([str(idx)[:20] for idx in top_10.index])
            ax.set_xlabel("Mean Error", fontsize=12)
            ax.set_title(f"Top 10 by {key}", fontsize=14, fontweight="bold")
            ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(
            output_dir / "failure_error_by_metadata.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    logger.info(f"Saved 5 failure mode plots to {output_dir}")
