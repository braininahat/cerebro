"""Visualization functions for diagnostic results."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_distribution(diagnostics: dict, output_path: Path) -> None:
    """Plots prediction vs target distributions and residual analysis.

    Creates a 2-panel figure:
    - Left: Overlayed histograms of predictions vs targets
    - Right: Residual plot (errors vs predicted values)

    Args:
        diagnostics: Output from analyze_predictions()
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Distribution comparison
    ax = axes[0]
    ax.hist(
        [diagnostics["targets"], diagnostics["predictions"]],
        bins=50,
        label=["Targets", "Predictions"],
        alpha=0.7,
        color=["#2E86AB", "#A23B72"],
    )
    ax.axvline(
        diagnostics["target_mean"],
        color="#2E86AB",
        linestyle="--",
        linewidth=2,
        label=f'Target mean: {diagnostics["target_mean"]:.3f}',
    )
    ax.axvline(
        diagnostics["pred_mean"],
        color="#A23B72",
        linestyle="--",
        linewidth=2,
        label=f'Pred mean: {diagnostics["pred_mean"]:.3f}',
    )
    ax.set_xlabel("Response Time (s)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right panel: Residual plot
    ax = axes[1]
    ax.scatter(
        diagnostics["predictions"],
        diagnostics["residuals"],
        alpha=0.4,
        s=20,
        color="#F18F01",
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=2, label="Zero error")
    ax.set_xlabel("Predicted RT (s)", fontsize=11)
    ax.set_ylabel("Residual (Pred - True)", fontsize=11)
    ax.set_title("Residual Analysis", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Add text box with key statistics
    textstr = "\n".join(
        [
            f"NRMSE: {diagnostics['nrmse']:.4f}",
            f"RMSE: {diagnostics['rmse']:.4f}",
            f"Variance Ratio: {diagnostics['variance_ratio']:.2f}",
        ]
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gradient_flow(diagnostics: dict, output_path: Path) -> None:
    """Plots gradient norms per layer.

    Creates a bar chart showing gradient L2 norms for each layer,
    using log scale for better visibility of small gradients.

    Args:
        diagnostics: Output from analyze_gradient_flow()
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    layer_names = diagnostics["layer_names"]
    grad_norms = diagnostics["grad_norms"]

    # Shorten layer names for display (keep last 2 parts)
    short_names = [
        ".".join(name.split(".")[-2:]) if len(name.split(".")) > 2 else name
        for name in layer_names
    ]

    # Color bars based on magnitude (red = very small, green = healthy)
    colors = [
        "#E63946" if gn < 1e-6 else "#06D6A0" if gn > 1e-4 else "#FFB703"
        for gn in grad_norms
    ]

    bars = ax.bar(
        range(len(grad_norms)),
        grad_norms,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(grad_norms)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Gradient L2 Norm", fontsize=11)
    ax.set_title("Gradient Flow Through Layers", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#E63946", label="Dead (<1e-6)"),
        Patch(facecolor="#FFB703", label="Small (1e-6 to 1e-4)"),
        Patch(facecolor="#06D6A0", label="Healthy (>1e-4)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Highlight dead layers
    if diagnostics["dead_layers"]:
        dead_indices = [
            i
            for i, name in enumerate(layer_names)
            if name in diagnostics["dead_layers"]
        ]
        for idx in dead_indices:
            ax.text(
                idx,
                grad_norms[idx] if grad_norms[idx] > 0 else 1e-8,
                "⚠",
                ha="center",
                va="bottom",
                fontsize=14,
                color="red",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_activation_stats(diagnostics: dict, output_path: Path) -> None:
    """Plots dead neuron percentages per layer.

    Creates a bar chart showing % of dead neurons (activation ≈ 0) for each layer.

    Args:
        diagnostics: Output from analyze_activations()
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    layer_names = diagnostics["layer_names"]
    dead_pcts = diagnostics["dead_neuron_pcts"]

    # Shorten layer names
    short_names = [
        ".".join(name.split(".")[-2:]) if len(name.split(".")) > 2 else name
        for name in layer_names
    ]

    # Color bars based on severity (red = many dead, green = healthy)
    colors = [
        "#E63946" if dp > 20 else "#FFB703" if dp > 10 else "#06D6A0"
        for dp in dead_pcts
    ]

    bars = ax.bar(
        range(len(dead_pcts)), dead_pcts, color=colors, edgecolor="black", linewidth=0.5
    )
    ax.set_xticks(range(len(dead_pcts)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Dead Neurons (%)", fontsize=11)
    ax.set_title(
        "Dead Neuron Detection (Activation < 1e-6)", fontsize=13, fontweight="bold"
    )
    ax.axhline(
        10,
        color="orange",
        linestyle="--",
        linewidth=1,
        label="10% threshold",
        alpha=0.7,
    )
    ax.axhline(
        20, color="red", linestyle="--", linewidth=1, label="20% threshold", alpha=0.7
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_integrated_gradients(ig_results: dict, output_path: Path) -> None:
    """Plots Integrated Gradients temporal and spatial attribution profiles.

    Creates a 3-panel figure:
    - Left: Temporal profile (importance vs time)
    - Middle: Spatial profile (importance per channel)
    - Right: Sample attribution heatmap (channels × time)

    Args:
        ig_results: Output from compute_integrated_gradients()
        output_path: Path to save figure
    """
    fig = plt.figure(figsize=(18, 5))

    # Left panel: Temporal profile
    ax1 = plt.subplot(1, 3, 1)
    time_steps = np.arange(len(ig_results["temporal_profile"]))
    time_sec = time_steps / 100.0 + 0.5  # Convert to seconds post-stimulus

    ax1.plot(time_sec, ig_results["temporal_profile"], color="#2E86AB", linewidth=2)
    ax1.axvline(
        ig_results["peak_time_sec"],
        color="#E63946",
        linestyle="--",
        linewidth=2,
        label=f'Peak: {ig_results["peak_time_sec"]:.2f}s',
    )

    # Highlight P300 window (0.8-1.3s post-stimulus)
    ax1.axvspan(0.8, 1.3, alpha=0.2, color="green", label="P300 window")

    ax1.set_xlabel("Time Post-Stimulus (s)", fontsize=11)
    ax1.set_ylabel("Attribution Magnitude", fontsize=11)
    ax1.set_title("Temporal Attribution Profile", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Middle panel: Spatial profile
    ax2 = plt.subplot(1, 3, 2)
    channel_indices = np.arange(len(ig_results["spatial_profile"]))

    # Color channels by region
    colors = []
    for idx in channel_indices:
        if idx <= 40:
            colors.append("#FFB703")  # Frontal (yellow)
        elif 40 < idx < 60:
            colors.append("#06D6A0")  # Central (green)
        elif 60 <= idx <= 90:
            colors.append("#2E86AB")  # Parietal (blue)
        else:
            colors.append("#A23B72")  # Other (purple)

    ax2.bar(
        channel_indices,
        ig_results["spatial_profile"],
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax2.axvline(
        ig_results["peak_channel_idx"],
        color="#E63946",
        linestyle="--",
        linewidth=2,
        label=f'Peak: Ch{ig_results["peak_channel_idx"]}',
    )

    ax2.set_xlabel("Channel Index", fontsize=11)
    ax2.set_ylabel("Attribution Magnitude", fontsize=11)
    ax2.set_title("Spatial Attribution Profile", fontsize=13, fontweight="bold")

    # Legend for regions
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#FFB703", label="Frontal (0-40)"),
        Patch(facecolor="#06D6A0", label="Central (41-59)"),
        Patch(facecolor="#2E86AB", label="Parietal (60-90)"),
        Patch(facecolor="#A23B72", label="Other (91+)"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    # Right panel: Sample attribution heatmap
    ax3 = plt.subplot(1, 3, 3)

    # Show first sample attribution (or average over samples)
    sample_attr = ig_results["attributions"][0]  # (129, 200)

    im = ax3.imshow(
        sample_attr,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        extent=[0.5, 2.5, 129, 0],  # Time: 0.5-2.5s, Channels: 0-129
    )
    ax3.set_xlabel("Time Post-Stimulus (s)", fontsize=11)
    ax3.set_ylabel("Channel Index", fontsize=11)
    ax3.set_title("Sample Attribution Heatmap", fontsize=13, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label("Attribution", fontsize=10)

    # Mark peak
    ax3.scatter(
        [ig_results["peak_time_sec"]],
        [ig_results["peak_channel_idx"]],
        color="red",
        s=100,
        marker="x",
        linewidths=3,
        label="Peak",
    )
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_layer_gradcam(gradcam_results: dict, output_path: Path) -> None:
    """Plots Layer GradCAM importance hierarchy.

    Creates a bar chart showing aggregate importance per layer,
    sorted by importance to visualize feature hierarchy.

    Args:
        gradcam_results: Output from compute_layer_gradcam()
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Sort layers by importance
    layer_names = list(gradcam_results["layer_importance"].keys())
    importances = [gradcam_results["layer_importance"][name] for name in layer_names]

    sorted_indices = np.argsort(importances)[::-1]
    sorted_names = [layer_names[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]

    # Shorten layer names
    short_names = [
        ".".join(name.split(".")[-2:]) if len(name.split(".")) > 2 else name
        for name in sorted_names
    ]

    # Color gradient: most important = dark blue, least important = light blue
    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=0, vmax=len(sorted_importances) - 1)
    colors = [cmap(norm(i)) for i in range(len(sorted_importances))]

    bars = ax.bar(
        range(len(sorted_importances)),
        sorted_importances,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(sorted_importances)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Aggregate Importance (Mean |Attribution|)", fontsize=11)
    ax.set_title("Layer GradCAM Importance Hierarchy", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Highlight most important layer
    most_important = gradcam_results.get("most_important_layer", None)
    if most_important:
        most_important_idx = sorted_names.index(most_important)
        ax.text(
            most_important_idx,
            sorted_importances[most_important_idx],
            "★",
            ha="center",
            va="bottom",
            fontsize=20,
            color="gold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_layer_temporal_profiles(layer_patterns: dict, output_path: Path) -> None:
    """Plots temporal attribution profiles for all layers.

    Creates overlayed line plots showing how each layer attends
    to different temporal windows.

    Args:
        layer_patterns: Output from summarize_layer_patterns()
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    temporal_profiles = layer_patterns["layer_temporal_profiles"]
    peak_times = layer_patterns["layer_peak_times"]

    # Plot each layer's temporal profile
    colors = plt.cm.viridis(np.linspace(0, 1, len(temporal_profiles)))

    for (layer_name, profile), color in zip(temporal_profiles.items(), colors):
        # Normalize profile for visualization
        profile_norm = profile / profile.max() if profile.max() > 0 else profile

        # Convert time indices to seconds (assuming 100 Hz, starting at 0.5s)
        time_sec = np.arange(len(profile)) / 100.0 + 0.5

        # Shorten layer name for legend
        short_name = (
            ".".join(layer_name.split(".")[-2:])
            if len(layer_name.split(".")) > 2
            else layer_name
        )

        ax.plot(
            time_sec,
            profile_norm,
            label=short_name,
            color=color,
            linewidth=2,
            alpha=0.7,
        )

    # Highlight P300 window
    ax.axvspan(0.8, 1.3, alpha=0.1, color="green", label="P300 window")

    ax.set_xlabel("Time Post-Stimulus (s)", fontsize=11)
    ax.set_ylabel("Normalized Attribution", fontsize=11)
    ax.set_title(
        "Layer-wise Temporal Attribution Profiles", fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_channel_ablation(ablation_results: dict, output_path: Path) -> None:
    """Plots channel importance from ablation study.

    Creates a bar chart showing NRMSE increase when each channel is ablated.
    Highlights most and least important channels.

    Args:
        ablation_results: Output from ablate_channels()
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    channel_importance = ablation_results["channel_importance"]
    most_important = ablation_results["most_important_channels"]
    least_important = ablation_results["least_important_channels"]

    channel_indices = np.arange(len(channel_importance))

    # Color channels by importance
    colors = []
    for idx in channel_indices:
        if idx in most_important[:3]:
            colors.append("#E63946")  # Top 3: red
        elif idx in most_important:
            colors.append("#F18F01")  # Top 10: orange
        elif idx in least_important:
            colors.append("#A8DADC")  # Bottom 10: light blue
        else:
            colors.append("#457B9D")  # Others: blue

    bars = ax.bar(
        channel_indices,
        channel_importance,
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("Channel Index", fontsize=11)
    ax.set_ylabel("ΔNRMSE (Importance)", fontsize=11)
    ax.set_title(
        f"Channel Ablation Study ({ablation_results['ablation_strategy']} strategy)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(alpha=0.3, axis="y")

    # Highlight baseline
    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Baseline (no ablation)",
    )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#E63946", label="Top 3 channels"),
        Patch(facecolor="#F18F01", label="Top 4-10 channels"),
        Patch(facecolor="#457B9D", label="Other channels"),
        Patch(facecolor="#A8DADC", label="Bottom 10 channels"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Annotate top 3
    for i, ch_idx in enumerate(most_important[:3], 1):
        ax.text(
            ch_idx,
            channel_importance[ch_idx],
            f"{ch_idx}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_temporal_ablation(ablation_results: dict, output_path: Path) -> None:
    """Plots temporal window importance from ablation study.

    Creates a line plot showing NRMSE increase when each temporal window is ablated.
    Highlights P300 window and most critical window.

    Args:
        ablation_results: Output from ablate_temporal_windows()
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    window_importance = ablation_results["window_importance"]
    window_centers_sec = ablation_results["window_centers_sec"]
    most_important_time = ablation_results["most_important_time_sec"]

    ax.plot(
        window_centers_sec,
        window_importance,
        color="#2E86AB",
        linewidth=2,
        marker="o",
        markersize=4,
    )

    # Highlight P300 window
    ax.axvspan(0.8, 1.3, alpha=0.2, color="green", label="P300 window (expected)")

    # Highlight most important window
    ax.axvline(
        most_important_time,
        color="#E63946",
        linestyle="--",
        linewidth=2,
        label=f"Most critical: {most_important_time:.2f}s",
    )

    ax.set_xlabel("Time Post-Stimulus (s)", fontsize=11)
    ax.set_ylabel("ΔNRMSE (Importance)", fontsize=11)
    ax.set_title(
        f"Temporal Window Ablation Study ({ablation_results['ablation_strategy']} strategy)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Baseline
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
