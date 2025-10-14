"""Ablation studies for EEG channel and temporal importance analysis.

Performs systematic feature ablation to determine which channels and
time windows are critical for model predictions.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple


def ablate_channels(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
    num_trials: int = 10,
    ablation_strategy: str = "zero",
) -> dict:
    """Performs systematic channel ablation to measure importance.

    For each channel, ablates (zeros out or randomizes) that channel
    and measures performance degradation. Larger degradation = more important.

    Args:
        model: PyTorch model (not LightningModule)
        dataloader: Validation DataLoader
        device: Device to run on
        num_samples: Number of samples to test (100 recommended)
        num_trials: Number of ablation trials per channel (10 recommended)
        ablation_strategy: How to ablate:
            - "zero": Set channel to zero
            - "mean": Replace with channel mean
            - "random": Replace with random Gaussian noise

    Returns:
        Dictionary with keys:
            - channel_importance: (129,) - NRMSE increase per channel
            - baseline_nrmse: float - NRMSE without ablation
            - ablated_nrmse: (129,) - NRMSE with each channel ablated
            - most_important_channels: List[int] - Top 10 channels
            - least_important_channels: List[int] - Bottom 10 channels
    """
    model.eval()

    # Collect samples
    samples = []
    targets = []
    total_collected = 0

    for batch in dataloader:
        X, y = batch[0], batch[1]
        samples.append(X)
        targets.append(y)
        total_collected += X.shape[0]
        if total_collected >= num_samples:
            break

    samples = torch.cat(samples, dim=0)[:num_samples].to(device)
    targets = torch.cat(targets, dim=0)[:num_samples].cpu().numpy()
    n_chans = samples.shape[1]  # Should be 129

    # Baseline performance (no ablation)
    with torch.no_grad():
        baseline_preds = model(samples).cpu().numpy().squeeze()
    baseline_nrmse = compute_nrmse(baseline_preds, targets)

    # Ablate each channel
    channel_importance = np.zeros(n_chans)
    ablated_nrmse = np.zeros(n_chans)

    for ch_idx in range(n_chans):
        trial_nrmses = []

        for trial in range(num_trials):
            # Create ablated copy
            samples_ablated = samples.clone()

            if ablation_strategy == "zero":
                samples_ablated[:, ch_idx, :] = 0.0
            elif ablation_strategy == "mean":
                ch_mean = samples_ablated[:, ch_idx, :].mean()
                samples_ablated[:, ch_idx, :] = ch_mean
            elif ablation_strategy == "random":
                ch_std = samples_ablated[:, ch_idx, :].std()
                ch_mean = samples_ablated[:, ch_idx, :].mean()
                samples_ablated[:, ch_idx, :] = torch.randn_like(
                    samples_ablated[:, ch_idx, :]
                ) * ch_std + ch_mean
            else:
                raise ValueError(f"Unknown ablation strategy: {ablation_strategy}")

            # Predict with ablated input
            with torch.no_grad():
                ablated_preds = model(samples_ablated).cpu().numpy().squeeze()

            trial_nrmse = compute_nrmse(ablated_preds, targets)
            trial_nrmses.append(trial_nrmse)

        # Average over trials
        avg_nrmse = np.mean(trial_nrmses)
        ablated_nrmse[ch_idx] = avg_nrmse
        channel_importance[ch_idx] = avg_nrmse - baseline_nrmse  # Increase in NRMSE

    # Rank channels by importance
    importance_ranking = np.argsort(channel_importance)[::-1]  # Descending
    most_important = importance_ranking[:10].tolist()
    least_important = importance_ranking[-10:].tolist()

    return {
        "channel_importance": channel_importance,
        "baseline_nrmse": baseline_nrmse,
        "ablated_nrmse": ablated_nrmse,
        "most_important_channels": most_important,
        "least_important_channels": least_important,
        "ablation_strategy": ablation_strategy,
    }


def ablate_temporal_windows(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
    num_trials: int = 10,
    window_size: int = 20,  # 200ms windows at 100 Hz
    ablation_strategy: str = "zero",
) -> dict:
    """Performs systematic temporal window ablation to measure importance.

    Slides a window across time and ablates each window to measure
    performance degradation. Identifies critical temporal regions.

    Args:
        model: PyTorch model (not LightningModule)
        dataloader: Validation DataLoader
        device: Device to run on
        num_samples: Number of samples to test (100 recommended)
        num_trials: Number of ablation trials per window (10 recommended)
        window_size: Size of ablation window in samples (20 = 200ms at 100Hz)
        ablation_strategy: "zero", "mean", or "random"

    Returns:
        Dictionary with keys:
            - window_importance: (n_windows,) - NRMSE increase per window
            - baseline_nrmse: float - NRMSE without ablation
            - window_centers_sec: (n_windows,) - Window center times in seconds
            - most_important_window: int - Index of most critical window
            - most_important_time_sec: float - Time of most critical window
    """
    model.eval()

    # Collect samples
    samples = []
    targets = []
    total_collected = 0

    for batch in dataloader:
        X, y = batch[0], batch[1]
        samples.append(X)
        targets.append(y)
        total_collected += X.shape[0]
        if total_collected >= num_samples:
            break

    samples = torch.cat(samples, dim=0)[:num_samples].to(device)
    targets = torch.cat(targets, dim=0)[:num_samples].cpu().numpy()
    n_times = samples.shape[2]  # Should be 200

    # Baseline performance
    with torch.no_grad():
        baseline_preds = model(samples).cpu().numpy().squeeze()
    baseline_nrmse = compute_nrmse(baseline_preds, targets)

    # Compute sliding windows
    stride = window_size // 2  # 50% overlap
    n_windows = (n_times - window_size) // stride + 1
    window_importance = np.zeros(n_windows)
    window_centers_sec = np.zeros(n_windows)

    for win_idx in range(n_windows):
        start_idx = win_idx * stride
        end_idx = start_idx + window_size

        # Window center time in seconds (0.5s offset + sample_idx/100Hz)
        center_idx = (start_idx + end_idx) // 2
        window_centers_sec[win_idx] = 0.5 + center_idx / 100.0

        trial_nrmses = []

        for trial in range(num_trials):
            # Create ablated copy
            samples_ablated = samples.clone()

            if ablation_strategy == "zero":
                samples_ablated[:, :, start_idx:end_idx] = 0.0
            elif ablation_strategy == "mean":
                window_mean = samples_ablated[:, :, start_idx:end_idx].mean()
                samples_ablated[:, :, start_idx:end_idx] = window_mean
            elif ablation_strategy == "random":
                window_std = samples_ablated[:, :, start_idx:end_idx].std()
                window_mean = samples_ablated[:, :, start_idx:end_idx].mean()
                samples_ablated[:, :, start_idx:end_idx] = torch.randn_like(
                    samples_ablated[:, :, start_idx:end_idx]
                ) * window_std + window_mean
            else:
                raise ValueError(f"Unknown ablation strategy: {ablation_strategy}")

            # Predict with ablated input
            with torch.no_grad():
                ablated_preds = model(samples_ablated).cpu().numpy().squeeze()

            trial_nrmse = compute_nrmse(ablated_preds, targets)
            trial_nrmses.append(trial_nrmse)

        # Average over trials
        avg_nrmse = np.mean(trial_nrmses)
        window_importance[win_idx] = avg_nrmse - baseline_nrmse

    # Find most important window
    most_important_window = int(np.argmax(window_importance))
    most_important_time_sec = window_centers_sec[most_important_window]

    return {
        "window_importance": window_importance,
        "baseline_nrmse": baseline_nrmse,
        "window_centers_sec": window_centers_sec,
        "most_important_window": most_important_window,
        "most_important_time_sec": most_important_time_sec,
        "window_size": window_size,
        "ablation_strategy": ablation_strategy,
    }


def compute_nrmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Computes Normalized RMSE.

    Args:
        predictions: Model predictions (N,)
        targets: Ground truth targets (N,)

    Returns:
        NRMSE value (float)
    """
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    nrmse = rmse / np.std(targets)
    return float(nrmse)


def interpret_channel_importance(
    channel_importance: np.ndarray,
    most_important_channels: List[int],
    least_important_channels: List[int],
) -> str:
    """Interprets channel importance for neuroscience plausibility.

    Args:
        channel_importance: (129,) array of NRMSE increases
        most_important_channels: Top 10 channel indices
        least_important_channels: Bottom 10 channel indices

    Returns:
        Interpretation string
    """
    # Categorize channels by region (simplified mapping)
    def get_region(ch_idx: int) -> str:
        if ch_idx <= 40:
            return "Frontal"
        elif 40 < ch_idx < 60:
            return "Central"
        elif 60 <= ch_idx <= 90:
            return "Parietal"
        else:
            return "Other"

    # Count regions in top 10
    region_counts = {"Frontal": 0, "Central": 0, "Parietal": 0, "Other": 0}
    for ch_idx in most_important_channels:
        region = get_region(ch_idx)
        region_counts[region] += 1

    interpretation = "Channel Importance Analysis:\n\n"
    interpretation += f"Top 10 most important channels:\n"
    for i, ch_idx in enumerate(most_important_channels, 1):
        region = get_region(ch_idx)
        importance = channel_importance[ch_idx]
        interpretation += f"  {i}. Channel {ch_idx} ({region}): ΔNRMSE = {importance:.4f}\n"

    interpretation += f"\nRegion distribution in top 10:\n"
    for region, count in region_counts.items():
        interpretation += f"  - {region}: {count}/10\n"

    # Neuroscience interpretation
    interpretation += "\nNeuroscience Interpretation:\n"
    if region_counts["Parietal"] >= 5:
        interpretation += "✓ Model relies on parietal channels (expected for RT/decision tasks).\n"
    elif region_counts["Frontal"] >= 5:
        interpretation += "⚠ Model relies on frontal channels (unusual for RT prediction).\n"
    else:
        interpretation += "~ Model uses mixed regions (no clear spatial focus).\n"

    # Check if importance is distributed or concentrated
    importance_std = np.std(channel_importance)
    importance_mean = np.mean(channel_importance)
    if importance_mean > 0 and importance_std / importance_mean < 0.3:
        interpretation += "⚠ Channel importance is relatively uniform (no clear spatial selectivity).\n"
    else:
        interpretation += "✓ Model shows clear spatial selectivity (some channels much more important).\n"

    return interpretation


def interpret_temporal_importance(
    window_importance: np.ndarray,
    window_centers_sec: np.ndarray,
    most_important_time_sec: float,
) -> str:
    """Interprets temporal window importance for neuroscience plausibility.

    Args:
        window_importance: (n_windows,) array of NRMSE increases
        window_centers_sec: (n_windows,) window centers in seconds
        most_important_time_sec: Time of most critical window

    Returns:
        Interpretation string
    """
    interpretation = "Temporal Window Importance Analysis:\n\n"
    interpretation += f"Most critical window: {most_important_time_sec:.2f}s post-stimulus\n"

    # Find top 3 windows
    top_indices = np.argsort(window_importance)[-3:][::-1]
    interpretation += f"\nTop 3 most important windows:\n"
    for i, win_idx in enumerate(top_indices, 1):
        time_sec = window_centers_sec[win_idx]
        importance = window_importance[win_idx]
        interpretation += f"  {i}. {time_sec:.2f}s: ΔNRMSE = {importance:.4f}\n"

    # Neuroscience interpretation
    interpretation += f"\nNeuroscience Interpretation:\n"
    if 0.8 <= most_important_time_sec <= 1.3:
        interpretation += "✓ Most critical window aligns with P300 (0.8-1.3s), expected for RT tasks.\n"
    elif most_important_time_sec < 0.8:
        interpretation += "⚠ Model relies on early window (<0.8s). May be learning stimulus features instead of decision.\n"
    elif most_important_time_sec > 1.8:
        interpretation += "⚠ Model relies on late window (>1.8s). May be learning motor response instead of decision.\n"
    else:
        interpretation += "~ Model relies on intermediate window. Partially aligned with RT neuroscience.\n"

    # Check if importance is concentrated or distributed
    importance_std = np.std(window_importance)
    importance_mean = np.mean(window_importance)
    if importance_mean > 0 and importance_std / importance_mean < 0.3:
        interpretation += "⚠ Temporal importance is relatively uniform (no clear critical window).\n"
    else:
        interpretation += "✓ Model shows clear temporal selectivity (specific windows are critical).\n"

    return interpretation
