"""Captum-based attribution analysis for EEG models.

Implements Integrated Gradients and other attribution methods to understand
which spatiotemporal regions of the EEG signal the model attends to.
"""

import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch import nn
from torch.utils.data import DataLoader


class SqueezeOutputWrapper(nn.Module):
    """Wraps a model to squeeze (batch, 1) outputs to (batch) for Captum."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # Squeeze last dim if shape is (batch, 1)
        if out.dim() == 2 and out.shape[-1] == 1:
            return out.squeeze(-1)
        return out


def compute_integrated_gradients(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
    n_steps: int = 50,
    baseline_type: str = "zero",
) -> dict:
    """Computes Integrated Gradients attributions for EEG input.

    Integrated Gradients reveals which (channel, time) locations in the EEG signal
    contribute most to the model's predictions. For RT prediction, we expect:
    - Temporal: Peak attribution in P300 window (~300-500ms post-stimulus)
    - Spatial: Peak attribution in parietal channels (Pz, P3, P4)

    Args:
        model: PyTorch model (not LightningModule)
        dataloader: Validation DataLoader
        device: Device to run on
        num_samples: Number of samples to analyze (100-500 recommended)
        n_steps: IG integration steps (50 = good tradeoff speed/accuracy)
        baseline_type: Baseline for attribution:
            - "zero": All zeros (default)
            - "mean": Mean EEG signal from batch
            - "random": Small random noise

    Returns:
        Dictionary with keys:
            - attributions: np.array of shape (num_samples, 129, 200)
            - temporal_profile: np.array of shape (200,) - importance vs time
            - spatial_profile: np.array of shape (129,) - importance per channel
            - predictions: np.array of shape (num_samples,)
            - targets: np.array of shape (num_samples,)
            - peak_time_idx: int - index of peak temporal attribution
            - peak_channel_idx: int - index of peak spatial attribution
            - peak_time_sec: float - peak time in seconds post-stimulus
    """
    model.eval()
    model = model.to(device)

    # Wrap model to squeeze outputs for Captum
    wrapped_model = SqueezeOutputWrapper(model).to(device)
    wrapped_model.eval()

    # Initialize Integrated Gradients
    ig = IntegratedGradients(wrapped_model)

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

    # Concatenate and move to device
    samples = torch.cat(samples, dim=0)[:num_samples].to(device)
    targets = torch.cat(targets, dim=0)[:num_samples].to(device)

    # Compute baseline
    if baseline_type == "zero":
        baseline = torch.zeros_like(samples)
    elif baseline_type == "mean":
        baseline = samples.mean(dim=0, keepdim=True).expand_as(samples)
    elif baseline_type == "random":
        baseline = torch.randn_like(samples) * 0.001
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")

    baseline = baseline.to(device)

    # Enable gradients for attribution
    samples.requires_grad = True

    # Compute attributions (batched for memory efficiency)
    attributions_list = []
    predictions_list = []

    batch_size = 4  # Reduced for IG memory efficiency (4x reduction from 16)
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_samples = samples[i:end_idx]
        batch_baseline = baseline[i:end_idx]

        # Skip empty batches (edge case protection)
        if batch_samples.shape[0] == 0:
            continue

        # Compute IG for this batch
        # internal_batch_size processes n_steps in smaller chunks (12.5x memory reduction)
        # Without it, all 50 steps loaded simultaneously → 9.65 GB peak memory
        # With internal_batch_size=4: only 0.77 GB peak (safe for RTX 4090 with 22 GB free)
        batch_attr = ig.attribute(
            batch_samples,
            baselines=batch_baseline,
            n_steps=n_steps,
            internal_batch_size=4,
        )

        attributions_list.append(batch_attr.detach().cpu())

        # Get predictions for this batch
        with torch.no_grad():
            batch_pred = model(batch_samples)
            predictions_list.append(batch_pred.cpu())

    # Concatenate results
    attributions = torch.cat(
        attributions_list, dim=0
    ).numpy()  # (num_samples, 129, 200)
    predictions = torch.cat(predictions_list, dim=0).numpy().squeeze()  # (num_samples,)
    targets_np = targets.cpu().numpy().squeeze()  # (num_samples,)

    # Aggregate attributions across samples
    # Use absolute values to capture both positive and negative contributions
    abs_attr = np.abs(attributions)

    # Temporal profile: Sum over samples and channels
    temporal_profile = abs_attr.sum(axis=(0, 1))  # (200,)

    # Spatial profile: Sum over samples and time
    spatial_profile = abs_attr.sum(axis=(0, 2))  # (129,)

    # Find peaks
    peak_time_idx = int(np.argmax(temporal_profile))
    peak_channel_idx = int(np.argmax(spatial_profile))

    # Convert peak time to seconds (relative to stimulus)
    # Window starts at +0.5s post-stimulus, 100 Hz sampling
    peak_time_sec = float(peak_time_idx / 100.0 + 0.5)  # Absolute time post-stimulus

    return {
        "attributions": attributions,
        "temporal_profile": temporal_profile,
        "spatial_profile": spatial_profile,
        "predictions": predictions,
        "targets": targets_np,
        "peak_time_idx": peak_time_idx,
        "peak_channel_idx": peak_channel_idx,
        "peak_time_sec": peak_time_sec,
    }


def interpret_temporal_pattern(
    peak_time_sec: float, temporal_profile: np.ndarray
) -> str:
    """Interprets temporal attribution pattern for neuroscience plausibility.

    Args:
        peak_time_sec: Peak time in seconds post-stimulus
        temporal_profile: Attribution profile over time

    Returns:
        Interpretation string
    """
    interpretation = "📌 **Note**: P300 expectations based on oddball/decision tasks. Your task may have different ERP signatures.\n\n"

    # P300 component typically occurs 300-800ms post-stimulus
    # For RT tasks, expect peak in this window
    if 0.8 <= peak_time_sec <= 1.3:
        interpretation += (
            f"✓ Model attends to P300 window (peak at {peak_time_sec:.2f}s post-stimulus). "
            "This is neuroscientifically plausible for RT prediction tasks."
        )
    elif 0.5 <= peak_time_sec < 0.8:
        interpretation += (
            f"⚠ Model attends to early window (peak at {peak_time_sec:.2f}s post-stimulus). "
            "This is slightly early for P300 but may capture stimulus processing."
        )
    else:
        interpretation += (
            f"✗ Model attends to unexpected window (peak at {peak_time_sec:.2f}s post-stimulus). "
            "Expected peak in P300 window (0.8-1.3s). Model may be learning artifacts."
        )

    # Check if attribution is uniform (no clear peak)
    profile_std = np.std(temporal_profile)
    profile_mean = np.mean(temporal_profile)
    if profile_std / profile_mean < 0.2:  # Low relative variance
        interpretation += "\n⚠ Temporal profile is relatively uniform (no clear peak). Model may not be learning temporal structure."

    return interpretation


def interpret_spatial_pattern(
    spatial_profile: np.ndarray, peak_channel_idx: int, top_k: int = 10
) -> str:
    """Interprets spatial attribution pattern for neuroscience plausibility.

    Args:
        spatial_profile: Attribution profile over channels
        peak_channel_idx: Index of peak channel
        top_k: Number of top channels to analyze

    Returns:
        Interpretation string
    """
    # Get top-K channels
    top_indices = np.argsort(spatial_profile)[-top_k:][::-1]

    # Neuroscientifically relevant channels for RT prediction:
    # - Parietal: channels 60-90 (rough estimate for 129-channel system)
    # - Frontal: channels 0-40
    # - Central: channels 40-60
    # Note: Without channel names, this is approximate

    parietal_count = sum(60 <= idx <= 90 for idx in top_indices)
    frontal_count = sum(idx <= 40 for idx in top_indices)
    central_count = sum(40 < idx < 60 for idx in top_indices)

    interpretation = "⚠️ **Approximate regions** (no channel names available - using index-based estimates)\n\n"
    interpretation += f"Top {top_k} channels by importance:\n"
    interpretation += f"  - Parietal region: {parietal_count}/{top_k}\n"
    interpretation += f"  - Frontal region: {frontal_count}/{top_k}\n"
    interpretation += f"  - Central region: {central_count}/{top_k}\n"

    if parietal_count >= top_k // 2:
        interpretation += "✓ Model prioritizes parietal channels (expected for attention/decision tasks)."
    elif frontal_count >= top_k // 2:
        interpretation += (
            "⚠ Model prioritizes frontal channels (unusual for RT prediction)."
        )
    else:
        interpretation += "⚠ No clear spatial pattern detected. Model may not be learning spatial structure."

    # Check if attribution is uniform
    profile_std = np.std(spatial_profile)
    profile_mean = np.mean(spatial_profile)
    if profile_std / profile_mean < 0.2:
        interpretation += "\n⚠ Spatial profile is relatively uniform. Model may not be learning channel-specific patterns."

    return interpretation
