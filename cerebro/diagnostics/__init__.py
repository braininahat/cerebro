"""Diagnostic modules for model analysis."""

from cerebro.diagnostics.ablation import (
    ablate_channels,
    ablate_temporal_windows,
    interpret_channel_importance,
    interpret_temporal_importance,
)
from cerebro.diagnostics.activations import analyze_activations
from cerebro.diagnostics.captum_attributions import (
    compute_integrated_gradients,
    interpret_spatial_pattern,
    interpret_temporal_pattern,
)
from cerebro.diagnostics.captum_layers import (
    compute_layer_gradcam,
    detect_conv_layers,
    interpret_layer_hierarchy,
    summarize_layer_patterns,
)
from cerebro.diagnostics.gradients import analyze_gradient_flow
from cerebro.diagnostics.predictions import analyze_predictions, compute_baseline_scores

__all__ = [
    "analyze_predictions",
    "compute_baseline_scores",
    "analyze_gradient_flow",
    "analyze_activations",
    "compute_integrated_gradients",
    "interpret_temporal_pattern",
    "interpret_spatial_pattern",
    "compute_layer_gradcam",
    "detect_conv_layers",
    "interpret_layer_hierarchy",
    "summarize_layer_patterns",
    "ablate_channels",
    "ablate_temporal_windows",
    "interpret_channel_importance",
    "interpret_temporal_importance",
]
