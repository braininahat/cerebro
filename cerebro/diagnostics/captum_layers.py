"""Layer-wise attribution analysis using GradCAM for EEG models.

Implements GradCAM to visualize which spatial/temporal regions activate
in convolutional layers, helping understand hierarchical feature learning.
"""

import numpy as np
import torch
from captum.attr import LayerGradCam
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


def detect_conv_layers(model: nn.Module, max_layers: int = 10) -> list[str]:
    """Auto-detect layers suitable for GradCAM analysis (architecture-agnostic).

    Tries multiple strategies in priority order:
    1. Convolutional layers (best for CNNs)
    2. Attention/transformer layers (for attention-based models)
    3. All parameterized layers (fallback for RNNs, etc.)

    Args:
        model: PyTorch model (not LightningModule)
        max_layers: Maximum number of layers to return (default: 10)

    Returns:
        List of layer names suitable for GradCAM

    Raises:
        ValueError: If no suitable layers found
    """
    import logging
    logger = logging.getLogger(__name__)

    # Strategy 1: Convolutional layers (best for CNNs like EEGNeX)
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_layers.append(name)

    if conv_layers:
        logger.info(f"Detected {len(conv_layers)} convolutional layers for GradCAM")
        return conv_layers[:max_layers]

    # Strategy 2: Attention/Transformer layers (for SignalJEPA, Conformer)
    attn_layers = []
    for name, module in model.named_modules():
        # Match common attention layer naming patterns
        if any(keyword in name.lower() for keyword in ['attention', 'attn', 'multihead', 'self_attn']):
            # Verify it has parameters
            if len(list(module.parameters())) > 0:
                attn_layers.append(name)

    if attn_layers:
        logger.warning(
            f"No conv layers found. Using {len(attn_layers)} attention layers instead. "
            f"GradCAM results may differ from CNN interpretations."
        )
        return attn_layers[:max_layers]

    # Strategy 3: All parameterized layers (fallback for RNNs, custom architectures)
    param_layers = []
    for name, module in model.named_modules():
        # Skip container modules (Sequential, ModuleList, etc.)
        if len(list(module.children())) > 0:
            continue
        # Only include layers with parameters
        if len(list(module.parameters())) > 0:
            param_layers.append(name)

    if param_layers:
        logger.warning(
            f"No conv/attention layers found. Using {len(param_layers)} parameterized layers. "
            f"GradCAM results may be difficult to interpret for this architecture."
        )
        return param_layers[:max_layers]

    # No suitable layers found
    raise ValueError(
        "No suitable layers found for GradCAM. Model may not be compatible with layer attribution."
    )


def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """Retrieve layer module by dot-separated name.

    Args:
        model: PyTorch model
        layer_name: Dot-separated layer name (e.g., "model.conv1")

    Returns:
        Layer module

    Raises:
        AttributeError: If layer not found
    """
    parts = layer_name.split(".")
    layer = model
    for part in parts:
        layer = getattr(layer, part)
    return layer


def compute_layer_gradcam(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_layers: list[str] | None = None,
    num_samples: int = 100,
) -> dict:
    """Computes GradCAM attributions for convolutional layers.

    GradCAM reveals which spatial/temporal regions activate in each layer,
    helping diagnose:
    - Early layers: Should detect local temporal patterns (edges, transients)
    - Middle layers: Should detect intermediate patterns (oscillations, ERPs)
    - Late layers: Should detect global patterns (decision-relevant features)

    If model attends to correct regions in late layers but predictions collapse,
    this suggests the architecture is adequate but optimization failed.

    Args:
        model: PyTorch model (not LightningModule)
        dataloader: Validation DataLoader
        device: Device to run on
        target_layers: List of layer names to analyze (None = auto-detect conv layers)
        num_samples: Number of samples to analyze (100 recommended)

    Returns:
        Dictionary with keys:
            - layer_attributions: Dict[layer_name, np.array] of shape (num_samples, C, T)
            - layer_importance: Dict[layer_name, float] - aggregate importance per layer
            - layer_shapes: Dict[layer_name, tuple] - output shapes
            - most_important_layer: str - layer with highest aggregate importance
    """
    model.eval()
    model = model.to(device)

    # Wrap model to squeeze outputs for Captum
    wrapped_model = SqueezeOutputWrapper(model).to(device)
    wrapped_model.eval()

    # Auto-detect conv layers if not specified (from original model)
    if target_layers is None:
        target_layers = detect_conv_layers(model)
        if not target_layers:
            raise ValueError("No convolutional layers found in model")

    # Collect samples
    samples = []
    total_collected = 0

    for batch in dataloader:
        X = batch[0]
        samples.append(X)
        total_collected += X.shape[0]
        if total_collected >= num_samples:
            break

    samples = torch.cat(samples, dim=0)[:num_samples].to(device)
    samples.requires_grad = True

    # Compute GradCAM for each layer
    layer_attributions = {}
    layer_importance = {}
    layer_shapes = {}

    for layer_name in target_layers:
        try:
            # Get layer from wrapped model's inner model
            layer = get_layer_by_name(wrapped_model.model, layer_name)
        except AttributeError:
            print(f"Warning: Layer '{layer_name}' not found, skipping")
            continue

        # Initialize GradCAM for this layer
        gradcam = LayerGradCam(wrapped_model, layer)

        # Compute attributions (batched for memory)
        attributions_list = []
        batch_size = 4  # Reduced for GradCAM memory efficiency (4x reduction from 16)

        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_samples = samples[i:end_idx]

            # Skip empty batches (edge case protection)
            if batch_samples.shape[0] == 0:
                continue

            # Forward func handles output squeezing, no target needed
            # Note: LayerGradCam doesn't have internal_batch_size parameter like IG,
            # so we rely on reduced batch_size for memory efficiency
            batch_attr = gradcam.attribute(
                batch_samples,
            )

            attributions_list.append(batch_attr.detach().cpu())

        # Concatenate and store
        layer_attr = torch.cat(attributions_list, dim=0).numpy()  # (num_samples, C, T)
        layer_attributions[layer_name] = layer_attr
        layer_shapes[layer_name] = layer_attr.shape[1:]  # (C, T)

        # Aggregate importance: mean absolute attribution
        layer_importance[layer_name] = float(np.abs(layer_attr).mean())

    # Find most important layer
    if layer_importance:
        most_important_layer = max(layer_importance, key=layer_importance.get)
    else:
        most_important_layer = None

    return {
        "layer_attributions": layer_attributions,
        "layer_importance": layer_importance,
        "layer_shapes": layer_shapes,
        "most_important_layer": most_important_layer,
        "target_layers": target_layers,
    }


def interpret_layer_hierarchy(
    layer_importance: dict[str, float], layer_shapes: dict[str, tuple]
) -> str:
    """Interprets layer importance hierarchy for neuroscience plausibility.

    Args:
        layer_importance: Dict[layer_name, importance_score]
        layer_shapes: Dict[layer_name, (C, T)] output shapes

    Returns:
        Interpretation string
    """
    if not layer_importance:
        return "⚠ No layers analyzed"

    # Sort layers by importance
    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)

    interpretation = "Layer Importance Hierarchy:\n"
    for i, (layer_name, importance) in enumerate(sorted_layers, 1):
        shape = layer_shapes.get(layer_name, "unknown")
        interpretation += f"  {i}. {layer_name}: {importance:.6f} (shape: {shape})\n"

    # Check if importance increases with depth (expected for hierarchical learning)
    # Strategy: Infer layer depth from layer names (e.g., "feature_encoder.5" > "feature_encoder.1")
    # Extract numeric indices from layer names to determine depth ordering
    layer_names = list(layer_importance.keys())

    if len(layer_names) >= 2:
        def extract_depth_idx(layer_name: str) -> int:
            """Extract numeric depth indicator from layer name (e.g., 'encoder.5.0' -> 5)."""
            import re
            # Find all numbers in layer name
            numbers = re.findall(r'\d+', layer_name)
            # Use first number as depth proxy (higher = deeper)
            return int(numbers[0]) if numbers else 0

        # Sort layers by inferred depth (lowest depth = earliest layer)
        layers_by_depth = sorted(layer_names, key=extract_depth_idx)
        early_layer = layers_by_depth[0]
        late_layer = layers_by_depth[-1]

        early_importance = layer_importance[early_layer]
        late_importance = layer_importance[late_layer]

        if late_importance > early_importance * 1.5:
            interpretation += f"\n✓ Late layers have higher importance (hierarchical feature learning detected). "
            interpretation += f"Late layer '{late_layer}' ({late_importance:.6f}) > Early layer '{early_layer}' ({early_importance:.6f})."
        elif early_importance > late_importance * 1.5:
            interpretation += f"\n⚠ Early layers dominate (model may not be learning hierarchical features). "
            interpretation += f"Early layer '{early_layer}' ({early_importance:.6f}) > Late layer '{late_layer}' ({late_importance:.6f})."
        else:
            interpretation += "\n~ Importance distributed across layers (mixed hierarchical learning)."

    # Check if importance is uniform (no clear hierarchy)
    importances = list(layer_importance.values())
    importance_std = np.std(importances)
    importance_mean = np.mean(importances)
    if importance_mean > 0 and importance_std / importance_mean < 0.2:
        interpretation += (
            "\n⚠ Layer importance is relatively uniform (no clear feature hierarchy)."
        )

    return interpretation


def summarize_layer_patterns(layer_attributions: dict[str, np.ndarray]) -> dict:
    """Summarizes attribution patterns for each layer.

    Args:
        layer_attributions: Dict[layer_name, np.array of shape (num_samples, C, T)]

    Returns:
        Dictionary with keys:
            - layer_temporal_profiles: Dict[layer_name, np.array of shape (T,)]
            - layer_spatial_profiles: Dict[layer_name, np.array of shape (C,)]
            - layer_peak_times: Dict[layer_name, int] - peak time index per layer
            - layer_sparsity: Dict[layer_name, float] - fraction of activations near zero
    """
    layer_temporal_profiles = {}
    layer_spatial_profiles = {}
    layer_peak_times = {}
    layer_sparsity = {}

    for layer_name, attr in layer_attributions.items():
        # Temporal profile: sum over samples and channels
        temporal_profile = np.abs(attr).sum(axis=(0, 1))  # (T,)
        layer_temporal_profiles[layer_name] = temporal_profile

        # Spatial profile: sum over samples and time
        spatial_profile = np.abs(attr).sum(axis=(0, 2))  # (C,)
        layer_spatial_profiles[layer_name] = spatial_profile

        # Peak time
        peak_time_idx = int(np.argmax(temporal_profile))
        layer_peak_times[layer_name] = peak_time_idx

        # Sparsity: fraction of activations with |attr| < 1e-6
        sparsity = float((np.abs(attr) < 1e-6).mean())
        layer_sparsity[layer_name] = sparsity

    return {
        "layer_temporal_profiles": layer_temporal_profiles,
        "layer_spatial_profiles": layer_spatial_profiles,
        "layer_peak_times": layer_peak_times,
        "layer_sparsity": layer_sparsity,
    }
