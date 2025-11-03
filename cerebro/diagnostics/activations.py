"""Activation statistics and dead neuron detection."""

import torch
import torch.nn as nn


def analyze_activations(
    model: torch.nn.Module, batch: tuple, device: torch.device
) -> dict:
    """Analyzes activation statistics through model layers.

    Uses forward hooks to capture activations from linear and convolutional layers.
    Computes statistics to detect dead neurons and layer saturation.

    Args:
        model: PyTorch model
        batch: Single batch tuple (X, y, ...) from DataLoader
        device: Device to run on

    Returns:
        Dictionary with keys:
            - layer_names: List[str]
            - activation_means: List[float] - mean activation per layer
            - activation_stds: List[float] - std activation per layer
            - dead_neuron_pcts: List[float] - % of neurons with output ≈ 0
            - sparsity: List[float] - % of activations near zero (<0.01)
    """
    model.eval()

    # Storage for activations
    activations = {}

    def get_activation(name):
        """Hook function to capture activations"""

        def hook(module, input, output):
            activations[name] = output.detach()

        return hook

    # Register hooks on linear and conv layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            hooks.append(module.register_forward_hook(get_activation(name)))

    # Forward pass
    X, y = batch[0].to(device), batch[1].to(device)
    with torch.no_grad():
        _ = model(X)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute statistics
    layer_names = []
    activation_means = []
    activation_stds = []
    dead_neuron_pcts = []
    sparsity = []

    for name, act in activations.items():
        layer_names.append(name)
        activation_means.append(float(act.mean().item()))
        activation_stds.append(float(act.std().item()))

        # Dead neurons: activations < 1e-6
        dead_pct = float((act.abs() < 1e-6).float().mean().item() * 100)
        dead_neuron_pcts.append(dead_pct)

        # Sparsity: activations < 0.01
        sparse_pct = float((act.abs() < 0.01).float().mean().item() * 100)
        sparsity.append(sparse_pct)

    return {
        "layer_names": layer_names,
        "activation_means": activation_means,
        "activation_stds": activation_stds,
        "dead_neuron_pcts": dead_neuron_pcts,
        "sparsity": sparsity,
    }
