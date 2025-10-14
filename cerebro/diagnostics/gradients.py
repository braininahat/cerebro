"""Gradient flow analysis through model layers."""

import torch


def analyze_gradient_flow(
    model: torch.nn.Module, batch: tuple, device: torch.device
) -> dict:
    """Analyzes gradient flow through model layers.

    Performs a backward pass on a single batch and extracts gradient statistics
    per layer. Useful for detecting vanishing/exploding gradients and dead layers.

    Args:
        model: PyTorch model
        batch: Single batch tuple (X, y, ...) from DataLoader
        device: Device to run on

    Returns:
        Dictionary with keys:
            - layer_names: List[str] - parameter names
            - grad_norms: List[float] - L2 norm of gradients per layer
            - param_norms: List[float] - L2 norm of parameters per layer
            - grad_to_param_ratio: List[float] - grad_norm / param_norm
            - dead_layers: List[str] - layers with grad_norm < 1e-7
    """
    model.train()  # Need training mode for gradients
    model.zero_grad()

    # Forward + backward pass
    X, y = batch[0].to(device), batch[1].to(device)
    X = X.float()
    y = y.float().view(-1, 1)

    y_pred = model(X)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    loss.backward()

    # Collect gradient statistics
    layer_names = []
    grad_norms = []
    param_norms = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_names.append(name)
            grad_norms.append(float(param.grad.norm().item()))
            param_norms.append(float(param.data.norm().item()))

    # Compute grad/param ratio (indicates learning effectiveness)
    grad_to_param_ratio = [
        g / p if p > 0 else 0.0 for g, p in zip(grad_norms, param_norms)
    ]

    # Identify dead layers (grad_norm ≈ 0)
    dead_layers = [name for name, gn in zip(layer_names, grad_norms) if gn < 1e-7]

    return {
        "layer_names": layer_names,
        "grad_norms": grad_norms,
        "param_norms": param_norms,
        "grad_to_param_ratio": grad_to_param_ratio,
        "dead_layers": dead_layers,
    }
