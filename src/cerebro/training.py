from typing import Optional, Tuple

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .constants import *


def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    optimizer,
    scheduler: Optional[LRScheduler],
    device,
    print_batch_stats: bool = True,
) -> Tuple[float, float]:
    """Train model for one epoch.

    Args:
        dataloader: Training data loader
        model: Model to train
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        device: Device to use (cuda/cpu)
        print_batch_stats: Whether to show progress bar

    Returns:
        Tuple of (average_loss, rmse)
    """
    model.train()

    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples = 0

    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats
    )

    for batch_idx, batch in progress_bar:
        # Support datasets that may return (X, y) or (X, y, ...)
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Flatten to 1D for regression metrics and accumulate squared error
        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()

        if print_batch_stats:
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            progress_bar.set_description(
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}, RMSE: {running_rmse:.6f}"
            )

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
    return avg_loss, rmse


def validate(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    device,
    print_batch_stats: bool = True,
    return_predictions: bool = False,
) -> Tuple[float, float] | Tuple[float, float, torch.Tensor, torch.Tensor]:
    """Validate model on dataset.

    Args:
        dataloader: Validation data loader
        model: Model to validate
        loss_fn: Loss function
        device: Device to use (cuda/cpu)
        print_batch_stats: Whether to show progress bar
        return_predictions: Whether to return all predictions and targets

    Returns:
        Tuple of (average_loss, rmse) or (average_loss, rmse, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    sum_sq_err = 0.0
    n_batches = len(dataloader)
    n_samples = 0

    all_preds = []
    all_targets = []

    iterator = tqdm(
        enumerate(dataloader), total=n_batches, disable=not print_batch_stats
    )

    with torch.no_grad():
        for batch_idx, batch in iterator:
            # Supports (X, y) or (X, y, ...)
            X, y = batch[0], batch[1]
            X, y = X.to(device).float(), y.to(device).float()

            preds = model(X)
            batch_loss = loss_fn(preds, y).item()
            total_loss += batch_loss

            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            n_samples += y_flat.numel()

            if return_predictions:
                all_preds.append(preds_flat)
                all_targets.append(y_flat)

            if print_batch_stats:
                running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
                iterator.set_description(
                    f"Val Batch {batch_idx + 1}/{n_batches}, "
                    f"Loss: {batch_loss:.6f}, RMSE: {running_rmse:.6f}"
                )

    avg_loss = total_loss / n_batches if n_batches else float("nan")
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5

    if return_predictions:
        return avg_loss, rmse, torch.cat(all_preds), torch.cat(all_targets)
    else:
        return avg_loss, rmse


def get_optimizer(
    model: Module,
    optimizer_name: str,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
):
    """Create optimizer by name.

    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay

    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name: str, **kwargs):
    """Create scheduler by name.

    Args:
        optimizer: Optimizer instance
        scheduler_name: Name of scheduler ('cosine', 'step', 'exponential')
        **kwargs: Additional scheduler-specific arguments

    Returns:
        Scheduler instance or None
    """
    if scheduler_name.lower() == "none":
        return None
    elif scheduler_name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs.get("T_max", DEFAULT_N_EPOCHS)
        )
    elif scheduler_name.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif scheduler_name.lower() == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=kwargs.get("gamma", 0.95)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
