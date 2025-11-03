"""Prediction distribution and baseline comparison diagnostics."""

import numpy as np
import torch
from torch.utils.data import DataLoader


def analyze_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = None,
) -> dict:
    """Computes predictions and analyzes distribution vs ground truth.

    Args:
        model: PyTorch model (not LightningModule wrapper)
        dataloader: Validation DataLoader
        device: Device to run inference on
        num_samples: Number of samples to analyze (None = all)

    Returns:
        Dictionary with keys:
            - predictions: np.array of shape (N,)
            - targets: np.array of shape (N,)
            - residuals: np.array of shape (N,) - errors
            - pred_mean: float
            - pred_std: float
            - target_mean: float
            - target_std: float
            - variance_ratio: float - pred_std / target_std
            - nrmse: float - Model NRMSE
            - rmse: float - Model RMSE
    """
    model.eval()
    preds = []
    targets = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[0].to(device), batch[1].to(device)
            y_pred = model(X)

            preds.append(y_pred.cpu().numpy())
            targets.append(y.cpu().numpy())

            total_samples += X.shape[0]

            # Early exit if num_samples specified
            if num_samples and total_samples >= num_samples:
                break

    # Concatenate and flatten
    preds = np.concatenate(preds).squeeze()
    targets = np.concatenate(targets).squeeze()

    # Trim to exact num_samples if specified
    if num_samples:
        preds = preds[:num_samples]
        targets = targets[:num_samples]

    # Compute statistics
    residuals = preds - targets
    rmse = np.sqrt(np.mean(residuals**2))
    target_std = np.std(targets)
    nrmse = rmse / target_std if target_std > 0 else rmse

    return {
        "predictions": preds,
        "targets": targets,
        "residuals": residuals,
        "pred_mean": float(np.mean(preds)),
        "pred_std": float(np.std(preds)),
        "target_mean": float(np.mean(targets)),
        "target_std": float(target_std),
        "variance_ratio": float(np.std(preds) / target_std) if target_std > 0 else 0.0,
        "nrmse": float(nrmse),
        "rmse": float(rmse),
    }


def compute_baseline_scores(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compare model performance vs naive baselines.

    Args:
        predictions: Model predictions, shape (N,)
        targets: Ground truth labels, shape (N,)

    Returns:
        Dictionary with keys:
            - naive_mean_rmse: float
            - naive_mean_nrmse: float
            - naive_median_rmse: float
            - naive_median_nrmse: float
            - model_nrmse: float
            - improvement_over_mean: float - (baseline - model) / baseline
    """
    target_mean = np.mean(targets)
    target_median = np.median(targets)
    target_std = np.std(targets)

    # Naive mean baseline (always predict mean)
    naive_mean_rmse = np.sqrt(np.mean((targets - target_mean) ** 2))
    naive_mean_nrmse = (
        naive_mean_rmse / target_std if target_std > 0 else naive_mean_rmse
    )

    # Naive median baseline (always predict median)
    naive_median_rmse = np.sqrt(np.mean((targets - target_median) ** 2))
    naive_median_nrmse = (
        naive_median_rmse / target_std if target_std > 0 else naive_median_rmse
    )

    # Model performance
    model_rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    model_nrmse = model_rmse / target_std if target_std > 0 else model_rmse

    # Improvement over mean baseline
    improvement = (
        (naive_mean_nrmse - model_nrmse) / naive_mean_nrmse
        if naive_mean_nrmse > 0
        else 0.0
    )

    return {
        "naive_mean_rmse": float(naive_mean_rmse),
        "naive_mean_nrmse": float(naive_mean_nrmse),
        "naive_median_rmse": float(naive_median_rmse),
        "naive_median_nrmse": float(naive_median_nrmse),
        "model_nrmse": float(model_nrmse),
        "improvement_over_mean": float(improvement),
    }
