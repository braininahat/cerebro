import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def to_numpy(tensor):
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


# Challenge 1 Metrics (Cross-Task Transfer Learning)


def calculate_mae(preds, targets):
    """Calculate Mean Absolute Error.

    Args:
        preds: Predictions (tensor or array)
        targets: Ground truth values (tensor or array)

    Returns:
        MAE value
    """
    preds = to_numpy(preds)
    targets = to_numpy(targets)
    return mean_absolute_error(targets, preds)


def calculate_r2(preds, targets):
    """Calculate R-squared score.

    Args:
        preds: Predictions (tensor or array)
        targets: Ground truth values (tensor or array)

    Returns:
        R² value
    """
    preds = to_numpy(preds)
    targets = to_numpy(targets)
    return r2_score(targets, preds)


def calculate_auc_roc(preds, targets):
    """Calculate Area Under ROC Curve.

    Args:
        preds: Predicted probabilities (tensor or array)
        targets: Binary ground truth (tensor or array)

    Returns:
        AUC-ROC value
    """
    preds = to_numpy(preds)
    targets = to_numpy(targets)
    return roc_auc_score(targets, preds)


def calculate_balanced_accuracy(preds, targets):
    """Calculate Balanced Accuracy.

    Args:
        preds: Predicted classes (tensor or array)
        targets: Ground truth classes (tensor or array)

    Returns:
        Balanced accuracy value
    """
    preds = to_numpy(preds)
    targets = to_numpy(targets)
    # Convert probabilities to binary predictions if needed
    if preds.ndim > 1 or (preds.min() >= 0 and preds.max() <= 1):
        preds = (preds > 0.5).astype(int)
    return balanced_accuracy_score(targets, preds)


# Challenge 2 Metrics (Psychopathology Prediction)


def calculate_ccc(preds, targets):
    """Calculate Concordance Correlation Coefficient.

    Args:
        preds: Predictions (tensor or array)
        targets: Ground truth values (tensor or array)

    Returns:
        CCC value
    """
    preds = to_numpy(preds).flatten()
    targets = to_numpy(targets).flatten()

    # Pearson correlation
    cor, _ = pearsonr(preds, targets)

    # Means
    mean_preds = np.mean(preds)
    mean_targets = np.mean(targets)

    # Variances
    var_preds = np.var(preds)
    var_targets = np.var(targets)

    # Standard deviations
    sd_preds = np.sqrt(var_preds)
    sd_targets = np.sqrt(var_targets)

    # CCC calculation
    numerator = 2 * cor * sd_preds * sd_targets
    denominator = var_preds + var_targets + (mean_preds - mean_targets) ** 2

    ccc = numerator / denominator if denominator != 0 else 0
    return ccc


def calculate_rmse(preds, targets):
    """Calculate Root Mean Squared Error.

    Args:
        preds: Predictions (tensor or array)
        targets: Ground truth values (tensor or array)

    Returns:
        RMSE value
    """
    preds = to_numpy(preds)
    targets = to_numpy(targets)
    return np.sqrt(mean_squared_error(targets, preds))


def calculate_spearman(preds, targets):
    """Calculate Spearman Correlation.

    Args:
        preds: Predictions (tensor or array)
        targets: Ground truth values (tensor or array)

    Returns:
        Spearman correlation coefficient
    """
    preds = to_numpy(preds).flatten()
    targets = to_numpy(targets).flatten()
    correlation, _ = spearmanr(preds, targets)
    return correlation


# Challenge Score Computation


def compute_challenge1_score(mae, r2, auc, bacc):
    """Compute weighted score for Challenge 1.

    Weights:
    - MAE: 40%
    - R²: 20%
    - AUC-ROC: 30%
    - Balanced Accuracy: 10%

    Args:
        mae: Mean Absolute Error (lower is better, so we negate)
        r2: R-squared score
        auc: AUC-ROC score
        bacc: Balanced accuracy

    Returns:
        Weighted score
    """
    # Note: MAE is inverted since lower is better
    # We might need to normalize MAE depending on the scale
    score = (
        -0.40 * mae  # Negative because lower is better
        + 0.20 * r2
        + 0.30 * auc
        + 0.10 * bacc
    )
    return score


def compute_challenge2_score(ccc, rmse, spearman):
    """Compute weighted score for Challenge 2.

    Weights:
    - CCC: 50%
    - RMSE: 30% (lower is better)
    - Spearman: 20%

    Args:
        ccc: Concordance Correlation Coefficient
        rmse: Root Mean Squared Error (lower is better, so we negate)
        spearman: Spearman correlation

    Returns:
        Weighted score
    """
    # Note: RMSE is inverted since lower is better
    score = (
        0.50 * ccc + -0.30 * rmse + 0.20 * spearman  # Negative because lower is better
    )
    return score


# Utility functions for batch processing


def compute_metrics_batch(preds, targets, metric_fns):
    """Compute multiple metrics for a batch.

    Args:
        preds: Predictions
        targets: Ground truth
        metric_fns: Dictionary of metric_name -> metric_function

    Returns:
        Dictionary of metric_name -> metric_value
    """
    results = {}
    for name, fn in metric_fns.items():
        try:
            results[name] = fn(preds, targets)
        except Exception as e:
            results[name] = np.nan
            print(f"Error computing {name}: {e}")
    return results
