"""Auxiliary task heads for multi-task learning.

These heads predict demographic and clinical variables to improve
representation learning during pretraining.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class DemographicHead(nn.Module):
    """Head for predicting demographic variables (age, sex, etc.).

    Args:
        input_dim: Dimensionality of encoder output
        output_dim: Output dimensionality (1 for regression, n_classes for classification)
        hidden_dim: Hidden layer size (default: 128)
        task_type: 'regression' or 'classification'
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> age_head = DemographicHead(input_dim=512, output_dim=1, task_type='regression')
        >>> sex_head = DemographicHead(input_dim=512, output_dim=2, task_type='classification')
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        task_type: str = 'regression',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task_type = task_type

        # 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder output (batch, input_dim)

        Returns:
            Predictions (batch, output_dim)
        """
        return self.mlp(x)


class MultiAuxiliaryHead(nn.Module):
    """Multiple auxiliary task heads for multi-task learning.

    Predicts demographics and clinical variables from shared encoder representations.

    Supported tasks:
    - age: Regression (years)
    - sex: Classification (2 classes: M/F)
    - p_factor: Regression (externalizing psychopathology)
    - attention: Regression (attention factor score)
    - internalizing: Regression (internalizing factor score)
    - externalizing: Regression (externalizing factor score)

    Args:
        input_dim: Dimensionality of encoder output
        tasks: Dictionary mapping task names to configs
            Example: {'age': {'type': 'regression', 'dim': 1},
                      'sex': {'type': 'classification', 'dim': 2}}
        hidden_dim: Hidden layer size for all heads (default: 128)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> aux_head = MultiAuxiliaryHead(
        ...     input_dim=512,
        ...     tasks={
        ...         'age': {'type': 'regression', 'dim': 1},
        ...         'sex': {'type': 'classification', 'dim': 2},
        ...         'p_factor': {'type': 'regression', 'dim': 1},
        ...     }
        ... )
        >>> encoder_out = torch.randn(32, 512)
        >>> predictions = aux_head(encoder_out)
        >>> # predictions = {'age': (32, 1), 'sex': (32, 2), 'p_factor': (32, 1)}
    """

    def __init__(
        self,
        input_dim: int,
        tasks: Dict[str, Dict[str, any]],
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task_names = list(tasks.keys())

        # Create head for each task
        self.heads = nn.ModuleDict()
        for task_name, task_config in tasks.items():
            self.heads[task_name] = DemographicHead(
                input_dim=input_dim,
                output_dim=task_config['dim'],
                hidden_dim=hidden_dim,
                task_type=task_config['type'],
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Encoder output (batch, input_dim)

        Returns:
            Dictionary of predictions {task_name: (batch, dim)}
        """
        predictions = {}
        for task_name in self.task_names:
            predictions[task_name] = self.heads[task_name](x)
        return predictions


class AuxiliaryTaskLoss(nn.Module):
    """Combined loss for multiple auxiliary tasks.

    Computes weighted combination of task-specific losses.

    Args:
        tasks: Dictionary mapping task names to configs
            Example: {'age': {'type': 'regression', 'weight': 1.0},
                      'sex': {'type': 'classification', 'weight': 0.5}}
        loss_weights: Optional dictionary of task-specific weights
            If None, uses weights from task configs

    Example:
        >>> loss_fn = AuxiliaryTaskLoss(
        ...     tasks={
        ...         'age': {'type': 'regression', 'weight': 1.0},
        ...         'sex': {'type': 'classification', 'weight': 0.5},
        ...     }
        ... )
        >>> predictions = {'age': torch.randn(32, 1), 'sex': torch.randn(32, 2)}
        >>> targets = {'age': torch.randn(32, 1), 'sex': torch.randint(0, 2, (32,))}
        >>> loss_dict = loss_fn(predictions, targets)
        >>> # loss_dict = {'age': ..., 'sex': ..., 'total': ...}
    """

    def __init__(
        self,
        tasks: Dict[str, Dict[str, any]],
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.tasks = tasks
        self.loss_weights = loss_weights or {
            name: config.get('weight', 1.0)
            for name, config in tasks.items()
        }

        # Create loss functions
        self.loss_fns = {}
        for task_name, task_config in tasks.items():
            if task_config['type'] == 'regression':
                self.loss_fns[task_name] = nn.MSELoss()
            elif task_config['type'] == 'classification':
                self.loss_fns[task_name] = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown task type: {task_config['type']}")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary task losses.

        Args:
            predictions: Dictionary of predictions {task_name: (batch, dim)}
            targets: Dictionary of targets {task_name: (batch,) or (batch, dim)}
            mask: Optional dictionary of masks {task_name: (batch,)}
                  Use to handle missing labels (e.g., mask out n/a values)

        Returns:
            Dictionary of losses {task_name: scalar, 'total': scalar}
        """
        losses = {}
        total_loss = 0.0

        for task_name in predictions.keys():
            if task_name not in targets:
                continue

            pred = predictions[task_name]
            target = targets[task_name]

            # Apply mask if provided (for missing labels)
            if mask is not None and task_name in mask:
                task_mask = mask[task_name]
                pred = pred[task_mask]
                target = target[task_mask]

                # Skip if no valid samples
                if pred.size(0) == 0:
                    continue

            # Compute loss
            loss = self.loss_fns[task_name](pred, target)
            losses[task_name] = loss

            # Add to total with weight
            total_loss += self.loss_weights[task_name] * loss

        losses['total'] = total_loss
        return losses


# HBN-specific auxiliary task configurations
HBN_AUXILIARY_TASKS = {
    'age': {
        'type': 'regression',
        'dim': 1,
        'weight': 1.0,
        'description': 'Age in years',
    },
    'sex': {
        'type': 'classification',
        'dim': 2,  # M/F
        'weight': 0.5,
        'description': 'Biological sex',
    },
    'p_factor': {
        'type': 'regression',
        'dim': 1,
        'weight': 1.0,
        'description': 'General psychopathology factor',
    },
    'attention': {
        'type': 'regression',
        'dim': 1,
        'weight': 0.5,
        'description': 'Attention factor score',
    },
    'internalizing': {
        'type': 'regression',
        'dim': 1,
        'weight': 0.5,
        'description': 'Internalizing factor score',
    },
    'externalizing': {
        'type': 'regression',
        'dim': 1,
        'weight': 0.5,
        'description': 'Externalizing factor score',
    },
}
