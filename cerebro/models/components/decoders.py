"""Decoder components that transform features to task-specific outputs.

Decoders are lightweight heads that adapt encoder features to specific
tasks. They are pure nn.Module classes with no training logic.

Key design principles:
- Decoders are task-specific (regression, classification, projection)
- They are typically much smaller than encoders (few parameters)
- They can be swapped out for different tasks while keeping the encoder fixed
"""

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """Linear regression head for continuous value prediction.

    Simple linear transformation from features to predictions.
    Used for tasks like response time or p-factor prediction.

    Args:
        input_dim: Dimension of input features from encoder
        output_dim: Number of regression targets (default: 1)
        dropout: Dropout probability before linear layer (default: 0)
    """

    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Transform features to regression predictions.

        Args:
            features: Feature tensor of shape (batch_size, input_dim)

        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        features = self.dropout(features)
        return self.linear(features)


class ClassificationHead(nn.Module):
    """Linear classification head for categorical prediction.

    Simple linear transformation from features to class logits.
    Used for tasks requiring discrete class predictions.

    Args:
        input_dim: Dimension of input features from encoder
        n_classes: Number of output classes
        dropout: Dropout probability before linear layer (default: 0)
    """

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Transform features to class logits.

        Args:
            features: Feature tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, n_classes)
        """
        features = self.dropout(features)
        return self.linear(features)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Projects encoder features to a lower-dimensional embedding space
    optimized for contrastive objectives. The projection head is typically
    discarded after pretraining, keeping only the encoder.

    Architecture: Linear -> ReLU -> Linear

    Args:
        input_dim: Dimension of input features from encoder
        hidden_dim: Hidden layer dimension (default: 256)
        output_dim: Output embedding dimension (default: 128)
        dropout: Dropout probability after ReLU (default: 0.1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to embedding space.

        Args:
            features: Feature tensor of shape (batch_size, input_dim)

        Returns:
            Embeddings of shape (batch_size, output_dim)
        """
        return self.mlp(features)


class MultiTaskHead(nn.Module):
    """Multi-task head with separate branches for different tasks.

    Supports multiple regression or classification tasks from the same
    encoder features. Each task gets its own linear transformation.

    Args:
        input_dim: Dimension of input features from encoder
        task_dims: Dictionary mapping task names to output dimensions
        dropout: Dropout probability before each task head (default: 0)

    Example:
        >>> head = MultiTaskHead(
        ...     input_dim=256,
        ...     task_dims={'rt': 1, 'p_factor': 1, 'age_group': 3}
        ... )
        >>> outputs = head(features)
        >>> # outputs is dict with keys 'rt', 'p_factor', 'age_group'
    """

    def __init__(
        self,
        input_dim: int,
        task_dims: dict[str, int],
        dropout: float = 0.0
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.task_heads = nn.ModuleDict({
            task_name: nn.Linear(input_dim, output_dim)
            for task_name, output_dim in task_dims.items()
        })

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Transform features to multiple task outputs.

        Args:
            features: Feature tensor of shape (batch_size, input_dim)

        Returns:
            Dictionary mapping task names to predictions
        """
        features = self.dropout(features)
        return {
            task_name: head(features)
            for task_name, head in self.task_heads.items()
        }