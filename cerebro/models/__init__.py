"""Model components and architectures.

This module provides:
- Components: Encoders and decoders (building blocks)
- Architectures: Complete models (encoder + decoder compositions)
- Builders: Factory functions for model construction
"""

from .architectures import (
    ContrastiveModel,
    LinearProbeModel,
    MultitaskModel,
    RegressorModel,
)
from .builders import build_encoder, get_encoder_info, list_encoders, register_encoder
from .components import (
    BaseEncoder,
    ClassificationHead,
    EEGNeXEncoder,
    IdentityEncoder,
    MultiTaskHead,
    ProjectionHead,
    RegressionHead,
    SignalJEPAEncoder,
)

__all__ = [
    # Architectures
    "RegressorModel",
    "ContrastiveModel",
    "MultitaskModel",
    "LinearProbeModel",
    # Components
    "BaseEncoder",
    "EEGNeXEncoder",
    "SignalJEPAEncoder",
    "IdentityEncoder",
    "RegressionHead",
    "ClassificationHead",
    "ProjectionHead",
    "MultiTaskHead",
    # Builders
    "build_encoder",
    "register_encoder",
    "list_encoders",
    "get_encoder_info",
]
