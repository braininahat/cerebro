"""Model components: encoders and decoders.

Components are building blocks for complete models:
- Encoders: Transform raw input to features
- Decoders: Transform features to task outputs
"""

from .decoders import (
    ClassificationHead,
    MultiTaskHead,
    ProjectionHead,
    RegressionHead,
)
from .encoders import (
    BaseEncoder,
    EEGNeXEncoder,
    IdentityEncoder,
    SignalJEPAEncoder,
)
from .jepa_components import (
    JEPAEncoder,
    MambaEncoder,
)
from .jepa_predictors import (
    EventPredictor,
    MambaBlock,
    StatePredictor,
    TraitPredictor,
)

__all__ = [
    # Encoders
    "BaseEncoder",
    "EEGNeXEncoder",
    "SignalJEPAEncoder",
    "IdentityEncoder",
    "JEPAEncoder",
    "MambaEncoder",
    # Decoders
    "RegressionHead",
    "ClassificationHead",
    "ProjectionHead",
    "MultiTaskHead",
    # JEPA Predictors
    "TraitPredictor",
    "StatePredictor",
    "EventPredictor",
    "MambaBlock",
]