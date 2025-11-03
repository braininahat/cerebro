"""Model components: encoders, decoders, and adapters.

Components are building blocks for complete models:
- Encoders: Transform raw input to features
- Decoders: Transform features to task outputs
- Adapters: Handle input/output transformations (e.g., channel adaptation)
"""

from .auxiliary_heads import (
    AuxiliaryTaskLoss,
    DemographicHead,
    HBN_AUXILIARY_TASKS,
    MultiAuxiliaryHead,
)
from .channel_adapter import (
    PerceiverChannelAdapter,
    ZeroPadChannelAdapter,
)
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
    SignalJEPAWithLearnedChannels,
    VanillaSignalJEPAEncoder,
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
    "SignalJEPAWithLearnedChannels",
    "VanillaSignalJEPAEncoder",
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
    # Adapters
    "PerceiverChannelAdapter",
    "ZeroPadChannelAdapter",
    # Auxiliary Tasks
    "DemographicHead",
    "MultiAuxiliaryHead",
    "AuxiliaryTaskLoss",
    "HBN_AUXILIARY_TASKS",
]