"""Factory functions for building model components.

Builders provide a centralized way to instantiate encoders, decoders,
and complete models by name. This enables configuration-driven model
selection without hardcoding class imports throughout the codebase.
"""

from typing import Any, Optional

import torch.nn as nn

from .components import (
    BaseEncoder,
    EEGNeXEncoder,
    IdentityEncoder,
    JEPAEncoder,
    SignalJEPAEncoder,
    SignalJEPAWithLearnedChannels,
    VanillaSignalJEPAEncoder,
)


# Registry of available encoders
ENCODER_REGISTRY = {
    "EEGNeX": EEGNeXEncoder,
    "SignalJEPA": SignalJEPAEncoder,
    "SignalJEPA_LearnedChannels": SignalJEPAWithLearnedChannels,  # SignalJEPA with Perceiver-style learned channels
    "VanillaSignalJEPA": VanillaSignalJEPAEncoder,  # Vanilla braindecode SignalJEPA with real electrode locations
    "Identity": IdentityEncoder,
    "JEPA": JEPAEncoder,  # JEPA foundation model encoder
    # Future additions:
    # "SlowFastMamba": SlowFastMambaEncoder,
    # "SpatialTransformer": SpatialTransformerEncoder,
}


def build_encoder(encoder_class: str, **kwargs) -> BaseEncoder:
    """Build an encoder by name.

    This factory function allows configuration files to specify encoders
    by string name rather than requiring imports.

    Args:
        encoder_class: Name of encoder class (e.g., "EEGNeX", "SignalJEPA")
        **kwargs: Arguments to pass to encoder constructor

    Returns:
        Instantiated encoder

    Raises:
        ValueError: If encoder_class is not registered

    Example:
        >>> encoder = build_encoder("EEGNeX", n_chans=129, n_times=200)
        >>> features = encoder(eeg_input)
    """
    if encoder_class not in ENCODER_REGISTRY:
        available = ", ".join(ENCODER_REGISTRY.keys())
        raise ValueError(
            f"Unknown encoder: {encoder_class}. "
            f"Available encoders: {available}"
        )

    encoder_cls = ENCODER_REGISTRY[encoder_class]
    return encoder_cls(**kwargs)


def register_encoder(name: str, encoder_cls: type[BaseEncoder]) -> None:
    """Register a new encoder class.

    This allows custom encoders to be added to the registry at runtime,
    enabling them to be used via configuration files.

    Args:
        name: Name to register encoder under
        encoder_cls: Encoder class (must subclass BaseEncoder)

    Raises:
        ValueError: If name is already registered
        TypeError: If encoder_cls doesn't subclass BaseEncoder

    Example:
        >>> from my_custom_encoders import SlowFastMambaEncoder
        >>> register_encoder("SlowFastMamba", SlowFastMambaEncoder)
        >>> # Now can use in configs: encoder_class: "SlowFastMamba"
    """
    if name in ENCODER_REGISTRY:
        raise ValueError(f"Encoder '{name}' is already registered")

    if not issubclass(encoder_cls, BaseEncoder):
        raise TypeError(
            f"Encoder class must subclass BaseEncoder, got {encoder_cls}"
        )

    ENCODER_REGISTRY[name] = encoder_cls


def list_encoders() -> list[str]:
    """List all available encoder names.

    Returns:
        List of registered encoder names

    Example:
        >>> encoders = list_encoders()
        >>> print(encoders)
        ['EEGNeX', 'SignalJEPA', 'Identity']
    """
    return list(ENCODER_REGISTRY.keys())


def get_encoder_info(encoder_class: str) -> dict[str, Any]:
    """Get information about an encoder.

    Args:
        encoder_class: Name of encoder

    Returns:
        Dictionary with encoder information including docstring and
        constructor parameters

    Example:
        >>> info = get_encoder_info("EEGNeX")
        >>> print(info['description'])
        >>> print(info['parameters'])
    """
    if encoder_class not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {encoder_class}")

    encoder_cls = ENCODER_REGISTRY[encoder_class]

    # Extract docstring
    description = encoder_cls.__doc__ or "No description available"

    # Extract constructor parameters
    import inspect
    sig = inspect.signature(encoder_cls.__init__)
    parameters = {
        name: {
            "default": param.default
            if param.default is not inspect.Parameter.empty
            else None,
            "annotation": str(param.annotation)
            if param.annotation is not inspect.Parameter.empty
            else None,
        }
        for name, param in sig.parameters.items()
        if name != "self"
    }

    return {
        "name": encoder_class,
        "class": encoder_cls.__name__,
        "description": description.strip(),
        "parameters": parameters,
    }