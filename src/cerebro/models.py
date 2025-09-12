from braindecode.models import (
    BDTCN,
    BIOT,
    TCN,
    ATCNet,  # Core models
    AttentionBaseNet,
    AttnSleep,
    ContraWR,
    CTNet,
    Deep4Net,
    DeepSleepNet,
    EEGConformer,
    EEGInceptionERP,
    EEGInceptionMI,
    EEGITNet,
    EEGMiner,
    EEGNet,
    EEGNetv4,
    EEGNeX,
    EEGSimpleConv,
    EEGTCNet,
    FBCNet,
    FBLightConvNet,
    FBMSNet,
    HybridNet,
    IFNet,
    Labram,
    MSVTNet,
    SCCNet,
    ShallowFBCSPNet,
    SignalJEPA,
    SignalJEPA_Contextual,
    SignalJEPA_PostLocal,
    SignalJEPA_PreLocal,
    SincShallowNet,
    SleepStagerBlanco2020,
    SleepStagerChambon2018,
    SPARCNet,
    SyncNet,
    TIDNet,
    TSception,
    USleep,
)

from .constants import N_CHANS, N_TIMES, SFREQ


def get_all_models():
    """Get dictionary of all available braindecode models.

    Returns:
        Dictionary mapping model names to model classes
    """
    return {
        # Attention-based models
        "ATCNet": ATCNet,
        "AttentionBaseNet": AttentionBaseNet,
        # Temporal and sequence models
        "BDTCN": BDTCN,
        "TCN": TCN,
        "TIDNet": TIDNet,
        "TSception": TSception,
        "HybridNet": HybridNet,
        # Transformer and conformer models
        "BIOT": BIOT,
        "EEGConformer": EEGConformer,
        "EEGITNet": EEGITNet,
        # Contrastive and self-supervised models
        "ContraWR": ContraWR,
        "SignalJEPA": SignalJEPA,
        "SignalJEPA_Contextual": SignalJEPA_Contextual,
        "SignalJEPA_PostLocal": SignalJEPA_PostLocal,
        "SignalJEPA_PreLocal": SignalJEPA_PreLocal,
        # Classic and foundational models
        "Deep4Net": Deep4Net,
        "ShallowFBCSPNet": ShallowFBCSPNet,
        "EEGNet": EEGNet,
        "EEGNetv4": EEGNetv4,
        # Inception-based models
        "EEGInceptionERP": EEGInceptionERP,
        "EEGInceptionMI": EEGInceptionMI,
        # Advanced architectures
        "EEGNeX": EEGNeX,
        "EEGMiner": EEGMiner,
        # Filter bank models
        "FBCNet": FBCNet,
        "FBLightConvNet": FBLightConvNet,
        "FBMSNet": FBMSNet,
        # Specialized architectures
        "CTNet": CTNet,
        "EEGSimpleConv": EEGSimpleConv,
        "EEGTCNet": EEGTCNet,
        "IFNet": IFNet,
        "Labram": Labram,
        "MSVTNet": MSVTNet,
        "SCCNet": SCCNet,
        "SincShallowNet": SincShallowNet,
        "SPARCNet": SPARCNet,
        "SyncNet": SyncNet,
        # Sleep staging models
        "AttnSleep": AttnSleep,
        "DeepSleepNet": DeepSleepNet,
        "SleepStagerBlanco2020": SleepStagerBlanco2020,
        "SleepStagerChambon2018": SleepStagerChambon2018,
        "USleep": USleep,
    }


def create_model(
    model_name: str,
    n_chans: int = N_CHANS,
    n_outputs: int = 1,
    n_times: int = N_TIMES,
    sfreq: float = SFREQ,
    **kwargs,
):
    """Create a model instance by name.

    Args:
        model_name: Name of the model (must be in get_all_models())
        n_chans: Number of EEG channels
        n_outputs: Number of outputs (1 for regression, n_classes for classification)
        n_times: Number of time points in input
        sfreq: Sampling frequency
        **kwargs: Additional model-specific parameters

    Returns:
        Model instance
    """
    models = get_all_models()

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(models.keys())}"
        )

    model_class = models[model_name]

    # Base parameters that most models accept
    params = {
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "n_times": n_times,
        "sfreq": sfreq,
    }

    # Add any additional parameters
    params.update(kwargs)

    # Create model - let it fail explicitly if parameters are wrong
    return model_class(**params)


def get_model_info(model_name: str):
    """Get information about a model's requirements.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model information
    """
    models = get_all_models()

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    model_class = models[model_name]

    # Get docstring and signature info
    info = {
        "name": model_name,
        "class": model_class,
        "doc": model_class.__doc__,
        "module": model_class.__module__,
    }

    # Try to get init signature
    import inspect

    try:
        sig = inspect.signature(model_class.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name != "self":
                params[name] = {
                    "default": (
                        param.default
                        if param.default != inspect.Parameter.empty
                        else None
                    ),
                    "annotation": (
                        str(param.annotation)
                        if param.annotation != inspect.Parameter.empty
                        else None
                    ),
                }
        info["parameters"] = params
    except:
        info["parameters"] = {}

    return info


def list_models_by_category():
    """List all models organized by category.

    Returns:
        Dictionary with categories as keys and model lists as values
    """
    return {
        "attention_based": ["ATCNet", "AttentionBaseNet"],
        "temporal": ["BDTCN", "TCN", "TIDNet", "TSception", "HybridNet"],
        "transformer_conformer": ["BIOT", "EEGConformer", "EEGITNet"],
        "self_supervised": [
            "ContraWR",
            "SignalJEPA",
            "SignalJEPA_Contextual",
            "SignalJEPA_PostLocal",
            "SignalJEPA_PreLocal",
        ],
        "classic": ["Deep4Net", "ShallowFBCSPNet", "EEGNet", "EEGNetv4"],
        "inception": ["EEGInceptionERP", "EEGInceptionMI"],
        "advanced": ["EEGNeX", "EEGMiner"],
        "filter_bank": ["FBCNet", "FBLightConvNet", "FBMSNet"],
        "specialized": [
            "CTNet",
            "EEGSimpleConv",
            "EEGTCNet",
            "IFNet",
            "Labram",
            "MSVTNet",
            "SCCNet",
            "SincShallowNet",
            "SPARCNet",
            "SyncNet",
        ],
        "sleep_staging": [
            "AttnSleep",
            "DeepSleepNet",
            "SleepStagerBlanco2020",
            "SleepStagerChambon2018",
            "USleep",
        ],
    }
