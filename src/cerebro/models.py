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

from .constants import N_CHANS, N_TIMES, SFREQ, DEFAULT_LR, DEFAULT_BATCH_SIZE, DEFAULT_WEIGHT_DECAY


# Model-specific default configurations optimized for EEG tasks
MODEL_CONFIGS = {
    # Lightweight classical models (can handle larger batches)
    "EEGNet": {"batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4, "optimizer": "adamw"},
    "EEGNetv4": {"batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4, "optimizer": "adamw"},
    "ShallowFBCSPNet": {"batch_size": 128, "lr": 2e-3, "weight_decay": 1e-5, "optimizer": "adamw"},
    "EEGSimpleConv": {"batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4, "optimizer": "adamw"},
    
    # Deep classical models (moderate batch sizes)
    "Deep4Net": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    
    # Temporal convolutional networks
    "BDTCN": {"batch_size": 128, "lr": 1e-3, "weight_decay": 1e-4, "optimizer": "adamw"},
    "TCN": {"batch_size": 128, "lr": 1e-3, "weight_decay": 1e-4, "optimizer": "adamw"},
    "EEGTCNet": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "TIDNet": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    
    # Attention-based models (memory intensive)
    "ATCNet": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "AttentionBaseNet": {"batch_size": 32, "lr": 3e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    
    # Transformer and conformer models (very memory intensive)
    "BIOT": {"batch_size": 32, "lr": 1e-4, "weight_decay": 1e-5, "optimizer": "adamw"},
    "EEGConformer": {"batch_size": 32, "lr": 1e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "EEGITNet": {"batch_size": 32, "lr": 1e-4, "weight_decay": 1e-5, "optimizer": "adamw"},
    
    # Inception-based models
    "EEGInceptionERP": {"batch_size": 128, "lr": 1e-3, "weight_decay": 1e-4, "optimizer": "adamw"},
    "EEGInceptionMI": {"batch_size": 128, "lr": 1e-3, "weight_decay": 1e-4, "optimizer": "adamw"},
    
    # Advanced architectures
    "EEGNeX": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "EEGMiner": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    
    # Filter bank models (moderate memory usage)
    "FBCNet": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "FBLightConvNet": {"batch_size": 128, "lr": 1e-3, "weight_decay": 1e-4, "optimizer": "adamw"},
    "FBMSNet": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    
    # Self-supervised models
    "ContraWR": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "SignalJEPA": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "SignalJEPA_Contextual": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "SignalJEPA_PostLocal": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "SignalJEPA_PreLocal": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    
    # Specialized architectures
    "CTNet": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "HybridNet": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "IFNet": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "Labram": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "MSVTNet": {"batch_size": 32, "lr": 3e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "SCCNet": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "SincShallowNet": {"batch_size": 128, "lr": 1e-3, "weight_decay": 1e-4, "optimizer": "adamw"},
    "SPARCNet": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "SyncNet": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "TSception": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    
    # Sleep staging models (moderate requirements)
    "AttnSleep": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "DeepSleepNet": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "SleepStagerBlanco2020": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "SleepStagerChambon2018": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    "USleep": {"batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw"},
    
    # Default fallback configuration
    "default": {
        "batch_size": DEFAULT_BATCH_SIZE,
        "lr": DEFAULT_LR,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "optimizer": "adamw"
    }
}


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


def get_model_config(model_name: str, use_defaults: bool = True, **overrides):
    """Get configuration parameters for a specific model.
    
    Args:
        model_name: Name of the model
        use_defaults: If True, use model-specific defaults; if False, use global defaults
        **overrides: Parameters to override in the configuration
        
    Returns:
        Dictionary with model configuration parameters
    """
    if use_defaults and model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_name].copy()
    else:
        config = MODEL_CONFIGS["default"].copy()
    
    # Apply any overrides
    config.update(overrides)
    
    return config


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
