"""LaBraM (Large Brain Model) components for EEG analysis.

This module contains implementations for:
- VQNSP: Vector Quantized Neural Signal Processing for EEG tokenization
- NeuralTransformer: Transformer architecture for EEG analysis
- NormEMAVectorQuantizer: Normalized EMA-based vector quantization
- MEMPretrainModule: Masked EEG Modeling pretraining module
"""

from .tokenizer import VQNSP
from .finetune import NeuralTransformer
from .norm_ema_quantizer import NormEMAVectorQuantizer
from .pretrain import MEMPretrainModule

__all__ = [
    "VQNSP",
    "NeuralTransformer",
    "NormEMAVectorQuantizer",
    "MEMPretrainModule",
]
