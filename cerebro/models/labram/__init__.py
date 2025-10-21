"""LaBraM (Large Brain Model) components for EEG analysis.

This module contains implementations for:
- VQNSP: Vector Quantized Neural Signal Processing for EEG tokenization
- NeuralTransformer: Transformer architecture for EEG analysis
- NormEMAVectorQuantizer: Normalized EMA-based vector quantization
"""

from .tokenizer import VQNSP
from .modeling_finetune import NeuralTransformer
from .norm_ema_quantizer import NormEMAVectorQuantizer

__all__ = [
    "VQNSP",
    "NeuralTransformer",
    "NormEMAVectorQuantizer",
]
