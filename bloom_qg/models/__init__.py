"""
Bloom-controlled Question Generation Models.

This package contains the neural components for the hybrid architecture:
- BloomBERT encoder for cognitive-level embeddings
- Fusion layer with learnable prototypes
- FLAN-T5 generator with LoRA and prefix injection
- Integrated hybrid model
"""

from .bloom_encoder import BloomEncoder
from .fusion_layer import FusionLayer, BLOOM_LEVELS
from .flan_generator import FlanGenerator
from .hybrid_model import HybridModel

__all__ = [
    'BloomEncoder',
    'FusionLayer',
    'FlanGenerator',
    'HybridModel',
    'BLOOM_LEVELS',
]
