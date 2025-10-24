"""
Bloom-Controlled Question Generation System.

A hybrid neural architecture combining BloomBERT and FLAN-T5 for
generating questions at specific Bloom's Taxonomy cognitive levels.

Components:
- models: Neural architecture (BloomBERT, Fusion Layer, FLAN-T5)
- data: Dataset preparation and loading utilities

Example usage:
    >>> from bloom_qg.models import HybridModel
    >>> 
    >>> model = HybridModel(device="cuda")
    >>> questions = model.generate(
    ...     contexts=["Paris is the capital of France."],
    ...     answers=["Paris"],
    ...     bloom_levels=["Remember"]
    ... )
    >>> print(questions[0])
"""

__version__ = "1.0.0"
__author__ = "Bloom-QG Team"

from .models import HybridModel, BloomEncoder, FusionLayer, FlanGenerator, BLOOM_LEVELS
from .data import BloomQGDataset, create_collate_fn

__all__ = [
    # Models
    'HybridModel',
    'BloomEncoder',
    'FusionLayer',
    'FlanGenerator',
    'BLOOM_LEVELS',
    # Data
    'BloomQGDataset',
    'create_collate_fn',
]
