"""
Fusion Layer with Learnable Bloom Prototypes.

This module implements the fusion mechanism between BloomBERT embeddings
and learnable prototype vectors for each Bloom taxonomy level.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Bloom's Taxonomy levels in order
BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


class FusionLayer(nn.Module):
    """
    Learnable fusion layer combining BloomBERT embeddings with prototypes.
    
    Components:
    - 6 prototype vectors (one per Bloom level)
    - Learnable alpha scalar for fusion strength
    - MLP projector for transformation
    
    Attributes:
        prototypes: Learnable prototype matrix (6, 768)
        alpha: Learnable fusion weight scalar
        projector: MLP for transforming fused embeddings
        level_to_idx: Mapping from level names to indices
    """
    
    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        """
        Initialize fusion layer with prototypes and MLP.
        
        Args:
            hidden_dim: Dimension of embeddings (768 for BERT/T5)
            dropout: Dropout probability in MLP
        """
        super(FusionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_levels = len(BLOOM_LEVELS)
        
        # Learnable prototype vectors for each Bloom level
        # Initialized with small random values
        self.prototypes = nn.Parameter(
            torch.randn(self.num_levels, hidden_dim) * 0.02
        )
        
        # Learnable fusion weight (alpha)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # MLP projector: 768 -> 1536 -> 768 with normalization
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Level name to index mapping
        self.level_to_idx = {level: idx for idx, level in enumerate(BLOOM_LEVELS)}
        
        logger.info(f"FusionLayer initialized with {self.num_levels} prototypes, "
                   f"alpha={self.alpha.item():.3f}")
    
    def get_prototype(self, bloom_levels: list) -> torch.Tensor:
        """
        Select prototypes based on Bloom levels.
        
        Args:
            bloom_levels: List of Bloom level names
            
        Returns:
            Stacked prototypes of shape (batch_size, 768)
            
        Raises:
            ValueError: If invalid Bloom level provided
        """
        indices = []
        for level in bloom_levels:
            if level not in self.level_to_idx:
                raise ValueError(
                    f"Invalid Bloom level '{level}'. "
                    f"Must be one of {BLOOM_LEVELS}"
                )
            indices.append(self.level_to_idx[level])
        
        # Index prototypes
        indices_tensor = torch.tensor(indices, device=self.prototypes.device)
        selected_prototypes = self.prototypes[indices_tensor]  # (batch, 768)
        
        return selected_prototypes
    
    def forward(self, h_bloom: torch.Tensor, bloom_levels: list) -> torch.Tensor:
        """
        Fuse BloomBERT embeddings with level-specific prototypes.
        
        Processing:
        1. Select prototype p_level for each sample
        2. Fuse: v_fused = h_bloom + alpha * p_level
        3. Project: v_prefix = MLP(v_fused)
        
        Args:
            h_bloom: BloomBERT embeddings (batch_size, 768)
            bloom_levels: List of Bloom level names
            
        Returns:
            Fused and projected embeddings (batch_size, 768)
        """
        batch_size = h_bloom.shape[0]
        
        # Validate input
        assert h_bloom.shape == (batch_size, self.hidden_dim), \
            f"Expected h_bloom shape ({batch_size}, {self.hidden_dim}), got {h_bloom.shape}"
        assert len(bloom_levels) == batch_size, \
            f"Mismatch: {len(bloom_levels)} levels for {batch_size} samples"
        
        # Get prototypes for each sample
        p_levels = self.get_prototype(bloom_levels)  # (batch, 768)
        
        # Fuse with learnable alpha
        v_fused = h_bloom + self.alpha * p_levels  # (batch, 768)
        
        # Project through MLP
        v_prefix = self.projector(v_fused)  # (batch, 768)
        
        # Validate output
        assert v_prefix.shape == (batch_size, self.hidden_dim), \
            f"Expected v_prefix shape ({batch_size}, {self.hidden_dim}), got {v_prefix.shape}"
        
        logger.debug(f"Fused {batch_size} embeddings with alpha={self.alpha.item():.3f}")
        
        return v_prefix
    
    def get_trainable_params(self) -> Dict[str, int]:
        """
        Count trainable parameters in fusion layer.
        
        Returns:
            Dictionary with parameter counts
        """
        proto_params = self.prototypes.numel()
        alpha_params = 1
        mlp_params = sum(p.numel() for p in self.projector.parameters())
        
        return {
            "prototypes": proto_params,
            "alpha": alpha_params,
            "mlp": mlp_params,
            "total": proto_params + alpha_params + mlp_params
        }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    fusion = FusionLayer()
    
    # Test forward
    batch_size = 4
    h_bloom = torch.randn(batch_size, 768)
    levels = ["Remember", "Understand", "Apply", "Analyze"]
    
    v_prefix = fusion(h_bloom, levels)
    print(f"Input shape: {h_bloom.shape}")
    print(f"Output shape: {v_prefix.shape}")
    
    # Show trainable params
    params = fusion.get_trainable_params()
    print(f"Trainable parameters: {params}")
    
    # Test invalid level
    try:
        fusion(h_bloom, ["InvalidLevel", "Remember", "Apply", "Analyze"])
    except ValueError as e:
        print(f"Caught expected error: {e}")
