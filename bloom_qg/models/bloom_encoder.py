"""
BloomBERT Encoder Module.

This module wraps the BloomBERT model (RyanLauQF/BloomBERT) to extract
cognitive-level embeddings. All parameters are frozen during training.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class BloomEncoder(nn.Module):
    """
    Frozen BloomBERT encoder for cognitive-level representation.
    
    Takes formatted input "[{bloom_level}] context: {context}" and returns
    mean-pooled embeddings from the last hidden state.
    
    Attributes:
        model: Pretrained BloomBERT model (frozen)
        tokenizer: BloomBERT tokenizer
        device: Computation device
    """
    
    def __init__(self, model_name: str = "RyanLauQF/BloomBERT", device: str = "cuda"):
        """
        Initialize BloomBERT encoder with frozen parameters.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device for computation (cuda/cpu/mps)
        """
        super(BloomEncoder, self).__init__()
        
        logger.info(f"Loading BloomBERT encoder from {model_name}")
        self.device = torch.device(device)
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Move to device and freeze all parameters
        self.model.to(self.device)
        self._freeze_parameters()
        
        logger.info(f"BloomBERT encoder loaded on {self.device}, all parameters frozen")
    
    def _freeze_parameters(self) -> None:
        """Freeze all model parameters to prevent training."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        logger.debug("All BloomBERT parameters frozen")
    
    def format_input(self, contexts: List[str], bloom_levels: List[str]) -> List[str]:
        """
        Format inputs with Bloom level prefix.
        
        Args:
            contexts: List of context strings
            bloom_levels: List of Bloom taxonomy levels
            
        Returns:
            Formatted strings as "[{level}] context: {context}"
        """
        return [f"[{level}] context: {ctx}" for ctx, level in zip(contexts, bloom_levels)]
    
    def forward(
        self, 
        contexts: List[str], 
        bloom_levels: List[str]
    ) -> torch.Tensor:
        """
        Encode contexts with Bloom level information.
        
        Args:
            contexts: Batch of context strings
            bloom_levels: Batch of Bloom levels
            
        Returns:
            Mean-pooled embeddings of shape (batch_size, 768)
        """
        # Format inputs
        formatted_inputs = self.format_input(contexts, bloom_levels)
        
        # Tokenize
        encoded = self.tokenizer(
            formatted_inputs,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Forward pass (no gradients)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # Mean pooling over sequence length
        # Mask padded tokens before averaging
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        h_bloom = sum_embeddings / sum_mask  # (batch, 768)
        
        # Validate output shape
        batch_size = len(contexts)
        assert h_bloom.shape == (batch_size, 768), \
            f"Expected shape ({batch_size}, 768), got {h_bloom.shape}"
        
        logger.debug(f"Encoded {batch_size} contexts to shape {h_bloom.shape}")
        
        return h_bloom


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = BloomEncoder(device=device)
    
    # Test encoding
    contexts = [
        "The capital of France is Paris.",
        "Photosynthesis converts light energy into chemical energy."
    ]
    levels = ["Remember", "Understand"]
    
    embeddings = encoder(contexts, levels)
    print(f"Output shape: {embeddings.shape}")
    print(f"Sample embedding (first 5 dims): {embeddings[0, :5]}")
