"""
Hybrid Model: Integrated Bloom-Controlled Question Generation.

Combines BloomBERT encoder, fusion layer, and FLAN-T5 generator
into a unified architecture for Bloom-taxonomy controlled question generation.
"""

import logging
from typing import List, Optional, Dict

import torch
import torch.nn as nn

from .bloom_encoder import BloomEncoder
from .fusion_layer import FusionLayer, BLOOM_LEVELS
from .flan_generator import FlanGenerator

logger = logging.getLogger(__name__)


class HybridModel(nn.Module):
    """
    Complete hybrid architecture for Bloom-controlled question generation.
    
    Pipeline:
    1. BloomBERT encodes "[{level}] context: {context}" -> h_bloom
    2. FusionLayer fuses h_bloom with level prototypes -> v_prefix
    3. FlanGenerator generates question with v_prefix injection
    
    Attributes:
        encoder: Frozen BloomBERT encoder
        fusion: Learnable fusion layer
        generator: FLAN-T5 with LoRA
        device: Computation device
    """
    
    def __init__(
        self,
        bloom_model: str = "RyanLauQF/BloomBERT",
        flan_model: str = "google/flan-t5-base",
        lora_r: int = 16,
        lora_alpha: int = 32,
        device: str = "cuda"
    ):
        """
        Initialize hybrid model components.
        
        Args:
            bloom_model: BloomBERT model name
            flan_model: FLAN-T5 model name
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            device: Computation device
        """
        super(HybridModel, self).__init__()
        
        self.device = torch.device(device)
        
        logger.info("Initializing Hybrid Model...")
        
        # Component 1: BloomBERT Encoder (frozen)
        self.encoder = BloomEncoder(model_name=bloom_model, device=device)
        
        # Component 2: Fusion Layer (trainable)
        self.fusion = FusionLayer(hidden_dim=768, dropout=0.1)
        self.fusion.to(self.device)
        
        # Component 3: FLAN-T5 Generator with LoRA (trainable)
        self.generator = FlanGenerator(
            model_name=flan_model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=device
        )
        
        # Level validation mapping
        self.level_to_idx = {level: idx for idx, level in enumerate(BLOOM_LEVELS)}
        
        # Log model info
        self._log_model_info()
    
    def _log_model_info(self) -> None:
        """Log model architecture and trainable parameters."""
        fusion_params = self.fusion.get_trainable_params()
        generator_params = sum(
            p.numel() for p in self.generator.parameters() if p.requires_grad
        )
        total_trainable = fusion_params["total"] + generator_params
        
        logger.info("=" * 60)
        logger.info("Hybrid Model Architecture")
        logger.info("=" * 60)
        logger.info(f"1. BloomBERT Encoder: FROZEN (0 trainable params)")
        logger.info(f"2. Fusion Layer: {fusion_params['total']:,} params")
        logger.info(f"   - Prototypes: {fusion_params['prototypes']:,}")
        logger.info(f"   - Alpha: {fusion_params['alpha']}")
        logger.info(f"   - MLP: {fusion_params['mlp']:,}")
        logger.info(f"3. FLAN-T5 Generator (LoRA): {generator_params:,} params")
        logger.info(f"Total Trainable: {total_trainable:,} (~{total_trainable/1e6:.1f}M)")
        logger.info("=" * 60)
    
    def validate_levels(self, bloom_levels: List[str]) -> None:
        """
        Validate that all Bloom levels are valid.
        
        Args:
            bloom_levels: List of level names to validate
            
        Raises:
            ValueError: If any invalid level found
        """
        for level in bloom_levels:
            if level not in self.level_to_idx:
                raise ValueError(
                    f"Invalid Bloom level '{level}'. "
                    f"Must be one of {BLOOM_LEVELS}"
                )
    
    def forward(
        self,
        contexts: List[str],
        answers: List[str],
        bloom_levels: List[str],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete pipeline.
        
        Args:
            contexts: Batch of context strings
            answers: Batch of answer strings
            bloom_levels: Batch of Bloom taxonomy levels
            labels: Target question tokens for training (optional)
            
        Returns:
            Dictionary with loss (training) or encoder outputs (inference)
        """
        batch_size = len(contexts)
        
        # Validate inputs
        assert len(answers) == batch_size, "Contexts and answers must match"
        assert len(bloom_levels) == batch_size, "Contexts and levels must match"
        self.validate_levels(bloom_levels)
        
        # Step 1: BloomBERT encoding
        h_bloom = self.encoder(contexts, bloom_levels)  # (batch, 768)
        
        # Step 2: Fusion with prototypes
        v_prefix = self.fusion(h_bloom, bloom_levels)  # (batch, 768)
        
        # Step 3: FLAN-T5 generation
        outputs = self.generator(
            contexts=contexts,
            answers=answers,
            v_prefix=v_prefix,
            labels=labels
        )
        
        return outputs
    
    def generate(
        self,
        contexts: List[str],
        answers: List[str],
        bloom_levels: List[str],
        num_beams: int = 4,
        max_new_tokens: int = 128
    ) -> List[str]:
        """
        Generate questions for given inputs.
        
        Args:
            contexts: Batch of contexts
            answers: Batch of answers
            bloom_levels: Batch of Bloom levels
            num_beams: Beam search width
            max_new_tokens: Max tokens to generate
            
        Returns:
            List of generated questions
        """
        self.eval()
        batch_size = len(contexts)
        
        # Validate inputs
        assert len(answers) == batch_size
        assert len(bloom_levels) == batch_size
        self.validate_levels(bloom_levels)
        
        with torch.no_grad():
            # Encode and fuse
            h_bloom = self.encoder(contexts, bloom_levels)
            v_prefix = self.fusion(h_bloom, bloom_levels)
            
            # Generate
            questions = self.generator.generate(
                contexts=contexts,
                answers=answers,
                v_prefix=v_prefix,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens
            )
        
        logger.info(f"Generated {len(questions)} questions")
        return questions
    
    def save_trainable(self, path: str) -> None:
        """
        Save only trainable components (fusion + LoRA).
        
        Args:
            path: Save path for checkpoint
        """
        checkpoint = {
            "fusion_state_dict": self.fusion.state_dict(),
            "generator_state_dict": self.generator.model.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved trainable weights to {path}")
    
    def load_trainable(self, path: str) -> None:
        """
        Load trainable components from checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.fusion.load_state_dict(checkpoint["fusion_state_dict"])
        self.generator.model.load_state_dict(checkpoint["generator_state_dict"])
        logger.info(f"Loaded trainable weights from {path}")


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridModel(device=device)
    
    # Test forward (inference mode)
    contexts = ["Albert Einstein developed the theory of relativity."]
    answers = ["Albert Einstein"]
    levels = ["Remember"]
    
    questions = model.generate(contexts, answers, levels)
    print(f"\nGenerated question: {questions[0]}")
    
    # Test invalid level
    try:
        model.generate(contexts, answers, ["InvalidLevel"])
    except ValueError as e:
        print(f"\nCaught expected error: {e}")
