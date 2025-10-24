"""
FLAN-T5 Generator with LoRA and Prefix Injection.

This module implements the FLAN-T5 generation component with:
- LoRA adapters on query and value projections
- Prefix injection mechanism for Bloom-controlled generation
"""

import logging
from typing import List, Optional, Dict

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


class FlanGenerator(nn.Module):
    """
    FLAN-T5 generator with LoRA and prefix injection.
    
    The generator:
    1. Encodes input prompt with FLAN-T5 encoder
    2. Prepends v_prefix to encoder hidden states
    3. Generates questions via cross-attention with extended hidden states
    
    Attributes:
        model: FLAN-T5 model with LoRA adapters
        tokenizer: FLAN-T5 tokenizer
        device: Computation device
    """
    
    def __init__(
        self, 
        model_name: str = "google/flan-t5-base",
        lora_r: int = 16,
        lora_alpha: int = 32,
        device: str = "cuda"
    ):
        """
        Initialize FLAN-T5 generator with LoRA.
        
        Args:
            model_name: HuggingFace model identifier
            lora_r: LoRA rank
            lora_alpha: LoRA scaling parameter
            device: Device for computation
        """
        super(FlanGenerator, self).__init__()
        
        logger.info(f"Loading FLAN-T5 generator from {model_name}")
        self.device = torch.device(device)
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q", "v"],  # Query and value projections only
            lora_dropout=0.1,
            bias="none",
        )
        
        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        self.model.to(self.device)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"FLAN-T5 with LoRA loaded: {trainable_params:,} trainable / "
            f"{total_params:,} total params ({100 * trainable_params / total_params:.2f}%)"
        )
    
    def format_prompt(
        self, 
        contexts: List[str], 
        answers: List[str]
    ) -> List[str]:
        """
        Format input prompts for FLAN-T5.
        
        Args:
            contexts: List of context strings
            answers: List of answer strings
            
        Returns:
            Formatted prompts as "Generate question. Context: {ctx} Answer: {ans}"
        """
        prompts = [
            f"Generate question. Context: {ctx} Answer: {ans}"
            for ctx, ans in zip(contexts, answers)
        ]
        return prompts
    
    def encode_with_prefix(
        self,
        contexts: List[str],
        answers: List[str],
        v_prefix: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Encode prompts and inject prefix into encoder hidden states.
        
        Args:
            contexts: Batch of contexts
            answers: Batch of answers
            v_prefix: Prefix embeddings (batch_size, 768)
            
        Returns:
            Dictionary with extended encoder_outputs and attention_mask
        """
        batch_size = len(contexts)
        
        # Format and tokenize prompts
        prompts = self.format_prompt(contexts, answers)
        encoded = self.tokenizer(
            prompts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get encoder hidden states
        encoder = self.model.get_encoder()
        encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden = encoder_outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # Validate prefix shape
        assert v_prefix.shape == (batch_size, 768), \
            f"Expected v_prefix shape ({batch_size}, 768), got {v_prefix.shape}"
        
        # Prepend prefix to encoder hidden states
        # v_prefix: (batch, 768) -> (batch, 1, 768)
        v_prefix_expanded = v_prefix.unsqueeze(1)
        extended_hidden = torch.cat([v_prefix_expanded, encoder_hidden], dim=1)
        # extended_hidden: (batch, seq_len + 1, 768)
        
        # Extend attention mask to include prefix position
        prefix_mask = torch.ones(batch_size, 1, device=self.device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        # extended_mask: (batch, seq_len + 1)
        
        logger.debug(
            f"Encoded {batch_size} prompts, extended hidden from "
            f"{encoder_hidden.shape[1]} to {extended_hidden.shape[1]} tokens"
        )
        
        return {
            "encoder_hidden_states": extended_hidden,
            "attention_mask": extended_mask
        }
    
    def forward(
        self,
        contexts: List[str],
        answers: List[str],
        v_prefix: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or generation.
        
        Args:
            contexts: Batch of context strings
            answers: Batch of answer strings
            v_prefix: Prefix embeddings (batch_size, 768)
            labels: Target question token IDs for training (optional)
            
        Returns:
            Dictionary with loss (if labels provided) or logits
        """
        # Encode with prefix injection
        encoder_outputs = self.encode_with_prefix(contexts, answers, v_prefix)
        
        if labels is not None:
            # Training mode: compute loss
            labels = labels.to(self.device)
            
            outputs = self.model(
                attention_mask=encoder_outputs["attention_mask"],
                encoder_outputs=(encoder_outputs["encoder_hidden_states"],),
                labels=labels
            )
            
            return {"loss": outputs.loss}
        else:
            # Inference mode: return encoder outputs for generation
            return encoder_outputs
    
    def generate(
        self,
        contexts: List[str],
        answers: List[str],
        v_prefix: torch.Tensor,
        num_beams: int = 4,
        max_new_tokens: int = 128
    ) -> List[str]:
        """
        Generate questions using beam search.
        
        Args:
            contexts: Batch of contexts
            answers: Batch of answers
            v_prefix: Prefix embeddings (batch_size, 768)
            num_beams: Number of beams for beam search
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of generated question strings
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get extended encoder outputs
            encoder_outputs = self.encode_with_prefix(contexts, answers, v_prefix)
            
            # Create a proper encoder output object for T5
            from transformers.modeling_outputs import BaseModelOutput
            encoder_output_obj = BaseModelOutput(
                last_hidden_state=encoder_outputs["encoder_hidden_states"]
            )
            
            # Generate with beam search
            generated_ids = self.model.generate(
                attention_mask=encoder_outputs["attention_mask"],
                encoder_outputs=encoder_output_obj,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            
            # Decode generated tokens
            questions = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
        
        logger.debug(f"Generated {len(questions)} questions")
        return questions


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = FlanGenerator(device=device)
    
    # Test generation
    contexts = ["The Eiffel Tower is in Paris, France."]
    answers = ["Paris"]
    v_prefix = torch.randn(1, 768).to(device)
    
    questions = generator.generate(contexts, answers, v_prefix)
    print(f"Generated question: {questions[0]}")
