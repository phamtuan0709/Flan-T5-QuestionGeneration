"""
Unit tests for neural models (BloomEncoder, FusionLayer, FlanGenerator, HybridModel).

Run with: pytest tests/test_models.py -v
"""

import pytest
import torch
from bloom_qg.models import BloomEncoder, FusionLayer, FlanGenerator, HybridModel, BLOOM_LEVELS


@pytest.fixture
def device():
    """Get available device (CPU for testing)."""
    return torch.device("cpu")


class TestBloomEncoder:
    """Test BloomBERT encoder."""
    
    def test_initialization(self, device):
        """Test encoder initialization."""
        encoder = BloomEncoder(device=device)
        assert encoder is not None
        assert encoder.device == device
    
    def test_frozen_parameters(self, device):
        """Test that encoder parameters are frozen."""
        encoder = BloomEncoder(device=device)
        for param in encoder.model.parameters():
            assert not param.requires_grad, "BloomBERT should be frozen"
    
    def test_forward_pass(self, device):
        """Test forward pass with sample data."""
        encoder = BloomEncoder(device=device)
        
        contexts = ["This is a test context.", "Another context here."]
        bloom_levels = ["Remember", "Understand"]
        
        h_bloom = encoder(contexts, bloom_levels)
        
        assert h_bloom.shape == (2, 768), f"Expected (2, 768), got {h_bloom.shape}"
        assert h_bloom.dtype == torch.float32
    
    def test_input_formatting(self, device):
        """Test input format string."""
        encoder = BloomEncoder(device=device)
        
        contexts = ["Test context"]
        bloom_levels = ["Apply"]
        
        formatted = encoder.format_input(contexts, bloom_levels)
        
        assert len(formatted) == 1
        assert "Apply" in formatted[0]
        assert "Test context" in formatted[0]


class TestFusionLayer:
    """Test Fusion Layer with prototypes."""
    
    def test_initialization(self):
        """Test fusion layer initialization."""
        fusion = FusionLayer()
        
        assert fusion.prototypes.shape == (6, 768), "Should have 6 prototypes"
        assert fusion.alpha.requires_grad, "Alpha should be learnable"
    
    def test_prototype_retrieval(self):
        """Test prototype selection by Bloom level."""
        fusion = FusionLayer()
        
        bloom_levels = ["Remember", "Create"]
        prototypes = fusion.get_prototype(bloom_levels)
        
        assert prototypes.shape == (2, 768)
    
    def test_forward_pass(self):
        """Test forward pass."""
        fusion = FusionLayer()
        
        h_bloom = torch.randn(3, 768)
        bloom_levels = ["Remember", "Understand", "Apply"]
        
        v_prefix = fusion(h_bloom, bloom_levels)
        
        assert v_prefix.shape == (3, 768)
        assert v_prefix.dtype == torch.float32
    
    def test_trainable_parameters(self):
        """Test parameter count."""
        fusion = FusionLayer()
        params = fusion.get_trainable_params()
        
        assert "prototypes" in params
        assert "alpha" in params
        assert "mlp" in params
        assert "total" in params
        assert params["total"] > 0


class TestFlanGenerator:
    """Test FLAN-T5 generator with LoRA."""
    
    def test_initialization(self, device):
        """Test generator initialization."""
        generator = FlanGenerator(device=device)
        
        assert generator is not None
        assert generator.device == device
    
    def test_lora_applied(self, device):
        """Test that LoRA adapters are applied."""
        generator = FlanGenerator(lora_r=8, lora_alpha=16, device=device)
        
        # Check if model has LoRA parameters
        has_lora = any("lora" in name.lower() for name, _ in generator.model.named_parameters())
        assert has_lora, "Model should have LoRA parameters"
    
    def test_prompt_formatting(self, device):
        """Test prompt format string."""
        generator = FlanGenerator(device=device)
        
        contexts = ["Context text"]
        answers = ["answer"]
        
        prompts = generator.format_prompt(contexts, answers)
        
        assert len(prompts) == 1
        assert "Context:" in prompts[0]
        assert "Answer:" in prompts[0]
    
    def test_encode_with_prefix(self, device):
        """Test encoding with prefix injection."""
        generator = FlanGenerator(device=device)
        
        contexts = ["Test context"]
        answers = ["test"]
        v_prefix = torch.randn(1, 768)
        
        result = generator.encode_with_prefix(contexts, answers, v_prefix)
        
        assert "encoder_hidden_states" in result
        assert "attention_mask" in result
        
        # Prefix should add 1 token to sequence length
        hidden = result["encoder_hidden_states"]
        assert hidden.shape[0] == 1  # batch size
        assert hidden.shape[-1] == 768  # hidden dim


class TestHybridModel:
    """Test complete hybrid pipeline."""
    
    def test_initialization(self, device):
        """Test hybrid model initialization."""
        model = HybridModel(device=device)
        
        assert model.bloom_encoder is not None
        assert model.fusion_layer is not None
        assert model.flan_generator is not None
    
    def test_forward_pass(self, device):
        """Test forward pass (loss computation)."""
        model = HybridModel(device=device)
        
        contexts = ["Albert Einstein developed relativity."]
        answers = ["Albert Einstein"]
        bloom_levels = ["Remember"]
        questions = ["Who developed relativity?"]
        
        # Tokenize labels
        tokenizer = model.flan_generator.tokenizer
        labels_encoded = tokenizer(
            questions,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        labels = labels_encoded["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        loss = model(contexts, answers, bloom_levels, labels)
        
        assert loss is not None
        assert loss.item() > 0
    
    def test_generate(self, device):
        """Test question generation."""
        model = HybridModel(device=device)
        
        contexts = ["Water boils at 100 degrees Celsius."]
        answers = ["100 degrees Celsius"]
        bloom_levels = ["Remember"]
        
        questions = model.generate(
            contexts,
            answers,
            bloom_levels,
            num_beams=2,
            max_new_tokens=32
        )
        
        assert len(questions) == 1
        assert isinstance(questions[0], str)
        assert len(questions[0]) > 0
    
    def test_trainable_parameters(self, device):
        """Test parameter count."""
        model = HybridModel(device=device)
        params = model.get_trainable_params()
        
        assert "bloom_encoder" in params
        assert "fusion_layer" in params
        assert "flan_generator" in params
        assert "total" in params
        
        # BloomBERT should be frozen
        assert params["bloom_encoder"] == 0
        
        # Total should be around 6.5M
        total = params["total"]
        assert 6_000_000 < total < 7_000_000, f"Expected ~6.5M params, got {total:,}"


class TestBloomLevels:
    """Test Bloom level constants."""
    
    def test_bloom_levels_constant(self):
        """Test BLOOM_LEVELS constant."""
        assert len(BLOOM_LEVELS) == 6
        assert "Remember" in BLOOM_LEVELS
        assert "Understand" in BLOOM_LEVELS
        assert "Apply" in BLOOM_LEVELS
        assert "Analyze" in BLOOM_LEVELS
        assert "Evaluate" in BLOOM_LEVELS
        assert "Create" in BLOOM_LEVELS


# Performance tests (optional, can be slow)
@pytest.mark.slow
class TestModelPerformance:
    """Performance tests for models."""
    
    def test_inference_speed(self, device):
        """Test inference latency."""
        import time
        
        model = HybridModel(device=device)
        
        contexts = ["Test context"] * 4
        answers = ["test"] * 4
        bloom_levels = ["Remember"] * 4
        
        start = time.time()
        questions = model.generate(contexts, answers, bloom_levels, num_beams=1, max_new_tokens=20)
        elapsed = time.time() - start
        
        assert len(questions) == 4
        assert elapsed < 30, f"Inference too slow: {elapsed:.2f}s for 4 samples"
    
    def test_memory_usage(self, device):
        """Test memory footprint."""
        model = HybridModel(device=device)
        
        # Get model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # FLAN-T5-base has ~250M params, with LoRA should add ~4M trainable
        assert total_params > 240_000_000, "Model seems incomplete"
        assert 6_000_000 < trainable_params < 7_000_000, f"Expected ~6.5M trainable, got {trainable_params:,}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
