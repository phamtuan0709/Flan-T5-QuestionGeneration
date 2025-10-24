"""
Integration tests for end-to-end workflows.

Run with: pytest tests/test_integration.py -v
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path


@pytest.fixture
def sample_data():
    """Create sample training data."""
    return [
        {
            "context": "Albert Einstein developed the theory of relativity in 1915.",
            "answer": "Albert Einstein",
            "question": "Who developed the theory of relativity?",
            "bloom_level": "Remember",
            "source": "test"
        },
        {
            "context": "Photosynthesis is the process plants use to convert sunlight into energy.",
            "answer": "photosynthesis",
            "question": "What process do plants use to convert sunlight?",
            "bloom_level": "Remember",
            "source": "test"
        }
    ]


class TestEndToEndPipeline:
    """Test complete training and inference pipeline."""
    
    def test_full_pipeline_cpu(self, sample_data):
        """Test full pipeline on CPU."""
        from transformers import AutoTokenizer
        from torch.utils.data import DataLoader
        from bloom_qg.models import HybridModel
        from bloom_qg.data import BloomQGDataset, create_collate_fn
        
        device = torch.device("cpu")
        
        # Create temp data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_path = f.name
        
        try:
            # 1. Load data
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            dataset = BloomQGDataset(temp_path, tokenizer, max_length=128)
            collate_fn = create_collate_fn(tokenizer, max_length=128)
            loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
            
            # 2. Initialize model
            model = HybridModel(device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # 3. Training step
            batch = next(iter(loader))
            
            loss = model(
                batch["contexts"],
                batch["answers"],
                batch["bloom_levels"],
                batch["labels"]
            )
            
            assert loss is not None
            assert loss.item() > 0
            
            # 4. Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # 5. Inference
            questions = model.generate(
                ["Test context about relativity theory"],
                ["relativity"],
                ["Remember"],
                num_beams=1,
                max_new_tokens=20
            )
            
            assert len(questions) == 1
            assert isinstance(questions[0], str)
            assert len(questions[0]) > 0
            
        finally:
            Path(temp_path).unlink()
    
    def test_checkpoint_save_load(self, sample_data):
        """Test checkpoint saving and loading."""
        from bloom_qg.models import HybridModel
        
        device = torch.device("cpu")
        
        # Create model
        model = HybridModel(device=device)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": 1,
                "best_loss": 2.5
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint
            loaded = torch.load(checkpoint_path, map_location=device)
            
            assert "model_state_dict" in loaded
            assert "epoch" in loaded
            assert "best_loss" in loaded
            
            # Load state dict
            new_model = HybridModel(device=device)
            new_model.load_state_dict(loaded["model_state_dict"])
            
            # Verify parameters match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2), "Parameters don't match after loading"


class TestTrainingStability:
    """Test training stability and convergence."""
    
    def test_gradient_flow(self, sample_data):
        """Test that gradients flow correctly."""
        from transformers import AutoTokenizer
        from torch.utils.data import DataLoader
        from bloom_qg.models import HybridModel
        from bloom_qg.data import BloomQGDataset, create_collate_fn
        
        device = torch.device("cpu")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_path = f.name
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            dataset = BloomQGDataset(temp_path, tokenizer, max_length=128)
            collate_fn = create_collate_fn(tokenizer, max_length=128)
            loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
            
            model = HybridModel(device=device)
            
            batch = next(iter(loader))
            loss = model(
                batch["contexts"],
                batch["answers"],
                batch["bloom_levels"],
                batch["labels"]
            )
            
            loss.backward()
            
            # Check that gradients exist for trainable parameters
            fusion_layer = model.fusion_layer
            assert fusion_layer.prototypes.grad is not None, "Prototypes should have gradients"
            assert fusion_layer.alpha.grad is not None, "Alpha should have gradients"
            
            # Check gradient magnitudes
            total_grad = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad += param.grad.abs().sum().item()
            
            assert total_grad > 0, "Total gradient should be non-zero"
            
        finally:
            Path(temp_path).unlink()
    
    def test_loss_decreases(self, sample_data):
        """Test that loss decreases over multiple steps."""
        from transformers import AutoTokenizer
        from torch.utils.data import DataLoader
        from bloom_qg.models import HybridModel
        from bloom_qg.data import BloomQGDataset, create_collate_fn
        
        device = torch.device("cpu")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data * 10, f)  # Repeat data
            temp_path = f.name
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            dataset = BloomQGDataset(temp_path, tokenizer, max_length=128)
            collate_fn = create_collate_fn(tokenizer, max_length=128)
            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
            
            model = HybridModel(device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Higher LR for test
            
            losses = []
            for i, batch in enumerate(loader):
                if i >= 5:  # Only 5 steps
                    break
                
                loss = model(
                    batch["contexts"],
                    batch["answers"],
                    batch["bloom_levels"],
                    batch["labels"]
                )
                
                losses.append(loss.item())
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Check that average loss in last 2 steps < first 2 steps
            early_loss = sum(losses[:2]) / 2
            late_loss = sum(losses[-2:]) / 2
            
            # Loss should decrease (or at least not increase significantly)
            assert late_loss <= early_loss * 1.1, f"Loss increased: {early_loss:.3f} -> {late_loss:.3f}"
            
        finally:
            Path(temp_path).unlink()


class TestInferenceConsistency:
    """Test inference consistency and reproducibility."""
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic with fixed seed."""
        from bloom_qg.models import HybridModel
        import random
        import numpy as np
        
        device = torch.device("cpu")
        
        def set_seed(seed=42):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        contexts = ["Water boils at 100 degrees Celsius"]
        answers = ["100 degrees"]
        bloom_levels = ["Remember"]
        
        # Generate twice with same seed
        set_seed(42)
        model1 = HybridModel(device=device)
        q1 = model1.generate(contexts, answers, bloom_levels, num_beams=1, max_new_tokens=20)
        
        set_seed(42)
        model2 = HybridModel(device=device)
        q2 = model2.generate(contexts, answers, bloom_levels, num_beams=1, max_new_tokens=20)
        
        # Should be identical with greedy decoding
        assert q1[0] == q2[0], "Generation should be deterministic with same seed"


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for the pipeline."""
    
    def test_training_throughput(self, sample_data):
        """Test training throughput (samples/second)."""
        import time
        from transformers import AutoTokenizer
        from torch.utils.data import DataLoader
        from bloom_qg.models import HybridModel
        from bloom_qg.data import BloomQGDataset, create_collate_fn
        
        device = torch.device("cpu")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data * 50, f)  # 100 samples
            temp_path = f.name
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            dataset = BloomQGDataset(temp_path, tokenizer, max_length=128)
            collate_fn = create_collate_fn(tokenizer, max_length=128)
            loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
            
            model = HybridModel(device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            start = time.time()
            total_samples = 0
            
            for batch in loader:
                loss = model(
                    batch["contexts"],
                    batch["answers"],
                    batch["bloom_levels"],
                    batch["labels"]
                )
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_samples += len(batch["contexts"])
            
            elapsed = time.time() - start
            throughput = total_samples / elapsed
            
            print(f"\nTraining throughput: {throughput:.2f} samples/sec (CPU)")
            assert throughput > 0.1, "Training throughput too low"
            
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
