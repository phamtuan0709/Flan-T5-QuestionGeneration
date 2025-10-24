"""
Unit tests for data processing (Dataset, prepare_learningq).

Run with: pytest tests/test_data.py -v
"""

import pytest
import json
import tempfile
from pathlib import Path
from bloom_qg.data import BloomQGDataset, create_collate_fn


@pytest.fixture
def sample_learningq_data():
    """Create sample LearningQ data."""
    return [
        {
            "context": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "answer": "photosynthesis",
            "question": "What is the process by which plants convert light energy?",
            "bloom_level": "Remember",
            "source": "khan_academy"
        },
        {
            "context": "Machine learning algorithms can learn patterns from data without explicit programming.",
            "answer": "machine learning",
            "question": "What type of algorithms can learn from data automatically?",
            "bloom_level": "Understand",
            "source": "teded"
        },
        {
            "context": "The mitochondria is the powerhouse of the cell, producing ATP through cellular respiration.",
            "answer": "mitochondria",
            "question": "Which organelle is responsible for energy production in cells?",
            "bloom_level": "Remember",
            "source": "khan_academy"
        }
    ]


@pytest.fixture
def sample_squad_data():
    """Create sample SQuAD-like data (for backward compatibility)."""
    return [
        {
            "context": "Paris is the capital of France.",
            "answer": "Paris",
            "question": "What is the capital of France?",
            "bloom_level": "Remember"
        },
        {
            "context": "The Earth orbits around the Sun.",
            "answer": "the Sun",
            "question": "What does Earth orbit around?",
            "bloom_level": "Remember"
        }
    ]


class TestBloomQGDataset:
    """Test BloomQGDataset class."""
    
    def test_load_learningq_data(self, sample_learningq_data):
        """Test loading LearningQ format data."""
        from transformers import AutoTokenizer
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_learningq_data, f)
            temp_path = f.name
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            dataset = BloomQGDataset(temp_path, tokenizer, max_length=512)
            
            assert len(dataset) == 3
            assert dataset.dataset_type == "learningq"
            
            # Test __getitem__
            sample = dataset[0]
            assert "context" in sample
            assert "answer" in sample
            assert "question" in sample
            assert "bloom_level" in sample
            assert "source" in sample
            
        finally:
            Path(temp_path).unlink()
    
    def test_load_squad_data(self, sample_squad_data):
        """Test loading SQuAD format data (backward compatibility)."""
        from transformers import AutoTokenizer
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_squad_data, f)
            temp_path = f.name
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            dataset = BloomQGDataset(temp_path, tokenizer, max_length=512)
            
            assert len(dataset) == 2
            assert dataset.dataset_type == "squad"
            
        finally:
            Path(temp_path).unlink()
    
    def test_context_truncation(self, sample_learningq_data):
        """Test context truncation for long sequences."""
        from transformers import AutoTokenizer
        
        # Create data with very long context
        long_data = [{
            "context": "This is a very long context. " * 100,  # ~500 tokens
            "answer": "test",
            "question": "What is this?",
            "bloom_level": "Remember",
            "source": "khan_academy"
        }]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(long_data, f)
            temp_path = f.name
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            dataset = BloomQGDataset(temp_path, tokenizer, max_length=512, context_max_length=384)
            
            sample = dataset[0]
            
            # Check that context was truncated
            context_tokens = tokenizer.encode(sample["context"], add_special_tokens=False)
            assert len(context_tokens) <= 384, f"Context not truncated: {len(context_tokens)} tokens"
            
        finally:
            Path(temp_path).unlink()
    
    def test_dataset_statistics(self, sample_learningq_data):
        """Test context length statistics logging."""
        from transformers import AutoTokenizer
        import io
        import sys
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_learningq_data, f)
            temp_path = f.name
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            
            # Capture log output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            dataset = BloomQGDataset(temp_path, tokenizer, max_length=512)
            
            sys.stdout = sys.__stdout__
            
            assert len(dataset) == 3
            
        finally:
            Path(temp_path).unlink()


class TestCollateFunction:
    """Test collate_fn for DataLoader."""
    
    def test_collate_fn_basic(self):
        """Test basic collate function."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        collate_fn = create_collate_fn(tokenizer, max_length=512)
        
        batch = [
            {
                "context": "Context 1",
                "answer": "answer1",
                "question": "Question 1?",
                "bloom_level": "Remember"
            },
            {
                "context": "Context 2",
                "answer": "answer2",
                "question": "Question 2?",
                "bloom_level": "Understand"
            }
        ]
        
        result = collate_fn(batch)
        
        assert "contexts" in result
        assert "answers" in result
        assert "bloom_levels" in result
        assert "labels" in result
        
        assert len(result["contexts"]) == 2
        assert len(result["answers"]) == 2
        assert len(result["bloom_levels"]) == 2
        assert result["labels"].shape[0] == 2
    
    def test_collate_fn_padding(self):
        """Test padding in collate function."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        collate_fn = create_collate_fn(tokenizer, max_length=128)
        
        batch = [
            {
                "context": "Short",
                "answer": "ans",
                "question": "Q?",
                "bloom_level": "Remember"
            },
            {
                "context": "This is a much longer context with many more words",
                "answer": "long answer here",
                "question": "What is this long question about?",
                "bloom_level": "Understand"
            }
        ]
        
        result = collate_fn(batch)
        labels = result["labels"]
        
        # Check padding tokens are replaced with -100
        assert (labels == -100).any(), "Should have padding tokens replaced with -100"
    
    def test_collate_fn_truncation(self):
        """Test truncation in collate function."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        collate_fn = create_collate_fn(tokenizer, max_length=32)  # Very short
        
        batch = [
            {
                "context": "Context " * 50,
                "answer": "answer " * 10,
                "question": "Question " * 20,
                "bloom_level": "Remember"
            }
        ]
        
        result = collate_fn(batch)
        labels = result["labels"]
        
        # Check truncation
        assert labels.shape[1] <= 32, f"Labels not truncated: {labels.shape[1]} tokens"


class TestDataPreparation:
    """Test data preparation functions."""
    
    def test_bloom_keyword_classification(self):
        """Test Bloom level keyword classification."""
        # This would test the classify_bloom_level function from prepare_learningq
        # For now, just test the keywords exist
        from bloom_qg.data.prepare_learningq import BLOOM_KEYWORDS
        
        assert len(BLOOM_KEYWORDS) == 6
        assert "Remember" in BLOOM_KEYWORDS
        assert "Create" in BLOOM_KEYWORDS
        
        # Check each level has keywords
        for level, keywords in BLOOM_KEYWORDS.items():
            assert len(keywords) > 0, f"No keywords for {level}"


@pytest.mark.slow
class TestDatasetPerformance:
    """Performance tests for data processing."""
    
    def test_dataset_loading_speed(self, sample_learningq_data):
        """Test dataset loading performance."""
        import time
        from transformers import AutoTokenizer
        
        # Create larger dataset
        large_data = sample_learningq_data * 1000  # 3000 samples
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(large_data, f)
            temp_path = f.name
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            
            start = time.time()
            dataset = BloomQGDataset(temp_path, tokenizer, max_length=512)
            elapsed = time.time() - start
            
            assert len(dataset) == 3000
            assert elapsed < 5, f"Loading too slow: {elapsed:.2f}s"
            
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
