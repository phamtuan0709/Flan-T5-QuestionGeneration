"""
Quick verification script to test installation and basic functionality.

Run this after installing dependencies to verify the setup.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_imports():
    """Verify all required packages can be imported."""
    logger.info("Checking imports...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
        
        import transformers
        logger.info(f"✓ Transformers {transformers.__version__}")
        
        import peft
        logger.info(f"✓ PEFT {peft.__version__}")
        
        import datasets
        logger.info(f"✓ Datasets {datasets.__version__}")
        
        import sacrebleu
        logger.info(f"✓ SacreBLEU {sacrebleu.__version__}")
        
        import rouge_score
        logger.info(f"✓ ROUGE Score")
        
        import nltk
        logger.info(f"✓ NLTK {nltk.__version__}")
        
        # NEW: Check LearningQ dependencies
        try:
            import sentence_transformers
            logger.info(f"✓ Sentence Transformers {sentence_transformers.__version__}")
        except ImportError:
            logger.warning("⚠️  Sentence Transformers not found (needed for LearningQ)")
            logger.warning("   Install with: pip install sentence-transformers")
        
        # Check for data-prep-kit (optional for now)
        try:
            import data_prep_toolkit
            logger.info(f"✓ Data Prep Kit installed")
        except ImportError:
            logger.warning("⚠️  Data Prep Kit not found (optional for LearningQ)")
            logger.warning("   Install with: pip install data-prep-kit")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        logger.error("Please run: pip install -r requirements.txt")
        return False


def check_device():
    """Check available compute devices."""
    logger.info("\nChecking compute devices...")
    
    import torch
    
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("✗ CUDA not available")
    
    if torch.backends.mps.is_available():
        logger.info("✓ MPS (Apple Silicon) available")
    else:
        logger.info("✗ MPS not available")
    
    logger.info(f"✓ CPU available: {torch.get_num_threads()} threads")


def check_bloom_qg():
    """Verify bloom_qg package can be imported."""
    logger.info("\nChecking bloom_qg package...")
    
    try:
        from bloom_qg.models import BLOOM_LEVELS
        logger.info(f"✓ BLOOM_LEVELS: {BLOOM_LEVELS}")
        
        from bloom_qg.models import HybridModel
        logger.info("✓ HybridModel imported")
        
        from bloom_qg.data import BloomQGDataset, create_collate_fn
        logger.info("✓ Data utilities imported")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        logger.error("Please ensure you're in the correct directory")
        return False


def quick_model_test():
    """Quick test of model initialization (CPU only)."""
    logger.info("\nRunning quick model test (CPU, may take a minute)...")
    
    try:
        import torch
        from bloom_qg.models.fusion_layer import FusionLayer, BLOOM_LEVELS
        
        # Test fusion layer (lightweight)
        logger.info("Testing Fusion Layer...")
        fusion = FusionLayer()
        
        # Test forward pass
        h_bloom = torch.randn(2, 768)
        levels = ["Remember", "Understand"]
        v_prefix = fusion(h_bloom, levels)
        
        assert v_prefix.shape == (2, 768), "Unexpected output shape"
        logger.info(f"✓ Fusion Layer works! Output shape: {v_prefix.shape}")
        
        # Count parameters
        params = fusion.get_trainable_params()
        logger.info(f"✓ Trainable parameters: {params['total']:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        return False


def main():
    """Run all verification checks."""
    logger.info("=" * 60)
    logger.info("BLOOM-QG INSTALLATION VERIFICATION")
    logger.info("=" * 60)
    
    all_ok = True
    
    # Check imports
    if not check_imports():
        all_ok = False
    
    # Check devices
    check_device()
    
    # Check package
    if not check_bloom_qg():
        all_ok = False
    
    # Quick model test
    if not quick_model_test():
        all_ok = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    if all_ok:
        logger.info("✓ ALL CHECKS PASSED!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Prepare dataset: make prepare-small   (or make prepare for full)")
        logger.info("2. Train model:     make train-small     (or make train-full)")
        logger.info("3. Test inference:  make infer-test")
        logger.info("\nOr use direct commands:")
        logger.info("  python -m bloom_qg.data.prepare_learningq --limit 10000")
        logger.info("  python -m bloom_qg.train_gpu --data_path data/learningq_small.json")
        return 0
    else:
        logger.error("✗ SOME CHECKS FAILED")
        logger.error("=" * 60)
        logger.error("\nPlease fix the errors above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
