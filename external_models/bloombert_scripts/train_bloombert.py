"""
Training script for BloomBERT classifier.

This script trains the BloomBERT model (DistilBERT-based classifier) 
on Bloom's Taxonomy dataset for use as the encoder in our QG system.

Usage:
    cd external_models/bloombert_scripts
    python train_bloombert.py [--epochs 50] [--batch_size 128] [--lr 1e-5]
"""

import sys
import argparse
import logging
from pathlib import Path

# Add BloomBERT src to path
bloombert_path = Path(__file__).parent.parent / "BloomBERT"
sys.path.insert(0, str(bloombert_path))

import torch
from transformers import DistilBertTokenizer

from src.dataset.process_data import load_dataset, calculate_class_weights
from bloombert_train_patch import train_model_bloombert
from src.helper.plots_helper import plot_training_history

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_device():
    """Setup computation device (MPS/CUDA/CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        logger.info("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def main():
    parser = argparse.ArgumentParser(
        description="Train BloomBERT classifier for Bloom's Taxonomy"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Training batch size (default: 128)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set ratio (default: 0.2)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use data augmentation (SMOTE)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.001,
        help="Minimum improvement for early stopping (default: 0.001)"
    )
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(1111)
    device = setup_device()
    
    # Paths relative to script location
    script_dir = Path(__file__).parent
    bloombert_dir = script_dir.parent / "BloomBERT"
    output_dir = bloombert_dir / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    data_path = bloombert_dir / "data" / "blooms_cleaned_dataset.csv"
    
    if not data_path.exists():
        logger.warning("Cleaned dataset not found, creating from raw data...")
        raw_data_path = bloombert_dir / "data" / "blooms_dataset.csv"
        df = load_dataset(str(raw_data_path), clean=True)
        df = df.sort_values(by=['Label', 'Text'], ascending=[True, True])
        df.to_csv(data_path, index=False)
        logger.info(f"Saved cleaned dataset to {data_path}")
    else:
        import pandas as pd
        df = pd.read_csv(data_path)
    
    logger.info(f"Dataset loaded: {len(df)} samples")
    logger.info(f"Label distribution:\n{df['Label'].value_counts().sort_index()}")
    
    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(df["Label"].to_numpy()).to(device)
    logger.info(f"Class weights: {class_weights}")
    
    # Load tokenizer
    logger.info("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Training configuration
    config = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "device": device,
        "patience": args.patience,
        "min_delta": args.min_delta
    }
    
    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Epochs: {config['epochs']}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Device: {config['device']}")
    logger.info(f"  Test size: {args.test_size}")
    logger.info(f"  Augmentation: {args.augment}")
    logger.info(f"  Early stopping patience: {config['patience']}")
    logger.info(f"  Min improvement delta: {config['min_delta']}")
    logger.info("=" * 60)
    
    # Train model
    logger.info("Starting training...")
    best_model, history, best_val_acc = train_model_bloombert(
        df,
        tokenizer,
        config,
        class_weights=class_weights,
        test_size=args.test_size,
        augment=args.augment
    )
    
    # Save model
    model_path = output_dir / "bloombert_model.pt"
    torch.save(best_model.state_dict(), model_path)
    logger.info(f"✓ Model saved to {model_path}")
    logger.info(f"✓ Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training history
    logger.info("Generating training plots...")
    plot_training_history(history)
    
    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Check training plots in the current directory")
    logger.info(f"2. Model weights saved to: {model_path}")
    logger.info("3. You can now use this model for inference or as encoder")
    logger.info("\nTo test the model:")
    logger.info("  python test_bloombert_classifier.py")


if __name__ == "__main__":
    main()
