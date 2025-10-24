"""
GPU Training Script for Hybrid Bloom-QG Model.

Trains the hybrid model with FP16 mixed precision, gradient clipping,
and checkpoint saving.
"""

import argparse
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from bloom_qg.models import HybridModel
from bloom_qg.data import BloomQGDataset, create_collate_fn

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def train_epoch(
    model: HybridModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Hybrid model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for FP16
        device: Computation device
        accumulation_steps: Gradient accumulation steps
        max_grad_norm: Max gradient norm for clipping
        
    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        contexts = batch["contexts"]
        answers = batch["answers"]
        bloom_levels = batch["bloom_levels"]
        labels = batch["labels"].to(device)
        
        # Forward pass with mixed precision
        with autocast(dtype=torch.float16):
            outputs = model(
                contexts=contexts,
                answers=answers,
                bloom_levels=bloom_levels,
                labels=labels
            )
            loss = outputs["loss"]
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
        
        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        # Track loss
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item() * accumulation_steps:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: HybridModel,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """
    Validate model on validation set.
    
    Args:
        model: Hybrid model
        dataloader: Validation data loader
        device: Computation device
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            contexts = batch["contexts"]
            answers = batch["answers"]
            bloom_levels = batch["bloom_levels"]
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                contexts=contexts,
                answers=answers,
                bloom_levels=bloom_levels,
                labels=labels
            )
            
            total_loss += outputs["loss"].item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(
        description="Train Hybrid Bloom-QG Model on GPU"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/learningq_train.json",
        help="Path to annotated training data JSON (LearningQ or SQuAD format)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for scheduler (default: 0.1)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Check device
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This script requires GPU.")
        return
    device = torch.device("cuda")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to {output_dir}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    # Load dataset
    logger.info("Loading dataset...")
    logger.info(f"Dataset path: {args.data_path}")
    full_dataset = BloomQGDataset(args.data_path, tokenizer, args.max_length)
    logger.info(f"Dataset type: {full_dataset.dataset_type}")
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create dataloaders
    collate_fn = create_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = HybridModel(device="cuda")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Setup gradient scaler for FP16
    scaler = GradScaler()
    
    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            accumulation_steps=args.accumulation_steps
        )
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, device)
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        model.save_trainable(str(checkpoint_path))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_model.pt"
            model.save_trainable(str(best_path))
            logger.info(f"New best model saved! Val loss: {val_loss:.4f}")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
