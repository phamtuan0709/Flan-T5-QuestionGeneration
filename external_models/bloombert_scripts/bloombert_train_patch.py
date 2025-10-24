"""
Patch for BloomBERT training script to make nlpaug optional.

This file replaces the original train_bloombert.py to avoid dependency 
on nlpaug when augmentation is not needed.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

# Add BloomBERT to path
bloombert_path = Path(__file__).parent.parent / "BloomBERT"
sys.path.insert(0, str(bloombert_path))

from src.model.bloombert import BloomBERT
from src.dataset.process_data import BloomsDataset
from src.helper.plots_helper import plot_confusion_matrix


def train_model_bloombert(
    df, tokenizer, config, class_weights=None, test_size=0.2, augment=False
):
    """
    Train BloomBERT model with optional data augmentation.
    
    Args:
        df: DataFrame with Text and Label columns
        tokenizer: DistilBERT tokenizer
        config: Dict with learning_rate, batch_size, epochs, device, 
                patience (for early stopping), min_delta (for improvement threshold)
        class_weights: Tensor of class weights for imbalanced data
        test_size: Ratio for train/validation split
        augment: Whether to use SMOTE augmentation (requires nlpaug)
    
    Returns:
        best_model: Trained model with best validation accuracy
        history: Training history dict
        best_val_acc: Best validation accuracy achieved
    """
    X_train, X_val, y_train, y_val = train_test_split(
        df["Text"],
        df["Label"],
        test_size=test_size,
        random_state=1234,
        stratify=df["Label"],
        shuffle=True  # Ensure shuffling
    )

    train = pd.DataFrame({"Text": X_train, "Label": y_train})
    val = pd.DataFrame({"Text": X_val, "Label": y_val})
    
    # Shuffle training data again
    train = train.sample(frac=1, random_state=1234).reset_index(drop=True)

    if augment:
        # Import only when needed
        import nltk
        import nlpaug.augmenter.word as nlpaw
        from src.dataset.augment import augment_data
        
        nltk.download("wordnet")
        nltk.download("averaged_perceptron_tagger_eng")

        aug = nlpaw.SynonymAug(aug_src="wordnet", aug_max=3)
        max_count = max(train["Label"].value_counts())
        print("Oversampling training data")
        print("-- Initial training data distribution")
        print(train["Label"].value_counts().sort_index())
        train = augment_data(train, aug, target_count=max_count)
        print("-- Augmented training data distribution")
        print(train["Label"].value_counts().sort_index())

    train_encodings = tokenizer(list(train["Text"]), truncation=True, padding=True)
    val_encodings = tokenizer(list(val["Text"]), truncation=True, padding=True)

    train_dataset = BloomsDataset(train_encodings, list(train["Label"]))
    val_dataset = BloomsDataset(val_encodings, list(val["Label"]))

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # Initialize model
    model = BloomBERT(output_dim=6).to(config["device"])
    
    # Optimizer with weight decay for better regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=0.01,  # L2 regularization
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup
    from torch.optim.lr_scheduler import OneCycleLR
    total_steps = len(train_dataloader) * config["epochs"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_model = None
    
    # Early stopping parameters
    patience = config.get("patience", 10)
    min_delta = config.get("min_delta", 0.001)
    patience_counter = 0

    print(f"\nTraining for {config['epochs']} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Early stopping: patience={patience}, min_delta={min_delta}")
    print(f"Learning rate: OneCycleLR with warmup (max_lr={config['learning_rate']:.2e})")
    print(f"Weight decay: 0.01 (L2 regularization)")
    
    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        for batch in pbar:
            input_ids = batch["input_ids"].to(config["device"])
            attention_mask = batch["attention_mask"].to(config["device"])
            labels = batch["labels"].to(config["device"])

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct/train_total:.4f}"})
            
            # Step LR scheduler after each batch (OneCycleLR)
            scheduler.step()

        avg_train_loss = train_loss / len(train_dataloader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
                input_ids = batch["input_ids"].to(config["device"])
                attention_mask = batch["attention_mask"].to(config["device"])
                labels = batch["labels"].to(config["device"])

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_acc = val_correct / val_total
        
        # Get current learning rate (for logging only - OneCycleLR steps per batch)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e}")

        # Save best model and check early stopping
        improvement = val_acc - best_val_acc
        if improvement > min_delta:
            best_val_acc = val_acc
            best_model = BloomBERT(output_dim=6).to(config["device"])
            best_model.load_state_dict(model.state_dict())
            patience_counter = 0
            print(f"✓ New best model! Val Acc: {val_acc:.4f} (improvement: +{improvement:.4f})")
        else:
            patience_counter += 1
            print(f"⚠ No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs!")
                print(f"   Best validation accuracy: {best_val_acc:.4f}")
                break

    return best_model, history, best_val_acc
