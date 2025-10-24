"""
Test script for trained BloomBERT classifier.

This script loads the trained BloomBERT model and tests it on sample inputs.

Usage:
    cd external_models/bloombert_scripts
    python test_bloombert_classifier.py
"""

import sys
from pathlib import Path

# Add BloomBERT src to path
bloombert_path = Path(__file__).parent.parent / "BloomBERT"
sys.path.insert(0, str(bloombert_path))

import torch
import numpy as np
from transformers import DistilBertTokenizer
from src.model.bloombert import BloomBERT

# Bloom's Taxonomy levels mapping
BLOOM_LEVELS = [
    "Remember",   # 0
    "Understand", # 1
    "Apply",      # 2
    "Analyze",    # 3
    "Evaluate",   # 4
    "Create"      # 5
]


def setup_device():
    """Setup computation device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def load_model(model_path, device):
    """Load trained BloomBERT model."""
    model = BloomBERT(output_dim=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    return model


def predict(model, tokenizer, texts, device):
    """Predict Bloom's level for given texts."""
    # Tokenize
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    return predictions.cpu().numpy(), probabilities.cpu().numpy()


def main():
    # Setup
    device = setup_device()
    
    # Paths relative to script location
    script_dir = Path(__file__).parent
    model_path = script_dir.parent / "BloomBERT" / "model" / "bloombert_model.pt"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first:")
        print("  cd external_models/bloombert_scripts")
        print("  python train_bloombert.py")
        return
    
    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = load_model(model_path, device)
    
    # Test examples
    print("\n" + "=" * 80)
    print("BLOOMBERT CLASSIFIER - TEST MODE")
    print("=" * 80)
    
    test_examples = [
        # Remember (0)
        "Recall the definition of photosynthesis",
        "List the steps in the scientific method",
        "Remember the formula for calculating area",
        
        # Understand (1)
        "Explain how photosynthesis works",
        "Summarize the main points of the article",
        "Describe the water cycle process",
        
        # Apply (2)
        "Use the formula to calculate the area of a circle",
        "Apply the scientific method to design an experiment",
        "Implement the algorithm in Python code",
        
        # Analyze (3)
        "Compare and contrast mitosis and meiosis",
        "Analyze the causes of World War II",
        "Break down the components of the ecosystem",
        
        # Evaluate (4)
        "Critique the effectiveness of the marketing strategy",
        "Judge the validity of the experimental results",
        "Assess the quality of the proposed solution",
        
        # Create (5)
        "Design a new automated testing framework",
        "Develop a machine learning model for image recognition",
        "Create an innovative solution to reduce carbon emissions"
    ]
    
    predictions, probabilities = predict(model, tokenizer, test_examples, device)
    
    print("\nPredictions:\n")
    for i, (text, pred_idx, probs) in enumerate(zip(test_examples, predictions, probabilities)):
        predicted_level = BLOOM_LEVELS[pred_idx]
        confidence = probs[pred_idx] * 100
        
        print(f"{i+1}. [{predicted_level}] (confidence: {confidence:.1f}%)")
        print(f"   Task: {text}")
        
        # Show top 3 predictions
        top3_indices = np.argsort(probs)[-3:][::-1]
        print("   Top 3:", end=" ")
        for idx in top3_indices:
            print(f"{BLOOM_LEVELS[idx]}({probs[idx]*100:.1f}%)", end=" ")
        print("\n")
    
    print("=" * 80)
    
    # Interactive mode
    print("\n💡 Want to test your own examples? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        print("\nInteractive Mode - Enter tasks to classify (or 'quit' to exit)")
        print("-" * 80)
        
        while True:
            print("\nTask: ", end="")
            text = input().strip()
            
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            predictions, probabilities = predict(model, tokenizer, [text], device)
            pred_idx = predictions[0]
            probs = probabilities[0]
            
            print(f"\n✓ Predicted Level: {BLOOM_LEVELS[pred_idx]}")
            print(f"  Confidence: {probs[pred_idx]*100:.1f}%")
            print("\n  All probabilities:")
            for i, level in enumerate(BLOOM_LEVELS):
                bar = "█" * int(probs[i] * 50)
                print(f"    {level:12s}: {probs[i]*100:5.1f}% {bar}")
    
    print("\n✓ Testing completed!")


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    main()
