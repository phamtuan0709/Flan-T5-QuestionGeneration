# BloomBERT Training Scripts

This directory contains scripts to train and test the BloomBERT classifier model.

## Structure

```
bloombert_scripts/
├── train_bloombert.py           # Training script
├── test_bloombert_classifier.py # Testing/inference script
├── bloombert_train_patch.py     # Training loop (makes nlpaug optional)
├── requirements-bloombert.txt   # Additional dependencies
└── README.md                    # This file
```

## Installation

1. **Install dependencies** (from project root or this directory):
   ```bash
   pip install -r requirements-bloombert.txt
   ```

2. **Verify BloomBERT repository** is cloned:
   ```bash
   ls ../BloomBERT/
   # Should see: data/, src/, README.md, etc.
   ```

## Usage

### Training BloomBERT

Train the classifier on Bloom's Taxonomy dataset (6,175 samples):

```bash
# Quick training (10 epochs for testing)
python train_bloombert.py --epochs 10 --batch_size 64

# Full training (50 epochs as in paper)
python train_bloombert.py --epochs 50 --batch_size 128

# Custom configuration
python train_bloombert.py --epochs 30 --batch_size 64 --lr 1e-5
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Training batch size (default: 128)
- `--lr`: Learning rate (default: 1e-5)
- `--test_size`: Validation split ratio (default: 0.2)
- `--augment`: Enable data augmentation with SMOTE (requires nlpaug)

**Output:**
- Trained model saved to: `../BloomBERT/model/bloombert_model.pt`
- Training plots generated in current directory

**Expected Results** (from paper):
- Training Accuracy: ~98.6%
- Validation Accuracy: ~90.4%
- Training time: ~30-40 minutes (50 epochs on M4/MPS)

### Testing BloomBERT

Test the trained model on sample inputs:

```bash
python test_bloombert_classifier.py
```

**Features:**
- Tests on 18 predefined examples (3 per Bloom level)
- Shows top-3 predictions with confidence scores
- Interactive mode for custom inputs
- Works on MPS/CUDA/CPU

**Example Output:**
```
1. [Remember] (confidence: 92.3%)
   Task: Recall the definition of photosynthesis
   Top 3: Remember(92.3%) Understand(5.1%) Apply(1.8%)

2. [Create] (confidence: 95.7%)
   Task: Design a new automated testing framework
   Top 3: Create(95.7%) Evaluate(2.3%) Apply(1.2%)
```

## Integration with Flan-T5 Project

The trained BloomBERT model can be used as an encoder in the main Flan-T5 question generation system:

1. **After training**, the model weights are saved to `../BloomBERT/model/bloombert_model.pt`

2. **The main project** uses DistilBERT (same base model as BloomBERT) for encoding in `bloom_qg/models/bloom_encoder.py`

3. **Optional**: You can load the trained BloomBERT weights if you want to use the fine-tuned version instead of the base DistilBERT

## Dataset Information

**Source**: `../BloomBERT/data/blooms_cleaned_dataset.csv`

**Statistics**:
- Total samples: 6,175
- Train/Val split: 80/20 (4,940 / 1,235)

**Label Distribution**:
```
Remember (0):   1,532 samples (24.8%)
Understand (1): 2,348 samples (38.0%)
Apply (2):        671 samples (10.9%)
Analyze (3):      560 samples (9.1%)
Evaluate (4):     634 samples (10.3%)
Create (5):       430 samples (7.0%)
```

**Class Weights** (for handling imbalance):
- Applied during training via weighted CrossEntropyLoss
- Automatically calculated from label distribution

## Architecture

**Base Model**: DistilBERT (distilbert-base-uncased)
- 66M parameters (40% smaller than BERT-base)
- 6 transformer layers
- 768 hidden dimensions

**Classifier Head**:
```python
DistilBERT
    ↓
Attention Pooling
    ↓
Dropout (0.3)
    ↓
Linear(768 → 128) + ReLU + Dropout(0.3)
    ↓
Linear(128 → 6)  # 6 Bloom levels
```

**Training Details**:
- Optimizer: AdamW
- Learning rate: 1e-5
- Loss: Weighted CrossEntropyLoss (handles class imbalance)
- Device: Auto-detects MPS/CUDA/CPU

## Troubleshooting

**Import errors for `src.*`**: 
- Scripts automatically add `../BloomBERT` to Python path
- Run scripts from this directory: `cd external_models/bloombert_scripts`

**MPS errors on Mac M4**:
- Some operations may fall back to CPU (expected)
- Overall training still runs on MPS for major ops

**Out of memory**:
- Reduce `--batch_size` (try 32 or 64)
- DistilBERT is already lightweight, but you can reduce sequence length in tokenizer

**No improvements after N epochs**:
- Model might have converged
- Try reducing learning rate: `--lr 5e-6`
- Check for overfitting (train acc >> val acc)

## Notes

- Training is **deterministic** (seed=1111) for reproducibility
- **Data augmentation** (--augment flag) requires `nlpaug` package (not installed by default)
- **Plots** are saved as PNG files in the current directory after training
- **Model checkpoints**: Only the best model (highest val acc) is saved

---

**Last Updated**: October 24, 2025
**Tested on**: Mac M4 with MPS, Python 3.11
