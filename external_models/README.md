# External Models Directory

This directory contains external model repositories and training scripts.

## Structure

```
external_models/
├── BloomBERT/              # Cloned BloomBERT repository
│   ├── data/               # Training dataset (6,175 samples)
│   ├── src/                # BloomBERT source code
│   ├── model/              # Trained model weights (after training)
│   └── README.md
├── bloombert_scripts/      # Our training/testing scripts
│   ├── train_bloombert.py
│   ├── test_bloombert_classifier.py
│   ├── bloombert_train_patch.py
│   ├── requirements-bloombert.txt
│   └── README.md
└── README.md               # This file
```

## Quick Start

### 1. Train BloomBERT Classifier

```bash
# Install dependencies
cd bloombert_scripts
pip install -r requirements-bloombert.txt

# Train model (10 epochs for quick test)
python train_bloombert.py --epochs 10 --batch_size 64

# Or full training (50 epochs as in paper)
python train_bloombert.py --epochs 50 --batch_size 128
```

**Output**: Model saved to `BloomBERT/model/bloombert_model.pt`

### 2. Test Trained Model

```bash
cd bloombert_scripts
python test_bloombert_classifier.py
```

See `bloombert_scripts/README.md` for detailed usage instructions.

## BloomBERT

**Repository**: https://github.com/RyanLauQF/BloomBERT.git

**Purpose**: BloomBERT is a Bloom's Taxonomy classifier built on DistilBERT. While our Flan-T5 project doesn't use the BloomBERT classifier directly, we use the same base model (DistilBERT) for encoding.

### Architecture Difference:

1. **BloomBERT (in this repo)**: 
   - DistilBERT → Attention Pooling → 2-layer Classifier
   - Purpose: Classify tasks into 6 Bloom's Taxonomy levels
   - Output: Class probabilities (6 classes)

2. **Our BloomEncoder (in bloom_qg/models/)**:
   - DistilBERT → Mean Pooling → Embeddings
   - Purpose: Extract semantic embeddings for question generation
   - Output: 768-dim embeddings

### Using BloomBERT in Our Project:

The `bloom_qg/models/bloom_encoder.py` module uses `distilbert-base-uncased` as the base encoder, which is the same architecture that BloomBERT was built upon. This provides:

- ✅ **Lighter model**: 66M parameters (vs 110M for BERT-base)
- ✅ **Faster inference**: ~40% faster than BERT-base
- ✅ **Same quality**: Retains 97% of BERT-base's performance
- ✅ **Better for embedding extraction**: DistilBERT's architecture is well-suited for this task

### If You Want to Train BloomBERT Classifier:

If you need the actual BloomBERT classifier (not the encoder):

1. **Install dependencies**:
   ```bash
   cd external_models/BloomBERT
   pip install -r requirements.txt
   ```

2. **Prepare data**:
   - BloomBERT uses a dataset of 6,175 labeled tasks
   - Data is in `data/` directory

3. **Train the model**:
   ```bash
   # Follow instructions in BloomBERT/notebook_bloombert.ipynb
   # Or run the training script directly
   ```

4. **Save trained weights**:
   - The trained model will be saved as `.pth` or `.h5` file
   - You can then load it in your code

### Note:

⚠️ **This directory is excluded from git** (see `.gitignore`) to keep the repository clean. The `external_models/` folder will not be committed to GitHub.

If you need to share the project, others should clone BloomBERT separately:
```bash
mkdir -p external_models
cd external_models
git clone https://github.com/RyanLauQF/BloomBERT.git
```

---

**Current Status**: 
- ✅ BloomBERT repository cloned
- ✅ Using DistilBERT base model for encoding
- ⚠️ No pre-trained BloomBERT classifier weights (would need to train if needed)
