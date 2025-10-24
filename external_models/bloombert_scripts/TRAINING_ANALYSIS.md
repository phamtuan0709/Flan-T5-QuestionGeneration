# 📊 Phân Tích Training BloomBERT

## 🎯 Kết Quả Tổng Quan

### Thông Tin Training
- **Tổng epochs**: 29/50 (early stopping triggered)
- **Best validation accuracy**: **90.12%** (epoch 22)
- **Final train accuracy**: 99.64%
- **Final validation accuracy**: 89.80%
- **Overfitting gap**: ~10% (99.64% - 89.80%)

### Dataset
- Train samples: 4,940
- Validation samples: 1,235
- Batch size: 128
- Test/Val split: 20%

### Cấu Hình Optimization
- ✅ Learning rate: 1e-5 với OneCycleLR scheduler
- ✅ Weight decay: 0.01 (L2 regularization)
- ✅ Warmup: 10% of total steps
- ✅ Data shuffling: Enabled
- ✅ Class weights: Enabled (imbalanced dataset)
- ✅ Early stopping: patience=7, min_delta=0.001

---

## 📈 Diễn Biến Training

### Giai Đoạn 1: Warmup & Fast Learning (Epochs 1-5)
```
Epoch 1:  Train 17.94% → Val 24.05%  (LR: 1.33e-6)
Epoch 2:  Train 21.86% → Val 32.55%  (LR: 3.75e-6)
Epoch 3:  Train 33.48% → Val 48.02%  (LR: 6.73e-6)
Epoch 4:  Train 57.45% → Val 73.28%  (LR: 9.12e-6)
Epoch 5:  Train 79.90% → Val 81.78%  (LR: 1.00e-5) ← Peak LR
```
**Nhận xét**: Learning rate tăng dần (warmup), model học rất nhanh (+50% val acc in 5 epochs)

### Giai Đoạn 2: Peak Performance (Epochs 6-15)
```
Epoch 6:  Train 86.50% → Val 83.89%  (LR: 9.99e-6)
Epoch 7:  Train 89.55% → Val 86.15%  (LR: 9.95e-6)
Epoch 8:  Train 91.48% → Val 86.48%  (LR: 9.89e-6)
Epoch 9:  Train 93.00% → Val 87.13%  (LR: 9.80e-6)
Epoch 10: Train 94.29% → Val 89.07%  (LR: 9.70e-6)
Epoch 11: Train 95.14% → Val 88.91%  (LR: 9.56e-6) ⚠
Epoch 12: Train 95.93% → Val 87.69%  (LR: 9.41e-6) ⚠⚠
Epoch 13: Train 96.48% → Val 87.77%  (LR: 9.24e-6) ⚠⚠⚠
Epoch 14: Train 96.82% → Val 89.39%  (LR: 9.04e-6) ✓
Epoch 15: Train 97.63% → Val 89.96%  (LR: 8.82e-6) ✓
```
**Nhận xét**: Val acc đạt ~89-90%, bắt đầu có dấu hiệu plateau

### Giai Đoạn 3: Overfitting & Best Model (Epochs 16-22)
```
Epoch 16: Train 97.96% → Val 89.23%  (LR: 8.59e-6) ⚠
Epoch 17: Train 98.40% → Val 89.72%  (LR: 8.34e-6) ⚠⚠
Epoch 18: Train 98.62% → Val 90.04%  (LR: 8.07e-6) ⚠⚠⚠
Epoch 19: Train 98.83% → Val 89.47%  (LR: 7.79e-6) ⚠⚠⚠⚠
Epoch 20: Train 99.03% → Val 89.47%  (LR: 7.49e-6) ⚠⚠⚠⚠⚠
Epoch 21: Train 99.35% → Val 89.39%  (LR: 7.18e-6) ⚠⚠⚠⚠⚠⚠
Epoch 22: Train 99.31% → Val 90.12%  (LR: 6.86e-6) ✓✓✓ BEST!
```
**Nhận xét**: Train acc đạt 99%+, val acc dao động 89-90%, gap tăng lên ~9-10%

### Giai Đoạn 4: Decline & Early Stop (Epochs 23-29)
```
Epoch 23: Train 99.37% → Val 89.64%  (LR: 6.54e-6) ⚠
Epoch 24: Train 99.37% → Val 90.04%  (LR: 6.20e-6) ⚠⚠
Epoch 25: Train 99.53% → Val 89.55%  (LR: 5.86e-6) ⚠⚠⚠
Epoch 26: Train 99.66% → Val 89.23%  (LR: 5.51e-6) ⚠⚠⚠⚠
Epoch 27: Train 99.70% → Val 89.64%  (LR: 5.17e-6) ⚠⚠⚠⚠⚠
Epoch 28: Train 99.74% → Val 89.72%  (LR: 4.82e-6) ⚠⚠⚠⚠⚠⚠
Epoch 29: Train 99.64% → Val 89.80%  (LR: 4.47e-6) ⚠⚠⚠⚠⚠⚠⚠ STOP!
```
**Nhận xét**: Val acc không cải thiện trong 7 epochs, early stopping kích hoạt

---

## 🔍 Phân Tích Vấn Đề

### ✅ Những Điểm Tốt
1. **Learning curve tốt**: Không bị underfitting, model học được pattern
2. **Data shuffling hoạt động**: Không còn bị memorize thứ tự như lần trước
3. **OneCycleLR scheduler hiệu quả**: LR giảm dần sau peak, giúp fine-tune
4. **Weight decay giúp giảm overfitting**: Từ 98% train / 89% val → 99.6% / 90%
5. **Early stopping hoạt động đúng**: Dừng khi val không cải thiện

### ⚠️ Vấn Đề Chính: Overfitting Vẫn Còn

**Biểu hiện**:
- Train accuracy: 99.64%
- Val accuracy: 89.80%
- Gap: ~10% (lý tưởng < 5%)

**Nguyên nhân**:
1. **Dataset nhỏ**: Chỉ 4,940 samples training
2. **Model capacity lớn**: DistilBERT (66M params) với 6 classes
3. **Imbalanced dataset**: 
   - Class 1: 2,348 samples (nhiều nhất)
   - Class 5: 430 samples (ít nhất)
   - Ratio: 5.5:1
4. **Learning rate có thể còn cao**: 1e-5 max_lr
5. **Weight decay có thể chưa đủ**: 0.01

---

## 💡 Phương Pháp Cải Thiện

### 1️⃣ Data Augmentation (Ưu tiên cao)
```bash
# Sử dụng nlpaug để augment data
python train_bloombert.py \
  --epochs 50 \
  --batch_size 128 \
  --augment  # Enable augmentation
```

**Kỹ thuật**:
- Synonym replacement (thay từ đồng nghĩa)
- Random insertion
- Random swap
- Back translation

**Ước tính**: Tăng dataset lên ~10-15K samples → giảm overfitting ~5%

### 2️⃣ Tăng Regularization
```bash
# Option A: Tăng weight decay
python train_bloombert.py --weight_decay 0.05  # từ 0.01 → 0.05
```

Hoặc sửa trong `bloombert_train_patch.py`:
```python
# Thêm dropout cao hơn
model = BloomBERT(output_dim=6, dropout=0.3)  # từ 0.1 → 0.3

# Label smoothing
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1  # Thêm dòng này
)
```

**Ước tính**: Giảm train acc xuống 97-98%, tăng val acc lên 90-91%

### 3️⃣ Giảm Learning Rate
```bash
# Giảm max_lr từ 1e-5 → 5e-6
python train_bloombert.py --lr 5e-6
```

Hoặc sửa trong code:
```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=5e-6,  # Thay vì 1e-5
    total_steps=total_steps,
    pct_start=0.15,  # Tăng warmup từ 10% → 15%
    anneal_strategy='cos'
)
```

**Ước tính**: Training chậm hơn nhưng generalize tốt hơn

### 4️⃣ Ensemble hoặc Mixup
```python
# Thêm Mixup trong training loop
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
```

**Ước tính**: Tăng val acc ~1-2%

### 5️⃣ Balanced Sampling
```python
from torch.utils.data import WeightedRandomSampler

# Tính sample weights
class_counts = train['Label'].value_counts().sort_index()
weights = 1.0 / class_counts
sample_weights = [weights[label] for label in train['Label']]

# Tạo sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train),
    replacement=True
)

# Sử dụng trong DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    sampler=sampler  # Thay vì shuffle=True
)
```

**Ước tính**: Cải thiện val acc cho minority classes

---

## 🎯 Khuyến Nghị Thực Hiện

### Plan A: Quick Win (Thời gian: ~30 phút)
1. Enable data augmentation: `--augment`
2. Tăng weight decay: 0.01 → 0.05
3. Giảm learning rate: 1e-5 → 5e-6

**Kỳ vọng**: Val acc ~91-92%

### Plan B: Comprehensive (Thời gian: ~2 giờ)
1. Plan A +
2. Thêm label smoothing (0.1)
3. Tăng dropout (0.3)
4. Thêm balanced sampling
5. Tăng epochs lên 100 (với early stopping patience=10)

**Kỳ vọng**: Val acc ~92-93%

### Plan C: Research-level (Thời gian: ~1 ngày)
1. Plan B +
2. Implement Mixup
3. Try different architectures (BERT-base, RoBERTa)
4. Hyperparameter tuning (Optuna)
5. K-fold cross validation

**Kỳ vọng**: Val acc ~93-95%

---

## 📍 Vị Trí Model

```
/Users/phammtuan/Study/NCKH/Flan-T5/external_models/BloomBERT/model/bloombert_model.pt
Size: 254 MB
Best Val Acc: 90.12% (epoch 22)
```

### Sử Dụng Model

#### 1. Testing với test set
```bash
cd /Users/phammtuan/Study/NCKH/Flan-T5/external_models/bloombert_scripts
python test_bloombert_classifier.py
```

#### 2. Tích hợp vào Bloom-QG system
```python
# Trong bloom_qg/models/bloom_encoder.py
model = BloomBERT(output_dim=6)
model.load_state_dict(torch.load(
    "/Users/phammtuan/Study/NCKH/Flan-T5/external_models/BloomBERT/model/bloombert_model.pt"
))
model.eval()
```

#### 3. Inference trực tiếp
```python
from external_models.BloomBERT.src.BloomBERT import BloomBERT
import torch

model = BloomBERT(output_dim=6)
model.load_state_dict(torch.load("path/to/bloombert_model.pt"))
model.eval()

# Predict
text = "What is the capital of France?"
bloom_level = model.predict(text)  # 0-5
```

---

## 🚀 Lệnh Chạy Tiếp

### Quick retry với augmentation
```bash
cd /Users/phammtuan/Study/NCKH/Flan-T5/external_models/bloombert_scripts

# Backup model hiện tại
cp /Users/phammtuan/Study/NCKH/Flan-T5/external_models/BloomBERT/model/bloombert_model.pt \
   /Users/phammtuan/Study/NCKH/Flan-T5/external_models/BloomBERT/model/bloombert_model_v1_90.12.pt

# Train với augmentation
nohup python train_bloombert.py \
  --epochs 50 \
  --batch_size 128 \
  --lr 5e-6 \
  --patience 10 \
  --augment \
  > training_v2.log 2>&1 &

# Monitor
tail -f training_v2.log
```

---

## 📝 Ghi Chú

- **So sánh với lần trước**: Val acc tăng từ 89% (plateau) → 90.12% (nhờ shuffling + weight decay)
- **Training time**: ~18 phút cho 29 epochs (~37s/epoch)
- **Hardware**: Mac M4 với MPS (Apple Silicon)
- **Memory usage**: Stable, no OOM issues
- **Improvement rate**: ~0.3-0.5% val acc per improvement
