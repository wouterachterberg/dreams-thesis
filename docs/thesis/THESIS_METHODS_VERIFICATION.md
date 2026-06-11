> **Archival note.** This is a working document from an earlier stage of the project, retained for provenance. It has been superseded by the final thesis and the repository's top-level `README.md`; some details (fingerprint conditions, decoder architecture, split naming) may be out of date. Do not treat it as the current account.

# Thesis Methods Section - Implementation Verification

**Generated:** 13 January 2026  
**Purpose:** Verified implementation details from DreaMS codebase for Methods sections 2.4, 2.5, and 2.6

---

## AXIS 1: PROBING (Section 2.4)

### Linear Probes

| Parameter | Classification | Regression | Source |
|-----------|---------------|------------|--------|
| Model | `LogisticRegression` | `Ridge` | `simple_probing.py:37-50` |
| Regularization | C=1.0 (default) | `alpha=1.0` | Explicit in code |
| max_iter | `1000` | N/A | Explicit in code |
| Cross-validation | ❌ Not used | ❌ Not used (`Ridge`, not `RidgeCV`) | |

### MLP Probes

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | `(256, 128)` | Two hidden layers |
| max_iter | `500` | ⚠️ Not 1000 |
| early_stopping | `True` | Explicit |
| activation | `'relu'` | sklearn default (not explicit) |
| solver | `'adam'` | sklearn default (not explicit) |
| validation_fraction | `0.1` | sklearn default (not explicit) |
| Dropout | N/A | Not available in sklearn MLPClassifier/Regressor |

### Data Preprocessing

**Embeddings:** No normalization applied before probing.

**Regression Targets:** Standardized using `StandardScaler`
- ⚠️ **Issue:** Scaler fit on ALL data before train/test split (potential data leakage)
- Location: `simple_probing.py:113-114`

```python
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()  # Fits on entire dataset
```

### Evaluation

- Train/test split used (no explicit cross-validation for final metrics)
- Metrics computed on held-out test set

---

## AXIS 2: FINGERPRINT PREDICTION (Section 2.5)

### FeedForward Head Architecture

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | `FeedForward` | `heads.py:626-634` |
| Input dim | `backbone.d_model` | Dynamic |
| Output dim | `fp_size` (fingerprint size) | |
| hidden_dim | `'interpolated'` | Linear interpolation between in/out dims |
| depth | Configurable (default=1) | ⚠️ Check if you used depth=2 |
| Activation | `ReLU` | Between layers only |
| act_last | `False` | No activation on output |
| Dropout | Configurable | Applied between layers |
| Bias | `False` | Explicit |

**Interpolated Hidden Dims:** When `hidden_dim='interpolated'` and `depth=2`:
- Uses `utils.interpolate_interval(in_dim, out_dim, n=depth-1)` to compute intermediate sizes

### Training Configuration

| Parameter | Default Value | Recommended | Source |
|-----------|---------------|-------------|--------|
| Optimizer | `Adam` | - | `heads.py:687` |
| Learning rate | **Required** (no default) | `1e-4` | `train_argparse.py:22` |
| weight_decay | `0.0` | - | `train_argparse.py:26` |
| batch_size | `32` | 32-64 | `train_argparse.py:27` |
| max_epochs | `3000` | `50` | `train_argparse.py:23` |
| LR scheduler | ❌ Not implemented | - | |
| Gradient clipping | `None` (disabled) | - | `train_argparse.py:31` |

### Backbone Handling

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `unfreeze_backbone_at_epoch` | `0` (default) | Backbone trained from start |
| Backbone frozen | ❌ Never | Unless explicitly set |

Source: `heads.py:583`

### Loss Function

**CosSimLoss** (Cosine Similarity Loss):
```python
loss = 1 - F.cosine_similarity(inputs, targets).mean()
```
Source: `losses_metrics.py:14-17`

Alternative losses available: `'cross_entropy'` (BCELoss), `'smooth_iou'` (SmoothIoULoss)

### Evaluation Metrics

From `losses_metrics.py:35-50`:

| Metric | Implementation |
|--------|----------------|
| Tanimoto | `BinaryJaccardIndex` (torchmetrics) |
| Recall | `BinaryRecall` |
| Precision | `BinaryPrecision` |
| Accuracy | `BinaryAccuracy` |
| AUROC | `BinaryAUROC` |
| Cosine Similarity | `CosineSimilarity` |

**Binarization threshold:** `0.5` (torchmetrics default for binary metrics)

### Checkpointing & Early Stopping

| Component | Monitors | Other Settings |
|-----------|----------|----------------|
| ModelCheckpoint | `'Train loss'` ⚠️ | mode='min', every_n_train_steps=1000 |
| EarlyStopping | `'Val loss'` | patience=5 (default), min_delta=0.0 |

Source: `train.py:107-117`

### Data Loading

| Parameter | Value |
|-----------|-------|
| num_workers | `0` (default) |
| Data augmentation | ❌ Not implemented |

---

## AXIS 3: IMPLEMENTATION (Section 2.6)

### Package Versions

From `setup.py`:

| Package | Version |
|---------|---------|
| PyTorch | `2.2.1` |
| pytorch-lightning | `2.0.8` |
| RDKit | `2023.9.6` |
| torchmetrics | `1.3.2` |
| scikit-learn | Not pinned |

### Hardware Specifications

❌ **Not documented in codebase.** You will need to add from your experimental logs:
- GPU type (e.g., H100)
- Number of GPUs
- Cluster name (e.g., Snellius)
- GPU memory
- Training time

---

## Summary: Items to Verify/Update

| Claim to Check | Codebase Value | Action |
|----------------|----------------|--------|
| MLP max_iter | `500` (not 1000) | Verify your config |
| head_depth | `1` (default) | Confirm if you used 2 |
| batch_size | `32` (default) | Confirm if you used 64 |
| StandardScaler | Fits on all data | Note potential leakage |
| ModelCheckpoint | Monitors Train loss | Verify this is intended |
| Hardware specs | Not in code | Add from logs |

---

## Key Code References

| File | Content |
|------|---------|
| `dreams-thesis-wa/src/simple_probing.py` | Linear & MLP probe implementations |
| `dreams/models/heads/heads.py` | FingerprintHead class |
| `dreams/models/layers/feed_forward.py` | FeedForward architecture |
| `dreams/models/optimization/losses_metrics.py` | CosSimLoss, FingerprintMetrics |
| `dreams/training/train.py` | Trainer, callbacks |
| `dreams/training/train_argparse.py` | Training argument defaults |
| `setup.py` | Package versions |
