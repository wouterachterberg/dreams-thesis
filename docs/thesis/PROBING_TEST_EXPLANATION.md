# probing_test Dataset: Purpose and Usage

## What is probing_test.parquet?

**probing_test** is a **held-out test set** (15% of data) used for evaluation purposes:

- **Created by**: `align_splits_for_finetuning.py` 
- **Size**: 45,185 spectra over 6,147 unique molecules in the OOD split
- **Key property**: NEVER seen during fine-tuning (isolated from both ft_train and ft_val)
- **Purpose**: Provide unbiased evaluation on out-of-distribution data

## How it's Used

### Axis 1: Probing (Linear/MLP Probing)
- Train linear classifiers on top of frozen DreaMS embeddings
- Evaluate what molecular properties are encoded in the 1024-dim embeddings
- Uses simple_probing.py with LogisticRegression or MLP classifiers
- Tests ~200 molecular descriptors (Tanimoto similarity, fingerprints, etc.)

### Current Analysis: Fingerprint Prediction Evaluation
- Uses probing_test to evaluate fine-tuned FingerprintHead performance
- Measures cosine similarity between predicted and true Morgan fingerprints
- Computes per-bit AUROC for 2048 fingerprint bits

## Critical Finding

**Cosine Similarity Discrepancy:**

| Metric | Value | Source |
|--------|-------|--------|
| **Validation (during training)** | ~0.65 | WandB logs (ValCosineSimilarity) |
| **probing_test (current eval)** | 0.39 | `legacy/notebooks/per_bit_morgan_analysis.ipynb` |
| **DreaMS paper** | 0.646 | Equation 9, Extended Data Table 1 |

This **0.26 point discrepancy** suggests:

### Possible Explanations

1. **Wrong Checkpoint Selected** ⚠️ HIGH PRIORITY
   - Using `last.ckpt` (epoch 68, final training checkpoint)
   - May have overfit after best validation loss
   - ModelCheckpoint monitors 'Train loss' but EarlyStopping monitors 'Val loss'
   - Should verify best validation epoch from WandB logs

2. **Dataset Distribution Shift**
   - probing_test may have different spectrum characteristics than validation set
   - Check if preprocessing pipeline is identical
   - Validate that same data augmentation was used

3. **Architecture Verification**
   - ✓ FingerprintHead uses DeepSets pooling (`head_phi_depth=1`)
   - ✓ Uses all peak embeddings, not just precursor
   - ✓ Training reached ~0.65 cosine similarity

4. **Evaluation Protocol**
   - Current analysis binarizes predictions at optimal threshold
   - This **contradicts** cosine-loss training objective
   - DreaMS authors evaluated on **retrieval accuracy@k**, not binarized Tanimoto
   - Per-bit AUROC is fundamentally misaligned with training loss

## Recommendations

**Immediate Actions:**

1. **Test all checkpoints** on probing_test to find which one achieves 0.65:
   ```
   - epoch=53-step=7000.ckpt
   - epoch=60-step=8000.ckpt  ← likely best validation
   - epoch=68-step=9000.ckpt
   - last.ckpt
   ```

2. **Check WandB logs** for best validation loss epoch and corresponding metrics

3. **Implement retrieval evaluation** instead of (or in addition to) binarized metrics:
   - Rank candidates by cosine similarity
   - Compute accuracy@k for finding correct molecule
   - This aligns with training objective and DreaMS paper evaluation

## References

- **Dataset creation**: `dreams-thesis-wa/src/align_splits_for_finetuning.py` (lines 84-169)
- **Validation splits**: Lines 202-217 confirm probing_test isolation
- **FingerprintHead**: `dreams/models/heads/heads.py` lines 574-650
- **Probing implementation**: `dreams-thesis-wa/src/simple_probing.py`
- **Training script**: `dreams/training/train.py` lines 263-271 (EarlyStopping config)
