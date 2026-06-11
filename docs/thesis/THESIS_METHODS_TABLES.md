> **Archival note.** This is a working document from an earlier stage of the project, retained for provenance. It has been superseded by the final thesis and the repository's top-level `README.md`; some details (fingerprint conditions, decoder architecture, split naming) may be out of date. Do not treat it as the current account.

# Thesis Methods - Quick Reference Tables

**Generated**: January 9, 2026  
**Model**: Claude Opus 4.5

---

## Table 1: Dataset Specifications

| Parameter | Value | Source File | Line |
|-----------|-------|-------------|------|
| Dataset name | MassSpecGym | prepare_massspecgym_for_finetuning.py | L1-15 |
| Input format | TSV | prepare_massspecgym_for_finetuning.py | L51 |
| Output format | HDF5 (gzip compressed) | prepare_massspecgym_for_finetuning.py | L93 |
| Max peaks per spectrum | 128 | prepare_massspecgym_for_finetuning.py | L44 |
| Spectrum array shape | (N, 2, 128) | prepare_massspecgym_for_finetuning.py | L58 |
| Spectrum data type | float32 | prepare_massspecgym_for_finetuning.py | L58 |
| Intensity range | [0, 1] (normalized) | prepare_massspecgym_for_finetuning.py | L72-73 |
| Peak selection | Top 128 by intensity, re-sorted by m/z | prepare_massspecgym_for_finetuning.py | L76-80 |

---

## Table 2: Morgan Fingerprint Parameters

| Parameter | Value | Source File | Line |
|-----------|-------|-------------|------|
| Fingerprint type | Morgan (ECFP4) | add_morgan_fingerprints.py | L17 |
| Radius | 2 | add_morgan_fingerprints.py | L17 |
| Bit vector size | 2048 | add_morgan_fingerprints.py | L17 |
| Output data type | uint8 (binary) | add_morgan_fingerprints.py | L32 |
| RDKit function | `AllChem.GetMorganFingerprintAsBitVect` | add_morgan_fingerprints.py | L29 |
| Invalid SMILES handling | Zero vector | add_morgan_fingerprints.py | L25-26 |

---

## Table 3: RDKit Molecular Descriptors

| Descriptor | RDKit Function | Min | Max |
|------------|----------------|-----|-----|
| AlogP | `Crippen.MolLogP(mol)` | -13.05 | 26.85 |
| NumHAcceptors | `Lipinski.NumHAcceptors(mol)` | 0 | 36 |
| NumHDonors | `Lipinski.NumHDonors(mol)` | 0 | 20 |
| TPSA | `Descriptors.TPSA(mol)` | 0 | 585.03 |
| NumRotatableBonds | `Lipinski.NumRotatableBonds(mol)` | 0 | 68 |
| NumAromaticRings | `Lipinski.NumAromaticRings(mol)` | 0 | 8 |
| NumAliphaticRings | `Lipinski.NumAliphaticRings(mol)` | 0 | 22 |
| FractionCSP3 | `Lipinski.FractionCSP3(mol)` | 0 | 1 |
| QED | `QED.qed(mol)` | 0 | 1 |
| SA Score | `sascorer.calculateScore(mol)` | 1 | 10 |

*Source: dreams/utils/mols.py, lines 99-111 (MolPropertyCalculator.min_maxs)*

---

## Table 4: Data Split Parameters

| Parameter | Value | Source File | Line |
|-----------|-------|-------------|------|
| Splitting algorithm | `sklearn.model_selection.GroupShuffleSplit` | scaffold_splits.py | L26 |
| Grouping variable | scaffold_id (Bemis-Murcko) | scaffold_splits.py | L27 |
| Test set size | 20% | scaffold_splits.py | L14 |
| Validation set size | 10% | scaffold_splits.py | L15 |
| Training set size | 70% (derived) | - | - |
| Random seed | 42 | scaffold_splits.py | L16 |
| Leakage verification | Explicit assertions | scaffold_splits.py | L47-53 |
| n_splits | 1 | scaffold_splits.py | L26 |

---

## Table 5: Probing Configuration

### Linear Probing - Classification

| Parameter | Value | Source File | Line |
|-----------|-------|-------------|------|
| Model | `sklearn.linear_model.LogisticRegression` | simple_probing.py | L61 |
| Regularization (C) | 1.0 (default) | sklearn default | - |
| max_iter | 1000 | simple_probing.py | L61 |
| random_state | 42 | simple_probing.py | L27 |

### Linear Probing - Regression

| Parameter | Value | Source File | Line |
|-----------|-------|-------------|------|
| Model | `sklearn.linear_model.Ridge` | simple_probing.py | L118 |
| Alpha | 1.0 | simple_probing.py | L118 |
| Target preprocessing | StandardScaler | simple_probing.py | L105-106 |
| random_state | 42 | simple_probing.py | L27 |

### MLP Probing

| Parameter | Value | Source File | Line |
|-----------|-------|-------------|------|
| Hidden layers | (256, 128) | simple_probing.py | L64/L122 |
| max_iter | 500 | simple_probing.py | L65/L123 |
| early_stopping | True | simple_probing.py | L67/L125 |
| Activation | ReLU (default) | sklearn default | - |
| random_state | 42 | simple_probing.py | L27 |

---

## Table 6: Embedding Extraction

| Parameter | Value | Source File | Line |
|-----------|-------|-------------|------|
| Embedding dimension | 1024 | experiments/ablation/lr_bs.ipynb | d_model=1024 |
| Embedding source | Precursor peak (position 0) | heads.py | L90-91 |
| Backbone model | DreaMS (pre-trained) | heads.py | L53 |
| precursor_emb flag | True | heads.py | L55 |

---

## Table 7: Fine-tuning Hyperparameters

| Parameter | Value | Source File | Line |
|-----------|-------|-------------|------|
| Train objective | fp_morgan_2048 | finetune_massspecgym_morgan2048.sh | L82 |
| Learning rate | 1e-4 | finetune_massspecgym_morgan2048.sh | L87 |
| Batch size | 64 | finetune_massspecgym_morgan2048.sh | L88 |
| Weight decay | 0.0 (default) | train_argparse.py | L79 |
| Optimizer | Adam | heads.py | L143-144 |
| Max epochs | 100 | finetune_massspecgym_morgan2048.sh | L91 |
| Head depth | 2 | finetune_massspecgym_morgan2048.sh | L93 |
| Training seed | 3407 | finetune_massspecgym_morgan2048.sh | L94 |
| Training precision | 64-bit | finetune_massspecgym_morgan2048.sh | L95 |
| Validation frequency | 0.25 (every 25% epoch) | finetune_massspecgym_morgan2048.sh | L96 |
| Precursor intensity | 1.1 | finetune_massspecgym_morgan2048.sh | L89 |
| Max peaks | 128 | finetune_massspecgym_morgan2048.sh | L97 |
| Num devices | 1 | finetune_massspecgym_morgan2048.sh | L90 |
| Num workers | 16 | finetune_massspecgym_morgan2048.sh | L86 |
| Loss function | CosSimLoss | heads.py | L617 |

---

## Table 8: FeedForward Head Architecture

| Parameter | Value | Source File | Line |
|-----------|-------|-------------|------|
| Input dimension | 1024 (backbone.d_model) | heads.py | L630 |
| Output dimension | 2048 (fp_size) | heads.py | L631 |
| Hidden dimensions | 'interpolated' | heads.py | L631 |
| Depth | 2 | finetune_massspecgym_morgan2048.sh | L93 |
| act_last | False | heads.py | L633 |
| Activation | ReLU | feed_forward.py | L7 |
| Bias | False | heads.py | L635 |
| Dropout | 0 (default) | heads.py | L595 |

**Resulting architecture (depth=2):**
```
Linear(1024, ~1536) → ReLU → Dropout → Linear(~1536, 2048)
```

---

## Table 9: Loss Functions

| Loss | Class | Formula | Source File | Line |
|------|-------|---------|-------------|------|
| CosSimLoss | `CosSimLoss` | `1 - F.cosine_similarity(inputs, targets).mean()` | losses_metrics.py | L18-23 |
| SmoothIoU | `SmoothIoULoss` | `1 - (intersection + 1) / (union + 1)` | losses_metrics.py | L7-15 |
| BCE | `nn.BCELoss` | Standard binary cross-entropy | heads.py | L614 |

---

## Table 10: Evaluation Metrics

### Fingerprint Prediction Metrics

| Metric | TorchMetrics Class | Description |
|--------|-------------------|-------------|
| Tanimoto/Jaccard | `BinaryJaccardIndex` | Intersection over union |
| Cosine Similarity | `CosineSimilarity` | Cosine similarity between FP vectors |
| Binary AUROC | `BinaryAUROC` | Area under ROC curve |
| Binary Precision | `BinaryPrecision` | Precision for bit prediction |
| Binary Recall | `BinaryRecall` | Recall for bit prediction |
| Binary Accuracy | `BinaryAccuracy` | Per-bit accuracy |

### Probing Metrics - Classification

| Metric | Function | Source |
|--------|----------|--------|
| AUROC | `roc_auc_score(y_test, y_proba)` | simple_probing.py#L77 |
| Average Precision | `average_precision_score(y_test, y_proba)` | simple_probing.py#L78 |
| Accuracy | `(y_pred == y_test).mean()` | simple_probing.py#L79 |

### Probing Metrics - Regression

| Metric | Function | Source |
|--------|----------|--------|
| R² | `r2_score(y_test, y_pred)` | simple_probing.py#L139 |
| MSE | `mean_squared_error(y_test, y_pred)` | simple_probing.py#L140 |
| RMSE | `np.sqrt(mean_squared_error(y_test, y_pred))` | simple_probing.py#L141 |

---

## Table 11: Software Versions

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| Python | ≥3.11 | setup.py#L63 | Runtime |
| PyTorch | 2.2.1 | setup.py#L22 | Deep learning |
| PyTorch Lightning | 2.0.8 | setup.py#L23 | Training framework |
| torchmetrics | 1.3.2 | setup.py#L24 | Metrics |
| RDKit | 2023.9.6 | setup.py#L27 | Molecular processing |
| NumPy | 1.25.0 | setup.py#L20 | Numerical computing |
| Pandas | 2.2.1 | setup.py#L25 | Data manipulation |
| h5py | 3.11.0 | setup.py#L26 | HDF5 I/O |
| WandB | 0.16.4 | setup.py#L32 | Experiment tracking |
| UMAP | 0.5.6 | setup.py#L28 | Dimensionality reduction |
| matchms | 0.27.0 | setup.py#L34 | Mass spectrometry |
| numba | 0.58.0 | setup.py#L21 | JIT compilation |

---

## Table 12: Random Seeds

| Purpose | Value | Source File | Line |
|---------|-------|-------------|------|
| Training (model init, shuffling) | 3407 | finetune_massspecgym_morgan2048.sh | L94 |
| Data splitting | 42 | scaffold_splits.py | L16 |
| Probing | 42 | simple_probing.py | L27 |
| Default in train_argparse | 1 | train_argparse.py | L14 |

---

## Quick Copy-Paste Values

### For Methods Section

```
Morgan fingerprints: radius=2 (ECFP4), 2048 bits
Split ratios: 70% train / 10% val / 20% test
Split method: GroupShuffleSplit with Bemis-Murcko scaffolds
Training: Adam optimizer, lr=1e-4, batch_size=64, max_epochs=100
Loss: Cosine similarity loss (1 - cos_sim)
Head: FeedForward, depth=2, 1024→2048 dimensions
Seeds: 3407 (training), 42 (splitting)
Embedding dimension: 1024
Linear probe: LogisticRegression (C=1.0) / Ridge (α=1.0)
MLP probe: hidden_layers=(256, 128), early_stopping=True
```

---

*All values verified against source code. File paths are relative to repository root.*
