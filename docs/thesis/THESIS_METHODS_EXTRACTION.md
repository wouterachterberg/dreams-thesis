> **Archival note.** This is a working document from an earlier stage of the project, retained for provenance. It has been superseded by the final thesis and the repository's top-level `README.md`; some details (fingerprint conditions, decoder architecture, split naming) may be out of date. Do not treat it as the current account.

# Thesis Methods Section - Complete Extraction

**Generated**: January 12, 2026 (Updated)  
**Model**: Claude Opus 4.5  
**Source**: DreaMS Codebase Analysis

---

## 2.1 Dataset Definition

### 2.1.1 Source Dataset

The dataset is **MassSpecGym**, a mass spectrometry benchmark dataset. The raw data is provided as a tab-separated values (TSV) file containing tandem mass spectrometry (MS/MS) spectra with associated molecular structures.

**Source file**: [dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py](dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py)

### 2.1.2 Data Format Conversion

The TSV data is converted to HDF5 format for efficient loading during training:

```python
# From prepare_massspecgym_for_finetuning.py, lines 44-91
def convert_tsv_to_hdf5(tsv_path, output_path, n_highest_peaks=128):
    """
    Convert MassSpecGym.tsv to DreaMS-compatible HDF5 format.
    """
    df = pd.read_csv(tsv_path, sep='\t')
    num_spectra = len(df)
    
    # Prepare spectrum array (num_spectra, 2, n_highest_peaks)
    spectrum_array = np.zeros((num_spectra, 2, n_highest_peaks), dtype=np.float32)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        mzs = parse_array_string(row['mzs'])
        intensities = parse_array_string(row['intensities'])
        
        # Normalize intensities to [0, 1] range
        if intensities.max() > 0:
            intensities = intensities / intensities.max()
        
        # Sort by intensity (descending) and keep top n_highest_peaks
        if len(mzs) > n_highest_peaks:
            sorted_indices = np.argsort(intensities)[::-1][:n_highest_peaks]
            sorted_indices = np.sort(sorted_indices)  # Re-sort by m/z
            mzs = mzs[sorted_indices]
            intensities = intensities[sorted_indices]
```

### 2.1.3 Spectrum Representation

| Parameter | Value | Source |
|-----------|-------|--------|
| **Max peaks per spectrum** | 128 | [prepare_massspecgym_for_finetuning.py#L44](dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py#L44) |
| **Array shape** | (N, 2, 128) where N = number of spectra | [prepare_massspecgym_for_finetuning.py#L58](dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py#L58) |
| **Data type** | float32 | [prepare_massspecgym_for_finetuning.py#L58](dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py#L58) |
| **Intensity normalization** | [0, 1] relative to base peak | [prepare_massspecgym_for_finetuning.py#L72-L73](dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py#L72-L73) |
| **Peak selection** | Top 128 by intensity, re-sorted by m/z | [prepare_massspecgym_for_finetuning.py#L76-L80](dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py#L76-L80) |
| **Compression** | gzip | [prepare_massspecgym_for_finetuning.py#L93](dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py#L93) |

### 2.1.4 HDF5 Structure

The HDF5 file contains the following datasets:

| Dataset Name | Description | Data Type |
|--------------|-------------|-----------|
| `spectrum` | m/z and intensity arrays | float32 |
| `precursor_mz` | Precursor m/z values | float32 |
| `charge` | Charge states (default: 1) | int32 |
| `adduct` | Adduct type (default: [M+H]+) | string |
| `smiles` | Molecular structures | string |
| `identifier` | Unique spectrum identifiers | string |
| `inchikey` | InChIKey identifiers | string |
| `fold` | Train/val/test split labels | string |

---

## 2.2 Molecular Pre-processing

### 2.2.1 Morgan Fingerprints (ECFP4)

Molecular fingerprints are computed using the Extended-Connectivity Fingerprint algorithm (ECFP4) via RDKit's Morgan fingerprint implementation.

**Source file**: [dreams-thesis-wa/src/add_morgan_fingerprints.py](dreams-thesis-wa/src/add_morgan_fingerprints.py)

```python
# From add_morgan_fingerprints.py, lines 17-32
def compute_morgan_fingerprint(smiles, fp_size=2048, radius=2):
    """
    Compute Morgan ECFP4 fingerprint from SMILES.
    
    Args:
        smiles: SMILES string
        fp_size: Fingerprint size (default 2048)
        radius: Radius for Morgan algorithm (default 2 = ECFP4)
    
    Returns:
        numpy array of fingerprint as uint8 (binary)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(fp_size, dtype=np.uint8)
    
    # Generate Morgan fingerprint (binary, radius=2 for ECFP4)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    
    # Convert to numpy array
    fp_array = np.array(fp, dtype=np.uint8)
    return fp_array
```

| Parameter | Value | Source |
|-----------|-------|--------|
| **Fingerprint type** | Morgan (ECFP4) | [add_morgan_fingerprints.py#L17](dreams-thesis-wa/src/add_morgan_fingerprints.py#L17) |
| **Radius** | 2 (equivalent to ECFP4) | [add_morgan_fingerprints.py#L17](dreams-thesis-wa/src/add_morgan_fingerprints.py#L17) |
| **Bit vector size** | 2048 | [add_morgan_fingerprints.py#L17](dreams-thesis-wa/src/add_morgan_fingerprints.py#L17) |
| **Output data type** | uint8 (binary: 0 or 1) | [add_morgan_fingerprints.py#L32](dreams-thesis-wa/src/add_morgan_fingerprints.py#L32) |
| **RDKit function** | `AllChem.GetMorganFingerprintAsBitVect` | [add_morgan_fingerprints.py#L29](dreams-thesis-wa/src/add_morgan_fingerprints.py#L29) |
| **Invalid SMILES handling** | Zero vector | [add_morgan_fingerprints.py#L25-L26](dreams-thesis-wa/src/add_morgan_fingerprints.py#L25-L26) |

### 2.2.2 RDKit Molecular Descriptors

All available RDKit molecular descriptors are calculated for each molecule (209 total, 201 valid after filtering). These descriptors span multiple categories:

**Source file**: [dreams-thesis-wa/notebooks/exploratory/probe_all_rdkit_descriptors.ipynb](dreams-thesis-wa/notebooks/exploratory/probe_all_rdkit_descriptors.ipynb)

```python
# From probe_all_rdkit_descriptors.ipynb
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# Get all 209 available descriptor names
descriptor_names = [desc[0] for desc in Descriptors.descList]

# Initialize calculator for all descriptors
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# Calculate all descriptors for a molecule
mol = Chem.MolFromSmiles(smiles)
desc_values = calc.CalcDescriptors(mol)
```

#### Descriptor Categories (~209 total)

| Category | Description | Examples |
|----------|-------------|----------|
| **Constitutional** | Basic counts (atoms, bonds, rings) | NumHeavyAtoms, NumRotatableBonds |
| **Topological** | Graph properties | Wiener index, Zagreb indices, Kappa indices |
| **Electronic** | Charge and electronegativity | MolLogP (AlogP), TPSA, MaxPartialCharge |
| **Geometric** | Shape, surface area, volume | PMI, RadiusOfGyration |
| **Pharmacophore** | Drug-likeness properties | QED, NumHAcceptors, NumHDonors |
| **Fragment-based** | Functional group counts | fr_Al_OH, fr_benzene, fr_ester |

#### Descriptor Filtering

Descriptors are filtered to remove invalid values before probing:

```python
# Filter criteria (from probe_all_rdkit_descriptors.ipynb)
# 1. NaN values: >10% missing → excluded
# 2. Inf values: any Inf → excluded  
# 3. Zero variance: constant value → excluded

# Result: 201 valid descriptors from 209 total
```

| Filtering Step | Count |
|----------------|-------|
| **Total RDKit descriptors** | 209 |
| **Valid after filtering** | 201 |
| **Removed (NaN/Inf/constant)** | 8 |

*Note: Full descriptor probing results available in [dreams-thesis-wa/notebooks/exploratory/analyze_descriptor_probing_results.ipynb](dreams-thesis-wa/notebooks/exploratory/analyze_descriptor_probing_results.ipynb).*

---

## 2.3 Data Splitting and Leakage Control

### 2.3.1 Murcko Histogram Splitting

To prevent data leakage, the dataset is split using the **Murcko histogram splitting algorithm** from the DreaMS codebase. This method groups molecules based on their Murcko scaffold histograms rather than exact scaffold identity, providing more robust structural separation.

**Source file**: [dreams-thesis-wa/src/prepare_massspecgym_murcko_split.py](dreams-thesis-wa/src/prepare_massspecgym_murcko_split.py)

```python
# From prepare_massspecgym_murcko_split.py, lines 69-99
def create_murcko_split(df, smiles_to_hist, val_frac=0.15):
    """
    Create train/val split based on Murcko histograms.
    
    This implements the algorithm from the DreaMS paper that ensures
    structurally similar molecules stay in the same split.
    """
    # Group by Murcko histogram
    df_gb = df_us.groupby('MurckoHistStr').agg(
        count=('smiles', 'count'),
        smiles_list=('smiles', list)
    ).reset_index()
    
    # Split algorithm from DreaMS tutorial
    median_i = len(df_gb) // 2
    for i in range(median_i, -1, -1):
        current_hist = df_gb.iloc[i]['MurckoHist']
        
        # Check if current histogram is a sub-histogram of any validation histogram
        is_val_subhist = any(
            are_sub_hists(current_hist, df_gb.iloc[j]['MurckoHist'], k=3, d=4)
            for j in val_idx
        )
        
        if is_val_subhist:
            train_idx.append(i)
        else:
            if cum_val_mols / len(df_us) <= val_frac:
                cum_val_mols += df_gb.iloc[i]['count']
                val_idx.append(i)
            else:
                train_idx.append(i)
```

**Core algorithm**: [dreams/algorithms/murcko_hist/murcko_hist.py](dreams/algorithms/murcko_hist/murcko_hist.py)

```python
# The are_sub_hists function with k=3, d=4 parameters determines
# whether two Murcko histograms are structurally related enough
# to potentially cause data leakage
are_sub_hists(hist1, hist2, k=3, d=4)
```

### 2.3.2 Split Statistics

The MassSpecGym dataset contains **231,104 spectra** from **31,602 unique molecules**, split into three mutually exclusive sets:

| Split | Spectra | % Spectra | Molecules | % Molecules | Spectra/Mol |
|-------|---------|-----------|-----------|-------------|-------------|
| **Train** | 159,271 | 68.9% | 21,471 | 67.9% | 7.4 |
| **Validation** | 45,185 | 19.6% | 6,147 | 19.5% | 7.4 |
| **Holdout** | 26,648 | 11.5% | 3,984 | 12.6% | 6.7 |
| **Total** | 231,104 | 100% | 31,602 | 100% | 7.3 |

*Note: Percentages differ slightly between spectra and molecules because the holdout set has fewer spectra per molecule on average (6.7 vs 7.4).*

**Important**: Each molecule may have multiple spectra acquired under different experimental conditions:
- Different collision energies
- Different instruments (Orbitrap, Q-TOF, etc.)
- Different ionization modes

**All spectra from the same molecule are assigned to the same fold** to prevent data leakage. The splitting is performed at the molecule level using Murcko histograms, then all associated spectra inherit their molecule's fold assignment.

| Statistic | Value |
|-----------|-------|
| **Average spectra/molecule** | 7.3 |
| **Min spectra/molecule** | 1 |
| **Max spectra/molecule** | 542 |
| **Molecules with 1 spectrum** | 5,469 (17.3%) |
| **Molecules with 2-5 spectra** | 18,644 (59.0%) |
| **Molecules with >50 spectra** | 670 (2.1%) |

| Parameter | Value | Source |
|-----------|-------|--------|
| **Splitting algorithm** | Murcko histogram | [prepare_massspecgym_murcko_split.py#L69](dreams-thesis-wa/src/prepare_massspecgym_murcko_split.py#L69) |
| **Leakage control** | `are_sub_hists(k=3, d=4)` | [prepare_massspecgym_murcko_split.py#L92](dreams-thesis-wa/src/prepare_massspecgym_murcko_split.py#L92) |
| **Grouping variable** | Murcko histogram string | [prepare_massspecgym_murcko_split.py#L78](dreams-thesis-wa/src/prepare_massspecgym_murcko_split.py#L78) |

### 2.3.3 Dataset Files

| File | Description | Contents |
|------|-------------|----------|
| `full.hdf5` | Complete dataset | train + val + holdout folds |
| `development.hdf5` | Development set | train + val only |
| `finetuning.hdf5` | Fine-tuning subset | Further split of train fold |
| `holdout.parquet` | Holdout evaluation | Unseen molecules for final testing |
| `probing_test.parquet` | Probing test set | For linear/MLP probing evaluation |

### 2.3.4 Superclass Distribution

Molecules were classified using **NPClassifier** (UCSD) into 73 chemical superclasses. The classification was performed via the NPClassifier API:

**Source file**: [dreams-thesis-wa/scripts/classify_superclasses_fast.py](dreams-thesis-wa/scripts/classify_superclasses_fast.py)

```python
# API endpoint for NPClassifier
url = f'https://npclassifier.ucsd.edu/classify?smiles={urllib.parse.quote(smiles)}'

# Using ThreadPoolExecutor with 30 workers for parallel classification
with ThreadPoolExecutor(max_workers=30) as executor:
    ...
```

**Top 10 Superclasses by Frequency:**

| Rank | Superclass | Train | Val | Holdout | Total | % |
|------|------------|-------|-----|---------|-------|---|
| 1 | Other/Synthetic | 6,250 | 2,110 | 1,228 | 9,588 | 30.3% |
| 2 | Tryptophan alkaloids | 1,251 | 434 | 299 | 1,984 | 6.3% |
| 3 | Flavonoids | 1,017 | 80 | 520 | 1,617 | 5.1% |
| 4 | Small peptides | 1,185 | 197 | 113 | 1,495 | 4.7% |
| 5 | Nicotinic acid alkaloids | 837 | 347 | 87 | 1,271 | 4.0% |
| 6 | Fatty amides | 1,167 | 75 | 0 | 1,242 | 3.9% |
| 7 | Anthranilic acid alkaloids | 624 | 249 | 304 | 1,177 | 3.7% |
| 8 | Coumarins | 540 | 342 | 285 | 1,167 | 3.7% |
| 9 | Steroids | 1,131 | 6 | 2 | 1,139 | 3.6% |
| 10 | Tyrosine alkaloids | 792 | 127 | 110 | 1,029 | 3.3% |

*Note: "Other/Synthetic" includes molecules not classified as natural products (synthetic compounds, drugs, etc.). Full distribution available in [superclass_distribution_full.csv](dreams-thesis-wa/data/processed/MassSpecGym_splits/superclass_distribution_full.csv).*

| Statistic | Value |
|-----------|-------|
| **Total superclasses** | 73 |
| **Natural products** | ~70% |
| **Other/Synthetic** | ~30% |
| **Classification method** | NPClassifier API |
| **Classification rate** | ~42 SMILES/second |

---

## 2.4 Axis 1: Probing Methods

### 2.4.1 Embedding Extraction

Embeddings are extracted from the pre-trained DreaMS model. The model uses the embedding at the **precursor peak position** (index 0) as the spectrum representation.

**Source file**: [dreams/models/heads/heads.py](dreams/models/heads/heads.py)

```python
# From heads.py, lines 85-95
def forward(self, spec, charge=None, no_head=False):
    # Get backbone embeddings
    embs = self.backbone(spec, charge)

    if self.precursor_emb:
        # Output projection from precursor peak
        embs = embs[:, 0, ...]  # Take embedding at position 0 (precursor)

    if no_head:
        return embs

    return self.head(embs)
```

| Parameter | Value | Source |
|-----------|-------|--------|
| **Embedding dimension** | 1024 | [experiments/ablation/lr_bs.ipynb](experiments/ablation/lr_bs.ipynb) - d_model=1024 |
| **Embedding source** | Precursor peak (position 0) | [heads.py#L90-L91](dreams/models/heads/heads.py#L90-L91) |
| **Backbone model** | DreaMS (pre-trained) | [heads.py#L53](dreams/models/heads/heads.py#L53) |

### 2.4.2 Linear Probing

Linear probes are used to evaluate what molecular properties are encoded in the embeddings.

**Source file**: [dreams-thesis-wa/src/simple_probing.py](dreams-thesis-wa/src/simple_probing.py)

#### Binary Classification (LogisticRegression)

```python
# From simple_probing.py, lines 60-66
if probe_type == 'linear':
    probe = LogisticRegression(max_iter=1000, random_state=self.random_state)
else:  # mlp
    probe = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=500,
        random_state=self.random_state,
        early_stopping=True
    )
```

| Parameter | Value | Source |
|-----------|-------|--------|
| **Classifier** | `sklearn.linear_model.LogisticRegression` | [simple_probing.py#L61](dreams-thesis-wa/src/simple_probing.py#L61) |
| **Regularization (C)** | 1.0 (default) | sklearn default |
| **Max iterations** | 1000 | [simple_probing.py#L61](dreams-thesis-wa/src/simple_probing.py#L61) |
| **Random state** | 42 | [simple_probing.py#L27](dreams-thesis-wa/src/simple_probing.py#L27) |

#### Regression (Ridge)

```python
# From simple_probing.py, lines 118-124
if probe_type == 'linear':
    probe = Ridge(alpha=1.0, random_state=self.random_state)
else:  # mlp
    probe = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        max_iter=500,
        random_state=self.random_state,
        early_stopping=True
    )
```

| Parameter | Value | Source |
|-----------|-------|--------|
| **Regressor** | `sklearn.linear_model.Ridge` | [simple_probing.py#L118](dreams-thesis-wa/src/simple_probing.py#L118) |
| **Alpha (regularization)** | 1.0 | [simple_probing.py#L118](dreams-thesis-wa/src/simple_probing.py#L118) |
| **Target preprocessing** | StandardScaler | [simple_probing.py#L105-L106](dreams-thesis-wa/src/simple_probing.py#L105-L106) |

### 2.4.3 MLP Probing

Non-linear probes use scikit-learn's MLP implementations.

| Parameter | Value | Source |
|-----------|-------|--------|
| **Hidden layers** | (256, 128) | [simple_probing.py#L64](dreams-thesis-wa/src/simple_probing.py#L64) |
| **Max iterations** | 500 | [simple_probing.py#L65](dreams-thesis-wa/src/simple_probing.py#L65) |
| **Early stopping** | True | [simple_probing.py#L67](dreams-thesis-wa/src/simple_probing.py#L67) |
| **Activation** | ReLU (default) | sklearn default |

### 2.4.4 Evaluation Metrics

**Classification metrics:**
```python
# From simple_probing.py, lines 76-80
metrics = {
    'auroc': roc_auc_score(y_test, y_proba),
    'avg_precision': average_precision_score(y_test, y_proba),
    'accuracy': (y_pred == y_test).mean()
}
```

**Regression metrics:**
```python
# From simple_probing.py, lines 138-142
metrics = {
    'r2': r2_score(y_test, y_pred),
    'mse': mean_squared_error(y_test, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
}
```

| Metric | Type | Description |
|--------|------|-------------|
| **AUROC** | Classification | Area Under ROC Curve |
| **Average Precision** | Classification | Area under precision-recall curve |
| **Accuracy** | Classification | Fraction of correct predictions |
| **R²** | Regression | Coefficient of determination |
| **MSE** | Regression | Mean squared error |
| **RMSE** | Regression | Root mean squared error |

---

## 2.5 Axis 2: Fingerprint Prediction (Fine-tuning)

### 2.5.1 Model Architecture

The fine-tuning head is a **FeedForward network** attached to the pre-trained DreaMS backbone.

**Source file**: [dreams/models/heads/heads.py](dreams/models/heads/heads.py)

```python
# From heads.py, lines 593-642
class FingerprintHead(FineTuningHead):
    def __init__(self, backbone: Path, fp_str: str, lr, batch_size, weight_decay, dropout=0, loss='cos',
                 retrieval_val_pth=None, retrieval_epoch_freq=10, unfreeze_backbone_at_epoch=0,
                 head_depth=1, store_val_out_dir: Path = None, head_phi_depth: int = 0):
        
        super().__init__(backbone=backbone, lr=lr, weight_decay=weight_decay, 
                         precursor_emb=not head_phi_depth,
                         unfreeze_backbone_at_epoch=unfreeze_backbone_at_epoch)

        self.fp_str = fp_str
        self.fp_size = int(self.fp_str.split('_')[-1])  # e.g., "fp_morgan_2048" -> 2048
        
        # Define head for the backbone
        if self.head_phi_depth > 1:
            raise NotImplementedError
        if self.head_phi_depth == 1:
            self.head = DeepSets(...)
        else:
            self.head = FeedForward(
                in_dim=self.backbone.d_model,       # 1024
                out_dim=self.fp_size,               # 2048
                hidden_dim='interpolated',          # Linearly interpolated dimensions
                depth=self.head_depth,              # 2
                act_last=False,                     # No activation after final layer
                dropout=dropout,
                bias=False
            )
```

### 2.5.2 FeedForward Architecture

**Source file**: [dreams/models/layers/feed_forward.py](dreams/models/layers/feed_forward.py)

```python
# From feed_forward.py, lines 6-34
class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, depth=None, act_last=True, act=nn.ReLU, bias=True, dropout=0):
        super().__init__()

        if hidden_dim == 'interpolated':
            assert depth is not None
            # Linearly interpolate dimensions between in_dim and out_dim
            hidden_dim = utils.interpolate_interval(a=in_dim, b=out_dim, n=depth - 1, only_inter=True, rounded=True)
        
        self.ff = nn.ModuleList([])
        for l in range(depth):
            d1 = hidden_dim[l - 1] if l != 0 else in_dim
            d2 = hidden_dim[l] if l != depth - 1 else out_dim
            self.ff.append(nn.Linear(d1, d2, bias=bias))
            if l != depth - 1:
                self.ff.append(nn.Dropout(p=dropout))
            if l != depth - 1 or act_last:
                self.ff.append(act())
```

**Architecture with depth=2, in_dim=1024, out_dim=2048:**
```
Input: 1024
  └── Linear(1024, ~1536) + ReLU + Dropout
      └── Linear(~1536, 2048)  [No activation - raw logits]
Output: 2048
```

### 2.5.3 Loss Function

**Source file**: [dreams/models/optimization/losses_metrics.py](dreams/models/optimization/losses_metrics.py)

```python
# From losses_metrics.py, lines 18-23
class CosSimLoss(nn.Module):
    def __init__(self):
        super(CosSimLoss, self).__init__()

    def forward(self, inputs, targets):
        return 1 - F.cosine_similarity(inputs, targets).mean()
```

| Parameter | Value | Source |
|-----------|-------|--------|
| **Loss function** | Cosine Similarity Loss | [heads.py#L617](dreams/models/heads/heads.py#L617) |
| **Formula** | `1 - cos_sim(pred, target).mean()` | [losses_metrics.py#L22](dreams/models/optimization/losses_metrics.py#L22) |
| **Alternative losses** | BCELoss, SmoothIoULoss | [heads.py#L614-L621](dreams/models/heads/heads.py#L614-L621) |

### 2.5.4 Training Hyperparameters

**Source file**: [dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh)

```bash
# From finetune_massspecgym_morgan2048.sh, lines 78-99
python3 dreams/training/train.py \
 --train_objective fp_morgan_2048 \
 --train_regime fine-tuning \
 --lr 1e-4 \
 --batch_size 64 \
 --prec_intens 1.1 \
 --num_devices 1 \
 --max_epochs 100 \
 --log_every_n_steps 50 \
 --head_depth 2 \
 --seed 3407 \
 --train_precision 64 \
 --val_check_interval 0.25 \
 --max_peaks_n 128
```

| Parameter | Value | Source |
|-----------|-------|--------|
| **Learning rate** | 1e-4 | [finetune_massspecgym_morgan2048.sh#L87](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L87) |
| **Batch size** | 64 | [finetune_massspecgym_morgan2048.sh#L88](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L88) |
| **Weight decay** | 0.0 (default) | [train_argparse.py#L79](dreams/training/train_argparse.py#L79) |
| **Optimizer** | Adam | [heads.py#L143-L144](dreams/models/heads/heads.py#L143-L144) |
| **Max epochs** | 100 | [finetune_massspecgym_morgan2048.sh#L91](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L91) |
| **Head depth** | 2 | [finetune_massspecgym_morgan2048.sh#L93](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L93) |
| **Training seed** | 3407 | [finetune_massspecgym_morgan2048.sh#L94](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L94) |
| **Training precision** | 64-bit | [finetune_massspecgym_morgan2048.sh#L95](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L95) |
| **Validation frequency** | Every 25% of epoch | [finetune_massspecgym_morgan2048.sh#L96](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L96) |
| **Precursor intensity** | 1.1 | [finetune_massspecgym_morgan2048.sh#L89](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L89) |
| **Backbone freezing** | Frozen (epoch 0), unfreeze at epoch 0 (never frozen) | [train_argparse.py#L41](dreams/training/train_argparse.py#L41) |

### 2.5.5 Evaluation Metrics

**Source file**: [dreams/models/optimization/losses_metrics.py](dreams/models/optimization/losses_metrics.py)

```python
# From losses_metrics.py, lines 87-97
class FingerprintMetrics(torchmetrics.MetricCollection):
    def __init__(self, prefix=None, device=None):
        super(FingerprintMetrics, self).__init__([
            torchmetrics.classification.BinaryJaccardIndex(),  # Tanimoto similarity
            torchmetrics.classification.BinaryRecall(),
            torchmetrics.classification.BinaryPrecision(),
            torchmetrics.classification.BinaryAccuracy(),
            torchmetrics.classification.BinaryAUROC(),
            torchmetrics.CosineSimilarity(reduction='mean')
        ])
```

| Metric | Description |
|--------|-------------|
| **Tanimoto/Jaccard** | Intersection over union of fingerprint bits |
| **Cosine Similarity** | Cosine similarity between predicted and true fingerprints |
| **Binary AUROC** | Area under ROC curve for bit prediction |
| **Binary Precision/Recall** | Precision and recall for bit prediction |
| **Binary Accuracy** | Per-bit accuracy |

---

## 2.6 Implementation Details

### 2.6.1 Software Versions

**Source file**: [setup.py](setup.py)

| Package | Version | Purpose |
|---------|---------|---------|
| **Python** | ≥3.11 | [setup.py#L63](setup.py#L63) |
| **PyTorch** | 2.2.1 | [setup.py#L22](setup.py#L22) |
| **PyTorch Lightning** | 2.0.8 | [setup.py#L23](setup.py#L23) |
| **torchmetrics** | 1.3.2 | [setup.py#L24](setup.py#L24) |
| **RDKit** | 2023.9.6 | [setup.py#L27](setup.py#L27) |
| **NumPy** | 1.25.0 | [setup.py#L20](setup.py#L20) |
| **Pandas** | 2.2.1 | [setup.py#L25](setup.py#L25) |
| **h5py** | 3.11.0 | [setup.py#L26](setup.py#L26) |
| **scikit-learn** | (via torchmetrics) | [simple_probing.py#L7-L9](dreams-thesis-wa/src/simple_probing.py#L7-L9) |
| **WandB** | 0.16.4 | [setup.py#L32](setup.py#L32) |
| **UMAP** | 0.5.6 | [setup.py#L28](setup.py#L28) |
| **matchms** | 0.27.0 | [setup.py#L34](setup.py#L34) |

### 2.6.2 Random Seeds

| Seed | Purpose | Value | Source |
|------|---------|-------|--------|
| **Training seed** | Model initialization, data shuffling | 3407 | [finetune_massspecgym_morgan2048.sh#L94](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L94) |
| **Split seed** | Murcko histogram splitting | N/A (deterministic algorithm) | [prepare_massspecgym_murcko_split.py](dreams-thesis-wa/src/prepare_massspecgym_murcko_split.py) |
| **Probing seed** | Probe training | 42 | [simple_probing.py#L27](dreams-thesis-wa/src/simple_probing.py#L27) |

### 2.6.3 Spectrum Preprocessing

**Source file**: [dreams/utils/data.py](dreams/utils/data.py)

```python
# From data.py, lines 45-88
class SpectrumPreprocessor:
    def __init__(self, dformat: DataFormat, prec_intens=1.1, n_highest_peaks=None, 
                 spec_entropy_cleaning=False, normalize_mzs=False, precision=32, 
                 mz_shift_aug_p=0, mz_shift_aug_max=0, to_relative_intensities=True):
        
        self.dformat = dformat
        self.prec_intens = prec_intens
        self.n_highest_peaks = n_highest_peaks
        self.spec_entropy_cleaning = spec_entropy_cleaning
        self.normalize_mzs = normalize_mzs
        self.to_relative_intensities = to_relative_intensities
        self.precision = precision
```

**Preprocessing pipeline:**

1. **Peak trimming**: Keep top N peaks by intensity
2. **Peak padding**: Pad to fixed length with zeros
3. **Intensity normalization**: Convert to relative intensities [0, 1]
4. **Precursor prepending**: Add precursor m/z at position 0 with intensity 1.1

| Parameter | Value | Source |
|-----------|-------|--------|
| **Max peaks** | 128 | [finetune_massspecgym_morgan2048.sh#L97](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L97) |
| **Precursor intensity** | 1.1 | [finetune_massspecgym_morgan2048.sh#L89](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L89) |
| **Relative intensities** | True (default) | [data.py#L66](dreams/utils/data.py#L66) |
| **Precision** | 64-bit (training) | [finetune_massspecgym_morgan2048.sh#L95](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L95) |

### 2.6.4 Hardware and Training Infrastructure

| Parameter | Value | Source |
|-----------|-------|--------|
| **Number of devices** | 1 | [finetune_massspecgym_morgan2048.sh#L90](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L90) |
| **Data workers** | 16 | [finetune_massspecgym_morgan2048.sh#L86](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L86) |
| **Accelerator** | GPU (CUDA) | [heads.py#L72](dreams/models/heads/heads.py#L72) |
| **Experiment tracking** | WandB | [finetune_massspecgym_morgan2048.sh#L24-L40](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L24-L40) |

---

## Summary Table

| Section | Key Parameters |
|---------|----------------|
| **2.1 Dataset** | MassSpecGym (231,104 spectra, 31,602 molecules), TSV→HDF5, 128 peaks |
| **2.2 Preprocessing** | Morgan ECFP4 (r=2, 2048-bit), 209 RDKit descriptors (201 valid) |
| **2.3 Splitting** | Murcko histogram split; molecules: 67.9%/19.5%/12.6%, spectra: 68.9%/19.6%/11.5%; 73 NPClassifier superclasses |
| **2.4 Probing** | 1024-dim embeddings, LogReg (C=1.0), Ridge (α=1.0), MLP (256-128) |
| **2.5 Fine-tuning** | FeedForward head (1024→2048, depth=2), CosSimLoss, Adam (lr=1e-4) |
| **2.6 Implementation** | Python ≥3.11, PyTorch 2.2.1, seed=3407 (training), 64-bit precision |

---

*Document generated from DreaMS codebase analysis. All file references are relative to repository root.*
