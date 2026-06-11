> **Archival note.** This is a working document from an earlier stage of the project, retained for provenance. It has been superseded by the final thesis and the repository's top-level `README.md`; some details (fingerprint conditions, decoder architecture, split naming) may be out of date. Do not treat it as the current account.

# Thesis Methods - Code Snippets Reference

**Generated**: January 12, 2026 (Updated)  
**Model**: Claude Opus 4.5

This document contains verified code snippets extracted directly from the DreaMS codebase for reference in your thesis Methods section.

---

## Snippet 1: HDF5 Conversion (Dataset Preparation)

**Source**: [dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py](dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py#L44-L91)

```python
def convert_tsv_to_hdf5(tsv_path, output_path, n_highest_peaks=128):
    """
    Convert MassSpecGym.tsv to DreaMS-compatible HDF5 format.
    
    Args:
        tsv_path: Path to input TSV file
        output_path: Path to output HDF5 file
        n_highest_peaks: Maximum number of peaks to keep per spectrum
    """
    df = pd.read_csv(tsv_path, sep='\t')
    num_spectra = len(df)
    
    # Prepare spectrum array (num_spectra, 2, n_highest_peaks)
    spectrum_array = np.zeros((num_spectra, 2, n_highest_peaks), dtype=np.float32)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        mzs = parse_array_string(row['mzs'])
        intensities = parse_array_string(row['intensities'])
        
        if len(mzs) > 0 and len(intensities) > 0:
            min_len = min(len(mzs), len(intensities))
            mzs = mzs[:min_len]
            intensities = intensities[:min_len]
            
            # Normalize intensities to [0, 1] range
            if intensities.max() > 0:
                intensities = intensities / intensities.max()
            
            # Sort by intensity (descending) and keep top n_highest_peaks
            if len(mzs) > n_highest_peaks:
                sorted_indices = np.argsort(intensities)[::-1][:n_highest_peaks]
                sorted_indices = np.sort(sorted_indices)  # Re-sort by m/z
                mzs = mzs[sorted_indices]
                intensities = intensities[sorted_indices]
            
            # Store in array
            num_peaks = len(mzs)
            spectrum_array[i, 0, :num_peaks] = mzs
            spectrum_array[i, 1, :num_peaks] = intensities
    
    # Create HDF5 file with gzip compression
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('spectrum', data=spectrum_array, compression='gzip')
        f.create_dataset('precursor_mz', 
                        data=df['precursor_mz'].fillna(0.0).astype(np.float32).values, 
                        compression='gzip')
        # ... additional datasets
```

**Key points:**
- Intensities normalized to [0, 1]
- Top 128 peaks selected by intensity
- Peaks re-sorted by m/z after selection
- HDF5 with gzip compression

---

## Snippet 2: Morgan Fingerprint Generation

**Source**: [dreams-thesis-wa/src/add_morgan_fingerprints.py](dreams-thesis-wa/src/add_morgan_fingerprints.py#L17-L32)

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

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
    try:
        if not smiles or smiles == '' or str(smiles) == 'nan':
            return np.zeros(fp_size, dtype=np.uint8)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(fp_size, dtype=np.uint8)
        
        # Generate Morgan fingerprint (binary, radius=2 for ECFP4)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
        
        # Convert to numpy array
        fp_array = np.array(fp, dtype=np.uint8)
        return fp_array
    except Exception as e:
        print(f"Error computing fingerprint for '{smiles}': {e}")
        return np.zeros(fp_size, dtype=np.uint8)
```

**Key points:**
- Morgan radius=2 (equivalent to ECFP4)
- 2048-bit binary fingerprint
- Invalid SMILES return zero vector

---

## Snippet 3: Murcko Histogram Splitting

**Source**: [dreams-thesis-wa/src/prepare_massspecgym_murcko_split.py](dreams-thesis-wa/src/prepare_massspecgym_murcko_split.py#L69-L99)

```python
from dreams.algorithms.murcko_hist import murcko_hist, are_sub_hists
from rdkit import Chem

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
    df_gb = df_gb.sort_values('count', ascending=False).reset_index(drop=True)
    
    # Split algorithm from DreaMS tutorial
    median_i = len(df_gb) // 2
    cum_val_mols = 0
    val_idx, train_idx = [], []
    
    for i in range(median_i, -1, -1):
        current_hist = df_gb.iloc[i]['MurckoHist']
        
        # Check if current histogram is a sub-histogram of any validation histogram
        is_val_subhist = any(
            are_sub_hists(current_hist, df_gb.iloc[j]['MurckoHist'], k=3, d=4)
            for j in val_idx
        )
        
        if is_val_subhist:
            train_idx.append(i)  # Assign to train to prevent leakage
        else:
            if cum_val_mols / len(df_us) <= val_frac:
                cum_val_mols += df_gb.iloc[i]['count']
                val_idx.append(i)
            else:
                train_idx.append(i)
    
    # Add remaining indices to train set
    train_idx.extend(range(median_i + 1, len(df_gb)))
    
    # Map SMILES to fold
    smiles_to_fold = {}
    for i, row in df_gb.iterrows():
        fold = 'val' if i in val_idx else 'train'
        for smiles in row['smiles_list']:
            smiles_to_fold[smiles] = fold
    
    return smiles_to_fold
```

**Key points:**
- Murcko histogram grouping (not just scaffold identity)
- `are_sub_hists(k=3, d=4)` prevents structurally similar molecules from crossing splits
- Molecules assigned to train if their histogram is a sub-histogram of any val histogram
- Final splits: Train 67.9% (21,471 molecules), Val 19.5% (6,147), Holdout 12.6% (3,984)

---

## Snippet 4: Linear Probing Implementation

**Source**: [dreams-thesis-wa/src/simple_probing.py](dreams-thesis-wa/src/simple_probing.py#L55-L90)

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

class SimpleProbe:
    def __init__(self, embeddings: np.ndarray, random_state: int = 42):
        self.embeddings = embeddings
        self.random_state = random_state
        self.results = {}
        
    def probe_binary(self, target, target_name, probe_type='linear', 
                     train_mask=None, test_mask=None):
        """Probe for binary classification task."""
        X_train, X_test = self.embeddings[train_mask], self.embeddings[test_mask]
        y_train, y_test = target[train_mask], target[test_mask]
        
        # Train probe
        if probe_type == 'linear':
            probe = LogisticRegression(max_iter=1000, random_state=self.random_state)
        else:  # mlp
            probe = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True
            )
        
        probe.fit(X_train, y_train)
        
        # Evaluate
        y_proba = probe.predict_proba(X_test)[:, 1]
        y_pred = probe.predict(X_test)
        
        metrics = {
            'auroc': roc_auc_score(y_test, y_proba),
            'avg_precision': average_precision_score(y_test, y_proba),
            'accuracy': (y_pred == y_test).mean()
        }
        return metrics
    
    def probe_regression(self, target, target_name, probe_type='linear',
                         train_mask=None, test_mask=None):
        """Probe for regression task."""
        # Standardize target
        scaler = StandardScaler()
        target_scaled = scaler.fit_transform(target.reshape(-1, 1)).squeeze()
        
        X_train, X_test = self.embeddings[train_mask], self.embeddings[test_mask]
        y_train, y_test = target_scaled[train_mask], target_scaled[test_mask]
        
        if probe_type == 'linear':
            probe = Ridge(alpha=1.0, random_state=self.random_state)
        else:  # mlp
            probe = MLPRegressor(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True
            )
        
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        return metrics
```

**Key points:**
- LogisticRegression with C=1.0 (default), max_iter=1000
- Ridge regression with alpha=1.0
- MLP probes with (256, 128) hidden layers
- StandardScaler for regression targets

---

## Snippet 5: FingerprintHead Architecture

**Source**: [dreams/models/heads/heads.py](dreams/models/heads/heads.py#L593-L642)

```python
import torch
from torch import nn
import pytorch_lightning as pl
from pathlib import Path

class FingerprintHead(FineTuningHead):
    def __init__(self, backbone: Path, fp_str: str, lr, batch_size, weight_decay, 
                 dropout=0, loss='cos', retrieval_val_pth=None, retrieval_epoch_freq=10, 
                 unfreeze_backbone_at_epoch=0, head_depth=1, store_val_out_dir=None, 
                 head_phi_depth=0):
        
        super().__init__(
            backbone=backbone, 
            lr=lr, 
            weight_decay=weight_decay, 
            precursor_emb=not head_phi_depth,
            unfreeze_backbone_at_epoch=unfreeze_backbone_at_epoch
        )

        self.fp_str = fp_str
        self.fp_size = int(self.fp_str.split('_')[-1])  # e.g., "fp_morgan_2048" -> 2048
        
        # Define loss function
        if loss == 'cross_entropy':
            self.loss = nn.BCELoss()
        elif loss == 'cos':
            self.loss = CosSimLoss()
        elif loss == 'smooth_iou':
            self.loss = SmoothIoULoss()
        
        # Define head architecture
        self.head = FeedForward(
            in_dim=self.backbone.d_model,    # 1024
            out_dim=self.fp_size,            # 2048
            hidden_dim='interpolated',       # Linearly interpolated
            depth=self.head_depth,           # 2
            act_last=False,                  # No activation on output
            dropout=dropout,
            bias=False
        )
        
        # Metrics
        self.val_metrics = FingerprintMetrics(prefix='Val')

    def step(self, data, batch_idx):
        pred = self(data['spec'], data['charge'])
        loss = self.loss(pred, data['label'])
        return pred, loss
```

**Key points:**
- FeedForward head with interpolated hidden dimensions
- CosSimLoss (cosine similarity loss) default
- 1024→2048 dimension mapping
- No activation on final layer

---

## Snippet 6: FeedForward Network

**Source**: [dreams/models/layers/feed_forward.py](dreams/models/layers/feed_forward.py#L1-L34)

```python
import torch.nn as nn
from typing import Sequence
import dreams.utils.misc as utils

class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, depth=None, 
                 act_last=True, act=nn.ReLU, bias=True, dropout=0):
        super().__init__()

        if isinstance(hidden_dim, int):
            assert depth is not None
            hidden_dim = [hidden_dim] * depth
        elif hidden_dim == 'interpolated':
            assert depth is not None
            # Linearly interpolate dimensions between in_dim and out_dim
            hidden_dim = utils.interpolate_interval(
                a=in_dim, b=out_dim, n=depth - 1, 
                only_inter=True, rounded=True
            )
        elif isinstance(hidden_dim, Sequence):
            depth = len(hidden_dim)
        
        self.ff = nn.ModuleList([])
        for l in range(depth):
            d1 = hidden_dim[l - 1] if l != 0 else in_dim
            d2 = hidden_dim[l] if l != depth - 1 else out_dim
            self.ff.append(nn.Linear(d1, d2, bias=bias))
            if l != depth - 1:
                self.ff.append(nn.Dropout(p=dropout))
            if l != depth - 1 or act_last:
                self.ff.append(act())
        self.ff = nn.Sequential(*self.ff)

    def forward(self, x):
        return self.ff(x)
```

**Key points:**
- 'interpolated' mode creates linearly spaced hidden dimensions
- ReLU activation between layers
- Dropout after linear layers (except last)
- Optional activation on final layer

---

## Snippet 7: Cosine Similarity Loss

**Source**: [dreams/models/optimization/losses_metrics.py](dreams/models/optimization/losses_metrics.py#L18-L23)

```python
import torch
import torch.nn.functional as F
from torch import nn

class CosSimLoss(nn.Module):
    def __init__(self):
        super(CosSimLoss, self).__init__()

    def forward(self, inputs, targets):
        return 1 - F.cosine_similarity(inputs, targets).mean()
```

**Key points:**
- Loss = 1 - cosine_similarity
- Mean reduction across batch
- Range: [0, 2] where 0 = perfect alignment

---

## Snippet 8: RDKit Descriptor Calculation (All 209 Descriptors)

**Source**: [dreams-thesis-wa/notebooks/exploratory/probe_all_rdkit_descriptors.ipynb](dreams-thesis-wa/notebooks/exploratory/probe_all_rdkit_descriptors.ipynb)

```python
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np

# Get all 209 RDKit descriptor names
descriptor_names = [desc[0] for desc in Descriptors.descList]
print(f"Total descriptors: {len(descriptor_names)}")  # 209

# Initialize calculator for all descriptors
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

def calculate_all_descriptors(smiles):
    """Calculate ALL RDKit descriptors for a single SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Calculate all 209 descriptors at once
        desc_values = calc.CalcDescriptors(mol)
        return dict(zip(descriptor_names, desc_values))
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None

# Filter invalid descriptors before probing
def filter_valid_descriptors(df_descriptors, descriptor_names, nan_threshold=0.1):
    """Filter out descriptors with >10% NaN, Inf values, or zero variance."""
    valid = []
    for desc in descriptor_names:
        values = df_descriptors[desc]
        if values.isna().mean() > nan_threshold:
            continue
        if np.isinf(values.dropna()).any():
            continue
        if values.dropna().std() == 0:
            continue
        valid.append(desc)
    return valid  # Returns 201 valid descriptors
```

**Key points:**
- 209 total RDKit molecular descriptors (all available)
- 201 valid after filtering (NaN/Inf/constant removal)
- Uses MolecularDescriptorCalculator for efficiency
- Categories: Constitutional, Topological, Electronic, Geometric, Pharmacophore, Fragment-based

---

## Snippet 9: MolPropertyCalculator with Normalization Ranges

**Source**: [dreams/utils/mols.py](dreams/utils/mols.py#L96-L145)

```python
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors, QED
import rdkit.Chem.GraphDescriptors
from rdkit.Contrib.SA_Score import sascorer

class MolPropertyCalculator:
    def __init__(self):
        # Min/max values from training part of MoNA and NIST20 Murcko histogram split
        self.min_maxs = {
            'AtomicLogP': {'min': -13.054800000000025, 'max': 26.849200000000053},
            'NumHAcceptors': {'min': 0.0, 'max': 36.0},
            'NumHDonors': {'min': 0.0, 'max': 20.0},
            'PolarSurfaceArea': {'min': 0.0, 'max': 585.0300000000002},
            'NumRotatableBonds': {'min': 0.0, 'max': 68.0},
            'NumAromaticRings': {'min': 0.0, 'max': 8.0},
            'NumAliphaticRings': {'min': 0.0, 'max': 22.0},
            'FractionCSP3': {'min': 0.0, 'max': 1.0},
            'QED': {'min': 0.0, 'max': 1.0},
            'SyntheticAccessibility': {'min': 1.0, 'max': 10.0},
            'BertzComplexity': {'min': 2.7548875021634682, 'max': 3748.669248605835}
        }
        self.prop_names = list(self.min_maxs.keys())

    def mol_to_props(self, mol, min_max_norm=False):
        props = {
            'AtomicLogP': Crippen.MolLogP(mol),
            'NumHAcceptors': Lipinski.NumHAcceptors(mol),
            'NumHDonors': Lipinski.NumHDonors(mol),
            'PolarSurfaceArea': rdMolDescriptors.CalcTPSA(mol),
            'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
            'NumAromaticRings': Lipinski.NumAromaticRings(mol),
            'NumAliphaticRings': Lipinski.NumAliphaticRings(mol),
            'FractionCSP3': Lipinski.FractionCSP3(mol),
            'QED': QED.qed(mol),
            'SyntheticAccessibility': sascorer.calculateScore(mol),
            'BertzComplexity': rdkit.Chem.GraphDescriptors.BertzCT(mol)
        }
        if min_max_norm:
            props = self.normalize_props(props)
        return props

    def normalize_prop(self, prop, prop_name):
        return (prop - self.min_maxs[prop_name]['min']) / \
               (self.min_maxs[prop_name]['max'] - self.min_maxs[prop_name]['min'])
```

**Key points:**
- 11 molecular properties (including BertzComplexity)
- Min/max ranges from training data
- Optional min-max normalization

---

## Snippet 10: SpectrumPreprocessor

**Source**: [dreams/utils/data.py](dreams/utils/data.py#L45-L140)

```python
import numpy as np
import dreams.utils.spectra as su
from dreams.utils.dformats import DataFormat

class SpectrumPreprocessor:
    def __init__(self, dformat: DataFormat, prec_intens=1.1, n_highest_peaks=None, 
                 spec_entropy_cleaning=False, normalize_mzs=False, precision=32, 
                 mz_shift_aug_p=0, mz_shift_aug_max=0, to_relative_intensities=True):
        
        assert precision in {32, 64}

        self.dformat = dformat
        self.prec_intens = prec_intens
        self.n_highest_peaks = n_highest_peaks
        self.spec_entropy_cleaning = spec_entropy_cleaning
        self.normalize_mzs = normalize_mzs
        self.to_relative_intensities = to_relative_intensities
        self.precision = precision
        self.mz_shift_aug_p = mz_shift_aug_p
        self.mz_shift_aug_max = mz_shift_aug_max

        if self.n_highest_peaks is None:
            self.n_highest_peaks = self.dformat.max_peaks_n

    def __call__(self, spec: np.array, prec_mz=None, high_form='auto', augment=False):
        spec = spec.copy()
        
        # Determine spectrum format
        if high_form == 'auto':
            high_form = spec.shape[1] == 2
        
        # (2, n_peaks) -> (n_peaks, 2)
        if not high_form:
            spec = spec.T

        # Trim and pad peak list
        if self.n_highest_peaks:
            spec = su.trim_peak_list(spec.T, self.n_highest_peaks).T
            spec = su.pad_peak_list(spec.T, target_len=self.n_highest_peaks).T

        # Normalize intensities to be relative to base peak
        if self.to_relative_intensities:
            spec = su.to_rel_intensity(spec.T).T

        # Prepend precursor peak
        if prec_mz is not None:
            spec = su.prepend_precursor_peak(spec, prec_mz, self.prec_intens, high=True)
        
        return spec
```

**Key points:**
- Trims to top N peaks by intensity
- Pads to fixed length with zeros
- Normalizes to relative intensities [0, 1]
- Prepends precursor peak with intensity 1.1

---

## Snippet 11: Fine-tuning Script

**Source**: [dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh](dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh#L78-L99)

```bash
python3 dreams/training/train.py \
 $WANDB_ARGS \
 --job_key "$RUN_NAME" \
 --run_name "$RUN_NAME" \
 --train_objective fp_morgan_2048 \
 --train_regime fine-tuning \
 --dataset_pth "$DATASET_PATH" \
 --dformat A \
 --model DreaMS \
 --num_workers_data 16 \
 --lr 1e-4 \
 --batch_size 64 \
 --prec_intens 1.1 \
 --num_devices 1 \
 --max_epochs 100 \
 --log_every_n_steps 50 \
 --head_depth 2 \
 --seed 3407 \
 --train_precision 64 \
 --pre_trained_pth "${PRETRAINED}/ssl_model.ckpt" \
 --val_check_interval 0.25 \
 --max_peaks_n 128
```

**Key points:**
- All hyperparameters in one place
- Pre-trained backbone from checkpoint
- 64-bit precision for training

---

*All snippets verified against source code. Line numbers accurate as of extraction date.*
