#!/usr/bin/env python3
"""
Complete retrieval evaluation analysis for DreaMS Morgan fingerprint predictions.
Standalone script that loads data, computes ranks, metrics, and generates visualizations.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import torch
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from rdkit import Chem
from rdkit.Chem import AllChem

# Project root
PROJECT_ROOT = Path('/Users/wouterachterberg/coding/DreaMS')
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
PROBING_TEST_PATH = PROJECT_ROOT / 'dreams-thesis-wa/data/processed/MassSpecGym_splits/probing_test.parquet'
FINETUNING_HDF5_PATH = PROJECT_ROOT / 'dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5'
OUTPUT_DIR = PROJECT_ROOT / 'dreams-thesis-wa/results/per_bit_analysis'
RETRIEVAL_DIR = OUTPUT_DIR / 'retrieval'
RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)

FP_SIZE = 2048
RADIUS = 2

print("="*80)
print("RETRIEVAL EVALUATION: Complete Analysis")
print("="*80)

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load data and predictions (from cache if available, else from notebook)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/5] Loading predictions and ground truth...")

# First, check if predictions are already computed and saved in notebook 
# (they should be if you've run cells 1-10)
y_pred_found = (OUTPUT_DIR / 'y_pred.npy').exists() and (OUTPUT_DIR / 'y_true.npy').exists()

if not y_pred_found:
    print("\n⚠️  Prediction cache not found in notebook output directory.")
    print("    Run the notebook cells 1-10 first to generate y_pred/y_true.")
    print("    This script will now check if SMILES files exist to load from original data...")
    sys.exit(1)

# Load OOD (probing_test) predictions - already computed by notebook
print("  Loading OOD (probing_test) predictions...")
y_pred_ood = np.load(OUTPUT_DIR / 'y_pred.npy')
y_true_ood = np.load(OUTPUT_DIR / 'y_true.npy')

# Load probing_test SMILES
df_test = pd.read_parquet(PROBING_TEST_PATH)
smiles_ood = df_test['smiles'].tolist()
print(f"  ✓ OOD: {len(smiles_ood):,} spectra, predictions shape {y_pred_ood.shape}")

# Now check if validation predictions exist - if not, we need to load from notebook kernel state
# For safety, check if they're saved with a known cache name
py_cache_val = OUTPUT_DIR / 'y_pred_val.npy'
py_cache_true_val = OUTPUT_DIR / 'y_true_val.npy'

if py_cache_val.exists() and py_cache_true_val.exists():
    print("  Loading validation (in-dist) predictions from cache...")
    y_pred_val = np.load(py_cache_val)
    y_true_val = np.load(py_cache_true_val)
    print(f"  ✓ Validation: loaded from cache")
else:
    print(f"\n  ⚠️  Cache files not found:")
    print(f"    - {py_cache_val}")
    print(f"    - {py_cache_true_val}")
    print(f"\n  Attempting to generate validation predictions from scratch...")
    
    # Load model and generate predictions
    from dreams.models.heads.heads import FingerprintHead
    from dreams.models.dreams.dreams import DreaMS
    import dreams.utils.data as du
    
    # Load checkpoint
    ckpt_path = Path('/Volumes/NVMe_Wouter/THESIS/snellius_output/MorganFingerprints/massspecgym_morgan2048_finetune_20260223_130317/epoch=53-step=7000.ckpt')
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model_config = checkpoint['hyper_parameters'].get('model_cfg', {})
    head_config = checkpoint['hyper_parameters'].get('head_config', {})
    
    # Build model
    model = DreaMS(**model_config)
    head = FingerprintHead(**head_config)
    model.head = head
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(DEVICE).eval()
    
    print("  ⚠️  Model generation from checkpoint requires spectrum preprocessing setup.")
    print("  This is complex - please run notebook cells 1-10 to generate and cache predictions.")
    raise FileNotFoundError(
        f"Validation prediction cache not found at {py_cache_val}.\n"
        f"Please run notebook cells 1-10 to generate validation predictions, "
        f"then the caching cell will save them for this script to use."
    )

# Load validation SMILES from HDF5
print("  Loading validation SMILES from HDF5...")
with h5py.File(FINETUNING_HDF5_PATH, 'r') as h5f:
    fold_array = h5f['fold'][()].astype(str)
    all_smiles = h5f['smiles'][()].astype(str)
    
    # Filter for validation split
    val_mask = fold_array == 'val'
    if val_mask.sum() == 0:
        val_mask = fold_array == 'test'
    
    smiles_val = all_smiles[val_mask]

print(f"  ✓ Validation: {len(smiles_val):,} spectra, predictions shape {y_pred_val.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Build fingerprint libraries
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/5] Building fingerprint libraries...")

def compute_morgan_fp(smi, fp_size=FP_SIZE, radius=RADIUS):
    """Compute binary Morgan fingerprint as numpy array."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(fp_size, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
    return np.array(fp, dtype=np.float32)

def build_fp_library(smiles_list):
    """Build fingerprint library for all unique molecules."""
    unique_smiles = list(set(smiles_list))
    smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}
    
    fp_matrix = np.zeros((len(unique_smiles), FP_SIZE), dtype=np.float32)
    for i, smi in enumerate(tqdm(unique_smiles, desc="Computing Morgan FPs", leave=False)):
        fp_matrix[i] = compute_morgan_fp(smi)
    
    spec_to_mol = np.array([smi_to_idx[s] for s in smiles_list], dtype=np.int32)
    return fp_matrix, spec_to_mol, unique_smiles

fp_library_val, spec_to_mol_val, unique_smi_val = build_fp_library(smiles_val)
fp_library_ood, spec_to_mol_ood, unique_smi_ood = build_fp_library(smiles_ood)

print(f"  ✓ Validation library: {len(unique_smi_val)} unique molecules")
print(f"  ✓ OOD library: {len(unique_smi_ood)} unique molecules")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Compute retrieval ranks
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3/5] Computing retrieval ranks via cosine similarity...")

def compute_retrieval_ranks(y_pred, fp_library, spec_to_mol, batch_size=500):
    """For each spectrum, rank molecules by cosine sim to predicted FP."""
    n_spectra = len(y_pred)
    ranks = np.zeros(n_spectra, dtype=np.int32)
    
    for start in tqdm(range(0, n_spectra, batch_size), desc="Computing ranks", leave=False):
        end = min(start + batch_size, n_spectra)
        sims = cosine_similarity(y_pred[start:end], fp_library)
        
        for i, row_idx in enumerate(range(start, end)):
            correct_mol = spec_to_mol[row_idx]
            score = sims[i, correct_mol]
            rank = (sims[i] > score).sum() + 1  # 1-indexed
            ranks[row_idx] = rank
    
    return ranks

ranks_val = compute_retrieval_ranks(y_pred_val, fp_library_val, spec_to_mol_val)
ranks_ood = compute_retrieval_ranks(y_pred_ood, fp_library_ood, spec_to_mol_ood)

# Save
np.save(RETRIEVAL_DIR / 'ranks_val.npy', ranks_val)
np.save(RETRIEVAL_DIR / 'ranks_ood.npy', ranks_ood)
print(f"  ✓ Saved ranks to {RETRIEVAL_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Compute retrieval metrics
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4/5] Computing retrieval metrics...")

def compute_retrieval_metrics(ranks, library_size):
    """Compute accuracy@k, MRR, and other metrics."""
    metrics = {}
    for k in [1, 5, 10, 20, 50, 100]:
        metrics[f'acc@{k}'] = float((ranks <= k).mean())
    metrics['mrr'] = float((1.0 / ranks).mean())
    metrics['mean_rank'] = float(ranks.mean())
    metrics['median_rank'] = float(np.median(ranks))
    metrics['min_rank'] = int(ranks.min())
    metrics['max_rank'] = int(ranks.max())
    metrics['std_rank'] = float(ranks.std())
    metrics['library_size'] = library_size
    metrics['n_spectra'] = len(ranks)
    metrics['random_acc@1'] = 1.0 / library_size
    metrics['pct_better_than_random'] = (ranks == 1).sum() / len(ranks) * 100
    return metrics

metrics_val = compute_retrieval_metrics(ranks_val, len(fp_library_val))
metrics_ood = compute_retrieval_metrics(ranks_ood, len(fp_library_ood))

print(f"\n  Validation:")
print(f"    Acc@1: {metrics_val['acc@1']:.4f}")
print(f"    Acc@10: {metrics_val['acc@10']:.4f}")
print(f"    MRR: {metrics_val['mrr']:.4f}")
print(f"    Library size: {metrics_val['library_size']:,}")

print(f"\n  OOD:")
print(f"    Acc@1: {metrics_ood['acc@1']:.4f}")
print(f"    Acc@10: {metrics_ood['acc@10']:.4f}")
print(f"    MRR: {metrics_ood['mrr']:.4f}")
print(f"    Library size: {metrics_ood['library_size']:,}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Generate visualizations and tables
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5/5] Generating visualizations and tables...")

# Rank distribution histograms
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
ax.hist(ranks_val, bins=np.logspace(0, np.log10(metrics_val['max_rank']), 50),
        color='tab:blue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xscale('log')
ax.set_xlabel('Rank (log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Validation: Rank Distribution', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
for k in [1, 5, 10, 20]:
    ax.axvline(k, color='darkblue', linestyle='--', linewidth=1.5, alpha=0.6)

ax = axes[1]
ax.hist(ranks_ood, bins=np.logspace(0, np.log10(metrics_ood['max_rank']), 50),
        color='tab:orange', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xscale('log')
ax.set_xlabel('Rank (log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('OOD: Rank Distribution', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
for k in [1, 5, 10, 20]:
    ax.axvline(k, color='darkorange', linestyle='--', linewidth=1.5, alpha=0.6)

plt.tight_layout()
plt.savefig(RETRIEVAL_DIR / 'rank_distribution.png', dpi=200, bbox_inches='tight')
plt.savefig(RETRIEVAL_DIR / 'rank_distribution.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved rank distribution plots")

# Accuracy@k curve
def compute_accuracy_at_k(ranks, max_k=None):
    if max_k is None:
        max_k = int(np.percentile(ranks, 99))
    ks = np.arange(1, max_k + 1)
    accs = np.array([(ranks <= k).mean() for k in ks])
    return ks, accs

max_k_val = int(min(200, int(np.percentile(ranks_val, 99.5))))
max_k_ood = int(min(200, int(np.percentile(ranks_ood, 99.5))))
max_k = max(max_k_val, max_k_ood)

ks_val, accs_val = compute_accuracy_at_k(ranks_val, max_k)
ks_ood, accs_ood = compute_accuracy_at_k(ranks_ood, max_k)

fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(ks_val, accs_val, 'o-', linewidth=2.5, markersize=5,
        color='tab:blue', label=f'Validation (n={len(ranks_val):,})', alpha=0.8)
ax.plot(ks_ood, accs_ood, 's-', linewidth=2.5, markersize=5,
        color='tab:orange', label=f'OOD (n={len(ranks_ood):,})', alpha=0.8)

for k in [1, 5, 10, 20, 50]:
    if k <= max_k:
        ax.axvline(k, color='grey', linestyle=':', linewidth=1.2, alpha=0.4)
        ax.text(k, 0.02, f'k={k}', rotation=0, fontsize=9, alpha=0.6)

random_baseline_val = metrics_val['random_acc@1']
ax.axhline(random_baseline_val, color='darkblue', linestyle=':', linewidth=1.5, alpha=0.4,
           label=f'Random (val, {random_baseline_val:.4f})')

random_baseline_ood = metrics_ood['random_acc@1']
ax.axhline(random_baseline_ood, color='darkorange', linestyle=':', linewidth=1.5, alpha=0.4,
           label=f'Random (OOD, {random_baseline_ood:.4f})')

ax.set_xlabel('Rank cutoff (k)', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy@k (fraction with rank ≤ k)', fontsize=13, fontweight='bold')
ax.set_title('Retrieval Performance: Accuracy@k Curve\n(higher is better)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_k)
ax.set_ylim(-0.02, 1.05)

plt.tight_layout()
plt.savefig(RETRIEVAL_DIR / 'accuracy_at_k_curve.png', dpi=200, bbox_inches='tight')
plt.savefig(RETRIEVAL_DIR / 'accuracy_at_k_curve.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved accuracy@k curve")

# Comparison table
comparison_data = [
    {'Metric': 'Library Size', 'Validation': f"{metrics_val['library_size']:,}", 'OOD': f"{metrics_ood['library_size']:,}"},
    {'Metric': 'Spectra (n)', 'Validation': f"{metrics_val['n_spectra']:,}", 'OOD': f"{metrics_ood['n_spectra']:,}"},
    {'Metric': '', 'Validation': '', 'OOD': ''},
    {'Metric': 'Accuracy@1', 'Validation': f"{metrics_val['acc@1']:.4f}", 'OOD': f"{metrics_ood['acc@1']:.4f}"},
    {'Metric': 'Accuracy@5', 'Validation': f"{metrics_val['acc@5']:.4f}", 'OOD': f"{metrics_ood['acc@5']:.4f}"},
    {'Metric': 'Accuracy@10', 'Validation': f"{metrics_val['acc@10']:.4f}", 'OOD': f"{metrics_ood['acc@10']:.4f}"},
    {'Metric': 'Accuracy@20', 'Validation': f"{metrics_val['acc@20']:.4f}", 'OOD': f"{metrics_ood['acc@20']:.4f}"},
    {'Metric': 'Accuracy@50', 'Validation': f"{metrics_val['acc@50']:.4f}", 'OOD': f"{metrics_ood['acc@50']:.4f}"},
    {'Metric': 'Accuracy@100', 'Validation': f"{metrics_val['acc@100']:.4f}", 'OOD': f"{metrics_ood['acc@100']:.4f}"},
    {'Metric': '', 'Validation': '', 'OOD': ''},
    {'Metric': 'Mean Reciprocal Rank', 'Validation': f"{metrics_val['mrr']:.4f}", 'OOD': f"{metrics_ood['mrr']:.4f}"},
    {'Metric': 'Mean Rank', 'Validation': f"{metrics_val['mean_rank']:.1f}", 'OOD': f"{metrics_ood['mean_rank']:.1f}"},
    {'Metric': 'Median Rank', 'Validation': f"{metrics_val['median_rank']:.0f}", 'OOD': f"{metrics_ood['median_rank']:.0f}"},
    {'Metric': '', 'Validation': '', 'OOD': ''},
    {'Metric': 'Random Baseline (Acc@1)', 'Validation': f"{metrics_val['random_acc@1']:.2e}", 'OOD': f"{metrics_ood['random_acc@1']:.2e}"},
    {'Metric': '% Better than Random', 'Validation': f"{metrics_val['pct_better_than_random']:.1f}%", 'OOD': f"{metrics_ood['pct_better_than_random']:.1f}%"},
]

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(RETRIEVAL_DIR / 'retrieval_comparison_table.csv', index=False)

print("\n" + "="*90)
print("RETRIEVAL EVALUATION: COMPREHENSIVE METRICS COMPARISON")
print("="*90)
print(comparison_df.to_string(index=False))
print("="*90)

# Key insights
print("\nKEY INSIGHTS FROM RETRIEVAL EVALUATION:")
print("─"*90)

if metrics_val['acc@1'] > 0:
    acc_drop_1 = (metrics_val['acc@1'] - metrics_ood['acc@1']) / metrics_val['acc@1'] * 100
    print(f"• Accuracy@1 (exact match): {metrics_val['acc@1']:.4f} (val) → {metrics_ood['acc@1']:.4f} (OOD) = {acc_drop_1:.1f}% drop")

if metrics_val['acc@5'] > 0:
    acc_drop_5 = (metrics_val['acc@5'] - metrics_ood['acc@5']) / metrics_val['acc@5'] * 100
    print(f"• Accuracy@5 (top-5 match): {metrics_val['acc@5']:.4f} (val) → {metrics_ood['acc@5']:.4f} (OOD) = {acc_drop_5:.1f}% drop")

if metrics_val['mrr'] > 0:
    mrr_drop = (metrics_val['mrr'] - metrics_ood['mrr']) / metrics_val['mrr'] * 100
    print(f"• Mean Reciprocal Rank: {metrics_val['mrr']:.4f} (val) → {metrics_ood['mrr']:.4f} (OOD) = {mrr_drop:.1f}% drop")

print(f"• Median rank: {metrics_val['median_rank']:.0f} (val) → {metrics_ood['median_rank']:.0f} (OOD)")
print(f"\n• {metrics_val['pct_better_than_random']:.1f}% of validation spectra beat random baseline")
print(f"• {metrics_ood['pct_better_than_random']:.1f}% of OOD spectra beat random baseline")
print(f"• Random baseline: 1 in {metrics_val['library_size']:,} (val) vs 1 in {metrics_ood['library_size']:,} (OOD)")

print("\n✓ Retrieval evaluation complete!")
print(f"✓ All outputs saved to: {RETRIEVAL_DIR}")
