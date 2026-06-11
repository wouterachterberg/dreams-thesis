"""Create thesis-ready superclass distribution table across all splits.

The MassSpecGym dataset uses scaffold-based splitting:
- Train: 80% of scaffolds (used for pre-training/fine-tuning)
- Validation (Probing Test): 20% of scaffolds (used for evaluation)
- Holdout: Separate held-out set for final evaluation

Scaffold splitting ensures molecules in test sets have different core 
structures than training molecules, testing true generalization.
"""

import pandas as pd
import pyarrow.parquet as pq
import h5py
from pathlib import Path

DATA_DIR = Path('/Users/wouterachterberg/coding/DreaMS/dreams-thesis-wa/data/processed/MassSpecGym_splits')

# Load superclass mapping
smiles_superclass = pd.read_csv(DATA_DIR / 'smiles_with_superclass.csv')

# Load full dataset for train/val split
with h5py.File(DATA_DIR / 'full.hdf5', 'r') as hf:
    smiles_arr = hf['smiles'][:]
    folds_arr = hf['fold'][:]
    smiles = [s.decode() if isinstance(s, bytes) else s for s in smiles_arr]
    folds = [f.decode() if isinstance(f, bytes) else f for f in folds_arr]

full_df = pd.DataFrame({'smiles': smiles, 'fold': folds})

# Get unique SMILES per fold from full dataset (train/val splits based on scaffolds)
train_smiles = set(full_df[full_df['fold'] == 'train']['smiles'].unique())
val_smiles = set(full_df[full_df['fold'] == 'val']['smiles'].unique())

# The original train/val split is what we report (no overlap by scaffold design)
# Holdout is a SUBSET of these for final evaluation (separate concern)

# Create dataframes for each split
def get_superclass_counts(smiles_set, name):
    df = pd.DataFrame({'smiles': list(smiles_set)})
    merged = df.merge(smiles_superclass, on='smiles', how='left')
    counts = merged['superclass'].value_counts()
    return counts.rename(name)

# Get counts for train/val (the main scaffold-split)
train_counts = get_superclass_counts(train_smiles, 'Train')
val_counts = get_superclass_counts(val_smiles, 'Validation')

# Combine into single table (train + val = all unique molecules)
all_superclasses = set(train_counts.index) | set(val_counts.index)
result = pd.DataFrame(index=sorted(all_superclasses))
result['Train'] = train_counts
result['Validation'] = val_counts
result = result.fillna(0).astype(int)
result['Total'] = result.sum(axis=1)

# Sort by total descending
result = result.sort_values('Total', ascending=False)

# Add percentages
total_train = result['Train'].sum()
total_val = result['Validation'].sum()
total_all = result['Total'].sum()

result['Train %'] = (result['Train'] / total_train * 100).round(1)
result['Val %'] = (result['Validation'] / total_val * 100).round(1)

# Print summary stats
print("=" * 80)
print("DATASET SPLIT SUMMARY (MassSpecGym with Scaffold Splitting)")
print("=" * 80)
print(f"\nSpectra counts:")
print(f"  Total:           231,104")
print(f"  Train:           185,919 (80.4%)")
print(f"  Validation:       45,185 (19.6%)")
print()
print(f"Unique molecules:")
print(f"  Total:           {total_all:,}")
print(f"  Train:           {total_train:,} ({100*total_train/total_all:.1f}%)")
print(f"  Validation:      {total_val:,} ({100*total_val/total_all:.1f}%)")
print()
print(f"Superclass coverage:")
print(f"  Total classes:   {len(all_superclasses)}")
print(f"  In Train:        {(result['Train'] > 0).sum()}")
print(f"  In Validation:   {(result['Validation'] > 0).sum()}")

# Create clean table for thesis (top 15 + others)
print("\n" + "=" * 80)
print("TOP 15 SUPERCLASSES BY MOLECULE COUNT")
print("=" * 80)

top_n = 15
top = result.head(top_n)[['Train', 'Validation', 'Total']].copy()
others = result.iloc[top_n:][['Train', 'Validation', 'Total']].sum()
n_other = len(result) - top_n

# Print as nice table
print(f"\n{'Superclass':<40} {'Train':>8} {'Val':>8} {'Total':>8}")
print("-" * 70)
for idx, row in top.iterrows():
    print(f"{idx:<40} {row['Train']:>8,} {row['Validation']:>8,} {row['Total']:>8,}")
print(f"{'Other classes (' + str(n_other) + ')':<40} {int(others['Train']):>8,} {int(others['Validation']):>8,} {int(others['Total']):>8,}")
print("-" * 70)
print(f"{'Total':<40} {total_train:>8,} {total_val:>8,} {total_all:>8,}")

# Save LaTeX table
latex_table = r"""\begin{table}[htbp]
\centering
\caption{Distribution of molecular superclasses across scaffold-based data splits. Classification was performed using NPClassifier \citep{npclassifier}. Molecules not classifiable as natural products (e.g., synthetic compounds) are grouped as ``Other/Synthetic''.}
\label{tab:superclass_distribution}
\begin{tabular}{lrrr}
\toprule
\textbf{Superclass} & \textbf{Train} & \textbf{Validation} & \textbf{Total} \\
\midrule
"""

for idx, row in top.iterrows():
    name = idx.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
    latex_table += f"{name} & {row['Train']:,} & {row['Validation']:,} & {row['Total']:,} \\\\\n"

latex_table += f"Other classes ({n_other}) & {int(others['Train']):,} & {int(others['Validation']):,} & {int(others['Total']):,} \\\\\n"
latex_table += r"""\midrule
\textbf{Total} & \textbf{""" + f"{total_train:,}" + r"""} & \textbf{""" + f"{total_val:,}" + r"""} & \textbf{""" + f"{total_all:,}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""

# Save
with open(DATA_DIR / 'superclass_distribution_thesis.tex', 'w') as f:
    f.write(latex_table)

# Also save full CSV
result.to_csv(DATA_DIR / 'superclass_by_split_full.csv')

print(f"\n\nSaved:")
print(f"  - {DATA_DIR / 'superclass_distribution_thesis.tex'}")
print(f"  - {DATA_DIR / 'superclass_by_split_full.csv'}")

# Print the LaTeX table
print("\n" + "=" * 80)
print("LATEX TABLE")
print("=" * 80)
print(latex_table)
