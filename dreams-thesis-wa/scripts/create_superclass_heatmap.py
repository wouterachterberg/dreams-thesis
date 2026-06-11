import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from pathlib import Path

"""Generate the thesis superclass heatmap with thesis evaluation splits."""

THESIS_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = THESIS_ROOT / "data" / "processed" / "MassSpecGym_splits"
FIG_DIR = THESIS_ROOT / "results" / "shared" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load superclass mapping
smiles_superclass = pd.read_csv(DATA_DIR / 'smiles_with_superclass.csv')

def decode_utf8(values):
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def load_finetuning_smiles():
    with h5py.File(DATA_DIR / "finetuning.hdf5", "r") as hf:
        smiles = decode_utf8(hf["smiles"][:])
        folds = decode_utf8(hf["fold"][:])
    df = pd.DataFrame({"smiles": smiles, "fold": folds})
    return {
        "Train": set(df.loc[df["fold"] == "train", "smiles"].unique()),
        "ID-Val": set(df.loc[df["fold"] == "val", "smiles"].unique()),
    }


source_split_smiles = load_finetuning_smiles()
source_split_smiles["OOD"] = set(pd.read_parquet(DATA_DIR / "probing_test.parquet")["smiles"].astype(str).unique())
source_split_smiles["Holdout"] = set(pd.read_parquet(DATA_DIR / "holdout.parquet")["smiles"].astype(str).unique())

source_expected_counts = {"Train": 18251, "ID-Val": 3220, "OOD": 6147, "Holdout": 3984}
source_observed_counts = {name: len(smiles) for name, smiles in source_split_smiles.items()}
assert source_observed_counts == source_expected_counts, f"Unexpected source split counts: {source_observed_counts}"

split_smiles = {
    "Train + ID-Val": source_split_smiles["Train"] | source_split_smiles["ID-Val"],
    "OOD": source_split_smiles["OOD"],
    "Holdout": source_split_smiles["Holdout"],
}

# Verify no molecule-level overlap across displayed splits.
for left_name, left_smiles in split_smiles.items():
    for right_name, right_smiles in split_smiles.items():
        if left_name >= right_name:
            continue
        overlap = left_smiles & right_smiles
        assert not overlap, f"{left_name} and {right_name} overlap by {len(overlap)} molecules"

split_order = ["Train + ID-Val", "OOD", "Holdout"]
expected_counts = {"Train + ID-Val": 21471, "OOD": 6147, "Holdout": 3984}
observed_counts = {name: len(smiles) for name, smiles in split_smiles.items()}
assert observed_counts == expected_counts, f"Unexpected split counts: {observed_counts}"

total_unique = len(set().union(*split_smiles.values()))
print(
    "Splits: "
    + ", ".join(f"{name}: {count:,}" for name, count in observed_counts.items())
)
print(f"Total unique (no overlap): {total_unique:,}")

# Get superclass counts for each split
def get_superclass_counts(smiles_set):
    df = pd.DataFrame({'smiles': list(smiles_set)})
    merged = df.merge(smiles_superclass, on='smiles', how='left')
    return merged['superclass'].value_counts()

split_counts = {
    split_name: get_superclass_counts(smiles_set)
    for split_name, smiles_set in split_smiles.items()
}

# Combine into DataFrame
all_superclasses = set().union(*(set(counts.index) for counts in split_counts.values()))
df = pd.DataFrame(index=sorted(all_superclasses))
for split_name, counts in split_counts.items():
    df[split_name] = counts
df = df.fillna(0).astype(int)
df['Total'] = df.sum(axis=1)
df = df.sort_values('Total', ascending=False)

# Save updated CSV with correct splits
for split_name in split_smiles:
    df[f"{split_name} %"] = (df[split_name] / df[split_name].sum() * 100).round(2)
df['Total %'] = (df['Total'] / df['Total'].sum() * 100).round(2)
df.to_csv(DATA_DIR / 'superclass_distribution_full.csv')
print(f"Saved: {DATA_DIR / 'superclass_distribution_full.csv'}")

# Take top 15 superclasses for readability
top_n = 15
top = df.head(top_n).copy()

# Create a matrix for the heatmap (using raw counts)
heatmap_data = top[split_order].copy()
heatmap_data.columns = [
    f"{split_name}\n(n={observed_counts[split_name]:,})"
    for split_name in split_order
]

# Create figure
fig, ax = plt.subplots(figsize=(10.5, 9))

# Create heatmap with log scale for better color distribution
# Handle zeros by replacing with 0.5 for log scale
heatmap_values = heatmap_data.values.astype(float)
heatmap_values[heatmap_values == 0] = 0.5  # Small value for log scale

im = ax.imshow(heatmap_values, cmap='Blues', aspect='auto', 
               norm=LogNorm(vmin=0.5, vmax=heatmap_data.values.max()))

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label('Number of molecules (log scale)', fontsize=10)

# Set ticks
ax.set_xticks(range(len(split_order)))
ax.set_xticklabels(heatmap_data.columns, fontsize=11, fontweight='bold')
ax.set_yticks(range(len(heatmap_data)))
ax.set_yticklabels(heatmap_data.index, fontsize=10)

# Add text annotations with counts
for i in range(len(heatmap_data)):
    for j in range(len(split_order)):
        val = int(heatmap_data.iloc[i, j])
        # Use white text for dark cells, black for light
        color = 'white' if val > 300 else 'black'
        ax.text(j, i, f'{val:,}', ha='center', va='center', 
                fontsize=9, color=color, fontweight='bold')

# Add note about Other classes
other_train = df.iloc[top_n:]['Train + ID-Val'].sum()
other_ood = df.iloc[top_n:]['OOD'].sum()
other_holdout = df.iloc[top_n:]['Holdout'].sum()
fig.text(0.5, 0.02, 
         f'Note: {len(df)-top_n} additional classes not shown '
         f'(Train + ID-Val: {other_train:,}, OOD: {other_ood:,}, Holdout: {other_holdout:,})',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)

# Save
plt.savefig(FIG_DIR / 'superclass_heatmap.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f'Saved: {FIG_DIR / "superclass_heatmap.pdf"}')
