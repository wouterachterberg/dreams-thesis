#!/usr/bin/env python3
"""
Quick fix: Flip the sign of Spearman correlations in nn_descriptor_consistency.csv
This corrects the sign error without re-running the full TASK 2 analysis.
"""

import pandas as pd
from pathlib import Path

# Load the CSV
results_dir = Path('dreams-thesis-wa/results/indicators')
csv_path = results_dir / 'nn_descriptor_consistency.csv'

print(f"Loading: {csv_path}")
df = pd.read_csv(csv_path)

print(f"Shape: {df.shape}")
print(f"\nBefore fix:")
print(f"  Spearman range: [{df['spearman_corr'].min():.4f}, {df['spearman_corr'].max():.4f}]")
print(f"  Spearman mean: {df['spearman_corr'].mean():.4f}")

# Flip the sign
df['spearman_corr'] = -df['spearman_corr']

print(f"\nAfter fix:")
print(f"  Spearman range: [{df['spearman_corr'].min():.4f}, {df['spearman_corr'].max():.4f}]")
print(f"  Spearman mean: {df['spearman_corr'].mean():.4f}")

# Save back
df.to_csv(csv_path, index=False)
print(f"\n✅ Updated: {csv_path}")
print(f"   Effect ratios unchanged, Spearman signs flipped")
