"""
Scaffold-based splitting utilities to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple
from pathlib import Path


def create_scaffold_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test splits ensuring no scaffold appears in multiple splits.
    
    This prevents data leakage: molecules with the same scaffold (core structure)
    should not appear in both training and test sets.
    
    Args:
        df: DataFrame with 'scaffold_id' column
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
        
    Returns:
        train_df, val_df, test_df
    """
    # First split: train+val vs test
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss.split(df, groups=df['scaffold_id']))
    
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)  # Adjust for already split test
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['scaffold_id']))
    
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()
    
    # Add fold labels
    train_df['fold'] = 'train'
    val_df['fold'] = 'val'
    test_df['fold'] = 'test'
    
    # Verify no scaffold leakage
    train_scaffolds = set(train_df['scaffold_id'].dropna())
    val_scaffolds = set(val_df['scaffold_id'].dropna())
    test_scaffolds = set(test_df['scaffold_id'].dropna())
    
    assert len(train_scaffolds & val_scaffolds) == 0, "Scaffold leakage: train-val"
    assert len(train_scaffolds & test_scaffolds) == 0, "Scaffold leakage: train-test"
    assert len(val_scaffolds & test_scaffolds) == 0, "Scaffold leakage: val-test"
    
    print(f"✅ Scaffold-based splits created:")
    print(f"  Train: {len(train_df):,} samples, {len(train_scaffolds):,} scaffolds")
    print(f"  Val:   {len(val_df):,} samples, {len(val_scaffolds):,} scaffolds")
    print(f"  Test:  {len(test_df):,} samples, {len(test_scaffolds):,} scaffolds")
    print(f"  No scaffold leakage detected ✓")
    
    return train_df, val_df, test_df


def verify_no_leakage(df: pd.DataFrame, fold_column: str = 'fold') -> dict:
    """
    Verify that scaffolds don't leak across folds.
    
    Args:
        df: DataFrame with scaffold_id and fold columns
        fold_column: Name of the fold column
        
    Returns:
        Dictionary with leakage statistics
    """
    folds = df[fold_column].unique()
    leakage_stats = {}
    
    for fold1 in folds:
        for fold2 in folds:
            if fold1 < fold2:  # Avoid duplicate comparisons
                scaffolds1 = set(df[df[fold_column] == fold1]['scaffold_id'].dropna())
                scaffolds2 = set(df[df[fold_column] == fold2]['scaffold_id'].dropna())
                overlap = scaffolds1 & scaffolds2
                leakage_stats[f"{fold1}_vs_{fold2}"] = len(overlap)
    
    return leakage_stats


def fix_massspecgym_splits(input_path: str, output_path: str):
    """
    Fix MassSpecGym splits to be scaffold-based.
    
    Args:
        input_path: Path to enriched MassSpecGym TSV
        output_path: Where to save scaffold-based splits
    """
    print("Loading MassSpecGym dataset...")
    df = pd.read_csv(input_path, sep='\t')
    
    print(f"Original dataset: {len(df):,} samples")
    
    # Remove old fold column if it exists
    if 'fold' in df.columns:
        df = df.drop(columns=['fold'])
    
    # Create new scaffold-based splits
    train_df, val_df, test_df = create_scaffold_splits(df, test_size=0.15, val_size=0.08)
    
    # Combine back
    df_fixed = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Verify
    leakage = verify_no_leakage(df_fixed)
    print(f"\nLeakage check: {leakage}")
    
    # Save
    df_fixed.to_csv(output_path, sep='\t', index=False)
    print(f"\n✅ Saved scaffold-based splits to: {output_path}")
    
    return df_fixed


if __name__ == "__main__":
    # Get the dreams-thesis-wa directory (parent of src/)
    THESIS_DIR = Path(__file__).parent.parent
    
    # Fix the MassSpecGym splits
    fix_massspecgym_splits(
        input_path=str(THESIS_DIR / "data/processed/MassSpecGym_enriched.tsv"),
        output_path=str(THESIS_DIR / "data/processed/MassSpecGym_scaffold_splits.tsv")
    )
