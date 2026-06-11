"""
Create Murcko histogram-based splits following the DreaMS tutorial approach.

This is more rigorous than simple scaffold splits because it prevents not just
identical scaffolds, but also structurally similar scaffolds from appearing
in different splits.

Reference: https://dreams-docs.readthedocs.io/en/latest/tutorials/murcko_hist_split.html
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm
from dreams.algorithms.murcko_hist import murcko_hist, are_sub_hists
from dreams.definitions import SMILES, FOLD

tqdm.pandas()


def create_murcko_histogram_splits(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    k: int = 3,
    d: int = 4,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create train/val/test splits based on Murcko histograms.
    
    This prevents data leakage by ensuring molecules with similar scaffolds
    don't appear in different splits.
    
    Args:
        df: DataFrame with SMILES column
        val_frac: Fraction for validation set (default 0.15)
        test_frac: Fraction for test set (default 0.15)
        k: Parameter for are_sub_hists (default 3)
        d: Parameter for are_sub_hists (default 4)
        random_state: Random seed
        
    Returns:
        DataFrame with added 'fold' column
    """
    np.random.seed(random_state)
    
    print("="*80)
    print("CREATING MURCKO HISTOGRAM-BASED SPLITS")
    print("="*80)
    
    # Step 1: Compute Murcko histograms for unique SMILES
    print("\n1. Computing Murcko histograms...")
    df_unique = df.drop_duplicates(subset=[SMILES]).copy()
    
    df_unique['MurckoHist'] = df_unique[SMILES].progress_apply(
        lambda x: murcko_hist(Chem.MolFromSmiles(x))
    )
    
    # Convert to string for grouping
    df_unique['MurckoHistStr'] = df_unique['MurckoHist'].astype(str)
    
    print(f"   Total unique SMILES: {len(df_unique):,}")
    print(f"   Unique Murcko histograms: {df_unique['MurckoHistStr'].nunique()}")
    
    # Step 2: Group by Murcko histogram
    print("\n2. Grouping by Murcko histogram...")
    df_gb = df_unique.groupby('MurckoHistStr').agg(
        count=(SMILES, 'count'),
        smiles_list=(SMILES, list)
    ).reset_index()
    
    df_gb['MurckoHist'] = df_gb['MurckoHistStr'].apply(eval)
    df_gb = df_gb.sort_values('count', ascending=False).reset_index(drop=True)
    
    print(f"   Created {len(df_gb)} histogram groups")
    print(f"\n   Top 10 most common Murcko histograms:")
    for i in range(min(10, len(df_gb))):
        print(f"     {i+1}. {df_gb.iloc[i]['MurckoHistStr']}: {df_gb.iloc[i]['count']} molecules")
    
    # Step 3: Split into test set first
    print(f"\n3. Creating test set (~{test_frac*100:.0f}% of molecules)...")
    median_i = len(df_gb) // 2
    cum_test_mols = 0
    test_idx, remaining_idx = [], []
    
    for i in range(median_i, -1, -1):
        current_hist = df_gb.iloc[i]['MurckoHist']
        is_test_subhist = any(
            are_sub_hists(current_hist, df_gb.iloc[j]['MurckoHist'], k=k, d=d)
            for j in test_idx
        )
        
        if is_test_subhist:
            remaining_idx.append(i)
        else:
            if cum_test_mols / len(df_unique) <= test_frac:
                cum_test_mols += df_gb.iloc[i]['count']
                test_idx.append(i)
            else:
                remaining_idx.append(i)
    
    # Add remaining indices to remaining set
    remaining_idx.extend(range(median_i + 1, len(df_gb)))
    
    print(f"   Test set: {cum_test_mols} molecules ({cum_test_mols/len(df_unique)*100:.1f}%)")
    print(f"   Test histogram groups: {len(test_idx)}")
    
    # Step 4: Split remaining into train/val
    print(f"\n4. Creating validation set from remaining (~{val_frac*100:.0f}% of total molecules)...")
    
    # Calculate how many molecules we need for validation
    target_val_mols = int(len(df_unique) * val_frac)
    cum_val_mols = 0
    val_idx, train_idx = [], []
    
    # Create a mapping of histogram group index to whether it's in test
    test_idx_set = set(test_idx)
    
    # Shuffle remaining indices for more randomness
    np.random.shuffle(remaining_idx)
    
    for i in remaining_idx:
        current_hist = df_gb.iloc[i]['MurckoHist']
        
        # Check if similar to any validation histograms
        is_val_subhist = any(
            are_sub_hists(current_hist, df_gb.iloc[j]['MurckoHist'], k=k, d=d)
            for j in val_idx
        )
        
        # Check if similar to any test histograms (should not be, but double-check)
        is_test_subhist = any(
            are_sub_hists(current_hist, df_gb.iloc[j]['MurckoHist'], k=k, d=d)
            for j in test_idx
        )
        
        if is_test_subhist:
            # This shouldn't happen, but if it does, put in train
            train_idx.append(i)
        elif is_val_subhist:
            train_idx.append(i)
        else:
            if cum_val_mols < target_val_mols:
                cum_val_mols += df_gb.iloc[i]['count']
                val_idx.append(i)
            else:
                train_idx.append(i)
    
    print(f"   Validation set: {cum_val_mols} molecules ({cum_val_mols/len(df_unique)*100:.1f}%)")
    print(f"   Validation histogram groups: {len(val_idx)}")
    print(f"   Training histogram groups: {len(train_idx)}")
    
    # Verify all indices are assigned
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df_gb), \
        "Not all histogram groups were assigned!"
    
    # Step 5: Map SMILES to their assigned fold
    print("\n5. Mapping SMILES to folds...")
    smiles_to_fold = {}
    
    for i, row in df_gb.iterrows():
        if i in test_idx:
            fold = 'test'
        elif i in val_idx:
            fold = 'val'
        else:
            fold = 'train'
        
        for smiles in row['smiles_list']:
            smiles_to_fold[smiles] = fold
    
    # Assign folds to all spectra (not just unique SMILES)
    df[FOLD] = df[SMILES].map(smiles_to_fold)
    
    # Step 6: Verify and report
    print("\n6. Final split statistics:")
    print("="*80)
    
    print("\nDistribution of spectra:")
    fold_counts = df[FOLD].value_counts()
    for fold in ['train', 'val', 'test']:
        count = fold_counts.get(fold, 0)
        pct = count / len(df) * 100
        print(f"  {fold:5s}: {count:6,} spectra ({pct:5.2f}%)")
    
    print("\nDistribution of unique SMILES:")
    df_unique_final = df.drop_duplicates(subset=[SMILES])
    fold_counts_unique = df_unique_final[FOLD].value_counts()
    for fold in ['train', 'val', 'test']:
        count = fold_counts_unique.get(fold, 0)
        pct = count / len(df_unique_final) * 100
        print(f"  {fold:5s}: {count:6,} molecules ({pct:5.2f}%)")
    
    # Verify no missing folds
    if df[FOLD].isna().any():
        n_missing = df[FOLD].isna().sum()
        print(f"\n⚠️  WARNING: {n_missing} spectra have no fold assignment!")
    else:
        print("\n✅ All spectra assigned to folds")
    
    print("="*80)
    
    return df


def verify_no_histogram_leakage(df: pd.DataFrame, k: int = 3, d: int = 4) -> dict:
    """
    Verify that similar Murcko histograms don't leak across folds.
    
    Args:
        df: DataFrame with SMILES and fold columns
        k: Parameter for are_sub_hists
        d: Parameter for are_sub_hists
        
    Returns:
        Dictionary with leakage statistics
    """
    print("\n" + "="*80)
    print("VERIFYING MURCKO HISTOGRAM LEAKAGE")
    print("="*80)
    
    # Get unique SMILES per fold
    df_unique = df.drop_duplicates(subset=[SMILES])
    
    fold_histograms = {}
    for fold in ['train', 'val', 'test']:
        df_fold = df_unique[df_unique[FOLD] == fold]
        print(f"\nComputing histograms for {fold} set ({len(df_fold)} molecules)...")
        fold_histograms[fold] = df_fold[SMILES].progress_apply(
            lambda x: murcko_hist(Chem.MolFromSmiles(x))
        ).tolist()
    
    # Check for similar histograms across folds
    leakage_stats = {}
    
    for fold1, fold2 in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        print(f"\nChecking {fold1} vs {fold2}...")
        similar_count = 0
        
        hists1 = fold_histograms[fold1]
        hists2 = fold_histograms[fold2]
        
        for h1 in tqdm(hists1, desc=f"  Comparing {fold1} histograms"):
            for h2 in hists2:
                if are_sub_hists(h1, h2, k=k, d=d):
                    similar_count += 1
                    break  # Found at least one similar, move to next h1
        
        leakage_stats[f"{fold1}_vs_{fold2}"] = similar_count
        pct = similar_count / len(hists1) * 100 if hists1 else 0
        print(f"  Similar histograms: {similar_count}/{len(hists1)} ({pct:.2f}%)")
    
    print("\n" + "="*80)
    return leakage_stats


def main():
    """Main function to create Murcko histogram-based splits."""
    # Paths
    THESIS_DIR = Path(__file__).parent.parent
    input_path = THESIS_DIR / "data/processed/massspecgym_complete/ssl_embs/MassSpecGym_with_SSL_embeddings.parquet"
    output_path = THESIS_DIR / "data/processed/massspecgym_complete/ssl_embs/MassSpecGym_with_SSL_embeddings_murcko_hist_splits.parquet"
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} spectra")
    
    # Remove old fold column if it exists
    if FOLD in df.columns:
        print(f"Removing old '{FOLD}' column...")
        df = df.drop(columns=[FOLD])
    
    # Create new splits
    df_split = create_murcko_histogram_splits(
        df,
        val_frac=0.08,    # ~8% for validation (to match your original)
        test_frac=0.15,   # ~15% for test (to match your original)
        k=3,              # DreaMS tutorial default
        d=4,              # DreaMS tutorial default
        random_state=42
    )
    
    # Verify no leakage (optional, can be slow)
    verify = input("\nVerify histogram leakage? This may take a while (y/n): ").lower().strip()
    if verify == 'y':
        leakage_stats = verify_no_histogram_leakage(df_split, k=3, d=4)
        print("\nLeakage statistics:", leakage_stats)
    
    # Save
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_split.to_parquet(output_path, index=False)
    
    print("\n✅ Done! New splits created with Murcko histogram method.")
    print(f"\nTo use these splits, update your notebook to load:")
    print(f"  {output_path.relative_to(THESIS_DIR / 'notebooks')}")


if __name__ == "__main__":
    main()
