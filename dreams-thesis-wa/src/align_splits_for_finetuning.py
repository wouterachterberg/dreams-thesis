#!/usr/bin/env python3
"""
Create aligned splits for fine-tuning that preserve probing evaluation set.

This script creates:
1. Fine-tuning HDF5 (from probing train 77% only):
   - 'train' = ~85% of probing train (~65% total)
   - 'val' = ~15% of probing train (~12% total) - for early stopping
   
2. Probing test parquet (15%):
   - Completely held out from fine-tuning
   - Use for evaluating both pre-trained and fine-tuned probing
   
3. Holdout parquet (8% = probing val):
   - Pristine, never touched

This ensures fair comparison: probing test is NEVER seen during fine-tuning.

Usage:
    python align_splits_for_finetuning.py
"""

import sys
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple

# Add DreaMS to path
dreams_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(dreams_path))

from dreams.definitions import SPECTRUM, PRECURSOR_MZ, CHARGE, ADDUCT, SMILES, FOLD


def load_probing_data(parquet_path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load probing parquet and extract SMILES→fold mapping.
    
    Returns:
        (DataFrame, Dictionary mapping SMILES to fold)
    """
    print("="*80)
    print("STEP 1: Loading Probing Data")
    print("="*80)
    
    print(f"\nReading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print(f"Total spectra: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Get unique SMILES→fold mapping
    df_unique = df[[SMILES, FOLD]].drop_duplicates(subset=[SMILES])
    smiles_to_fold = dict(zip(df_unique[SMILES], df_unique[FOLD]))
    
    print(f"\nUnique molecules: {len(smiles_to_fold):,}")
    
    # Report fold distribution
    fold_counts = Counter(smiles_to_fold.values())
    print("\nOriginal fold distribution (unique molecules):")
    total = sum(fold_counts.values())
    for fold in ['train', 'val', 'test']:
        count = fold_counts.get(fold, 0)
        pct = count / total * 100
        print(f"  {fold:5s}: {count:6,} molecules ({pct:5.2f}%)")
    
    # Report spectra distribution
    print("\nOriginal fold distribution (all spectra):")
    spectra_counts = df[FOLD].value_counts()
    for fold in ['train', 'val', 'test']:
        count = spectra_counts.get(fold, 0)
        pct = count / len(df) * 100
        print(f"  {fold:5s}: {count:6,} spectra ({pct:5.2f}%)")
    
    return df, smiles_to_fold


def create_finetuning_splits(df: pd.DataFrame, smiles_to_fold: Dict[str, str], 
                             ft_val_frac: float = 0.15, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data for fine-tuning while preserving probing test set.
    
    Strategy:
    - probing 'train' (77%) → Split into fine-tuning train/val
      - ft_train: ~85% of probing train (~65% total)
      - ft_val:   ~15% of probing train (~12% total) - for early stopping
    - probing 'test'  (15%) → Separate file (probing_test) - NEVER seen during fine-tuning
    - probing 'val'   (8%)  → Holdout parquet (pristine)
    
    Returns:
        (finetuning_df, probing_test_df, holdout_df)
    """
    print("\n" + "="*80)
    print("STEP 2: Creating Fine-tuning Splits")
    print("="*80)
    
    # Map SMILES to original probing folds
    df = df.copy()
    df['probing_fold'] = df[SMILES].map(smiles_to_fold)
    
    # Check for unmapped
    unmapped = df['probing_fold'].isna()
    if unmapped.any():
        n_unmapped = unmapped.sum()
        print(f"\n⚠️  WARNING: {n_unmapped} spectra could not be mapped!")
        print(f"   They will be excluded.")
        df = df[~unmapped].copy()
    
    # Separate the three probing splits
    probing_train_df = df[df['probing_fold'] == 'train'].copy()
    probing_test_df = df[df['probing_fold'] == 'test'].copy()
    holdout_df = df[df['probing_fold'] == 'val'].copy()
    
    print(f"\nOriginal probing splits:")
    print(f"  probing train: {len(probing_train_df):,} spectra")
    print(f"  probing test:  {len(probing_test_df):,} spectra")
    print(f"  probing val:   {len(holdout_df):,} spectra (→ holdout)")
    
    # Split probing train into fine-tuning train/val (by molecule, not spectrum)
    print(f"\nSplitting probing train into fine-tuning train/val...")
    print(f"  Fine-tuning val fraction: {ft_val_frac:.0%}")
    
    np.random.seed(seed)
    unique_train_smiles = probing_train_df[SMILES].unique()
    n_ft_val = int(len(unique_train_smiles) * ft_val_frac)
    n_ft_train = len(unique_train_smiles) - n_ft_val
    
    # Random shuffle and split
    shuffled_smiles = np.random.permutation(unique_train_smiles)
    ft_val_smiles = set(shuffled_smiles[:n_ft_val])
    ft_train_smiles = set(shuffled_smiles[n_ft_val:])
    
    # Assign fine-tuning folds
    probing_train_df[FOLD] = probing_train_df[SMILES].apply(
        lambda s: 'val' if s in ft_val_smiles else 'train'
    )
    
    # Create fine-tuning dataset (only from probing train)
    finetuning_df = probing_train_df.copy()
    
    # Set fold for probing test and holdout
    probing_test_df[FOLD] = 'test'
    probing_test_df['original_fold'] = 'probing_test'
    holdout_df[FOLD] = 'holdout'
    holdout_df['original_fold'] = 'probing_val'
    
    # Report fine-tuning splits
    ft_train_count = len(finetuning_df[finetuning_df[FOLD] == 'train'])
    ft_val_count = len(finetuning_df[finetuning_df[FOLD] == 'val'])
    
    print(f"\nFine-tuning dataset (from probing train only):")
    print(f"  train: {ft_train_count:6,} spectra ({len(ft_train_smiles):,} molecules)")
    print(f"  val:   {ft_val_count:6,} spectra ({len(ft_val_smiles):,} molecules) - for early stopping")
    
    print(f"\nProbing test (HELD OUT from fine-tuning):")
    print(f"  test:  {len(probing_test_df):6,} spectra ({probing_test_df[SMILES].nunique():,} molecules)")
    
    print(f"\nHoldout (pristine):")
    print(f"  holdout: {len(holdout_df):6,} spectra ({holdout_df[SMILES].nunique():,} molecules)")
    
    # Clean up temp column
    finetuning_df = finetuning_df.drop(columns=['probing_fold'])
    probing_test_df = probing_test_df.drop(columns=['probing_fold'])
    holdout_df = holdout_df.drop(columns=['probing_fold'])
    
    return finetuning_df, probing_test_df, holdout_df


def validate_splits(finetuning_df: pd.DataFrame, probing_test_df: pd.DataFrame,
                   holdout_df: pd.DataFrame, smiles_to_fold: Dict[str, str]) -> Tuple[bool, Dict]:
    """
    Validate that splits have no leakage.
    
    Returns:
        (is_valid, stats_dict)
    """
    print("\n" + "="*80)
    print("STEP 3: Validating Splits")
    print("="*80)
    
    is_valid = True
    stats = {}
    
    # Get unique SMILES per split
    ft_train_smiles = set(finetuning_df[finetuning_df[FOLD] == 'train'][SMILES].unique())
    ft_val_smiles = set(finetuning_df[finetuning_df[FOLD] == 'val'][SMILES].unique())
    test_smiles = set(probing_test_df[SMILES].unique())
    holdout_smiles = set(holdout_df[SMILES].unique())
    
    # Check 1: No overlap within fine-tuning (train vs val)
    print("\n1. Checking fine-tuning train-val separation...")
    overlap = ft_train_smiles & ft_val_smiles
    if overlap:
        print(f"   ❌ FAIL: {len(overlap)} molecules overlap between ft_train and ft_val")
        is_valid = False
    else:
        print("   ✅ PASS: No overlap between ft_train and ft_val")
    
    # Check 2: Probing test is isolated from fine-tuning
    print("\n2. Checking probing test isolation from fine-tuning...")
    overlap_ft_train = ft_train_smiles & test_smiles
    overlap_ft_val = ft_val_smiles & test_smiles
    
    if overlap_ft_train:
        print(f"   ❌ FAIL: {len(overlap_ft_train)} molecules overlap between ft_train and probing_test")
        is_valid = False
    else:
        print("   ✅ PASS: No overlap between ft_train and probing_test")
    
    if overlap_ft_val:
        print(f"   ❌ FAIL: {len(overlap_ft_val)} molecules overlap between ft_val and probing_test")
        is_valid = False
    else:
        print("   ✅ PASS: No overlap between ft_val and probing_test")
    
    # Check 3: Holdout is completely isolated
    print("\n3. Checking holdout isolation...")
    all_other_smiles = ft_train_smiles | ft_val_smiles | test_smiles
    overlap_holdout = all_other_smiles & holdout_smiles
    
    if overlap_holdout:
        print(f"   ❌ FAIL: {len(overlap_holdout)} molecules overlap between holdout and other sets")
        is_valid = False
    else:
        print("   ✅ PASS: Holdout is completely isolated")
    
    # Check 4: Fine-tuning train+val should equal probing train
    print("\n4. Checking fine-tuning comes from probing train only...")
    probing_train_count = sum(1 for f in smiles_to_fold.values() if f == 'train')
    ft_total = len(ft_train_smiles) + len(ft_val_smiles)
    
    if ft_total == probing_train_count:
        print(f"   ✅ ft_train + ft_val = {ft_total:,} molecules (matches probing train)")
    else:
        print(f"   ❌ ft_train + ft_val = {ft_total:,} molecules (expected {probing_train_count:,})")
        is_valid = False
    
    # Check 5: Probing test count
    probing_test_count = sum(1 for f in smiles_to_fold.values() if f == 'test')
    if len(test_smiles) == probing_test_count:
        print(f"   ✅ probing_test: {len(test_smiles):,} molecules (matches)")
    else:
        print(f"   ❌ probing_test: {len(test_smiles):,} molecules (expected {probing_test_count:,})")
        is_valid = False
    
    # Check 6: Holdout count
    probing_val_count = sum(1 for f in smiles_to_fold.values() if f == 'val')
    if len(holdout_smiles) == probing_val_count:
        print(f"   ✅ holdout: {len(holdout_smiles):,} molecules (matches probing val)")
    else:
        print(f"   ❌ holdout: {len(holdout_smiles):,} molecules (expected {probing_val_count:,})")
        is_valid = False
    
    # Store stats
    stats['ft_train_smiles'] = len(ft_train_smiles)
    stats['ft_val_smiles'] = len(ft_val_smiles)
    stats['test_smiles'] = len(test_smiles)
    stats['holdout_smiles'] = len(holdout_smiles)
    stats['ft_train_spectra'] = len(finetuning_df[finetuning_df[FOLD] == 'train'])
    stats['ft_val_spectra'] = len(finetuning_df[finetuning_df[FOLD] == 'val'])
    stats['test_spectra'] = len(probing_test_df)
    stats['holdout_spectra'] = len(holdout_df)
    
    return is_valid, stats
    
    return is_valid, stats


def parse_spectrum_string(spec_str: str, n_peaks: int = 128) -> np.ndarray:
    """Parse spectrum string and convert to (2, 128) array."""
    if pd.isna(spec_str) or spec_str == '':
        return np.zeros((2, n_peaks), dtype=np.float32)
    
    # Parse m/z and intensity pairs
    pairs = []
    for pair in spec_str.split():
        if ':' in pair:
            mz, intensity = pair.split(':')
            pairs.append([float(mz), float(intensity)])
    
    if not pairs:
        return np.zeros((2, n_peaks), dtype=np.float32)
    
    # Convert to array
    spec_array = np.array(pairs, dtype=np.float32).T  # (2, n_peaks)
    
    # Trim to n_peaks highest intensity
    if spec_array.shape[1] > n_peaks:
        intensities = spec_array[1, :]
        top_indices = np.argsort(intensities)[-n_peaks:]
        spec_array = spec_array[:, top_indices]
        # Sort by m/z
        mz_order = np.argsort(spec_array[0, :])
        spec_array = spec_array[:, mz_order]
    
    # Pad if needed
    if spec_array.shape[1] < n_peaks:
        padding = np.zeros((2, n_peaks - spec_array.shape[1]), dtype=np.float32)
        spec_array = np.concatenate([spec_array, padding], axis=1)
    
    return spec_array


def save_to_hdf5(df: pd.DataFrame, output_path: Path, n_highest_peaks: int = 128):
    """
    Save DataFrame to HDF5 format for fine-tuning.
    Only contains 2 folds: 'train' and 'val'.
    
    Args:
        df: DataFrame with spectra and fold assignments
        output_path: Path to output HDF5 file
        n_highest_peaks: Maximum number of peaks to keep
    """
    print("\n" + "="*80)
    print("STEP 4: Saving Fine-tuning HDF5")
    print("="*80)
    
    print(f"\nOutput: {output_path}")
    
    # Verify only train and val folds
    folds = df[FOLD].unique()
    assert set(folds) == {'train', 'val'}, f"Expected only train/val folds, got: {folds}"
    
    # Prepare spectrum data - handle different formats
    print("\nPreparing spectrum data...")
    
    if SPECTRUM in df.columns:
        # Already have spectrum column
        spectra = df[SPECTRUM].values
        if not isinstance(spectra[0], np.ndarray):
            print("Converting spectrum strings to arrays...")
            spectra = np.array([parse_spectrum_string(s, n_highest_peaks) for s in spectra])
    elif 'mzs' in df.columns and 'intensities' in df.columns:
        # Have separate mzs and intensities columns (parquet format)
        print("Found 'mzs' and 'intensities' columns, combining...")
        spectra = []
        for i, (mzs, intensities) in enumerate(zip(df['mzs'].values, df['intensities'].values)):
            if i % 50000 == 0:
                print(f"  Processing spectrum {i:,}/{len(df):,}")
            
            # Convert to arrays if needed
            if isinstance(mzs, str):
                # Try comma-separated first, then space-separated
                if ',' in mzs:
                    mzs = np.array([float(x) for x in mzs.split(',')])
                else:
                    mzs = np.array([float(x) for x in mzs.split()])
            elif isinstance(mzs, (list, tuple)):
                mzs = np.array(mzs)
            
            if isinstance(intensities, str):
                if ',' in intensities:
                    intensities = np.array([float(x) for x in intensities.split(',')])
                else:
                    intensities = np.array([float(x) for x in intensities.split()])
            elif isinstance(intensities, (list, tuple)):
                intensities = np.array(intensities)
            
            # Stack mzs and intensities
            if len(mzs) > 0:
                spec = np.stack([mzs, intensities], axis=0)  # (2, n_peaks)
                
                # Trim to n_highest_peaks if needed
                if spec.shape[1] > n_highest_peaks:
                    top_idx = np.argsort(spec[1])[-n_highest_peaks:]
                    spec = spec[:, top_idx]
                    # Sort by m/z
                    mz_order = np.argsort(spec[0])
                    spec = spec[:, mz_order]
                
                # Pad if needed
                if spec.shape[1] < n_highest_peaks:
                    padding = np.zeros((2, n_highest_peaks - spec.shape[1]), dtype=np.float32)
                    spec = np.concatenate([spec, padding], axis=1)
            else:
                spec = np.zeros((2, n_highest_peaks), dtype=np.float32)
            
            spectra.append(spec.astype(np.float32))
        
        spectra = np.array(spectra)
        print(f"  Converted {len(spectra):,} spectra to shape {spectra.shape}")
    else:
        raise ValueError("No spectrum column found! Expected 'spectrum' or 'mzs'+'intensities'")
    
    # Create HDF5
    print(f"\nCreating HDF5 file...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Save spectra
        print(f"  Writing spectrum data: shape {spectra.shape}")
        f.create_dataset(SPECTRUM, data=spectra, compression='gzip', compression_opts=9)
        
        # Save metadata columns
        for col in [SMILES, FOLD, PRECURSOR_MZ, ADDUCT, CHARGE]:
            if col in df.columns:
                data = df[col].values
                # Convert strings to bytes for HDF5 compatibility
                if data.dtype == object or str(data.dtype).startswith('<U') or str(data.dtype).startswith('str'):
                    data = np.array([str(x).encode('utf-8') for x in data])
                print(f"  Writing {col}: {len(data)} entries")
                f.create_dataset(col, data=data, compression='gzip', compression_opts=9)
    
    print(f"\n✅ HDF5 file created: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Folds: train + val (2 folds only)")


def save_holdout_parquet(df: pd.DataFrame, output_path: Path, label: str = "Holdout"):
    """
    Save data to parquet format.
    
    Args:
        df: DataFrame with spectra
        output_path: Path to output parquet file
        label: Label for logging
    """
    print(f"\nSaving {label} parquet...")
    print(f"  Output: {output_path}")
    print(f"  Spectra: {len(df):,}")
    
    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"  ✅ Created: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Main execution function."""
    # Paths
    THESIS_DIR = Path(__file__).parent.parent
    probing_path = THESIS_DIR / "data/processed/massspecgym_complete/ssl_embs/MassSpecGym_with_SSL_embeddings_murcko_hist_splits.parquet"
    
    # Output paths
    hdf5_output = THESIS_DIR / "data/processed/MassSpecGym_finetuning.hdf5"
    probing_test_output = THESIS_DIR / "data/processed/MassSpecGym_probing_test.parquet"
    holdout_output = THESIS_DIR / "data/processed/MassSpecGym_holdout.parquet"
    
    print("="*80)
    print("CREATING ALIGNED SPLITS FOR FINE-TUNING")
    print("="*80)
    print("\n📋 SPLIT STRATEGY:")
    print("   Probing train (77%) → Split into fine-tuning train/val")
    print("     └─ ft_train: ~85% of 77% (~65% total)")
    print("     └─ ft_val:   ~15% of 77% (~12% total) - for early stopping")
    print("   Probing test  (15%) → Separate file (probing_test.parquet)")
    print("     └─ NEVER seen during fine-tuning")
    print("     └─ Used to evaluate both pre-trained and fine-tuned probing")
    print("   Probing val   (8%)  → Separate file (holdout.parquet)")
    print("     └─ Pristine holdout, never touched")
    print()
    print(f"Input:         {probing_path}")
    print(f"Fine-tune HDF5: {hdf5_output}")
    print(f"Probing test:  {probing_test_output}")
    print(f"Holdout:       {holdout_output}")
    
    # Check input exists
    if not probing_path.exists():
        print(f"\n❌ ERROR: Probing parquet not found: {probing_path}")
        return
    
    # Step 1: Load probing data and splits
    df, smiles_to_fold = load_probing_data(probing_path)
    
    # Step 2: Create fine-tuning, probing test, and holdout splits
    finetuning_df, probing_test_df, holdout_df = create_finetuning_splits(df, smiles_to_fold)
    
    # Step 3: Validate splits
    is_valid, stats = validate_splits(finetuning_df, probing_test_df, holdout_df, smiles_to_fold)
    
    if not is_valid:
        print("\n❌ VALIDATION FAILED!")
        print("Please review the errors above before proceeding.")
        response = input("\nContinue anyway? (yes/no): ").lower().strip()
        if response != 'yes':
            print("Aborted.")
            return
    
    # Step 4: Save fine-tuning HDF5
    save_to_hdf5(finetuning_df, hdf5_output)
    
    # Step 5: Save probing test parquet
    print("\n" + "="*80)
    print("STEP 5: Saving Probing Test & Holdout Parquets")
    print("="*80)
    save_holdout_parquet(probing_test_df, probing_test_output, "Probing Test")
    
    # Step 6: Save holdout parquet
    save_holdout_parquet(holdout_df, holdout_output, "Holdout")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ SUMMARY")
    print("="*80)
    
    print("\n📁 FILES CREATED:")
    print(f"   1. {hdf5_output}")
    print(f"   2. {probing_test_output}")
    print(f"   3. {holdout_output}")
    
    print("\n📊 STATISTICS:")
    print(f"   Fine-tuning HDF5 (from probing train):")
    print(f"     ft_train: {stats['ft_train_smiles']:6,} molecules, {stats['ft_train_spectra']:6,} spectra")
    print(f"     ft_val:   {stats['ft_val_smiles']:6,} molecules, {stats['ft_val_spectra']:6,} spectra")
    print(f"   Probing test (held out from fine-tuning):")
    print(f"     test:     {stats['test_smiles']:6,} molecules, {stats['test_spectra']:6,} spectra")
    print(f"   Holdout (pristine):")
    print(f"     holdout:  {stats['holdout_smiles']:6,} molecules, {stats['holdout_spectra']:6,} spectra")
    
    print("\n📝 KEY POINTS:")
    print("   • Fine-tuning only sees probing train (split into ft_train/ft_val)")
    print("   • Probing test (15%) is NEVER seen during fine-tuning")
    print("   • Both pre-trained and fine-tuned probing evaluate on probing_test")
    print("   • Holdout (8%) is completely isolated for final validation")
    
    print("\n" + "="*80)
    print("USAGE")
    print("="*80)
    print(f"\n1. Fine-tuning:")
    print(f"   --dataset_pth {hdf5_output}")
    print(f"\n2. Probing (both pre-trained and fine-tuned embeddings):")
    print(f"   Train on: probing train (from original parquet, fold='train')")
    print(f"   Eval on:  {probing_test_output}")
    print(f"\n3. Do NOT touch the holdout until final thesis validation!")


if __name__ == "__main__":
    main()
