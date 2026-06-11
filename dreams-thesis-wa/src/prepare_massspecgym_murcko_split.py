#!/usr/bin/env python3
"""
Create Murcko histogram split for MassSpecGym dataset.

This script:
1. Reads MassSpecGym.tsv
2. Computes Murcko histograms for each unique molecule
3. Splits train/val using the DreaMS Murcko histogram algorithm (prevents data leakage)
4. Saves to HDF5 format for fine-tuning

Usage:
    python prepare_massspecgym_murcko_split.py
"""

import sys
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem

# Add DreaMS to path
dreams_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(dreams_path))

from dreams.algorithms.murcko_hist import murcko_hist, are_sub_hists
from dreams.definitions import SPECTRUM, PRECURSOR_MZ, CHARGE, ADDUCT, SMILES, FOLD


def parse_array_string(array_str):
    """Parse comma-separated string into numpy array."""
    if pd.isna(array_str) or array_str == '':
        return np.array([])
    array_str = str(array_str).strip('[]')
    if ',' in array_str:
        return np.array([float(x.strip()) for x in array_str.split(',') if x.strip()])
    else:
        return np.array([float(x.strip()) for x in array_str.split() if x.strip()])


def compute_murcko_histograms(df):
    """Compute Murcko histograms for unique SMILES."""
    print("\n=== Computing Murcko Histograms ===")
    
    # Get unique SMILES
    unique_smiles = df['smiles'].dropna().unique()
    print(f"Unique SMILES: {len(unique_smiles)}")
    
    # Compute histograms
    smiles_to_hist = {}
    failed = 0
    for smi in tqdm(unique_smiles, desc="Computing Murcko histograms"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                smiles_to_hist[smi] = murcko_hist(mol)
            else:
                smiles_to_hist[smi] = {}
                failed += 1
        except Exception:
            smiles_to_hist[smi] = {}
            failed += 1
    
    print(f"Successfully computed: {len(smiles_to_hist) - failed}")
    print(f"Failed (using empty hist): {failed}")
    
    return smiles_to_hist


def create_murcko_split(df, smiles_to_hist, val_frac=0.15):
    """
    Create train/val split based on Murcko histograms.
    
    This implements the algorithm from the DreaMS paper that ensures
    structurally similar molecules stay in the same split.
    """
    print(f"\n=== Creating Murcko Histogram Split (val_frac={val_frac}) ===")
    
    # Create dataframe with unique SMILES and their histograms
    df_us = df.drop_duplicates(subset=['smiles']).copy()
    df_us['MurckoHist'] = df_us['smiles'].map(smiles_to_hist)
    df_us['MurckoHistStr'] = df_us['MurckoHist'].astype(str)
    
    print(f"Unique Murcko histograms: {df_us['MurckoHistStr'].nunique()}")
    
    # Group by Murcko histogram
    df_gb = df_us.groupby('MurckoHistStr').agg(
        count=('smiles', 'count'),
        smiles_list=('smiles', list)
    ).reset_index()
    df_gb['MurckoHist'] = df_gb['MurckoHistStr'].apply(eval)
    df_gb = df_gb.sort_values('count', ascending=False).reset_index(drop=True)
    
    print(f"Murcko histogram groups: {len(df_gb)}")
    
    # Split algorithm from DreaMS tutorial
    median_i = len(df_gb) // 2
    cum_val_mols = 0
    val_idx, train_idx = [], []
    
    print("Running split algorithm...")
    for i in tqdm(range(median_i, -1, -1), desc="Assigning folds"):
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
    
    # Add remaining indices to train set
    train_idx.extend(range(median_i + 1, len(df_gb)))
    
    # Map SMILES to fold
    smiles_to_fold = {}
    for i, row in df_gb.iterrows():
        fold = 'val' if i in val_idx else 'train'
        for smiles in row['smiles_list']:
            smiles_to_fold[smiles] = fold
    
    # Apply to full dataframe
    df['fold_murcko'] = df['smiles'].map(smiles_to_fold)
    
    # Print statistics
    print(f"\n=== Split Statistics ===")
    fold_counts = df['fold_murcko'].value_counts()
    for fold, count in fold_counts.items():
        print(f"  {fold}: {count} spectra ({100*count/len(df):.1f}%)")
    
    unique_fold = df.drop_duplicates(subset=['smiles'])['fold_murcko'].value_counts()
    print("\nUnique molecules:")
    for fold, count in unique_fold.items():
        print(f"  {fold}: {count} molecules ({100*count/len(df_us):.1f}%)")
    
    return df


def save_to_hdf5(df, output_path, n_highest_peaks=128):
    """Save processed data to HDF5 format."""
    print(f"\n=== Saving to HDF5 ===")
    print(f"Output: {output_path}")
    
    num_spectra = len(df)
    spectrum_array = np.zeros((num_spectra, 2, n_highest_peaks), dtype=np.float32)
    
    # Process spectra
    skipped = 0
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=num_spectra, desc="Processing spectra")):
        mzs = parse_array_string(row['mzs'])
        intensities = parse_array_string(row['intensities'])
        
        if len(mzs) > 0 and len(intensities) > 0:
            min_len = min(len(mzs), len(intensities))
            mzs = mzs[:min_len]
            intensities = intensities[:min_len]
            
            if intensities.max() > 0:
                intensities = intensities / intensities.max()
            
            if len(mzs) > n_highest_peaks:
                sorted_indices = np.argsort(intensities)[::-1][:n_highest_peaks]
                sorted_indices = np.sort(sorted_indices)
                mzs = mzs[sorted_indices]
                intensities = intensities[sorted_indices]
            
            num_peaks = len(mzs)
            spectrum_array[i, 0, :num_peaks] = mzs
            spectrum_array[i, 1, :num_peaks] = intensities
        else:
            skipped += 1
    
    print(f"Skipped {skipped} spectra with empty peaks")
    
    # Create HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset(SPECTRUM, data=spectrum_array, compression='gzip')
        f.create_dataset(PRECURSOR_MZ, 
                        data=df['precursor_mz'].fillna(0.0).astype(np.float32).values,
                        compression='gzip')
        
        charges = np.ones(num_spectra, dtype=np.int32)
        f.create_dataset(CHARGE, data=charges, compression='gzip')
        
        adducts = df['adduct'].fillna('[M+H]+').astype(str).values
        f.create_dataset(ADDUCT, 
                        data=[s.encode('utf-8') for s in adducts],
                        compression='gzip')
        
        smiles = df['smiles'].fillna('').astype(str).values
        f.create_dataset(SMILES, 
                        data=[s.encode('utf-8') for s in smiles],
                        compression='gzip')
        
        # Use the Murcko histogram fold
        folds = df['fold_murcko'].fillna('train').astype(str).values
        f.create_dataset(FOLD, 
                        data=[s.encode('utf-8') for s in folds],
                        compression='gzip')
        
        # Additional metadata
        metadata_cols = ['identifier', 'inchikey', 'formula', 'instrument_type', 'collision_energy']
        for col in metadata_cols:
            if col in df.columns:
                values = df[col].fillna('').astype(str).values
                f.create_dataset(col, 
                               data=[s.encode('utf-8') for s in values],
                               compression='gzip')
    
    print(f"\n✓ Saved {num_spectra} spectra to {output_path.name}")


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    tsv_path = project_root / "data" / "raw" / "MassSpecGym.tsv"
    output_path = project_root / "data" / "processed" / "MassSpecGym_MurckoHist_split.hdf5"
    
    if not tsv_path.exists():
        print(f"Error: Input file not found: {tsv_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("MassSpecGym Murcko Histogram Split for Fine-Tuning")
    print("=" * 70)
    print(f"Input:  {tsv_path}")
    print(f"Output: {output_path}")
    
    # Load data
    print("\n=== Loading Data ===")
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Loaded {len(df)} spectra")
    
    # Compute Murcko histograms
    smiles_to_hist = compute_murcko_histograms(df)
    
    # Create split
    df = create_murcko_split(df, smiles_to_hist, val_frac=0.15)
    
    # Save to HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_hdf5(df, output_path)
    
    print("\n" + "=" * 70)
    print("✓ Done! Use this file for fine-tuning:")
    print(f"  --dataset_pth {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
