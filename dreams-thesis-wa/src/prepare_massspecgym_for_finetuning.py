#!/usr/bin/env python3
"""
Convert MassSpecGym.tsv to HDF5 format for DreaMS fine-tuning.

This script processes the raw MassSpecGym TSV file and creates an HDF5 file
containing only the necessary columns for fine-tuning with Morgan fingerprints:
- identifier
- mzs (m/z values)
- intensities
- smiles
- inchikey
- metadata (precursor_mz, adduct, instrument_type, collision_energy, fold)

Usage:
    python prepare_massspecgym_for_finetuning.py
"""

import sys
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

# Add DreaMS to path
dreams_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(dreams_path))

from dreams.definitions import SPECTRUM, PRECURSOR_MZ, CHARGE, ADDUCT, SMILES


def parse_array_string(array_str):
    """Parse comma-separated string into numpy array."""
    if pd.isna(array_str) or array_str == '':
        return np.array([])
    # Remove brackets and split by comma
    array_str = str(array_str).strip('[]')
    # Handle both comma-separated and space-separated values
    if ',' in array_str:
        return np.array([float(x.strip()) for x in array_str.split(',') if x.strip()])
    else:
        return np.array([float(x.strip()) for x in array_str.split() if x.strip()])


def convert_tsv_to_hdf5(tsv_path, output_path, n_highest_peaks=128):
    """
    Convert MassSpecGym.tsv to DreaMS-compatible HDF5 format.
    
    Args:
        tsv_path: Path to input TSV file
        output_path: Path to output HDF5 file
        n_highest_peaks: Maximum number of peaks to keep per spectrum
    """
    print(f"Reading {tsv_path.name}...")
    df = pd.read_csv(tsv_path, sep='\t')
    
    num_spectra = len(df)
    print(f"Processing {num_spectra} spectra...")
    print(f"Columns: {df.columns.tolist()}")
    
    # Prepare spectrum array (num_spectra, 2, n_highest_peaks)
    spectrum_array = np.zeros((num_spectra, 2, n_highest_peaks), dtype=np.float32)
    
    # Process each spectrum
    skipped = 0
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=num_spectra, desc="Converting spectra")):
        # Parse m/z and intensity arrays
        mzs = parse_array_string(row['mzs'])
        intensities = parse_array_string(row['intensities'])
        
        if len(mzs) > 0 and len(intensities) > 0:
            # Ensure arrays are same length
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
        else:
            skipped += 1
    
    print(f"Skipped {skipped} spectra with empty peaks")
    
    # Create HDF5 file
    print(f"\nCreating HDF5 file: {output_path.name}...")
    with h5py.File(output_path, 'w') as f:
        # Core DreaMS columns
        f.create_dataset(SPECTRUM, data=spectrum_array, compression='gzip')
        
        # Precursor m/z
        f.create_dataset(PRECURSOR_MZ, 
                        data=df['precursor_mz'].fillna(0.0).astype(np.float32).values, 
                        compression='gzip')
        
        # Charge (default to 1 if missing)
        charges = np.ones(num_spectra, dtype=np.int32)
        f.create_dataset(CHARGE, data=charges, compression='gzip')
        
        # Adduct
        if 'adduct' in df.columns:
            adducts = df['adduct'].fillna('[M+H]+').astype(str).values
        else:
            adducts = np.array(['[M+H]+'] * num_spectra)
        f.create_dataset(ADDUCT, 
                        data=[s.encode('utf-8') for s in adducts], 
                        compression='gzip')
        
        # SMILES (required for fine-tuning)
        if 'smiles' in df.columns:
            smiles = df['smiles'].fillna('').astype(str).values
        else:
            raise ValueError("SMILES column not found in TSV file")
        f.create_dataset(SMILES, 
                        data=[s.encode('utf-8') for s in smiles], 
                        compression='gzip')
        
        # Add essential metadata columns
        metadata_cols = {
            'identifier': 'identifier',
            'inchikey': 'inchikey',
            'formula': 'formula',
            'precursor_formula': 'precursor_formula',
            'parent_mass': 'parent_mass',
            'instrument_type': 'instrument_type',
            'collision_energy': 'collision_energy',
            'fold': 'fold',
            'simulation_challenge': 'simulation_challenge'
        }
        
        for hdf5_name, col_name in metadata_cols.items():
            if col_name in df.columns:
                values = df[col_name].fillna('').astype(str).values
                f.create_dataset(hdf5_name, 
                               data=[s.encode('utf-8') for s in values], 
                               compression='gzip')
                print(f"  Added {hdf5_name}: {len(values)} values")
    
    print(f"\n✓ Successfully created {output_path.name}")
    print(f"  Total spectra: {num_spectra}")
    print(f"  Spectrum shape: {spectrum_array.shape}")
    print(f"  Max peaks per spectrum: {n_highest_peaks}")
    print(f"\nFile saved to: {output_path}")
    

def main():
    """Main function to convert MassSpecGym.tsv to HDF5."""
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Input TSV file
    tsv_path = project_root / "data" / "raw" / "MassSpecGym.tsv"
    
    # Output HDF5 file
    output_path = project_root / "data" / "processed" / "MassSpecGym_finetuning.hdf5"
    
    # Check input file exists
    if not tsv_path.exists():
        print(f"Error: Input file not found: {tsv_path}")
        print(f"Please ensure MassSpecGym.tsv exists in data/raw/")
        sys.exit(1)
    
    print("=" * 70)
    print("MassSpecGym TSV → HDF5 Conversion for Fine-Tuning")
    print("=" * 70)
    print(f"Input:  {tsv_path}")
    print(f"Output: {output_path}")
    print()
    
    # Convert
    try:
        convert_tsv_to_hdf5(tsv_path, output_path, n_highest_peaks=128)
        print("\n" + "=" * 70)
        print("✓ Conversion completed successfully!")
        print("=" * 70)
        print(f"\nYou can now use this file for fine-tuning:")
        print(f"  --dataset_pth {output_path}")
        print(f"  --train_objective fp_morgan_2048")
        
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
