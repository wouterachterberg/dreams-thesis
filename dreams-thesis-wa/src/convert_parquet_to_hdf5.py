#!/usr/bin/env python3
"""
Convert MassSpecGym parquet files directly to HDF5 format for DreaMS.

This script reads parquet files and converts them to DreaMS-compatible HDF5 files.
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
    return np.array([float(x.strip()) for x in array_str.split(',')])


def convert_parquet_to_hdf5(parquet_path, output_path, n_highest_peaks=128):
    """
    Convert a single parquet file to DreaMS-compatible HDF5 format.
    
    Args:
        parquet_path: Path to input parquet file
        output_path: Path to output HDF5 file
        n_highest_peaks: Maximum number of peaks to keep per spectrum
    """
    print(f"Reading {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)
    
    num_spectra = len(df)
    print(f"Processing {num_spectra} spectra...")
    
    # Prepare spectrum array (num_spectra, 2, n_highest_peaks)
    spectrum_array = np.zeros((num_spectra, 2, n_highest_peaks), dtype=np.float32)
    
    # Process each spectrum
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=num_spectra, desc="Converting spectra")):
        # Parse m/z and intensity arrays
        mzs = parse_array_string(row['mzs'])
        intensities = parse_array_string(row['intensities'])
        
        if len(mzs) > 0 and len(intensities) > 0:
            # Ensure arrays are same length
            min_len = min(len(mzs), len(intensities))
            mzs = mzs[:min_len]
            intensities = intensities[:min_len]
            
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
    
    # Create HDF5 file
    print(f"Creating HDF5 file: {output_path.name}...")
    with h5py.File(output_path, 'w') as f:
        # Core DreaMS columns
        f.create_dataset(SPECTRUM, data=spectrum_array, compression='gzip')
        f.create_dataset(PRECURSOR_MZ, data=df['precursor_mz'].values.astype(np.float32), compression='gzip')
        
        # Handle charge (default to 1 if missing)
        if 'charge' in df.columns:
            charges = df['charge'].fillna(1).astype(np.int32).values
        else:
            charges = np.ones(num_spectra, dtype=np.int32)
        f.create_dataset(CHARGE, data=charges, compression='gzip')
        
        # Handle adduct (default to [M+H]+ if missing)
        if 'adduct' in df.columns:
            adducts = df['adduct'].fillna('[M+H]+').astype(str).values
        else:
            adducts = np.array(['[M+H]+'] * num_spectra)
        f.create_dataset(ADDUCT, data=[s.encode('utf-8') for s in adducts], compression='gzip')
        
        # Handle SMILES (default to empty string if missing)
        if 'smiles' in df.columns:
            smiles = df['smiles'].fillna('').astype(str).values
        else:
            smiles = np.array([''] * num_spectra)
        f.create_dataset(SMILES, data=[s.encode('utf-8') for s in smiles], compression='gzip')
        
        # Add other metadata columns
        for col in ['identifier', 'inchikey', 'formula', 'precursor_formula', 
                    'parent_mass', 'instrument_type', 'collision_energy', 
                    'fold', 'simulation_challenge']:
            if col in df.columns:
                values = df[col].fillna('').astype(str).values
                f.create_dataset(col, data=[s.encode('utf-8') for s in values], compression='gzip')
    
    print(f"✓ Successfully created {output_path.name}")
    print(f"  Spectra: {num_spectra}")
    print(f"  Shape: {spectrum_array.shape}")
    

def process_directory(input_dir, output_dir):
    """
    Process all parquet files in a directory.
    
    Args:
        input_dir: Directory containing parquet files
        output_dir: Directory to save HDF5 files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    parquet_files = sorted(input_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files to convert")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    successful = 0
    failed = 0
    
    for parquet_file in parquet_files:
        output_file = output_dir / f"{parquet_file.stem}.hdf5"
        
        try:
            convert_parquet_to_hdf5(parquet_file, output_file)
            successful += 1
            print()
        except Exception as e:
            print(f"✗ Error processing {parquet_file.name}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print(f"Conversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(parquet_files)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert MassSpecGym parquet files to DreaMS HDF5 format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing parquet files'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for HDF5 files'
    )
    parser.add_argument(
        '--n-peaks',
        type=int,
        default=128,
        help='Maximum number of peaks per spectrum (default: 128)'
    )
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
