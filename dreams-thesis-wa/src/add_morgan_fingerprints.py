#!/usr/bin/env python3
"""
Add Morgan ECFP4 (2048-bit) fingerprints to MassSpecGym HDF5 file.

This script reads the existing HDF5 file and computes Morgan fingerprints
for each SMILES string, then adds them as 'fp_morgan_2048' dataset.
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

def compute_morgan_fingerprint(smiles, fp_size=2048, radius=2):
    """
    Compute Morgan ECFP4 fingerprint from SMILES.
    
    Args:
        smiles: SMILES string
        fp_size: Fingerprint size (default 2048)
        radius: Radius for Morgan algorithm (default 2 = ECFP4)
    
    Returns:
        numpy array of fingerprint as uint8 (binary)
    """
    try:
        if not smiles or smiles == '' or str(smiles) == 'nan':
            return np.zeros(fp_size, dtype=np.uint8)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(fp_size, dtype=np.uint8)
        
        # Generate Morgan fingerprint (binary, radius=2 for ECFP4)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
        
        # Convert to numpy array
        fp_array = np.array(fp, dtype=np.uint8)
        return fp_array
    except Exception as e:
        print(f"Error computing fingerprint for '{smiles}': {e}")
        return np.zeros(fp_size, dtype=np.uint8)

def add_morgan_fingerprints_to_hdf5(hdf5_path, output_path=None, fp_size=2048):
    """
    Add Morgan fingerprints to existing HDF5 file.
    
    Args:
        hdf5_path: Path to input HDF5 file
        output_path: Path to output HDF5 file (default: same as input)
        fp_size: Fingerprint size (default 2048)
    """
    if output_path is None:
        output_path = hdf5_path
    
    hdf5_path = Path(hdf5_path)
    output_path = Path(output_path)
    
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Input file not found: {hdf5_path}")
    
    print(f"📖 Reading HDF5 file: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        num_spectra = len(f['smiles'])
        smiles_list = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in f['smiles'][:]]
    
    print(f"📊 Total spectra: {num_spectra}")
    print(f"🧬 Computing Morgan ECFP4 ({fp_size}-bit) fingerprints...")
    
    # Compute fingerprints
    fingerprints = []
    for smiles in tqdm(smiles_list, desc="Computing fingerprints"):
        fp = compute_morgan_fingerprint(smiles, fp_size=fp_size)
        fingerprints.append(fp)
    
    fingerprints = np.array(fingerprints, dtype=np.uint8)
    print(f"✅ Computed fingerprints shape: {fingerprints.shape}")
    
    # Check if we need to copy the file
    if output_path != hdf5_path:
        print(f"📋 Copying HDF5 file to: {output_path}")
        import shutil
        shutil.copy2(hdf5_path, output_path)
    
    # Add fingerprints to HDF5
    print(f"💾 Adding fingerprints to HDF5...")
    with h5py.File(output_path, 'a') as f:
        # Remove if already exists
        if 'fp_morgan_2048' in f:
            del f['fp_morgan_2048']
        
        # Add new dataset
        f.create_dataset(
            'fp_morgan_2048',
            data=fingerprints,
            compression='gzip',
            compression_opts=9
        )
        print(f"✅ Added fp_morgan_2048 dataset")
    
    print(f"✅ Done! Output file: {output_path}")
    
    # Verify
    print(f"\n📊 Verifying output:")
    with h5py.File(output_path, 'r') as f:
        print(f"Datasets in file: {list(f.keys())}")
        print(f"fp_morgan_2048 shape: {f['fp_morgan_2048'].shape}")
        print(f"fp_morgan_2048 dtype: {f['fp_morgan_2048'].dtype}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add Morgan ECFP4 fingerprints to MassSpecGym HDF5 file"
    )
    parser.add_argument(
        'input_hdf5',
        type=str,
        help='Path to input HDF5 file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output HDF5 file (default: same as input)'
    )
    parser.add_argument(
        '--fp-size',
        type=int,
        default=2048,
        help='Fingerprint size (default: 2048)'
    )
    
    args = parser.parse_args()
    
    add_morgan_fingerprints_to_hdf5(
        args.input_hdf5,
        output_path=args.output,
        fp_size=args.fp_size
    )
