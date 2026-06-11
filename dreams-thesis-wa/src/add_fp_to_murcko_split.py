#!/usr/bin/env python3
"""
Add pre-computed Morgan ECFP4 2048-bit fingerprints to MassSpecGym_MurckoHist_split.hdf5
This allows fast data loading without RDKit computation on every batch.
"""
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

def compute_morgan_fp(smiles, fp_size=2048, radius=2):
    """Compute Morgan ECFP4 fingerprint from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(fp_size, dtype=np.uint8)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
        return np.array(fp, dtype=np.uint8)
    except Exception as e:
        print(f"Error computing FP for {smiles}: {e}")
        return np.zeros(fp_size, dtype=np.uint8)

def main():
    hdf5_path = Path("dreams-thesis-wa/data/processed/MassSpecGym_MurckoHist_split.hdf5")
    
    if not hdf5_path.exists():
        print(f"Error: {hdf5_path} not found")
        return
    
    print(f"Opening {hdf5_path}...")
    with h5py.File(hdf5_path, 'a') as f:
        n_samples = len(f['spectrum'])
        print(f"Total samples: {n_samples}")
        
        # Check if fp_morgan_2048 already exists
        if 'fp_morgan_2048' in f:
            print("✅ fp_morgan_2048 already exists, skipping...")
            print(f"   Shape: {f['fp_morgan_2048'].shape}")
            return
        
        print("\nComputing Morgan ECFP4 fingerprints (2048-bit)...")
        
        # Create dataset for fingerprints
        fp_dataset = f.create_dataset(
            'fp_morgan_2048',
            shape=(n_samples, 2048),
            dtype=np.uint8,
            compression='gzip',
            compression_opts=4
        )
        
        # Process in batches for memory efficiency
        batch_size = 1000
        smiles_data = f['smiles'][:]  # Load all SMILES to avoid repeated HDF5 access
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Computing FPs"):
            batch_end = min(i + batch_size, n_samples)
            batch_smiles = smiles_data[i:batch_end]
            
            # Decode bytes if necessary
            if isinstance(batch_smiles[0], bytes):
                batch_smiles = [s.decode('utf-8') for s in batch_smiles]
            
            # Compute fingerprints
            fps = [compute_morgan_fp(s) for s in batch_smiles]
            fps = np.array(fps, dtype=np.uint8)
            
            # Write to HDF5
            fp_dataset[i:batch_end] = fps
        
        print("\n✅ Successfully added fp_morgan_2048 dataset!")
        print(f"   Shape: {fp_dataset.shape}")
        print(f"   Dtype: {fp_dataset.dtype}")
        print(f"   Compression: gzip (4)")

if __name__ == '__main__':
    main()
