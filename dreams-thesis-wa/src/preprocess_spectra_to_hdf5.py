#!/usr/bin/env python3
"""
Pre-process all spectra in MassSpecGym_MurckoHist_split.hdf5 and store as processed_spectrum.
This eliminates the need for on-the-fly preprocessing in the DataLoader, dramatically speeding up training.

Operations pre-computed:
- Trim to top 100 peaks
- Pad to 100 peaks
- Convert to relative intensities
- Prepend precursor peak
- Ensure float32 precision
"""
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreams.utils import spectra as su
from dreams.utils.dformats import DataFormatA

def preprocess_spectra():
    hdf5_path = Path("dreams-thesis-wa/data/processed/MassSpecGym_MurckoHist_split.hdf5")
    
    if not hdf5_path.exists():
        print(f"Error: {hdf5_path} not found")
        return
    
    print(f"Opening {hdf5_path}...")
    with h5py.File(hdf5_path, 'a') as f:
        n_samples = len(f['spectrum'])
        print(f"Total samples: {n_samples}")
        
        # Check if processed_spectrum already exists
        if 'processed_spectrum' in f:
            print("✅ processed_spectrum already exists, skipping...")
            print(f"   Shape: {f['processed_spectrum'].shape}")
            return
        
        print("\nPre-processing spectra (trim, pad, normalize, prepend precursor)...")
        print("This will take a few minutes...")
        
        # Create dataset for processed spectra: (n_samples, 101, 2)
        # 101 = 100 peaks + 1 precursor peak at index 0
        processed_spectrum = f.create_dataset(
            'processed_spectrum',
            shape=(n_samples, 101, 2),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4,
            chunks=(100, 101, 2)  # Chunk for efficient access
        )
        
        # Load necessary data
        spectra_data = f['spectrum'][:]  # (n, 2, 128)
        prec_mz_data = f['precursor_mz'][:]  # (n,)
        
        dformat = DataFormatA()
        prec_intens = 1.1
        
        # Process in batches
        batch_size = 500
        for i in tqdm(range(0, n_samples, batch_size), desc="Processing spectra"):
            batch_end = min(i + batch_size, n_samples)
            
            for j in range(i, batch_end):
                spec = spectra_data[j].copy()  # (2, 128)
                prec_mz = prec_mz_data[j]
                
                # Convert from (2, n_peaks) to (n_peaks, 2)
                spec = spec.T
                
                # Trim to top 100 peaks
                spec = su.trim_peak_list(spec, 100).T
                
                # Pad to 100 peaks
                spec = su.pad_peak_list(spec, target_len=100).T
                
                # Convert to relative intensities
                spec = su.to_rel_intensity(spec.T).T
                
                # Prepend precursor peak
                spec = su.prepend_precursor_peak(spec, prec_mz, prec_intens, high=True)
                
                # Ensure float32
                spec = spec.astype(np.float32, copy=False)
                
                # Store (101, 2) = precursor + 100 peaks
                processed_spectrum[j] = spec
        
        print("\n✅ Successfully added processed_spectrum dataset!")
        print(f"   Shape: {processed_spectrum.shape}")
        print(f"   Dtype: {processed_spectrum.dtype}")
        print(f"   First sample shape: {processed_spectrum[0].shape}")

if __name__ == '__main__':
    preprocess_spectra()
