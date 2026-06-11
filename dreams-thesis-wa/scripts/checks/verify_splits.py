import h5py
from pathlib import Path
from collections import Counter

DATA = Path('dreams-thesis-wa/data/processed/MassSpecGym_splits/full.hdf5')

with h5py.File(DATA, 'r') as hf:
    smiles = hf['smiles'][:]
    folds = hf['fold'][:]
    if isinstance(smiles[0], bytes):
        smiles = [s.decode() for s in smiles]
    if isinstance(folds[0], bytes):
        folds = [f.decode() for f in folds]
    
    # Spectra counts
    spectra_counts = Counter(folds)
    total_spectra = len(folds)
    print('=== SPECTRA-BASED SPLIT ===')
    for fold, count in spectra_counts.items():
        print(f'{fold}: {count:,} ({100*count/total_spectra:.1f}%)')
    print(f'Total: {total_spectra:,}')
    
    # Molecule counts
    mol_folds = {}
    for s, f in zip(smiles, folds):
        mol_folds[s] = f
    
    mol_counts = Counter(mol_folds.values())
    total_mol = len(mol_folds)
    print()
    print('=== MOLECULE-BASED SPLIT ===')
    for fold, count in mol_counts.items():
        print(f'{fold}: {count:,} ({100*count/total_mol:.1f}%)')
    print(f'Total: {total_mol:,}')
