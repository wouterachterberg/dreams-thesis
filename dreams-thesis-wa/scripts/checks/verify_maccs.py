#!/usr/bin/env python3
"""Quick check of MACCS fingerprint generation and bit density."""
import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem

hdf5_path = 'dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5'
with h5py.File(hdf5_path, 'r') as f:
    print('Datasets in HDF5:', list(f.keys()))
    print(f'fp_maccs_166 precomputed: {"fp_maccs_166" in f}')
    print(f'fp_morgan_2048 precomputed: {"fp_morgan_2048" in f}')

    smiles_raw = f['smiles'][:]
    smiles = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in smiles_raw]
    n_total = len(smiles)
    print(f'Total molecules: {n_total}')

    folds = None
    if 'fold' in f:
        folds_raw = f['fold'][:]
        folds = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in folds_raw]
        unique_folds = sorted(set(folds))
        print(f'Folds: {unique_folds}')
        for fold in unique_folds:
            count = sum(1 for x in folds if x == fold)
            print(f'  {fold}: {count} molecules')

    # Compute MACCS on all molecules
    print('\n--- MACCS 166-bit ---')
    maccs_fps = []
    failed = 0
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.array(fp, dtype=np.uint8)[1:]  # skip bit 0 (unused)
            maccs_fps.append(arr)
        else:
            failed += 1
            maccs_fps.append(np.zeros(166, dtype=np.uint8))
    maccs_fps = np.array(maccs_fps)
    print(f'Valid molecules: {n_total - failed}/{n_total} (failed: {failed})')
    print(f'Shape: {maccs_fps.shape}')
    print(f'Overall bit density: {maccs_fps.mean():.4f} ({maccs_fps.mean()*100:.2f}%)')
    print(f'Avg bits set per molecule: {maccs_fps.sum(axis=1).mean():.1f} / 166')

    if folds:
        for fold in unique_folds:
            mask = np.array([f == fold for f in folds])
            subset = maccs_fps[mask]
            density = subset.mean()
            print(f'  {fold}: density={density:.4f} ({density*100:.2f}%), avg bits={subset.sum(axis=1).mean():.1f}')

    # Compare with Morgan ECFP4
    print('\n--- Morgan/ECFP4 2048-bit (comparison) ---')
    morgan_fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            morgan_fps.append(np.array(fp, dtype=np.uint8))
        else:
            morgan_fps.append(np.zeros(2048, dtype=np.uint8))
    morgan_fps = np.array(morgan_fps)
    print(f'Overall bit density: {morgan_fps.mean():.4f} ({morgan_fps.mean()*100:.2f}%)')
    print(f'Avg bits set per molecule: {morgan_fps.sum(axis=1).mean():.1f} / 2048')
