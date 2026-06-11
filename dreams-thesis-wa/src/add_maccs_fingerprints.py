#!/usr/bin/env python3
"""
Add MACCS (166-bit) fingerprints to a MassSpecGym HDF5 file.

- Computes RDKit MACCS keys per molecule from `smiles`
- Stores them as a new dataset column `fp_maccs_166` (uint8)
- Reports bit density overall and by split (`fold`) when available

Note:
This script augments the HDF5 file with precomputed MACCS targets.
The current fine-tuning label path in `dreams.utils.data` computes `fp_*` labels
from SMILES on-the-fly; this file-level column is useful for inspection,
verification, and downstream analysis.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from tqdm import tqdm


def compute_maccs_166(smiles: str) -> np.ndarray:
    if not smiles or smiles == '' or str(smiles) == 'nan':
        return np.zeros(166, dtype=np.uint8)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(166, dtype=np.uint8)

    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.array(fp, dtype=np.uint8)
    return arr[1:]


def decode_utf8_column(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    if isinstance(values[0], (bytes, np.bytes_)):
        return np.array([v.decode('utf-8') for v in values], dtype=object)
    return values.astype(object, copy=False)


def report_density(fps: np.ndarray, folds: np.ndarray | None = None) -> None:
    overall_density = float(fps.mean())
    print(f"Overall MACCS bit density: {overall_density:.4f} ({overall_density * 100:.2f}%)")

    if folds is None:
        return

    for fold_name in sorted(set(folds.tolist())):
        mask = folds == fold_name
        if mask.sum() == 0:
            continue
        density = float(fps[mask].mean())
        print(f"  {fold_name:>8}: n={int(mask.sum()):>7}, density={density:.4f} ({density * 100:.2f}%)")


def add_maccs_to_hdf5(hdf5_path: Path, overwrite: bool = False) -> None:
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Input file not found: {hdf5_path}")

    print(f"Opening HDF5: {hdf5_path}")
    with h5py.File(hdf5_path, 'a') as f:
        if 'smiles' not in f:
            raise KeyError('Dataset must contain `smiles`.')

        smiles = decode_utf8_column(f['smiles'][:])
        n = len(smiles)
        print(f"Total molecules: {n}")

        if 'fp_maccs_166' in f:
            if not overwrite:
                print('`fp_maccs_166` already exists. Use --overwrite to recompute.')
                fps_existing = f['fp_maccs_166'][:]
                folds = decode_utf8_column(f['fold'][:]) if 'fold' in f else None
                report_density(fps_existing, folds)
                return
            del f['fp_maccs_166']

        fps = np.zeros((n, 166), dtype=np.uint8)
        for i, s in enumerate(tqdm(smiles, desc='Computing MACCS')):
            fps[i] = compute_maccs_166(str(s))

        f.create_dataset(
            'fp_maccs_166',
            data=fps,
            dtype=np.uint8,
            compression='gzip',
            compression_opts=4,
        )

        print('Saved dataset: fp_maccs_166')
        folds = decode_utf8_column(f['fold'][:]) if 'fold' in f else None
        report_density(fps, folds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add fp_maccs_166 to HDF5 and report density.')
    parser.add_argument('input_hdf5', type=Path, help='Path to input HDF5 file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite fp_maccs_166 if already present')
    args = parser.parse_args()

    add_maccs_to_hdf5(args.input_hdf5, overwrite=args.overwrite)
