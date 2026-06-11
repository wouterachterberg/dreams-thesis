#!/usr/bin/env python3
"""
Add MAP4 (2048-dim) fingerprints to a MassSpecGym HDF5 file.

- Computes MAP4 fingerprints per molecule from `smiles`
- Stores them as a new dataset column `fp_map4_2048` (float32)
- Reports basic statistics overall and by split (`fold`) when available

Requires:
    pip install map4

Usage:
    python add_map4_fingerprints.py data/processed/finetuning.hdf5
    python add_map4_fingerprints.py data/processed/finetuning.hdf5 --overwrite
    python add_map4_fingerprints.py data/processed/finetuning.hdf5 --fp_size 1024

Note:
    MAP4 fingerprints are MinHash-based and produce integer (not binary) vectors.
    They are stored as float32 for compatibility with the training pipeline.
    If the `map4` package is not available on Snellius, run this script locally
    and copy the updated HDF5 file to the cluster.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem
from tqdm import tqdm

try:
    from map4 import MAP4Calculator as MAP4
except ImportError:
    try:
        from map4 import MAP4
    except ImportError:
        MAP4 = None


def compute_map4(smiles: str, calculator) -> np.ndarray:
    """Compute MAP4 fingerprint from a SMILES string."""
    if not smiles or smiles == '' or str(smiles) == 'nan':
        return np.zeros(calculator.dimensions, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(calculator.dimensions, dtype=np.float32)

    try:
        fp = calculator.calculate(mol)
        return np.asarray(fp, dtype=np.float32)
    except Exception as e:
        print(f"  Warning: MAP4 failed for '{smiles}': {e}")
        return np.zeros(calculator.dimensions, dtype=np.float32)


def decode_utf8_column(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    if isinstance(values[0], (bytes, np.bytes_)):
        return np.array([v.decode('utf-8') for v in values], dtype=object)
    return values.astype(object, copy=False)


def report_stats(fps: np.ndarray, folds: np.ndarray | None = None) -> None:
    """Report basic statistics (MAP4 produces integer vectors, not binary)."""
    nonzero_frac = float((fps != 0).mean())
    print(f"Overall non-zero fraction: {nonzero_frac:.4f} ({nonzero_frac * 100:.2f}%)")
    print(f"Value range: [{fps.min():.1f}, {fps.max():.1f}]")
    print(f"Mean absolute value: {np.abs(fps).mean():.4f}")

    if folds is None:
        return

    for fold_name in sorted(set(folds.tolist())):
        mask = folds == fold_name
        if mask.sum() == 0:
            continue
        subset = fps[mask]
        nz = float((subset != 0).mean())
        print(f"  {fold_name:>8}: n={int(mask.sum()):>7}, non-zero={nz:.4f} ({nz * 100:.2f}%)")


def add_map4_to_hdf5(hdf5_path: Path, fp_size: int = 2048, overwrite: bool = False) -> None:
    if MAP4 is None:
        raise ImportError(
            'MAP4 fingerprint requested but `map4` package is not installed.\n'
            'Install with: pip install map4'
        )

    dataset_name = f'fp_map4_{fp_size}'

    if not hdf5_path.exists():
        raise FileNotFoundError(f"Input file not found: {hdf5_path}")

    print(f"Opening HDF5: {hdf5_path}")
    print(f"Target dataset: {dataset_name} (MAP4, {fp_size} dimensions)")

    calculator = MAP4(dimensions=fp_size, radius=2, is_folded=True)

    with h5py.File(hdf5_path, 'a') as f:
        if 'smiles' not in f:
            raise KeyError('Dataset must contain `smiles`.')

        smiles = decode_utf8_column(f['smiles'][:])
        n = len(smiles)
        print(f"Total molecules: {n}")

        if dataset_name in f:
            if not overwrite:
                print(f'`{dataset_name}` already exists. Use --overwrite to recompute.')
                fps_existing = f[dataset_name][:]
                folds = decode_utf8_column(f['fold'][:]) if 'fold' in f else None
                report_stats(fps_existing, folds)
                return
            del f[dataset_name]

        fps = np.zeros((n, fp_size), dtype=np.float32)
        for i, s in enumerate(tqdm(smiles, desc=f'Computing MAP4 ({fp_size}d)')):
            fps[i] = compute_map4(str(s), calculator)

        f.create_dataset(
            dataset_name,
            data=fps,
            dtype=np.float32,
            compression='gzip',
            compression_opts=4,
        )

        print(f'Saved dataset: {dataset_name}')
        folds = decode_utf8_column(f['fold'][:]) if 'fold' in f else None
        report_stats(fps, folds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add MAP4 fingerprints to HDF5 and report statistics.'
    )
    parser.add_argument('input_hdf5', type=Path, help='Path to input HDF5 file')
    parser.add_argument('--fp_size', type=int, default=2048, help='MAP4 fingerprint dimensions (default: 2048)')
    parser.add_argument('--overwrite', action='store_true', help=f'Overwrite if already present')
    args = parser.parse_args()

    add_map4_to_hdf5(args.input_hdf5, fp_size=args.fp_size, overwrite=args.overwrite)
