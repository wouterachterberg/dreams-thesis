#!/usr/bin/env python3
"""
Create fingerprint_cache.npz from a finetuning HDF5 file.

Outputs keys expected by frozen baselines:
- morgan_fps: [N, 2048]
- maccs_fps:  [N, 166]
- map4_fps:   [N, 2048]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem
from tqdm import tqdm

from dreams.utils.mols import morgan_fp, maccs_fp, map4_fp


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Create fingerprint cache for frozen baselines.")
    parser.add_argument(
        "--finetuning-hdf5",
        type=Path,
        default=root / "dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5",
        help="Input finetuning HDF5 containing a smiles dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "dreams-thesis-wa/data/processed/MassSpecGym_splits/fingerprint_cache.npz",
        help="Output .npz path.",
    )
    return parser.parse_args()


def decode_smiles(values: np.ndarray) -> list[str]:
    out = []
    for s in values:
        if isinstance(s, (bytes, np.bytes_)):
            out.append(s.decode("utf-8"))
        else:
            out.append(str(s))
    return out


def main() -> None:
    args = parse_args()

    if not args.finetuning_hdf5.is_file():
        raise FileNotFoundError(f"Missing finetuning HDF5: {args.finetuning_hdf5}")

    with h5py.File(args.finetuning_hdf5, "r") as f:
        if "smiles" not in f:
            raise KeyError(f"Dataset 'smiles' not found in: {args.finetuning_hdf5}")
        smiles = decode_smiles(f["smiles"][:])

    n = len(smiles)
    morgan = np.zeros((n, 2048), dtype=np.float32)
    maccs = np.zeros((n, 166), dtype=np.float32)
    map4 = np.zeros((n, 2048), dtype=np.float32)

    failed = 0
    for i, s in enumerate(tqdm(smiles, desc="Computing fingerprints")):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            failed += 1
            continue

        morgan[i] = morgan_fp(mol, binary=True, fp_size=2048, radius=2, as_numpy=True).astype(np.float32, copy=False)
        maccs[i] = maccs_fp(mol, as_numpy=True).astype(np.float32, copy=False)
        map4[i] = map4_fp(mol, fp_size=2048).astype(np.float32, copy=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, morgan_fps=morgan, maccs_fps=maccs, map4_fps=map4)

    print(f"Saved fingerprint cache: {args.output}")
    print(f"Rows: {n}")
    print(f"Failed SMILES: {failed}")


if __name__ == "__main__":
    main()
