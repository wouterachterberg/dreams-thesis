#!/usr/bin/env python3
"""
Build a Murcko-scaffold-aware subset from the MassSpecGym HDF5 file.

Features:
- Computes Murcko scaffolds for all SMILES
- Exports scaffold statistics to CSV
- Optional per-scaffold capping (train/val separately) to create a smaller subset
- Preserves existing `fold` labels (train/val) so Murcko-disjoint split stays intact
- Writes a new HDF5 subset (when sampling) and adds a `scaffold` dataset

Usage examples:
  # Just compute scaffold stats
  python build_murcko_subset.py \
    dreams-thesis-wa/data/processed/MassSpecGym_MurckoHist_split.hdf5

  # Cap at 200 spectra per scaffold (train and val separately) and write subset
  python build_murcko_subset.py \
    dreams-thesis-wa/data/processed/MassSpecGym_MurckoHist_split.hdf5 \
    --max-per-scaffold 200 \
    --output dreams-thesis-wa/data/processed/MassSpecGym_MurckoHist_subset.hdf5
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


def smiles_to_scaffold(smiles: str) -> str:
    """Return Murcko scaffold SMILES (non-isomeric). Empty string on failure."""
    try:
        if not smiles or smiles.strip() == "" or smiles == "nan":
            return ""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, isomericSmiles=False)
    except Exception:
        return ""


def compute_scaffolds(smiles_list: List[str]) -> List[str]:
    scaffolds = []
    for smi in tqdm(smiles_list, desc="Computing Murcko scaffolds"):
        scaffolds.append(smiles_to_scaffold(smi))
    return scaffolds


def scaffold_stats(scaffolds: List[str], folds: List[str]) -> List[Tuple[str, int, int, int]]:
    """Return list of (scaffold, total, train, val)."""
    stats: Dict[str, Dict[str, int]] = {}
    for scaf, fold in zip(scaffolds, folds):
        if scaf not in stats:
            stats[scaf] = {"total": 0, "train": 0, "val": 0}
        stats[scaf]["total"] += 1
        if "train" in fold:
            stats[scaf]["train"] += 1
        elif "val" in fold:
            stats[scaf]["val"] += 1
    rows = [(k, v["total"], v["train"], v["val"]) for k, v in stats.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def write_stats_csv(rows: List[Tuple[str, int, int, int]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scaffold", "total", "train", "val"])
        for r in rows:
            writer.writerow(r)


def cap_indices_by_scaffold(scaffolds: List[str], folds: List[str], max_per_scaffold: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep_indices: List[int] = []

    # Separate by fold to keep Murcko-disjoint split intact
    for fold_name in ("train", "val"):
        by_scaf: Dict[str, List[int]] = {}
        for idx, (scaf, fold) in enumerate(zip(scaffolds, folds)):
            if fold_name not in fold:
                continue
            by_scaf.setdefault(scaf, []).append(idx)
        for scaf, idxs in by_scaf.items():
            if len(idxs) <= max_per_scaffold:
                keep_indices.extend(idxs)
            else:
                keep_indices.extend(rng.choice(idxs, size=max_per_scaffold, replace=False))

    keep_indices = np.array(sorted(keep_indices), dtype=np.int64)
    return keep_indices


def write_subset_hdf5(in_path: Path, out_path: Path, keep_idx: np.ndarray, scaffolds: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
        for key in fin.keys():
            data = fin[key][...]
            if data.shape[0] != len(scaffolds):
                raise ValueError(f"Dataset {key} has unexpected leading dimension {data.shape[0]}")
            fout.create_dataset(key, data=data[keep_idx], compression="gzip")
        # Add scaffold dataset
        scaffold_arr = np.array(scaffolds, dtype=object)[keep_idx]
        fout.create_dataset("scaffold", data=scaffold_arr.astype("S"), compression="gzip")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute Murcko scaffolds and optionally build a capped subset")
    ap.add_argument("input_hdf5", type=str, help="Path to input HDF5 (MassSpecGym with fold column)")
    ap.add_argument("--output", type=str, default=None, help="Output HDF5 (if sampling). Default: no subset written")
    ap.add_argument("--max-per-scaffold", type=int, default=None, help="Cap per scaffold (applied separately to train and val)")
    ap.add_argument("--seed", type=int, default=3407, help="Random seed for sampling")
    args = ap.parse_args()

    in_path = Path(args.input_hdf5)
    if not in_path.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {in_path}")

    print(f"📖 Reading HDF5: {in_path}")
    with h5py.File(in_path, "r") as f:
        smiles_raw = f["smiles"][:]
        fold_raw = f["fold"][:]

    smiles = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in smiles_raw]
    folds = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in fold_raw]

    print(f"📊 Total entries: {len(smiles)}")

    scaffolds = compute_scaffolds(smiles)
    stats_rows = scaffold_stats(scaffolds, folds)

    stats_csv = in_path.with_name(in_path.stem + "_scaffold_stats.csv")
    write_stats_csv(stats_rows, stats_csv)
    print(f"✅ Scaffold stats written to: {stats_csv}")

    if args.max_per_scaffold is None:
        print("No sampling requested (--max-per-scaffold not set). Done.")
        exit(0)

    keep_idx = cap_indices_by_scaffold(scaffolds, folds, args.max_per_scaffold, args.seed)
    print(f"✅ Selected {len(keep_idx)} entries with cap {args.max_per_scaffold} per scaffold (per fold)")

    if args.output is None:
        raise ValueError("--output is required when sampling (max-per-scaffold set)")

    out_path = Path(args.output)
    write_subset_hdf5(in_path, out_path, keep_idx, scaffolds)
    print(f"✅ Subset written to: {out_path}")
