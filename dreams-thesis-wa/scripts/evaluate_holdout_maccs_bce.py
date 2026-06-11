#!/usr/bin/env python3
"""One-off Holdout evaluation for the finalized fine-tuned MACCS-BCE model.

This is inference-only and intentionally targets only:
dreams-thesis-wa/results/model_runs/maccs_166_bce
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "dreams-thesis-wa" / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from h100_batch_inference import infer_batches, load_model_compat, parse_spectrum_strings


SEED = 3407
FP_KIND = "maccs_166"
RUN_TAG = "maccs_166_bce"
MODEL_RUN_DIR = ROOT / "dreams-thesis-wa" / "results" / "model_runs" / RUN_TAG
ARTIFACTS_DIR = MODEL_RUN_DIR / "axis2_artifacts"
RETRIEVAL_DIR = ARTIFACTS_DIR / "retrieval"
CHECKPOINT_DIR = MODEL_RUN_DIR / "checkpoints"

HOLDOUT_PARQUET = ROOT / "dreams-thesis-wa" / "data" / "processed" / "MassSpecGym_splits" / "holdout.parquet"
FULL_HDF5 = ROOT / "dreams-thesis-wa" / "data" / "processed" / "MassSpecGym_splits" / "full.hdf5"

EXPECTED_HOLDOUT_SPECTRA = 26648
EXPECTED_HOLDOUT_UNIQUE_MOLECULES = 3984
FIXED_OOD_OPTIMAL_TAU = 0.27
INFERENCE_BATCH_SIZE_BY_DEVICE = {
    "cuda": 1024,
    "mps": 64,
    "cpu": 32,
}

METRICS_OUT = ARTIFACTS_DIR / "holdout_metrics_maccs_bce.csv"
PER_SPECTRUM_OUT = RETRIEVAL_DIR / "holdout_per_spectrum_ranks.csv"


def set_inference_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_val_loss_from_name(path: Path) -> float | None:
    match = re.search(r"val_loss=([0-9]*\.?[0-9]+)", path.name)
    if match is None:
        return None
    return float(match.group(1))


def resolve_best_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = []
    for path in checkpoint_dir.glob("*.ckpt"):
        if path.name == "last.ckpt":
            continue
        val_loss = parse_val_loss_from_name(path)
        if val_loss is not None:
            candidates.append((val_loss, path))

    if not candidates:
        raise FileNotFoundError(f"No validation-loss checkpoint found in {checkpoint_dir}")

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def decode_array(values: np.ndarray) -> np.ndarray:
    return np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def locate_holdout_label() -> str:
    with h5py.File(FULL_HDF5, "r") as handle:
        folds = decode_array(handle["fold"][:])
        smiles = decode_array(handle["smiles"][:])

    matches = []
    for label in sorted(set(folds.tolist())):
        mask = folds == label
        n_spectra = int(mask.sum())
        n_unique = int(pd.Series(smiles[mask]).nunique())
        if n_spectra == EXPECTED_HOLDOUT_SPECTRA and n_unique == EXPECTED_HOLDOUT_UNIQUE_MOLECULES:
            matches.append(label)

    if len(matches) != 1:
        counts = {
            label: {
                "spectra": int((folds == label).sum()),
                "unique_smiles": int(pd.Series(smiles[folds == label]).nunique()),
            }
            for label in sorted(set(folds.tolist()))
        }
        raise AssertionError(f"Could not uniquely locate Holdout fold label. Counts: {counts}")

    return matches[0]


def load_holdout_dataframe(holdout_label: str) -> pd.DataFrame:
    df = pd.read_parquet(HOLDOUT_PARQUET).reset_index(drop=True)
    split_cols = [c for c in df.columns if c.lower() in {"fold", "split", "partition"}]
    if not split_cols:
        raise KeyError(f"No fold/split/partition column found in {HOLDOUT_PARQUET}")

    fold_col = "fold" if "fold" in split_cols else split_cols[0]
    labels = set(df[fold_col].astype(str).unique().tolist())
    if labels != {holdout_label}:
        raise AssertionError(
            f"{HOLDOUT_PARQUET.name} {fold_col} labels {sorted(labels)} do not match located label {holdout_label!r}"
        )

    n_spectra = int(len(df))
    n_unique = int(df["smiles"].astype(str).nunique())
    if n_spectra != EXPECTED_HOLDOUT_SPECTRA or n_unique != EXPECTED_HOLDOUT_UNIQUE_MOLECULES:
        raise AssertionError(
            "Holdout count assertion failed: "
            f"spectra={n_spectra}, unique_smiles={n_unique}; "
            f"expected {EXPECTED_HOLDOUT_SPECTRA}, {EXPECTED_HOLDOUT_UNIQUE_MOLECULES}"
        )
    return df


def fp_from_smiles(smiles: str, fp_kind: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)

    if fp_kind == "maccs_166":
        n_bits = 166
    else:
        raise ValueError(f"Unsupported fp kind: {fp_kind}")

    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)

    if fp_kind == "maccs_166":
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp, dtype=np.float32)[1:]

    raise ValueError(f"Unsupported fp kind: {fp_kind}")


def compute_per_bit_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for b in range(y_true.shape[1]):
        yt = y_true[:, b]
        yp = y_pred[:, b]
        freq = float(yt.mean())
        if yt.min() == yt.max():
            rows.append((b, np.nan, freq))
        else:
            rows.append((b, float(roc_auc_score(yt, yp)), freq))
    return pd.DataFrame(rows, columns=["bit_index", "auroc", "freq"])


def compute_rowwise_cosine_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    numer = np.sum(y_true * y_pred, axis=1)
    denom = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    sims = numer / np.maximum(denom, 1e-8)
    return {
        "cosine_sim_mean": float(np.mean(sims)),
        "cosine_sim_median": float(np.median(sims)),
    }


def sweep_thresholds(y_pred: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    y_true_u8 = y_true.astype(np.uint8)
    for t in thresholds:
        yb = (y_pred >= t).astype(np.uint8)
        tp = (yb & y_true_u8).sum(axis=1)
        fp = (yb & (1 - y_true_u8)).sum(axis=1)
        fn = ((1 - yb) & y_true_u8).sum(axis=1)
        union = ((yb | y_true_u8)).sum(axis=1)

        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0).mean()
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0).mean()
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        tanimoto = np.where(union > 0, tp / union, 0.0).mean()
        rows.append((float(t), float(precision), float(recall), float(f1), float(tanimoto)))
    return pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1", "tanimoto_mean"])


def build_library(smiles: list[str], fp_kind: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    uniq = list(dict.fromkeys(smiles))
    smi_to_idx = {s: i for i, s in enumerate(uniq)}
    lib = np.stack([fp_from_smiles(s, fp_kind) for s in uniq]).astype(np.float32)
    spec_to_mol = np.array([smi_to_idx[s] for s in smiles], dtype=np.int32)
    return lib, spec_to_mol, uniq


def compute_ranks(y_pred: np.ndarray, lib: np.ndarray, spec_to_mol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sims = cosine_similarity(y_pred, lib)
    ranks = np.zeros((len(y_pred),), dtype=np.int32)
    true_sims = np.zeros((len(y_pred),), dtype=np.float32)
    rank_percentiles = np.zeros((len(y_pred),), dtype=np.float32)
    library_size = lib.shape[0]
    for i in range(len(y_pred)):
        s = sims[i, spec_to_mol[i]]
        ranks[i] = int((sims[i] > s).sum() + 1)
        true_sims[i] = float(s)
        rank_percentiles[i] = float((sims[i] < s).sum() / library_size)
    return ranks, true_sims, rank_percentiles


def retrieval_metrics(ranks: np.ndarray, library_size: int, rank_percentiles: np.ndarray) -> dict[str, Any]:
    out = {
        "library_size": library_size,
        "n_spectra": int(len(ranks)),
        "acc@1": float((ranks <= 1).mean()),
        "acc@5": float((ranks <= 5).mean()),
        "acc@10": float((ranks <= 10).mean()),
        "acc@20": float((ranks <= 20).mean()),
        "acc@50": float((ranks <= 50).mean()),
        "acc@100": float((ranks <= 100).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "mean_rank_percentile": float(np.mean(rank_percentiles)),
    }
    return out


def build_seen_flags(true_smiles: pd.Series, true_blocks: pd.Series) -> pd.DataFrame:
    with h5py.File(FULL_HDF5, "r") as handle:
        folds = decode_array(handle["fold"][:])
        smiles = decode_array(handle["smiles"][:])
        inchikeys = decode_array(handle["inchikey"][:])

    inchikey_blocks = pd.Series(inchikeys).astype(str).str.split("-", n=1).str[0].to_numpy()
    train_mask = folds == "train"
    val_mask = folds == "val"

    train_smiles = set(smiles[train_mask].tolist())
    val_smiles = set(smiles[val_mask].tolist())
    train_blocks = set(inchikey_blocks[train_mask].tolist())
    val_blocks = set(inchikey_blocks[val_mask].tolist())

    return pd.DataFrame(
        {
            "seen_smiles_train": true_smiles.astype(str).isin(train_smiles).astype(bool),
            "seen_smiles_val": true_smiles.astype(str).isin(val_smiles).astype(bool),
            "seen_inchikey_block_train": true_blocks.astype(str).isin(train_blocks).astype(bool),
            "seen_inchikey_block_val": true_blocks.astype(str).isin(val_blocks).astype(bool),
        }
    )


def main() -> None:
    set_inference_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    best_ckpt = resolve_best_checkpoint(CHECKPOINT_DIR)
    holdout_label = locate_holdout_label()
    df = load_holdout_dataframe(holdout_label)

    print(f"Resolved Holdout fold label: {holdout_label}")
    print(
        "Holdout count assertion passed: "
        f"{len(df):,} spectra, {df['smiles'].nunique():,} unique molecules"
    )
    print(f"Best checkpoint by validation loss: {best_ckpt}")
    print(f"Device: {device}")
    inference_batch_size = INFERENCE_BATCH_SIZE_BY_DEVICE[device]
    print(f"Inference batch size: {inference_batch_size}")

    t0 = time.perf_counter()
    model = load_model_compat(best_ckpt, device)
    model.eval()

    spec_holdout = np.stack(
        [parse_spectrum_strings(m, i) for m, i in zip(df["mzs"], df["intensities"], strict=False)],
        axis=0,
    )
    prec_holdout = pd.to_numeric(df["precursor_mz"], errors="coerce").to_numpy(dtype=np.float32)

    y_pred = infer_batches(
        model=model,
        spectra_np=spec_holdout,
        batch_size=inference_batch_size,
        device=device,
        apply_sigmoid_to_pred=True,
        progress_desc="maccs_166_bce | Holdout inference",
        prec_mz_np=prec_holdout,
        amp_dtype="none",
    )
    del model

    y_true = np.stack(
        [fp_from_smiles(s, FP_KIND) for s in tqdm(df["smiles"].astype(str).tolist(), desc="GT Holdout MACCS")],
        axis=0,
    ).astype(np.float32)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Holdout dimension mismatch: pred={y_pred.shape}, true={y_true.shape}")

    cos_stats = compute_rowwise_cosine_stats(y_true, y_pred)
    au = compute_per_bit_auroc(y_true, y_pred)
    thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)
    sweep = sweep_thresholds(y_pred, y_true, thresholds)
    best = sweep.loc[sweep["tanimoto_mean"].idxmax()]
    fixed_row = sweep[np.isclose(sweep["threshold"], FIXED_OOD_OPTIMAL_TAU, atol=1e-12)]
    if fixed_row.empty:
        raise AssertionError(f"Fixed OOD tau {FIXED_OOD_OPTIMAL_TAU} missing from threshold grid")
    fixed_tanimoto = float(fixed_row.iloc[0]["tanimoto_mean"])

    smiles = df["smiles"].astype(str).tolist()
    lib, spec_to_mol, unique_smiles = build_library(smiles, FP_KIND)
    if len(unique_smiles) != EXPECTED_HOLDOUT_UNIQUE_MOLECULES:
        raise AssertionError(f"Retrieval library size {len(unique_smiles)} != {EXPECTED_HOLDOUT_UNIQUE_MOLECULES}")
    ranks, true_sims, rank_percentiles = compute_ranks(y_pred, lib, spec_to_mol)
    retrieval = retrieval_metrics(ranks, len(lib), rank_percentiles)

    true_blocks = df["inchikey"].astype(str).str.split("-", n=1).str[0]
    seen_flags = build_seen_flags(df["smiles"], true_blocks)
    per_spectrum = pd.DataFrame(
        {
            "spectrum_id": df["identifier"].astype(str),
            "true_inchikey_block": true_blocks,
            "rank": ranks.astype(int),
            "reciprocal_rank": (1.0 / ranks).astype(float),
            "hit@1": (ranks <= 1).astype(bool),
            "hit@5": (ranks <= 5).astype(bool),
            "hit@10": (ranks <= 10).astype(bool),
            "rank_percentile": rank_percentiles.astype(float),
            "true_similarity": true_sims.astype(float),
            "adduct": df["adduct"].astype(str) if "adduct" in df.columns else "",
            "original_fold": df["original_fold"].astype(str) if "original_fold" in df.columns else "",
        }
    )
    per_spectrum = pd.concat([per_spectrum, seen_flags], axis=1)

    valid_aurocs = au["auroc"].dropna()
    metric_row = {
        "run_tag": RUN_TAG,
        "model_variant": "Fine-tuned",
        "fingerprint": "MACCS",
        "loss": "BCE",
        "split": "holdout",
        "fold_label": holdout_label,
        "seed": SEED,
        "max_peaks": 100,
        "inference_batch_size": int(inference_batch_size),
        "checkpoint": str(best_ckpt),
        "checkpoint_val_loss": parse_val_loss_from_name(best_ckpt),
        "apply_sigmoid": True,
        "n_bits": int(y_true.shape[1]),
        "n_spectra": int(y_true.shape[0]),
        "library_size": int(len(lib)),
        "unique_molecules": int(len(unique_smiles)),
        "bit_density": float(y_true.mean()),
        "cosine_sim_mean": cos_stats["cosine_sim_mean"],
        "cosine_sim_median": cos_stats["cosine_sim_median"],
        "mean_per_bit_auroc": float(valid_aurocs.mean()),
        "median_per_bit_auroc": float(valid_aurocs.median()),
        "valid_bits": int(valid_aurocs.shape[0]),
        "best_tanimoto": float(best["tanimoto_mean"]),
        "optimal_tau": float(best["threshold"]),
        "tanimoto_at_ood_tau_0.27": fixed_tanimoto,
        "precision_at_optimal_tau": float(best["precision"]),
        "recall_at_optimal_tau": float(best["recall"]),
        "f1_at_optimal_tau": float(best["f1"]),
        "acc@1": retrieval["acc@1"],
        "acc@5": retrieval["acc@5"],
        "acc@10": retrieval["acc@10"],
        "acc@20": retrieval["acc@20"],
        "acc@50": retrieval["acc@50"],
        "acc@100": retrieval["acc@100"],
        "median_rank": retrieval["median_rank"],
        "mean_rank": retrieval["mean_rank"],
        "mrr": retrieval["mrr"],
        "mean_rank_percentile": retrieval["mean_rank_percentile"],
        "seconds_total": float(time.perf_counter() - t0),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metric_row]).to_csv(METRICS_OUT, index=False)
    per_spectrum.to_csv(PER_SPECTRUM_OUT, index=False)

    metadata = {
        "guardrails": {
            "inference_only": True,
            "training_performed": False,
            "other_conditions_touched": False,
            "retrieval_similarity": "cosine",
            "prediction_values": "continuous_sigmoid",
        },
        "outputs": {
            "metrics": str(METRICS_OUT),
            "per_spectrum_ranks": str(PER_SPECTRUM_OUT),
        },
    }
    print(json.dumps(metadata, indent=2))
    print(pd.DataFrame([metric_row]).to_string(index=False))


if __name__ == "__main__":
    main()
