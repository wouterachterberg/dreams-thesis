from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


SCRIPT_FILE_DIR = Path(__file__).resolve().parent
if str(SCRIPT_FILE_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_FILE_DIR))

from config import REPO_ROOT

ROOT = REPO_ROOT
SCRIPT_DIR = ROOT / "dreams-thesis-wa/scripts"
SRC_DIR = ROOT / "dreams-thesis-wa/src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from h100_batch_inference import load_model_compat, predict_batch
from frozen_allpeaks_inference import (
    infer_with_head as infer_frozen_head_batches,
    load_head_checkpoint as load_frozen_head_checkpoint,
    load_ssl_model as load_frozen_ssl_model,
)


SEED = 3407
OUT_DIR = ROOT / "dreams-thesis-wa/results/axis3"
FIG_DIR = OUT_DIR / "figures"
PUBLIC_AXIS3_DIR = OUT_DIR / "specific"

MODEL_READY_HDF5 = OUT_DIR / "axis3_tier1_model_ready.hdf5"
RECORDS_PARQUET = OUT_DIR / "axis3_tier1_per_spectrum_records.parquet"
CLOSED_LIBRARY_PARQUET = OUT_DIR / "axis3_closed_reference_library.parquet"
CLOSED_LIBRARY_NPZ = OUT_DIR / "axis3_closed_reference_library.npz"
OPEN_LIBRARY_PARQUET = OUT_DIR / "axis3_open_reference_library.parquet"
OPEN_LIBRARY_NPZ = OUT_DIR / "axis3_open_reference_library.npz"
SEEN_FLAGS_CSV = OUT_DIR / "axis3_mac_compound_seen_flags.csv"

FINE_TUNED_RUN_DIR = ROOT / "dreams-thesis-wa/results/model_runs/maccs_166_bce"
FROZEN_RUN_DIR = ROOT / "dreams-thesis-wa/results/model_runs/maccs_166_bce_frozen"
TAU_TABLE = ROOT / "dreams-thesis-wa/results/axis2/figures/axis2_optimal_tau_precision_recall_table.csv"
AXIS2_AUROC_CSV = FINE_TUNED_RUN_DIR / "axis2_artifacts/per_bit_auroc/auroc_comparison.csv"

RESULTS_CSV = OUT_DIR / "axis3_tier1_retrieval_metrics.csv"
RESULTS_JSON = OUT_DIR / "axis3_tier1_retrieval_metrics.json"
PER_SPECTRUM_RETRIEVAL_CSV = OUT_DIR / "axis3_tier1_per_spectrum_retrieval_results.csv"
AXIS3_RETRIEVAL_METRICS_CSV = OUT_DIR / "axis3_retrieval_metrics.csv"
AXIS3_RECOVERY_CSV = OUT_DIR / "axis3_recovery.csv"
AXIS3_BY_ADDUCT_CSV = OUT_DIR / "axis3_by_adduct.csv"
AXIS3_BY_PEAKCOUNT_CSV = OUT_DIR / "axis3_by_peakcount.csv"
AXIS3_BY_SEEN_NOVEL_CSV = OUT_DIR / "axis3_by_seen_novel.csv"
AXIS3_PERBIT_AUROC_CSV = OUT_DIR / "axis3_perbit_auroc.csv"
AXIS3_SUBSTRUCTURE_TRANSFER_CSV = OUT_DIR / "axis3_substructure_transfer.csv"
AXIS3_ACC_CURVE_CSV = OUT_DIR / "axis3_acc_at_k_curve.csv"
AXIS3_SPECIFIC_RETRIEVAL_CSV = PUBLIC_AXIS3_DIR / "axis3_specific_retrieval_metrics.csv"
AXIS3_SPECIFIC_PER_SPECTRUM_CSV = PUBLIC_AXIS3_DIR / "axis3_specific_per_spectrum_ranks.csv"
AXIS3_SPECIFIC_PER_BIT_CSV = PUBLIC_AXIS3_DIR / "axis3_specific_per_bit_auroc.csv"
PRED_FINE_TUNED_NPY = OUT_DIR / "axis3_tier1_predictions_fine_tuned.npy"
PRED_FROZEN_NPY = OUT_DIR / "axis3_tier1_predictions_frozen.npy"
BIN_FINE_TUNED_NPY = OUT_DIR / "axis3_tier1_predictions_fine_tuned_binary.npy"
BIN_FROZEN_NPY = OUT_DIR / "axis3_tier1_predictions_frozen_binary.npy"
SSL_EMBEDDING_NPY = OUT_DIR / "axis3_tier1_ssl_embeddings_for_frozen_head.npy"
PER_BIT_AUROC_CSV = OUT_DIR / "axis3_tier1_per_bit_auroc_transfer.csv"
SUMMARY_TXT = OUT_DIR / "axis3_tier1_results_summary.txt"

DECOMP_ADDUCT_CSV = OUT_DIR / "axis3_tier1_decomposition_adduct.csv"
DECOMP_PEAK_CSV = OUT_DIR / "axis3_tier1_decomposition_peak_count_bin.csv"
DECOMP_SEEN_CSV = OUT_DIR / "axis3_tier1_decomposition_seen_novel.csv"
TRANSFER_CAVEAT_CSV = OUT_DIR / "axis3_tier1_seen_novel_transfer_caveat.csv"
ACC_CURVE_CSV = OUT_DIR / "axis3_tier1_acc_at_k_curve.csv"

FIG_ACC_CURVE = FIG_DIR / "axis3_tier1_acc_at_k_curve.pdf"
FIG_RANK_PERCENTILE = FIG_DIR / "axis3_tier1_open_rank_percentile_ecdf.pdf"
FIG_DECOMP = FIG_DIR / "axis3_tier1_decomposition_bars.pdf"
FIG_TRANSFER = FIG_DIR / "axis3_tier1_substructure_transfer_scatter.pdf"
FIG_AXIS3_ACC_CURVE = FIG_DIR / "axis3_acc_at_k_curves.pdf"
FIG_AXIS3_RANK_CDF = FIG_DIR / "axis3_rank_percentile_cdf.pdf"
FIG_AXIS3_TRANSFER = FIG_DIR / "axis3_substructure_transfer_scatter.pdf"
FIG_AXIS3_DECOMP = FIG_DIR / "axis3_decomposition_bars.pdf"

PROTON = 1.007276466812
SODIUM = 22.989218
NH3 = 17.026549101
H2O = 18.010564684
NH4 = NH3 + PROTON
MASS_PREFILTER_PPM = 10.0

ADDUCT_ORDER = ["[M+H]+", "[M+Na]+", "[M+NH4]+", "[M-H2O+H]+", "[M-NH3+H]+", "[M-2H2O+H]+"]
MODEL_LABELS = {
    "fine_tuned": "Fine-tuned",
    "frozen": "Frozen",
}
POOL_LABELS = {
    "closed": "Closed",
    "open": "Open",
    "open_prefilter_10ppm": "Open 10 ppm prefiltered",
}
PRIMARY_POOLS = ["closed", "open"]
MODEL_ORDER = ["fine_tuned", "frozen"]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value).strip()


def parse_val_loss(path: Path) -> float | None:
    match = re.search(r"val_loss=([0-9]*\.?[0-9]+)", path.name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def best_checkpoint_from_dir(run_dir: Path, allow_best_name: bool) -> Path:
    ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    ckpts = [path for path in ckpts if path.name != "last.ckpt"]
    with_loss = [(parse_val_loss(path), path) for path in ckpts]
    with_loss = [(loss, path) for loss, path in with_loss if loss is not None]
    if with_loss:
        with_loss.sort(key=lambda item: item[0])
        return with_loss[0][1]
    if allow_best_name:
        named_best = [path for path in ckpts if "best" in path.name.lower()]
        if named_best:
            named_best.sort(key=lambda path: path.name)
            return named_best[-1]
    raise FileNotFoundError(f"No usable checkpoint found in {run_dir / 'checkpoints'}")


def read_tau(run_tag: str) -> float:
    table = pd.read_csv(TAU_TABLE)
    rows = table.loc[table["run_tag"] == run_tag]
    if len(rows) != 1:
        raise ValueError(f"Expected one tau row for {run_tag}, found {len(rows)}")
    value = float(rows.iloc[0]["optimal_tau_val"])
    if not np.isfinite(value):
        raise ValueError(f"Tau for {run_tag} is not finite")
    return value


def load_axis3_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    records = pd.read_parquet(RECORDS_PARQUET)
    with h5py.File(MODEL_READY_HDF5, "r") as handle:
        spectra = handle["processed_spectrum"][:].astype(np.float32)
        frozen_ssl_input = spectra[:, 1:, :].astype(np.float32)
        true_bits = handle["fp_maccs_166"][:].astype(np.uint8)
        spectrum_ids = handle["spectrum_id"][:].astype(np.int64)
        charges = handle["charge"][:].astype(np.float32)
    if len(records) != spectra.shape[0]:
        raise ValueError("Record count does not match processed spectra")
    if not np.array_equal(records["spectrum_id"].to_numpy(dtype=np.int64), spectrum_ids):
        raise ValueError("Record order does not match HDF5 spectrum_id")
    return spectra, frozen_ssl_input, true_bits, charges, records


def load_library(parquet_path: Path, npz_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_parquet(parquet_path)
    npz = np.load(npz_path, allow_pickle=True)
    bits = npz["maccs_166"].astype(np.uint8)
    first_npz = [clean_text(value) for value in npz["first_block_inchikey"]]
    first_meta = meta["first_block_inchikey"].astype(str).tolist()
    if first_npz != first_meta:
        raise ValueError(f"Library order mismatch for {parquet_path}")
    if bits.shape != (len(meta), 166):
        raise ValueError(f"Unexpected library bit shape for {parquet_path}: {bits.shape}")
    return meta, bits


def infer_model(
    model_key: str,
    ckpt_path: Path,
    spectra: np.ndarray,
    charges: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model = load_model_compat(ckpt_path, device)
    outputs = []
    for start in tqdm(range(0, len(spectra), batch_size), desc=f"{MODEL_LABELS[model_key]} prediction"):
        end = min(start + batch_size, len(spectra))
        outputs.append(
            predict_batch(
                model=model,
                batch_spec_np=spectra[start:end],
                batch_charge_np=charges[start:end],
                device=device,
                apply_sigmoid_to_pred=True,
                amp_dtype="none",
            )
        )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0).astype(np.float32)


def compute_ssl_embeddings_for_frozen_head(
    spectra_for_ssl: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    ssl_model = load_frozen_ssl_model(device=device, n_highest_peaks=100)
    outputs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(spectra_for_ssl), batch_size), desc="Frozen SSL embedding"):
            end = min(start + batch_size, len(spectra_for_ssl))
            xb = torch.from_numpy(spectra_for_ssl[start:end].astype(np.float32)).to(device)
            emb = ssl_model.model(xb)
            outputs.append(emb[:, 0, :].detach().cpu().numpy().astype(np.float32))
    del ssl_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0).astype(np.float32)


def infer_frozen_embedding_head(
    ckpt_path: Path,
    spectra_for_ssl: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if SSL_EMBEDDING_NPY.exists():
        embeddings = np.load(SSL_EMBEDDING_NPY).astype(np.float32)
        if embeddings.shape != (len(spectra_for_ssl), 1024):
            raise ValueError(f"Cached SSL embeddings have unexpected shape {embeddings.shape}")
    else:
        embeddings = compute_ssl_embeddings_for_frozen_head(spectra_for_ssl, batch_size=batch_size, device=device)
        np.save(SSL_EMBEDDING_NPY, embeddings)
    head, _ = load_frozen_head_checkpoint(ckpt_path, device=device)
    pred = infer_frozen_head_batches(
        model=head,
        emb=embeddings,
        mask=None,
        apply_sigmoid=True,
        batch_size=batch_size,
        device=device,
    )
    del head
    if device == "cuda":
        torch.cuda.empty_cache()
    return pred.astype(np.float32)


def load_prediction_cache(path: Path, expected_rows: int) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path).astype(np.float32)
    if arr.shape != (expected_rows, 166):
        return None
    return arr


def cosine_similarity_matrix(pred_score: np.ndarray, cand_bits: np.ndarray) -> np.ndarray:
    pred = pred_score.astype(np.float32)
    cand = cand_bits.astype(np.float32)
    numerator = pred @ cand.T
    pred_norm = np.linalg.norm(pred, axis=1, keepdims=True)
    cand_norm = np.linalg.norm(cand, axis=1, keepdims=True).T
    denominator = pred_norm * cand_norm
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0,
    ).astype(np.float32)


def compute_ranks(
    sims: np.ndarray,
    true_keys: np.ndarray,
    candidate_keys: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    missing_rank_value: int | None = None,
) -> pd.DataFrame:
    rows = []
    n_queries = sims.shape[0]
    all_candidates = np.arange(sims.shape[1])
    for idx in range(n_queries):
        mask = candidate_mask[idx] if candidate_mask is not None else np.ones(sims.shape[1], dtype=bool)
        candidate_idx = all_candidates[mask]
        n_pool = int(len(candidate_idx))
        true_mask = mask & (candidate_keys == true_keys[idx])
        true_idx = all_candidates[true_mask]
        true_in_pool = bool(len(true_idx))
        if n_pool == 0 or not true_in_pool:
            rank_value = int(missing_rank_value) if missing_rank_value is not None else n_pool + 1
            rows.append(
                {
                    "query_index": idx,
                    "candidate_count": n_pool,
                    "true_in_pool": true_in_pool,
                    "true_similarity": np.nan,
                    "rank": rank_value,
                    "rank_percentile": 0.0,
                }
            )
            continue
        pool_sims = sims[idx, candidate_idx]
        true_sim = float(np.max(sims[idx, true_idx]))
        higher = int(np.sum(pool_sims > true_sim))
        lower = int(np.sum(pool_sims < true_sim))
        rows.append(
            {
                "query_index": idx,
                "candidate_count": n_pool,
                "true_in_pool": true_in_pool,
                "true_similarity": true_sim,
                "rank": higher + 1,
                "rank_percentile": lower / n_pool if n_pool else 0.0,
            }
        )
    return pd.DataFrame(rows)


def metric_row(model_key: str, pool_key: str, ranks: pd.DataFrame, tau: float) -> dict[str, Any]:
    rank_values = ranks["rank"].to_numpy(dtype=float)
    candidate_counts = ranks["candidate_count"].to_numpy(dtype=float)
    true_in_pool = ranks["true_in_pool"].to_numpy(dtype=bool)
    recoverable_ranks = rank_values[true_in_pool]
    valid_counts = candidate_counts[candidate_counts > 0]
    fixed_pool = len(set(candidate_counts.tolist())) == 1
    if fixed_pool:
        baseline_pool = float(candidate_counts[0])
        random_acc = {k: min(k, baseline_pool) / baseline_pool for k in [1, 5, 10]}
        random_median_rank = baseline_pool / 2.0
    else:
        random_acc = {
            k: float(
                np.mean(
                    [
                        min(k, count) / count if in_pool and count > 0 else 0.0
                        for count, in_pool in zip(candidate_counts, true_in_pool)
                    ]
                )
            )
            for k in [1, 5, 10]
        }
        random_median_rank = float(np.median(valid_counts / 2.0)) if len(valid_counts) else 0.0
    return {
        "model": model_key,
        "model_label": MODEL_LABELS[model_key],
        "pool": pool_key,
        "pool_label": POOL_LABELS[pool_key],
        "similarity": "cosine",
        "prediction_values": "continuous_sigmoid",
        "tau": tau,
        "n_queries": int(len(ranks)),
        "n_true_in_pool": int(true_in_pool.sum()),
        "true_in_pool_rate": float(np.mean(true_in_pool)),
        "pool_size": int(candidate_counts[0]) if fixed_pool else None,
        "candidate_count_min": int(np.min(candidate_counts)) if len(candidate_counts) else 0,
        "candidate_count_median": float(np.median(candidate_counts)) if len(candidate_counts) else 0.0,
        "candidate_count_max": int(np.max(candidate_counts)) if len(candidate_counts) else 0,
        "acc@1": float(np.mean(true_in_pool & (rank_values <= 1))),
        "acc@5": float(np.mean(true_in_pool & (rank_values <= 5))),
        "acc@10": float(np.mean(true_in_pool & (rank_values <= 10))),
        "median_rank": float(np.median(rank_values)),
        "median_rank_recoverable": float(np.median(recoverable_ranks)) if len(recoverable_ranks) else np.nan,
        "mrr": float(np.mean(np.where(true_in_pool, 1.0 / rank_values, 0.0))),
        "random_acc@1": float(random_acc[1]),
        "random_acc@5": float(random_acc[5]),
        "random_acc@10": float(random_acc[10]),
        "random_expected_median_rank": float(random_median_rank),
        "median_rank_percentile": float(np.median(ranks["rank_percentile"])),
        "mean_rank_percentile": float(np.mean(ranks["rank_percentile"])),
    }


def retrieval_metric_row(
    model_key: str,
    pool_key: str,
    ranks: pd.DataFrame,
    records: pd.DataFrame,
    stratum: str,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    sub = ranks.loc[row_mask].copy()
    sub_records = records.loc[row_mask].copy()
    if len(sub) == 0:
        return {
            "model": model_key,
            "model_label": MODEL_LABELS[model_key],
            "pool": pool_key,
            "pool_label": POOL_LABELS[pool_key],
            "stratum": stratum,
            "n_spectra": 0,
            "n_compounds": 0,
            "acc@1": np.nan,
            "acc@5": np.nan,
            "acc@10": np.nan,
            "median_rank": np.nan,
            "mrr": np.nan,
            "rank_pctile": np.nan,
            "mean_rank_percentile": np.nan,
            "median_rank_percentile": np.nan,
        }
    rank_values = sub["rank"].to_numpy(dtype=float)
    true_in_pool = sub["true_in_pool"].to_numpy(dtype=bool)
    rank_percentile = sub["rank_percentile"].to_numpy(dtype=float)
    return {
        "model": model_key,
        "model_label": MODEL_LABELS[model_key],
        "pool": pool_key,
        "pool_label": POOL_LABELS[pool_key],
        "stratum": stratum,
        "n_spectra": int(len(sub)),
        "n_compounds": int(sub_records["first_block_inchikey"].nunique()),
        "acc@1": float(np.mean(true_in_pool & (rank_values <= 1))),
        "acc@5": float(np.mean(true_in_pool & (rank_values <= 5))),
        "acc@10": float(np.mean(true_in_pool & (rank_values <= 10))),
        "median_rank": float(np.median(rank_values)),
        "mrr": float(np.mean(np.where(true_in_pool, 1.0 / rank_values, 0.0))),
        "rank_pctile": float(np.mean(rank_percentile)),
        "mean_rank_percentile": float(np.mean(rank_percentile)),
        "median_rank_percentile": float(np.median(rank_percentile)),
    }


def make_axis3_retrieval_metrics(
    records: pd.DataFrame,
    ranks_by_key: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    strata = [
        ("overall", np.ones(len(records), dtype=bool)),
        ("seen", records["training_status"].eq("Seen").to_numpy(dtype=bool)),
        ("novel", records["training_status"].eq("Novel").to_numpy(dtype=bool)),
    ]
    for model_key in MODEL_ORDER:
        for pool_key in PRIMARY_POOLS:
            ranks = ranks_by_key[(model_key, pool_key)]
            for stratum, mask in strata:
                rows.append(retrieval_metric_row(model_key, pool_key, ranks, records, stratum, mask))
    return pd.DataFrame(rows)


def make_stratified_metrics(
    records: pd.DataFrame,
    ranks_by_key: dict[tuple[str, str], pd.DataFrame],
    group_col: str,
    order: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    groups = order if order is not None else sorted(records[group_col].dropna().astype(str).unique())
    for model_key in MODEL_ORDER:
        for pool_key in PRIMARY_POOLS:
            ranks = ranks_by_key[(model_key, pool_key)]
            for group in groups:
                mask = records[group_col].eq(group).to_numpy(dtype=bool)
                if not np.any(mask):
                    continue
                row = retrieval_metric_row(model_key, pool_key, ranks, records, str(group), mask)
                row["grouping"] = group_col
                row["group"] = str(group)
                rows.append(row)
    return pd.DataFrame(rows)


def make_compound_recovery(
    records: pd.DataFrame,
    ranks_by_key: dict[tuple[str, str], pd.DataFrame],
    ks: list[int],
) -> pd.DataFrame:
    rows = []
    true_keys = records["first_block_inchikey"].astype(str).to_numpy()
    for model_key in MODEL_ORDER:
        for pool_key in PRIMARY_POOLS:
            ranks = ranks_by_key[(model_key, pool_key)]
            tmp = pd.DataFrame(
                {
                    "first_block_inchikey": true_keys,
                    "rank": ranks["rank"].to_numpy(dtype=float),
                    "true_in_pool": ranks["true_in_pool"].to_numpy(dtype=bool),
                }
            )
            n_compounds = int(tmp["first_block_inchikey"].nunique())
            for k in ks:
                recovered_by_compound = (
                    tmp.assign(recovered=lambda df: df["true_in_pool"] & (df["rank"] <= k))
                    .groupby("first_block_inchikey")["recovered"]
                    .any()
                )
                n_recovered = int(recovered_by_compound.sum())
                rows.append(
                    {
                        "model": model_key,
                        "model_label": MODEL_LABELS[model_key],
                        "pool": pool_key,
                        "pool_label": POOL_LABELS[pool_key],
                        "k": int(k),
                        "n_compounds": n_compounds,
                        "n_recovered": n_recovered,
                        "recovery_rate": float(n_recovered / n_compounds) if n_compounds else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def compound_recovery_rate(records: pd.DataFrame, ranks: pd.DataFrame, mask: np.ndarray, k: int) -> float:
    sub_records = records.loc[mask]
    sub_ranks = ranks.loc[mask]
    if len(sub_records) == 0:
        return np.nan
    recovered_by_compound = (
        pd.DataFrame(
            {
                "first_block_inchikey": sub_records["first_block_inchikey"].astype(str).to_numpy(),
                "recovered": sub_ranks["true_in_pool"].to_numpy(dtype=bool)
                & (sub_ranks["rank"].to_numpy(dtype=float) <= k),
            }
        )
        .groupby("first_block_inchikey")["recovered"]
        .any()
    )
    if len(recovered_by_compound) == 0:
        return np.nan
    return float(recovered_by_compound.mean())


def make_axis3_specific_retrieval_metrics(
    axis3_metrics_df: pd.DataFrame,
    records: pd.DataFrame,
    ranks_by_key: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    out = axis3_metrics_df.rename(
        columns={
            "stratum": "subset",
            "acc@1": "acc_at_1",
            "acc@5": "acc_at_5",
            "acc@10": "acc_at_10",
        }
    ).copy()
    out["subset"] = out["subset"].replace({"overall": "all"})
    subset_masks = {
        "all": np.ones(len(records), dtype=bool),
        "seen": records["training_status"].eq("Seen").to_numpy(dtype=bool),
        "novel": records["training_status"].eq("Novel").to_numpy(dtype=bool),
    }
    for k in [1, 5]:
        values = []
        for row in out.itertuples(index=False):
            ranks = ranks_by_key[(row.model, row.pool)]
            values.append(compound_recovery_rate(records, ranks, subset_masks[row.subset], k))
        out[f"compound_recovery_at_{k}"] = values
    return out[
        [
            "model",
            "pool",
            "subset",
            "n_spectra",
            "n_compounds",
            "acc_at_1",
            "acc_at_5",
            "acc_at_10",
            "median_rank",
            "mrr",
            "mean_rank_percentile",
            "compound_recovery_at_1",
            "compound_recovery_at_5",
        ]
    ]


def make_axis3_specific_per_spectrum_ranks(
    records: pd.DataFrame,
    ranks_by_key: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for idx, record in records.reset_index(drop=True).iterrows():
        seen_in_axis2 = record["training_status"] == "Seen"
        for model_key in MODEL_ORDER:
            for pool_key in PRIMARY_POOLS:
                ranks = ranks_by_key[(model_key, pool_key)]
                rank = int(ranks.iloc[idx]["rank"])
                true_in_pool = bool(ranks.iloc[idx]["true_in_pool"])
                rows.append(
                    {
                        "spectrum_id": int(record["spectrum_id"]),
                        "model": model_key,
                        "pool": pool_key,
                        "true_inchikey_block": str(record["first_block_inchikey"]),
                        "rank": rank,
                        "reciprocal_rank": float(1.0 / rank) if true_in_pool else 0.0,
                        "hit_at_1": bool(true_in_pool and rank <= 1),
                        "hit_at_5": bool(true_in_pool and rank <= 5),
                        "seen_in_axis2": bool(seen_in_axis2),
                        "adduct": str(record["adduct"]),
                        "n_peaks": int(record["peak_count_cleaned"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["spectrum_id", "model", "pool"], kind="stable").reset_index(drop=True)


def make_acc_curve(
    ranks_by_key: dict[tuple[str, str], pd.DataFrame],
    pool_sizes: dict[str, int],
    max_k: int,
) -> pd.DataFrame:
    rows = []
    for model_key in MODEL_ORDER:
        for pool_key in PRIMARY_POOLS:
            ranks = ranks_by_key[(model_key, pool_key)]
            true_in_pool = ranks["true_in_pool"].to_numpy(dtype=bool)
            rank_values = ranks["rank"].to_numpy(dtype=int)
            for k in range(1, max_k + 1):
                rows.append(
                    {
                        "model": model_key,
                        "model_label": MODEL_LABELS[model_key],
                        "pool": pool_key,
                        "pool_label": POOL_LABELS[pool_key],
                        "k": int(k),
                        "accuracy": float(np.mean(true_in_pool & (rank_values <= k))),
                        "random_accuracy": float(min(k, pool_sizes[pool_key]) / pool_sizes[pool_key]),
                    }
                )
    return pd.DataFrame(rows)


def adduct_to_neutral_mass(precursor_mz: float, adduct: str) -> float:
    if adduct == "[M+H]+":
        return precursor_mz - PROTON
    if adduct == "[M+Na]+":
        return precursor_mz - SODIUM
    if adduct == "[M+NH4]+":
        return precursor_mz - NH4
    if adduct == "[M-H2O+H]+":
        return precursor_mz + H2O - PROTON
    if adduct == "[M-NH3+H]+":
        return precursor_mz + NH3 - PROTON
    if adduct == "[M-2H2O+H]+":
        return precursor_mz + (2.0 * H2O) - PROTON
    return np.nan


def build_mass_prefilter(records: pd.DataFrame, library: pd.DataFrame) -> np.ndarray:
    query_mass = np.array(
        [adduct_to_neutral_mass(float(row.precursorMz), clean_text(row.adduct)) for row in records.itertuples()],
        dtype=np.float64,
    )
    candidate_mass = library["exactmass"].to_numpy(dtype=np.float64)
    mask = np.zeros((len(records), len(library)), dtype=bool)
    for idx, mass in enumerate(query_mass):
        if not np.isfinite(mass) or mass <= 0:
            continue
        tolerance = mass * MASS_PREFILTER_PPM * 1e-6
        mask[idx] = np.abs(candidate_mass - mass) <= tolerance
    return mask


def add_retrieval_columns(records: pd.DataFrame, ranks_by_key: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    out = records[[
        "spectrum_id",
        "compound",
        "base_compound_name",
        "adduct",
        "precursorMz",
        "exactmass",
        "peak_count_cleaned",
        "first_block_inchikey",
    ]].copy()
    for (model_key, pool_key), ranks in ranks_by_key.items():
        prefix = f"{model_key}_{pool_key}"
        out[f"{prefix}_rank"] = ranks["rank"].to_numpy(dtype=int)
        out[f"{prefix}_rank_percentile"] = ranks["rank_percentile"].to_numpy(dtype=float)
        out[f"{prefix}_true_similarity"] = ranks["true_similarity"].to_numpy(dtype=float)
        out[f"{prefix}_candidate_count"] = ranks["candidate_count"].to_numpy(dtype=int)
        out[f"{prefix}_true_in_pool"] = ranks["true_in_pool"].to_numpy(dtype=bool)
    return out


def peak_count_bin(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 5:
        return "4-5"
    if value <= 10:
        return "6-10"
    return ">10"


def decompose(records: pd.DataFrame, ranks: pd.DataFrame, group_col: str, order: list[str] | None = None) -> pd.DataFrame:
    df = records[[group_col]].copy()
    df["rank"] = ranks["rank"].to_numpy(dtype=int)
    df["true_in_pool"] = ranks["true_in_pool"].to_numpy(dtype=bool)
    rows = []
    groups = order if order is not None else sorted(df[group_col].dropna().astype(str).unique())
    for group in groups:
        sub = df.loc[df[group_col] == group]
        if len(sub) == 0:
            continue
        rank_values = sub["rank"].to_numpy(dtype=float)
        true_in_pool = sub["true_in_pool"].to_numpy(dtype=bool)
        rank_percentile = ranks.loc[sub.index, "rank_percentile"].to_numpy(dtype=float)
        rows.append(
            {
                "grouping": group_col,
                "group": group,
                "n_spectra": int(len(sub)),
                "acc@1": float(np.mean(true_in_pool & (rank_values <= 1))),
                "acc@5": float(np.mean(true_in_pool & (rank_values <= 5))),
                "acc@10": float(np.mean(true_in_pool & (rank_values <= 10))),
                "median_rank": float(np.median(rank_values)),
                "median_rank_percentile": float(np.median(rank_percentile)),
                "mean_rank_percentile": float(np.mean(rank_percentile)),
                "mrr": float(np.mean(np.where(true_in_pool, 1.0 / rank_values, 0.0))),
            }
        )
    return pd.DataFrame(rows)


def seen_novel_transfer_table(records: pd.DataFrame, ranks_by_key: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_key in ["fine_tuned", "frozen"]:
        ranks = ranks_by_key[(model_key, "open")]
        tmp = records[["training_status"]].copy()
        tmp["rank"] = ranks["rank"].to_numpy(dtype=float)
        tmp["rank_percentile"] = ranks["rank_percentile"].to_numpy(dtype=float)
        tmp["true_in_pool"] = ranks["true_in_pool"].to_numpy(dtype=bool)
        for status in ["Seen", "Novel", "Unknown"]:
            sub = tmp.loc[tmp["training_status"] == status]
            if len(sub) == 0:
                continue
            rank_values = sub["rank"].to_numpy(dtype=float)
            true_in_pool = sub["true_in_pool"].to_numpy(dtype=bool)
            rows.append(
                {
                    "model": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "training_status": status,
                    "n_spectra": int(len(sub)),
                    "acc@1": float(np.mean(true_in_pool & (rank_values <= 1))),
                    "acc@5": float(np.mean(true_in_pool & (rank_values <= 5))),
                    "acc@10": float(np.mean(true_in_pool & (rank_values <= 10))),
                    "median_rank": float(np.median(rank_values)),
                    "median_rank_percentile": float(np.median(sub["rank_percentile"])),
                    "mean_rank_percentile": float(np.mean(sub["rank_percentile"])),
                    "mrr": float(np.mean(np.where(true_in_pool, 1.0 / rank_values, 0.0))),
                }
            )
    return pd.DataFrame(rows)


def per_bit_auroc(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    rows = []
    for bit_idx in range(y_true.shape[1]):
        labels = y_true[:, bit_idx].astype(int)
        scores = y_score[:, bit_idx].astype(float)
        n_pos = int(labels.sum())
        n_neg = int(len(labels) - n_pos)
        if n_pos == 0 or n_neg == 0:
            auroc = np.nan
        else:
            auroc = float(roc_auc_score(labels, scores))
        rows.append(
            {
                "bit_index": bit_idx,
                "axis3_auroc": auroc,
                "axis3_frequency": float(np.mean(labels)),
                "axis3_n_positive": n_pos,
                "axis3_n_negative": n_neg,
            }
        )
    return pd.DataFrame(rows)


def make_perbit_auroc_table(true_bits: np.ndarray, pred_by_model: dict[str, np.ndarray]) -> pd.DataFrame:
    ft = per_bit_auroc(true_bits, pred_by_model["fine_tuned"]).rename(
        columns={
            "bit_index": "bit",
            "axis3_auroc": "auroc_mac_ft",
            "axis3_frequency": "freq_mac",
            "axis3_n_positive": "n_positive_mac",
            "axis3_n_negative": "n_negative_mac",
        }
    )
    frozen = per_bit_auroc(true_bits, pred_by_model["frozen"]).rename(
        columns={
            "bit_index": "bit",
            "axis3_auroc": "auroc_mac_frozen",
        }
    )[["bit", "auroc_mac_frozen"]]
    axis2 = pd.read_csv(AXIS2_AUROC_CSV)[["bit_index", "auroc_ood", "freq_ood"]].rename(
        columns={
            "bit_index": "bit",
            "auroc_ood": "auroc_axis2_ood",
            "freq_ood": "freq_axis2_ood",
        }
    )
    return ft.merge(frozen, on="bit", how="left").merge(axis2, on="bit", how="left")


def make_axis3_specific_per_bit_auroc(
    perbit_df: pd.DataFrame,
    true_bits: np.ndarray,
    fine_tuned_pred: np.ndarray,
    records: pd.DataFrame,
) -> pd.DataFrame:
    novel_mask = records["training_status"].eq("Novel").to_numpy(dtype=bool)
    novel_perbit = per_bit_auroc(true_bits[novel_mask], fine_tuned_pred[novel_mask]).rename(
        columns={"axis3_auroc": "mac_auroc_novel"}
    )
    return pd.DataFrame(
        {
            "bit_index": perbit_df["bit"].astype(int),
            "mac_auroc_all": perbit_df["auroc_mac_ft"].astype(float),
            "mac_auroc_novel": novel_perbit["mac_auroc_novel"].astype(float),
            "axis2_ood_auroc": perbit_df["auroc_axis2_ood"].astype(float),
            "mac_bit_prevalence": perbit_df["freq_mac"].astype(float),
        }
    )


def write_axis3_specific_exports(
    axis3_metrics_df: pd.DataFrame,
    records: pd.DataFrame,
    ranks_by_key: dict[tuple[str, str], pd.DataFrame],
    perbit_df: pd.DataFrame,
    true_bits: np.ndarray,
    pred_by_model: dict[str, np.ndarray],
) -> None:
    PUBLIC_AXIS3_DIR.mkdir(parents=True, exist_ok=True)
    make_axis3_specific_retrieval_metrics(axis3_metrics_df, records, ranks_by_key).to_csv(
        AXIS3_SPECIFIC_RETRIEVAL_CSV,
        index=False,
    )
    make_axis3_specific_per_spectrum_ranks(records, ranks_by_key).to_csv(
        AXIS3_SPECIFIC_PER_SPECTRUM_CSV,
        index=False,
    )
    make_axis3_specific_per_bit_auroc(
        perbit_df,
        true_bits,
        pred_by_model["fine_tuned"],
        records,
    ).to_csv(AXIS3_SPECIFIC_PER_BIT_CSV, index=False)


def make_substructure_transfer_table(perbit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_key, auroc_col in [("fine_tuned", "auroc_mac_ft"), ("frozen", "auroc_mac_frozen")]:
        valid = perbit_df.dropna(subset=["auroc_axis2_ood", auroc_col])
        if len(valid) >= 2:
            rho, p_value = spearmanr(valid["auroc_axis2_ood"], valid[auroc_col])
        else:
            rho, p_value = np.nan, np.nan
        rows.append(
            {
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "axis2_ood_column": "auroc_axis2_ood",
                "axis3_mac_column": auroc_col,
                "n_bits": int(len(valid)),
                "spearman_rho": float(rho),
                "spearman_p": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def save_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def plot_acc_curve(acc_curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    colours = {"fine_tuned": "#1b7f79", "frozen": "#b34748"}
    linestyles = {"closed": "-", "open": ":"}
    for (model_key, pool_key), sub in acc_curve.groupby(["model", "pool"]):
        label = f"{MODEL_LABELS[model_key]}, {POOL_LABELS[pool_key].lower()}"
        ax.plot(
            sub["k"],
            sub["accuracy"],
            marker="o",
            linewidth=1.8,
            color=colours[model_key],
            linestyle=linestyles[pool_key],
            label=label,
        )
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy at k")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(bottom=0)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(FIG_ACC_CURVE)
    plt.close(fig)


def plot_rank_percentile_ecdf(retrieval: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colours = {"fine_tuned": "#1b7f79", "frozen": "#b34748"}
    for model_key in ["fine_tuned", "frozen"]:
        values = retrieval[f"{model_key}_open_rank_percentile"].dropna().to_numpy(dtype=float)
        values = np.sort(values)
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", color=colours[model_key], linewidth=1.9, label=MODEL_LABELS[model_key])
    ax.set_xlabel("True compound rank percentile")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_RANK_PERCENTILE)
    plt.close(fig)


def plot_decompositions(adduct_df: pd.DataFrame, peak_df: pd.DataFrame, seen_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
    panels = [
        (adduct_df, "Adduct", axes[0]),
        (peak_df, "Cleaned peak count", axes[1]),
        (seen_df, "Axis 2 training status", axes[2]),
    ]
    for df, title, ax in panels:
        x = np.arange(len(df))
        ax.bar(x - 0.18, df["acc@1"], width=0.36, label="Accuracy at 1", color="#1b7f79")
        ax.bar(x + 0.18, df["acc@10"], width=0.36, label="Accuracy at 10", color="#e0a13a")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(df["group"], rotation=45, ha="right")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Accuracy")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG_DECOMP)
    plt.close(fig)


def plot_transfer_scatter(transfer_df: pd.DataFrame, rho: float, p_value: float) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    valid = transfer_df.dropna(subset=["axis2_ood_auroc", "axis3_auroc"])
    ax.scatter(valid["axis2_ood_auroc"], valid["axis3_auroc"], s=22, alpha=0.72, color="#4f6f91")
    ax.set_xlabel("Axis 2 OOD per-bit AUROC")
    ax.set_ylabel("Axis 3 MAC per-bit AUROC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.04,
        0.96,
        f"Spearman rho = {rho:.3f}\np = {p_value:.2g}\nvalid bits = {len(valid)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(FIG_TRANSFER)
    plt.close(fig)


def plot_axis3_acc_curves(acc_curve: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    colours = {"fine_tuned": "#0b6e4f", "frozen": "#6b7280"}
    widths = {"fine_tuned": 2.6, "frozen": 1.8}
    alphas = {"fine_tuned": 0.98, "frozen": 0.76}
    for ax, pool_key in zip(axes, PRIMARY_POOLS, strict=False):
        pool_df = acc_curve.loc[acc_curve["pool"] == pool_key]
        for model_key in MODEL_ORDER:
            sub = pool_df.loc[pool_df["model"] == model_key]
            ax.plot(
                sub["k"],
                sub["accuracy"],
                marker="o",
                markersize=3.8,
                linewidth=widths[model_key],
                alpha=alphas[model_key],
                color=colours[model_key],
                label=MODEL_LABELS[model_key],
            )
        ax.set_title(POOL_LABELS[pool_key])
        ax.set_xlabel("k")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(1, 20)
        ax.grid(alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel("Accuracy@k")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_AXIS3_ACC_CURVE)
    plt.close(fig)


def plot_axis3_rank_cdf(ranks_by_key: dict[tuple[str, str], pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colours = {"fine_tuned": "#0b6e4f", "frozen": "#6b7280"}
    linestyles = {"closed": "-", "open": (0, (4, 2))}
    for pool_key in PRIMARY_POOLS:
        for model_key in MODEL_ORDER:
            values = ranks_by_key[(model_key, pool_key)]["rank_percentile"].dropna().to_numpy(dtype=float)
            values = np.sort(values)
            y = np.arange(1, len(values) + 1) / len(values)
            ax.step(
                values,
                y,
                where="post",
                color=colours[model_key],
                linestyle=linestyles[pool_key],
                linewidth=2.2 if model_key == "fine_tuned" else 1.8,
                alpha=0.95 if model_key == "fine_tuned" else 0.78,
                label=f"{MODEL_LABELS[model_key]}, {POOL_LABELS[pool_key].lower()}",
            )
    ax.set_xlabel("True compound rank percentile")
    ax.set_ylabel("Cumulative fraction of spectra")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(FIG_AXIS3_RANK_CDF)
    plt.close(fig)


def plot_axis3_transfer_scatter(perbit_df: pd.DataFrame, transfer_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    series = [
        ("fine_tuned", "auroc_mac_ft", "#0b6e4f"),
        ("frozen", "auroc_mac_frozen", "#6b7280"),
    ]
    for model_key, column, colour in series:
        valid = perbit_df.dropna(subset=["auroc_axis2_ood", column])
        ax.scatter(
            valid["auroc_axis2_ood"],
            valid[column],
            s=24,
            alpha=0.68 if model_key == "fine_tuned" else 0.52,
            color=colour,
            label=MODEL_LABELS[model_key],
        )
    ax.set_xlabel("Axis 2 OOD per-bit AUROC")
    ax.set_ylabel("Axis 3 MAC per-bit AUROC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    annotation = []
    for row in transfer_df.itertuples(index=False):
        annotation.append(f"{row.model_label}: rho = {row.spearman_rho:.3f}, p = {row.spearman_p:.2g}")
    ax.text(
        0.04,
        0.96,
        "\n".join(annotation),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_AXIS3_TRANSFER)
    plt.close(fig)


def plot_axis3_decomposition(adduct_df: pd.DataFrame, peak_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    panels = [
        (peak_df, "Cleaned peak count", axes[0]),
        (adduct_df, "Adduct", axes[1]),
    ]
    for df, title, ax in panels:
        sub = df.loc[(df["model"] == "fine_tuned") & (df["pool"] == "open")].copy()
        x = np.arange(len(sub))
        ax.bar(x, sub["acc@1"], color="#0b6e4f", alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["group"], rotation=45, ha="right")
        ax.set_ylabel("Accuracy@1")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIG_AXIS3_DECOMP)
    plt.close(fig)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    if not MODEL_READY_HDF5.exists() or not RECORDS_PARQUET.exists():
        raise FileNotFoundError("Axis 3 tier-1 preparation artefacts are missing")
    fine_ckpt = best_checkpoint_from_dir(FINE_TUNED_RUN_DIR, allow_best_name=False)
    frozen_ckpt = best_checkpoint_from_dir(FROZEN_RUN_DIR, allow_best_name=True)
    tau_by_model = {
        "fine_tuned": read_tau("maccs_166_bce"),
        "frozen": read_tau("maccs_166_bce_frozen"),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    batch_size = 256 if device == "cuda" else 64

    spectra, frozen_ssl_input, true_bits, charges, records = load_axis3_data()
    closed_meta, closed_bits = load_library(CLOSED_LIBRARY_PARQUET, CLOSED_LIBRARY_NPZ)
    open_meta, open_bits = load_library(OPEN_LIBRARY_PARQUET, OPEN_LIBRARY_NPZ)
    seen_flags = pd.read_csv(SEEN_FLAGS_CSV)
    seen_map = dict(zip(seen_flags["first_block_inchikey"].astype(str), seen_flags["seen_in_axis2_train"].astype(bool)))
    records = records.copy()
    records["peak_count_bin"] = records["peak_count_cleaned"].astype(int).map(peak_count_bin)
    records["training_status"] = records["first_block_inchikey"].astype(str).map(seen_map)
    records["training_status"] = records["training_status"].map({True: "Seen", False: "Novel"}).fillna("Unknown")
    true_keys = records["first_block_inchikey"].astype(str).to_numpy()
    closed_keys = closed_meta["first_block_inchikey"].astype(str).to_numpy()
    open_keys = open_meta["first_block_inchikey"].astype(str).to_numpy()

    print("Axis 3 tier-1 retrieval run")
    print(f"Device: {device}")
    print(f"Fine-tuned checkpoint: {fine_ckpt}")
    print(f"Frozen checkpoint: {frozen_ckpt}")
    print(f"Tau fine-tuned: {tau_by_model['fine_tuned']:.2f}")
    print(f"Tau frozen: {tau_by_model['frozen']:.2f}")
    print("Retrieval similarity: cosine on continuous predictions")
    print("Tau is saved for binary prediction artefacts only")
    print(f"Spectra: {len(records)}")
    print(f"Closed pool: {len(closed_meta)}")
    print(f"Open pool: {len(open_meta)}")

    t0 = time.perf_counter()
    fine_pred = load_prediction_cache(PRED_FINE_TUNED_NPY, len(records))
    if fine_pred is None:
        fine_pred = infer_model("fine_tuned", fine_ckpt, spectra, charges, batch_size, device)
    else:
        print(f"Loaded cached fine-tuned predictions: {PRED_FINE_TUNED_NPY}")

    frozen_pred = load_prediction_cache(PRED_FROZEN_NPY, len(records))
    if frozen_pred is None:
        frozen_pred = infer_frozen_embedding_head(frozen_ckpt, frozen_ssl_input, batch_size, device)
    else:
        print(f"Loaded cached frozen predictions: {PRED_FROZEN_NPY}")

    pred_by_model = {
        "fine_tuned": fine_pred,
        "frozen": frozen_pred,
    }
    np.save(PRED_FINE_TUNED_NPY, pred_by_model["fine_tuned"])
    np.save(PRED_FROZEN_NPY, pred_by_model["frozen"])

    bin_by_model = {
        model_key: (pred >= tau_by_model[model_key]).astype(np.uint8)
        for model_key, pred in pred_by_model.items()
    }
    np.save(BIN_FINE_TUNED_NPY, bin_by_model["fine_tuned"])
    np.save(BIN_FROZEN_NPY, bin_by_model["frozen"])

    prefilter_mask = build_mass_prefilter(records, open_meta)
    ranks_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    metric_rows = []
    acc_curve_rows = []
    pools = {
        "closed": (closed_meta, closed_bits, closed_keys, None),
        "open": (open_meta, open_bits, open_keys, None),
        "open_prefilter_10ppm": (open_meta, open_bits, open_keys, prefilter_mask),
    }
    for model_key, pred_score in pred_by_model.items():
        for pool_key, (pool_meta, pool_bits, pool_keys, mask) in pools.items():
            sims = cosine_similarity_matrix(pred_score, pool_bits)
            missing_rank_value = len(pool_meta) + 1 if mask is not None else None
            ranks = compute_ranks(
                sims,
                true_keys=true_keys,
                candidate_keys=pool_keys,
                candidate_mask=mask,
                missing_rank_value=missing_rank_value,
            )
            ranks_by_key[(model_key, pool_key)] = ranks
            metric_rows.append(metric_row(model_key, pool_key, ranks, tau_by_model[model_key]))
            if pool_key in {"closed", "open"}:
                for k in range(1, 11):
                    acc_curve_rows.append(
                        {
                            "model": model_key,
                            "model_label": MODEL_LABELS[model_key],
                            "pool": pool_key,
                            "pool_label": POOL_LABELS[pool_key],
                            "similarity": "cosine",
                            "prediction_values": "continuous_sigmoid",
                            "k": k,
                            "accuracy": float(
                                np.mean(
                                    ranks["true_in_pool"].to_numpy(dtype=bool)
                                    & (ranks["rank"].to_numpy(dtype=int) <= k)
                                )
                            ),
                            "random_accuracy": min(k, len(pool_meta)) / len(pool_meta),
                        }
                    )
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULTS_CSV, index=False)
    save_json(metric_rows, RESULTS_JSON)
    axis3_metrics_df = make_axis3_retrieval_metrics(records, ranks_by_key)
    axis3_metrics_df.to_csv(AXIS3_RETRIEVAL_METRICS_CSV, index=False)
    recovery_df = make_compound_recovery(records, ranks_by_key, [1, 5, 10])
    recovery_df.to_csv(AXIS3_RECOVERY_CSV, index=False)
    axis3_acc_curve_df = make_acc_curve(
        ranks_by_key,
        pool_sizes={"closed": len(closed_meta), "open": len(open_meta)},
        max_k=20,
    )
    axis3_acc_curve_df.to_csv(AXIS3_ACC_CURVE_CSV, index=False)

    retrieval_df = add_retrieval_columns(records, ranks_by_key)
    retrieval_df.to_csv(PER_SPECTRUM_RETRIEVAL_CSV, index=False)
    acc_curve_df = pd.DataFrame(acc_curve_rows)
    acc_curve_df.to_csv(ACC_CURVE_CSV, index=False)

    primary_ranks = ranks_by_key[("fine_tuned", "open")]
    adduct_df = decompose(records, primary_ranks, "adduct", ADDUCT_ORDER)
    peak_df = decompose(records, primary_ranks, "peak_count_bin", ["1", "2-3", "4-5", "6-10", ">10"])
    seen_df = decompose(records, primary_ranks, "training_status", ["Seen", "Novel", "Unknown"])
    by_adduct_df = make_stratified_metrics(records, ranks_by_key, "adduct", ADDUCT_ORDER)
    by_peak_df = make_stratified_metrics(records, ranks_by_key, "peak_count_bin", ["1", "2-3", "4-5", "6-10", ">10"])
    by_seen_df = make_stratified_metrics(records, ranks_by_key, "training_status", ["Seen", "Novel", "Unknown"])
    transfer_caveat_df = seen_novel_transfer_table(records, ranks_by_key)
    adduct_df.to_csv(DECOMP_ADDUCT_CSV, index=False)
    peak_df.to_csv(DECOMP_PEAK_CSV, index=False)
    seen_df.to_csv(DECOMP_SEEN_CSV, index=False)
    by_adduct_df.to_csv(AXIS3_BY_ADDUCT_CSV, index=False)
    by_peak_df.to_csv(AXIS3_BY_PEAKCOUNT_CSV, index=False)
    by_seen_df.to_csv(AXIS3_BY_SEEN_NOVEL_CSV, index=False)
    transfer_caveat_df.to_csv(TRANSFER_CAVEAT_CSV, index=False)

    perbit_df = make_perbit_auroc_table(true_bits, pred_by_model)
    perbit_df.to_csv(AXIS3_PERBIT_AUROC_CSV, index=False)
    substructure_transfer_df = make_substructure_transfer_table(perbit_df)
    substructure_transfer_df.to_csv(AXIS3_SUBSTRUCTURE_TRANSFER_CSV, index=False)
    transfer_df = perbit_df.rename(
        columns={
            "bit": "bit_index",
            "auroc_mac_ft": "axis3_auroc",
            "freq_mac": "axis3_frequency",
            "n_positive_mac": "axis3_n_positive",
            "n_negative_mac": "axis3_n_negative",
            "auroc_axis2_ood": "axis2_ood_auroc",
            "freq_axis2_ood": "axis2_ood_frequency",
        }
    )[[
        "bit_index",
        "axis3_auroc",
        "axis3_frequency",
        "axis3_n_positive",
        "axis3_n_negative",
        "axis2_ood_auroc",
        "axis2_ood_frequency",
    ]]
    valid_transfer = transfer_df.dropna(subset=["axis3_auroc", "axis2_ood_auroc"])
    rho, p_value = spearmanr(valid_transfer["axis2_ood_auroc"], valid_transfer["axis3_auroc"])
    transfer_df["transfer_valid"] = transfer_df["bit_index"].isin(valid_transfer["bit_index"])
    transfer_df.to_csv(PER_BIT_AUROC_CSV, index=False)
    write_axis3_specific_exports(axis3_metrics_df, records, ranks_by_key, perbit_df, true_bits, pred_by_model)

    plot_acc_curve(acc_curve_df)
    plot_rank_percentile_ecdf(retrieval_df)
    plot_decompositions(adduct_df, peak_df, seen_df)
    plot_transfer_scatter(transfer_df, float(rho), float(p_value))
    plot_axis3_acc_curves(axis3_acc_curve_df)
    plot_axis3_rank_cdf(ranks_by_key)
    plot_axis3_transfer_scatter(perbit_df, substructure_transfer_df)
    plot_axis3_decomposition(by_adduct_df, by_peak_df)

    elapsed = time.perf_counter() - t0
    headline = metrics_df.loc[
        (metrics_df["model"] == "fine_tuned") & (metrics_df["pool"] == "open")
    ].iloc[0]
    frozen_open = metrics_df.loc[
        (metrics_df["model"] == "frozen") & (metrics_df["pool"] == "open")
    ].iloc[0]
    closed_headline = metrics_df.loc[
        (metrics_df["model"] == "fine_tuned") & (metrics_df["pool"] == "closed")
    ].iloc[0]
    prefilter_headline = metrics_df.loc[
        (metrics_df["model"] == "fine_tuned") & (metrics_df["pool"] == "open_prefilter_10ppm")
    ].iloc[0]
    seen_primary = transfer_caveat_df.loc[
        (transfer_caveat_df["model"] == "fine_tuned") & (transfer_caveat_df["training_status"] == "Seen")
    ].iloc[0]
    novel_primary = transfer_caveat_df.loc[
        (transfer_caveat_df["model"] == "fine_tuned") & (transfer_caveat_df["training_status"] == "Novel")
    ].iloc[0]
    axis2_retrieval = pd.read_csv(FINE_TUNED_RUN_DIR / "axis2_artifacts/retrieval/retrieval_metrics.csv")
    axis2_ood = dict(zip(axis2_retrieval["Metric"], axis2_retrieval["OOD"]))
    ft_transfer = substructure_transfer_df.loc[substructure_transfer_df["model"] == "fine_tuned"].iloc[0]
    frozen_transfer = substructure_transfer_df.loc[substructure_transfer_df["model"] == "frozen"].iloc[0]
    headline_table = axis3_metrics_df.loc[
        axis3_metrics_df["stratum"] == "overall",
        ["model_label", "pool_label", "n_spectra", "acc@1", "acc@5", "acc@10", "median_rank", "mrr"],
    ].copy()
    print("Headline retrieval table")
    print(headline_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    summary_lines = [
        "Axis 3 tier-1 retrieval summary",
        f"Generated in {elapsed:.1f} seconds",
        "Retrieval similarity cosine on continuous predictions",
        "Tau retained for saved binary prediction artefacts only",
        f"Fine-tuned open acc@1 {headline['acc@1']:.4f}, acc@10 {headline['acc@10']:.4f}, median rank {headline['median_rank']:.1f}, MRR {headline['mrr']:.4f}",
        f"Fine-tuned open seen acc@1 {seen_primary['acc@1']:.4f}, acc@10 {seen_primary['acc@10']:.4f}, median rank {seen_primary['median_rank']:.1f}, median percentile {seen_primary['median_rank_percentile']:.4f}",
        f"Fine-tuned open novel acc@1 {novel_primary['acc@1']:.4f}, acc@10 {novel_primary['acc@10']:.4f}, median rank {novel_primary['median_rank']:.1f}, median percentile {novel_primary['median_rank_percentile']:.4f}",
        f"Axis 2 OOD reference acc@1 {axis2_ood['acc@1']:.4f}, acc@10 {axis2_ood['acc@10']:.4f}, median rank {axis2_ood['median_rank']:.1f}",
        "Transfer caveat: aggregate Axis 3 acc@1 is inflated by seen compounds; the novel split is the defensible transfer comparison.",
        f"Frozen open acc@1 {frozen_open['acc@1']:.4f}, acc@10 {frozen_open['acc@10']:.4f}, median rank {frozen_open['median_rank']:.1f}, MRR {frozen_open['mrr']:.4f}",
        f"Fine-tuned closed acc@1 {closed_headline['acc@1']:.4f}, acc@10 {closed_headline['acc@10']:.4f}, median rank {closed_headline['median_rank']:.1f}, MRR {closed_headline['mrr']:.4f}",
        f"Fine-tuned open 10 ppm prefiltered acc@1 {prefilter_headline['acc@1']:.4f}, acc@10 {prefilter_headline['acc@10']:.4f}, median rank {prefilter_headline['median_rank']:.1f}, MRR {prefilter_headline['mrr']:.4f}",
        f"Substructure transfer fine-tuned Spearman rho {ft_transfer['spearman_rho']:.4f}, p {ft_transfer['spearman_p']:.3g}, valid bits {int(ft_transfer['n_bits'])}",
        f"Substructure transfer frozen Spearman rho {frozen_transfer['spearman_rho']:.4f}, p {frozen_transfer['spearman_p']:.3g}, valid bits {int(frozen_transfer['n_bits'])}",
    ]
    SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
