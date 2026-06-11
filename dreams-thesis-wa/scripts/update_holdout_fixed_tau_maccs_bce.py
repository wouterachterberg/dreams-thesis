#!/usr/bin/env python3
"""Update Holdout MACCS-BCE binarised metrics at the fixed OOD tau only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "dreams-thesis-wa" / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from h100_batch_inference import infer_batches, load_model_compat, parse_spectrum_strings
from evaluate_holdout_maccs_bce import (
    CHECKPOINT_DIR,
    EXPECTED_HOLDOUT_SPECTRA,
    EXPECTED_HOLDOUT_UNIQUE_MOLECULES,
    FP_KIND,
    HOLDOUT_PARQUET,
    INFERENCE_BATCH_SIZE_BY_DEVICE,
    METRICS_OUT,
    SEED,
    compute_per_bit_auroc,
    compute_rowwise_cosine_stats,
    fp_from_smiles,
    load_holdout_dataframe,
    locate_holdout_label,
    resolve_best_checkpoint,
    set_inference_seed,
)


FIXED_TAU = 0.27
TOL = 1e-4
HOLDOUT_PRED_CANDIDATES = [
    METRICS_OUT.parent / "y_pred_holdout.npy",
    METRICS_OUT.parent / "holdout_y_pred.npy",
]


def fixed_threshold_metrics(y_pred: np.ndarray, y_true: np.ndarray, tau: float) -> dict[str, float]:
    y_true_u8 = y_true.astype(np.uint8)
    yb = (y_pred >= tau).astype(np.uint8)
    tp = (yb & y_true_u8).sum(axis=1)
    fp = (yb & (1 - y_true_u8)).sum(axis=1)
    fn = ((1 - yb) & y_true_u8).sum(axis=1)
    union = ((yb | y_true_u8)).sum(axis=1)

    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0).mean()
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0).mean()
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    tanimoto = np.where(union > 0, tp / union, 0.0).mean()
    return {
        "fixed_tau": float(tau),
        "tanimoto_at_fixed_tau": float(tanimoto),
        "precision_at_fixed_tau": float(precision),
        "recall_at_fixed_tau": float(recall),
        "f1_at_fixed_tau": float(f1),
    }


def load_or_regenerate_predictions(df: pd.DataFrame, device: str, batch_size: int) -> np.ndarray:
    for path in HOLDOUT_PRED_CANDIDATES:
        if path.exists():
            y_pred = np.load(path).astype(np.float32)
            if y_pred.shape != (EXPECTED_HOLDOUT_SPECTRA, 166):
                raise AssertionError(f"Saved Holdout prediction cache has wrong shape: {path} {y_pred.shape}")
            print(f"Loaded saved Holdout predictions: {path}")
            return y_pred

    print("No saved Holdout prediction matrix found; regenerating deterministic inference.")
    best_ckpt = resolve_best_checkpoint(CHECKPOINT_DIR)
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
        batch_size=batch_size,
        device=device,
        apply_sigmoid_to_pred=True,
        progress_desc="maccs_166_bce | Holdout fixed-tau inference",
        prec_mz_np=prec_holdout,
        amp_dtype="none",
    )
    del model
    return y_pred.astype(np.float32)


def assert_close(name: str, observed: float, expected: float) -> None:
    delta = abs(float(observed) - float(expected))
    if delta >= TOL:
        raise AssertionError(f"{name} gate failed: observed={observed:.9f}, expected={expected:.9f}, delta={delta:.9g}")


def main() -> None:
    metrics = pd.read_csv(METRICS_OUT)
    if len(metrics) != 1:
        raise AssertionError(f"Expected one Holdout metrics row in {METRICS_OUT}, found {len(metrics)}")
    row = metrics.iloc[0].copy()

    set_inference_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    batch_size = int(row.get("inference_batch_size", INFERENCE_BATCH_SIZE_BY_DEVICE[device]))

    holdout_label = locate_holdout_label()
    df = load_holdout_dataframe(holdout_label)
    n_spectra = int(len(df))
    library_size = int(df["smiles"].astype(str).nunique())
    if n_spectra != EXPECTED_HOLDOUT_SPECTRA:
        raise AssertionError(f"n_spectra gate failed: {n_spectra} != {EXPECTED_HOLDOUT_SPECTRA}")
    if library_size != EXPECTED_HOLDOUT_UNIQUE_MOLECULES:
        raise AssertionError(f"library_size gate failed: {library_size} != {EXPECTED_HOLDOUT_UNIQUE_MOLECULES}")

    y_pred = load_or_regenerate_predictions(df, device=device, batch_size=batch_size)
    y_true = np.stack(
        [fp_from_smiles(s, FP_KIND) for s in tqdm(df["smiles"].astype(str).tolist(), desc="GT Holdout MACCS")],
        axis=0,
    ).astype(np.float32)
    if y_pred.shape != y_true.shape:
        raise AssertionError(f"Prediction/target shape mismatch: {y_pred.shape} vs {y_true.shape}")

    cos_stats = compute_rowwise_cosine_stats(y_true, y_pred)
    au = compute_per_bit_auroc(y_true, y_pred)
    mean_auroc = float(au["auroc"].dropna().mean())

    assert_close("cosine_sim_mean", cos_stats["cosine_sim_mean"], row["cosine_sim_mean"])
    assert_close("mean_per_bit_auroc", mean_auroc, row["mean_per_bit_auroc"])

    fixed = fixed_threshold_metrics(y_pred, y_true, FIXED_TAU)
    assert_close("tanimoto_at_ood_tau_0.27", fixed["tanimoto_at_fixed_tau"], row["tanimoto_at_ood_tau_0.27"])

    metrics.loc[0, "fixed_tau"] = fixed["fixed_tau"]
    metrics.loc[0, "tanimoto_at_fixed_tau"] = fixed["tanimoto_at_fixed_tau"]
    metrics.loc[0, "precision_at_fixed_tau"] = fixed["precision_at_fixed_tau"]
    metrics.loc[0, "recall_at_fixed_tau"] = fixed["recall_at_fixed_tau"]
    metrics.loc[0, "f1_at_fixed_tau"] = fixed["f1_at_fixed_tau"]
    metrics.loc[0, "threshold_dependent_headline"] = "fixed_tau_0.27_ood_derived"
    metrics.loc[
        0,
        "oracle_threshold_comment",
    ] = (
        "best_tanimoto/optimal_tau and *_at_optimal_tau are ORACLE Holdout-peeked "
        "upper-bound columns; use fixed_tau and *_at_fixed_tau for deployment-style binarised metrics."
    )

    metrics.to_csv(METRICS_OUT, index=False)

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
