#!/usr/bin/env python3
"""
Cross-axis bridge analysis connecting Axis 1 probing to Axis 2 fingerprint prediction.

This script performs three reproducible steps:

1. Model-independent point-biserial correlation matrices between fingerprint bits
   and RDKit descriptors on the fine-tuning training molecules.
2. MACCS SMARTS validation for the strongest MACCS descriptor matches.
3. Fine-tuned cross-axis scatter plots and Spearman correlations linking
   descriptor probeability (Axis 1) to per-bit AUROC (Axis 2).

The implementation uses a vectorized Pearson/point-biserial computation for
speed. For binary fingerprint bits, this is mathematically equivalent to
``scipy.stats.pointbiserialr``; the script saves a small sampled equivalence
check against SciPy for transparency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "cross_axis_bridge_mplconfig"),
)

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

try:
    from map4 import MAP4

    MAP4_AVAILABLE = True
except ImportError:
    try:
        from map4 import MAP4Calculator as MAP4

        MAP4_AVAILABLE = True
    except ImportError:
        MAP4 = None
        MAP4_AVAILABLE = False


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


FINGERPRINT_SPECS: dict[str, dict[str, Any]] = {
    "ecfp4": {
        "file_stem": "morgan_2048",
        "display": "ECFP4",
        "bits": 2048,
        "run_prefix": "morgan_2048",
    },
    "maccs": {
        "file_stem": "maccs_166",
        "display": "MACCS",
        "bits": 166,
        "run_prefix": "maccs_166",
    },
    "map4": {
        "file_stem": "map4_2048",
        "display": "MAP4",
        "bits": 2048,
        "run_prefix": "map4_2048",
    },
}

LOSS_DISPLAY = {"bce": "BCE", "cos": "Cosine"}
SPLIT_DISPLAY = {"val": "Val", "ood": "OOD"}

FINE_TUNED_RUN_TAGS = (
    "morgan_2048_cos",
    "morgan_2048_bce",
    "maccs_166_cos",
    "maccs_166_bce",
    "map4_2048_cos",
    "map4_2048_bce",
)

ALL_RUN_TAGS = FINE_TUNED_RUN_TAGS + (
    "morgan_2048_cos_frozen",
    "morgan_2048_bce_frozen",
    "maccs_166_cos_frozen",
    "maccs_166_bce_frozen",
    "map4_2048_cos_frozen",
    "map4_2048_bce_frozen",
)

SCATTER_PALETTE = {
    "bce": "#0b6e4f",
    "cos": "#b85c38",
}
SCATTER_MARKERS = {
    "bce": "o",
    "cos": "o",
}


@dataclass
class BridgeConfig:
    thesis_root: Path
    output_dir: Path
    descriptor_table_path: Path
    axis1_merged_path: Path
    axis1_linear_path: Path
    axis1_mlp_path: Path
    finetuning_hdf5_path: Path
    model_runs_dir: Path
    fine_tuned_run_tags: tuple[str, ...] = field(default_factory=lambda: FINE_TUNED_RUN_TAGS)
    all_run_tags: tuple[str, ...] = field(default_factory=lambda: ALL_RUN_TAGS)
    ecfp4_bits: int = 2048
    ecfp4_radius: int = 2
    maccs_bits: int = 166
    map4_bits: int = 2048
    map4_radius: int = 2
    top_maccs_validation: int = 10
    top_pairs_per_group: int = 10
    labels_per_category: int = 2
    pointbiserial_validation_pairs: int = 12
    random_state: int = 13

    @classmethod
    def from_thesis_root(cls, thesis_root: Path, output_dir: Path | None = None) -> "BridgeConfig":
        thesis_root = thesis_root.resolve()
        return cls(
            thesis_root=thesis_root,
            output_dir=(output_dir or thesis_root / "results" / "cross_axis").resolve(),
            descriptor_table_path=(
                thesis_root
                / "data"
                / "processed"
                / "massspecgym_complete"
                / "all_rdkit_descriptors.parquet"
            ).resolve(),
            axis1_merged_path=(
                thesis_root / "results" / "axis1" / "indicators" / "probe_indicator_merged.csv"
            ).resolve(),
            axis1_linear_path=(thesis_root / "results" / "all_descriptors_probing_results_linear.pkl").resolve(),
            axis1_mlp_path=(thesis_root / "results" / "all_descriptors_probing_results_mlp.pkl").resolve(),
            finetuning_hdf5_path=(
                thesis_root / "data" / "processed" / "MassSpecGym_splits" / "finetuning.hdf5"
            ).resolve(),
            model_runs_dir=(thesis_root / "results" / "model_runs").resolve(),
        )


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    thesis_root = script_path.parents[1]

    parser = argparse.ArgumentParser(
        description="Run the cross-axis bridge analysis linking probing R^2 to per-bit AUROC."
    )
    parser.add_argument(
        "--thesis-root",
        type=Path,
        default=thesis_root,
        help="Path to the dreams-thesis-wa project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to results/cross_axis.",
    )
    parser.add_argument(
        "--top-maccs-validation",
        type=int,
        default=10,
        help="Number of MACCS top matches to validate against SMARTS.",
    )
    parser.add_argument(
        "--top-pairs-per-group",
        type=int,
        default=10,
        help="Rows per fingerprint/split to keep in upper-right and disagreement tables.",
    )
    parser.add_argument(
        "--labels-per-category",
        type=int,
        default=2,
        help="Annotation count per category in each scatter panel.",
    )
    parser.add_argument(
        "--pointbiserial-validation-pairs",
        type=int,
        default=12,
        help="Random bit-descriptor pairs per fingerprint used for SciPy equivalence checks.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> BridgeConfig:
    cfg = BridgeConfig.from_thesis_root(args.thesis_root, args.output_dir)
    cfg.top_maccs_validation = args.top_maccs_validation
    cfg.top_pairs_per_group = args.top_pairs_per_group
    cfg.labels_per_category = args.labels_per_category
    cfg.pointbiserial_validation_pairs = args.pointbiserial_validation_pairs
    return cfg


def decode_utf8_array(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.asarray(values, dtype=object)
    first = values[0]
    if isinstance(first, (bytes, np.bytes_)):
        return np.array([value.decode("utf-8") for value in values], dtype=object)
    return np.asarray(values, dtype=object)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_probe_dataframe(df: pd.DataFrame, score_col_name: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if "descriptor_name" in out.columns and "descriptor" not in out.columns:
        out = out.rename(columns={"descriptor_name": "descriptor"})
    if "descriptor" not in out.columns:
        raise ValueError("Probe table must contain descriptor or descriptor_name.")
    if score_col_name and score_col_name not in out.columns:
        if "r2" in out.columns:
            out = out.rename(columns={"r2": score_col_name})
        else:
            raise ValueError(f"Probe table is missing {score_col_name} and r2.")
    return out


def apply_indicator_spearman_sign_convention(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the retired fix_spearman_sign.py convention once for raw negative inputs."""
    out = df.copy()
    if "spearman_corr" not in out.columns:
        return out

    spearman = pd.to_numeric(out["spearman_corr"], errors="coerce")
    if float(spearman.mean(skipna=True)) < 0.0:
        spearman = -spearman
    out["spearman_corr"] = spearman
    return out


def load_probe_result_table(path: Path, score_col_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Probe result file not found: {path}")
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".pkl":
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
        elif isinstance(obj, list):
            df = pd.DataFrame(obj)
        elif isinstance(obj, dict):
            df = pd.DataFrame(obj)
        else:
            raise TypeError(f"Unsupported pickle payload type for {path}: {type(obj)!r}")
    else:
        raise ValueError(f"Unsupported probe result file type: {path}")
    df = normalize_probe_dataframe(df, score_col_name=score_col_name)
    return df[["descriptor", score_col_name]].copy()


def load_axis1_scores(cfg: BridgeConfig) -> pd.DataFrame:
    if cfg.axis1_merged_path.exists():
        axis1 = pd.read_csv(cfg.axis1_merged_path)
        axis1 = normalize_probe_dataframe(axis1)
        axis1 = apply_indicator_spearman_sign_convention(axis1)
    else:
        linear_df = load_probe_result_table(cfg.axis1_linear_path, "r2_linear")
        if cfg.axis1_mlp_path.exists():
            mlp_df = load_probe_result_table(cfg.axis1_mlp_path, "r2_mlp")
            axis1 = linear_df.merge(mlp_df, on="descriptor", how="outer")
        else:
            axis1 = linear_df

    if "r2_linear" not in axis1.columns:
        if "r2" in axis1.columns:
            axis1 = axis1.rename(columns={"r2": "r2_linear"})
        else:
            raise ValueError("Axis 1 results must include r2_linear.")

    keep_cols = ["descriptor", "r2_linear"]
    if "r2_mlp" in axis1.columns:
        keep_cols.append("r2_mlp")
    axis1 = axis1[keep_cols].dropna(subset=["descriptor"]).copy()
    axis1 = axis1.sort_values("descriptor").drop_duplicates(subset="descriptor", keep="first")
    return axis1.reset_index(drop=True)


def load_training_descriptor_frame(cfg: BridgeConfig, axis1_scores: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    descriptor_table = pd.read_parquet(cfg.descriptor_table_path)
    if "smiles" not in descriptor_table.columns:
        raise ValueError(f"Descriptor table is missing smiles: {cfg.descriptor_table_path}")

    descriptor_cols = [col for col in axis1_scores["descriptor"].tolist() if col in descriptor_table.columns]
    missing = sorted(set(axis1_scores["descriptor"].tolist()) - set(descriptor_cols))

    descriptor_table = descriptor_table[["smiles"] + descriptor_cols].drop_duplicates(subset=["smiles"], keep="first")

    with h5py.File(cfg.finetuning_hdf5_path, "r") as handle:
        smiles = decode_utf8_array(handle["smiles"][:])
        folds = decode_utf8_array(handle["fold"][:])

    seen: set[str] = set()
    ordered_train_smiles: list[str] = []
    for smiles_value, fold_value in zip(smiles.tolist(), folds.tolist(), strict=False):
        smiles_str = str(smiles_value)
        fold_str = str(fold_value)
        if fold_str != "train":
            continue
        if smiles_str not in seen:
            seen.add(smiles_str)
            ordered_train_smiles.append(smiles_str)

    train_frame = pd.DataFrame({"smiles": ordered_train_smiles})
    aligned = train_frame.merge(descriptor_table, on="smiles", how="inner")
    aligned = aligned.dropna(subset=descriptor_cols).reset_index(drop=True)

    summary = {
        "axis1_descriptors_requested": int(axis1_scores["descriptor"].nunique()),
        "axis1_descriptors_used": len(descriptor_cols),
        "axis1_descriptors_missing_from_table": len(missing),
        "training_unique_smiles": len(ordered_train_smiles),
        "aligned_training_rows": int(len(aligned)),
    }
    if missing:
        summary["missing_descriptors"] = missing
    return aligned, descriptor_cols, summary


_MAP4_CACHE: dict[tuple[int, int], Any] = {}


def get_map4_calculator(dimensions: int, radius: int) -> Any:
    if not MAP4_AVAILABLE:
        raise RuntimeError("MAP4 package is unavailable in the current environment.")
    key = (dimensions, radius)
    if key not in _MAP4_CACHE:
        _MAP4_CACHE[key] = MAP4(dimensions=dimensions, radius=radius)
    return _MAP4_CACHE[key]


def build_fingerprint_matrices(smiles_list: list[str], cfg: BridgeConfig) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n_rows = len(smiles_list)
    morgan = np.zeros((n_rows, cfg.ecfp4_bits), dtype=np.uint8)
    maccs = np.zeros((n_rows, cfg.maccs_bits), dtype=np.uint8)
    map4_bits = np.zeros((n_rows, cfg.map4_bits), dtype=np.uint8)

    invalid_smiles = 0
    map4_calc = get_map4_calculator(cfg.map4_bits, cfg.map4_radius)

    for row_idx, smiles in enumerate(tqdm(smiles_list, desc="Computing aligned fingerprints")):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles += 1
            continue

        morgan[row_idx] = np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, cfg.ecfp4_radius, nBits=cfg.ecfp4_bits),
            dtype=np.uint8,
        )
        maccs[row_idx] = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8)[1:]
        map4_dense = np.asarray(map4_calc.calculate(mol), dtype=np.float32)
        if map4_dense.shape[0] != cfg.map4_bits:
            raise ValueError(
                f"MAP4 length mismatch for {smiles!r}: got {map4_dense.shape[0]}, expected {cfg.map4_bits}"
            )
        map4_bits[row_idx] = (map4_dense != 0).astype(np.uint8)

    summary = {
        "invalid_smiles_count": invalid_smiles,
        "ecfp4_shape": list(morgan.shape),
        "maccs_shape": list(maccs.shape),
        "map4_shape": list(map4_bits.shape),
        "ecfp4_mean_bit_density": float(morgan.mean()),
        "maccs_mean_bit_density": float(maccs.mean()),
        "map4_mean_bit_density": float(map4_bits.mean()),
    }
    return {"ecfp4": morgan, "maccs": maccs, "map4": map4_bits}, summary


def compute_pointbiserial_matrix_vectorized(x_bits: np.ndarray, x_desc: np.ndarray) -> np.ndarray:
    if x_bits.ndim != 2 or x_desc.ndim != 2:
        raise ValueError("Expected 2D matrices for x_bits and x_desc.")
    if x_bits.shape[0] != x_desc.shape[0]:
        raise ValueError(
            f"Row mismatch between fingerprint bits and descriptors: {x_bits.shape} vs {x_desc.shape}"
        )

    n_rows = x_bits.shape[0]
    if n_rows < 3:
        raise ValueError("Need at least three molecules to compute correlations.")

    x_bits_f = x_bits.astype(np.float32, copy=False)
    x_desc_f = x_desc.astype(np.float32, copy=False)

    desc_mean = x_desc_f.mean(axis=0, keepdims=True)
    desc_centered = x_desc_f - desc_mean
    desc_std = desc_centered.std(axis=0, ddof=1)

    bit_mean = x_bits_f.mean(axis=0, dtype=np.float64).astype(np.float32)
    bit_std = np.sqrt(np.clip(bit_mean * (1.0 - bit_mean) * (n_rows / (n_rows - 1.0)), 0.0, None))

    covariance = (x_bits_f.T @ desc_centered) / np.float32(n_rows - 1)
    denominator = bit_std[:, None] * desc_std[None, :]
    corr = np.divide(
        covariance,
        denominator,
        out=np.full(covariance.shape, np.nan, dtype=np.float32),
        where=denominator > 0,
    )
    return corr


def build_best_match_table(
    corr_matrix: np.ndarray,
    descriptor_cols: list[str],
    x_bits: np.ndarray,
    fp_family: str,
) -> pd.DataFrame:
    abs_corr = np.abs(corr_matrix)
    valid_rows = np.isfinite(abs_corr).any(axis=1)

    best_idx = np.full(corr_matrix.shape[0], -1, dtype=int)
    best_corr = np.full(corr_matrix.shape[0], np.nan, dtype=np.float32)
    best_abs = np.full(corr_matrix.shape[0], np.nan, dtype=np.float32)

    if valid_rows.any():
        best_idx[valid_rows] = abs_corr[valid_rows].argmax(axis=1)
        row_indices = np.where(valid_rows)[0]
        best_corr[valid_rows] = corr_matrix[row_indices, best_idx[valid_rows]]
        best_abs[valid_rows] = abs_corr[row_indices, best_idx[valid_rows]]

    descriptor_lookup = np.array(descriptor_cols, dtype=object)
    best_descriptor = [
        descriptor_lookup[idx] if idx >= 0 else None
        for idx in best_idx.tolist()
    ]

    table = pd.DataFrame(
        {
            "fp_family": fp_family,
            "bit_index": np.arange(corr_matrix.shape[0], dtype=int),
            "best_descriptor_idx": best_idx,
            "best_descriptor": best_descriptor,
            "best_corr": best_corr,
            "best_abs_corr": best_abs,
            "bit_frequency_train": x_bits.mean(axis=0),
        }
    )
    if fp_family == "maccs":
        table["rdkit_bit_number"] = table["bit_index"] + 1
    return table


def save_pointbiserial_matrix(
    corr_matrix: np.ndarray,
    descriptor_cols: list[str],
    output_csv: Path,
    output_npy: Path,
) -> None:
    matrix_df = pd.DataFrame(corr_matrix, columns=descriptor_cols)
    matrix_df.insert(0, "bit_index", np.arange(len(matrix_df), dtype=int))
    matrix_df.to_csv(output_csv, index=False)
    np.save(output_npy, corr_matrix)


def validate_pointbiserial_equivalence(
    fp_family: str,
    corr_matrix: np.ndarray,
    x_bits: np.ndarray,
    x_desc: np.ndarray,
    descriptor_cols: list[str],
    n_pairs: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    valid_positions = np.argwhere(np.isfinite(corr_matrix))
    if len(valid_positions) == 0:
        return pd.DataFrame(
            columns=[
                "fp_family",
                "bit_index",
                "descriptor",
                "vectorized_corr",
                "scipy_pointbiserial_corr",
                "abs_diff",
            ]
        )

    sample_size = min(n_pairs, len(valid_positions))
    sampled_positions = valid_positions[rng.choice(len(valid_positions), size=sample_size, replace=False)]

    rows: list[dict[str, Any]] = []
    for bit_idx, desc_idx in sampled_positions.tolist():
        scipy_corr, _ = pointbiserialr(x_bits[:, bit_idx].astype(np.float64), x_desc[:, desc_idx].astype(np.float64))
        vectorized_corr = float(corr_matrix[bit_idx, desc_idx])
        rows.append(
            {
                "fp_family": fp_family,
                "bit_index": int(bit_idx),
                "descriptor": descriptor_cols[desc_idx],
                "vectorized_corr": vectorized_corr,
                "scipy_pointbiserial_corr": float(scipy_corr),
                "abs_diff": abs(vectorized_corr - float(scipy_corr)),
            }
        )
    return pd.DataFrame(rows).sort_values(["fp_family", "abs_diff"], ascending=[True, False]).reset_index(drop=True)


def parse_run_tag(run_tag: str) -> tuple[str, str, bool]:
    is_frozen = run_tag.endswith("_frozen")
    base_tag = run_tag[:-7] if is_frozen else run_tag

    if base_tag.startswith("morgan_2048"):
        fp_family = "ecfp4"
    elif base_tag.startswith("maccs_166"):
        fp_family = "maccs"
    elif base_tag.startswith("map4_2048"):
        fp_family = "map4"
    else:
        raise ValueError(f"Unsupported run tag: {run_tag}")

    if base_tag.endswith("_bce"):
        loss_kind = "bce"
    elif base_tag.endswith("_cos"):
        loss_kind = "cos"
    else:
        raise ValueError(f"Unable to infer loss kind from run tag: {run_tag}")

    return fp_family, loss_kind, is_frozen


def compute_per_bit_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch for y_true/y_pred: {y_true.shape} vs {y_pred.shape}")

    n_bits = y_true.shape[1]
    aucs = np.full(n_bits, np.nan, dtype=np.float32)
    freqs = y_true.mean(axis=0).astype(np.float32)

    for bit_idx in range(n_bits):
        target = y_true[:, bit_idx]
        if np.unique(target).size < 2:
            continue
        aucs[bit_idx] = float(roc_auc_score(target, y_pred[:, bit_idx]))
    return aucs, freqs


def load_auroc_table(run_dir: Path) -> pd.DataFrame:
    axis2_dir = run_dir / "axis2_artifacts"
    auroc_csv = axis2_dir / "per_bit_auroc" / "auroc_comparison.csv"
    if auroc_csv.exists():
        auroc_df = pd.read_csv(auroc_csv)
    else:
        val_npy = axis2_dir / "per_bit_auroc_val.npy"
        ood_npy = axis2_dir / "per_bit_auroc_ood.npy"
        if val_npy.exists() and ood_npy.exists():
            auroc_val = np.load(val_npy)
            auroc_ood = np.load(ood_npy)
            if auroc_val.shape != auroc_ood.shape:
                raise ValueError(f"Per-bit AUROC array shape mismatch in {run_dir}")
            auroc_df = pd.DataFrame(
                {
                    "bit_index": np.arange(len(auroc_val), dtype=int),
                    "auroc_val": auroc_val.astype(np.float32),
                    "auroc_ood": auroc_ood.astype(np.float32),
                }
            )
        else:
            y_true_ood = np.load(axis2_dir / "y_true.npy")
            y_pred_ood = np.load(axis2_dir / "y_pred.npy")
            y_true_val = np.load(axis2_dir / "y_true_val.npy")
            y_pred_val = np.load(axis2_dir / "y_pred_val.npy")

            auroc_ood, freq_ood = compute_per_bit_auroc(y_true_ood, y_pred_ood)
            auroc_val, freq_val = compute_per_bit_auroc(y_true_val, y_pred_val)
            auroc_df = pd.DataFrame(
                {
                    "bit_index": np.arange(len(auroc_val), dtype=int),
                    "auroc_val": auroc_val,
                    "freq_val": freq_val,
                    "auroc_ood": auroc_ood,
                    "freq_ood": freq_ood,
                }
            )

    if "bit_index" not in auroc_df.columns:
        auroc_df.insert(0, "bit_index", np.arange(len(auroc_df), dtype=int))
    if "auroc_drop" not in auroc_df.columns and {"auroc_val", "auroc_ood"}.issubset(auroc_df.columns):
        auroc_df["auroc_drop"] = auroc_df["auroc_val"] - auroc_df["auroc_ood"]
    return auroc_df


def load_fine_tuned_scatter_table(
    cfg: BridgeConfig,
    axis1_scores: pd.DataFrame,
    best_match_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    descriptor_scores = axis1_scores[["descriptor", "r2_linear"]].copy()
    if "r2_mlp" in axis1_scores.columns:
        descriptor_scores["r2_mlp"] = axis1_scores["r2_mlp"]

    rows: list[pd.DataFrame] = []
    for run_tag in cfg.fine_tuned_run_tags:
        fp_family, loss_kind, is_frozen = parse_run_tag(run_tag)
        if is_frozen:
            continue

        run_dir = cfg.model_runs_dir / run_tag
        auroc_df = load_auroc_table(run_dir)
        expected_bits = FINGERPRINT_SPECS[fp_family]["bits"]
        if len(auroc_df) != expected_bits:
            raise ValueError(
                f"Bit count mismatch for {run_tag}: expected {expected_bits}, found {len(auroc_df)}"
            )

        merged = auroc_df.merge(best_match_tables[fp_family], on="bit_index", how="left")
        merged = merged.merge(
            descriptor_scores,
            left_on="best_descriptor",
            right_on="descriptor",
            how="left",
        )
        merged["run_tag"] = run_tag
        merged["fp_family"] = fp_family
        merged["fp_display"] = FINGERPRINT_SPECS[fp_family]["display"]
        merged["loss_kind"] = loss_kind
        merged["loss_display"] = LOSS_DISPLAY[loss_kind]
        merged["is_frozen"] = False

        base_cols = [
            "run_tag",
            "fp_family",
            "fp_display",
            "loss_kind",
            "loss_display",
            "is_frozen",
            "bit_index",
            "best_descriptor_idx",
            "best_descriptor",
            "best_corr",
            "best_abs_corr",
            "bit_frequency_train",
            "r2_linear",
        ]
        if "r2_mlp" in merged.columns:
            base_cols.append("r2_mlp")
        if "rdkit_bit_number" in merged.columns:
            base_cols.append("rdkit_bit_number")

        val_df = merged[base_cols + [col for col in ["auroc_val", "freq_val"] if col in merged.columns]].copy()
        val_df = val_df.rename(columns={"auroc_val": "auroc", "freq_val": "bit_frequency_eval"})
        val_df["split"] = "val"
        val_df["split_display"] = SPLIT_DISPLAY["val"]

        ood_df = merged[base_cols + [col for col in ["auroc_ood", "freq_ood"] if col in merged.columns]].copy()
        ood_df = ood_df.rename(columns={"auroc_ood": "auroc", "freq_ood": "bit_frequency_eval"})
        ood_df["split"] = "ood"
        ood_df["split_display"] = SPLIT_DISPLAY["ood"]

        rows.extend([val_df, ood_df])

    scatter_df = pd.concat(rows, ignore_index=True)
    scatter_df = scatter_df.dropna(subset=["best_descriptor", "r2_linear", "auroc"]).reset_index(drop=True)
    return scatter_df


def compute_spearman_table(scatter_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = scatter_df.groupby(["fp_family", "loss_kind", "split"], sort=False)
    for (fp_family, loss_kind, split), group in grouped:
        valid = group[["r2_linear", "auroc"]].dropna()
        if len(valid) < 2:
            rho = np.nan
            pvalue = np.nan
        else:
            rho, pvalue = spearmanr(valid["r2_linear"], valid["auroc"], nan_policy="omit")
        rows.append(
            {
                "fp_family": fp_family,
                "fp_display": FINGERPRINT_SPECS[fp_family]["display"],
                "loss_kind": loss_kind,
                "loss_display": LOSS_DISPLAY[loss_kind],
                "split": split,
                "split_display": SPLIT_DISPLAY[split],
                "n_pairs": int(len(valid)),
                "spearman_rho": float(rho) if not pd.isna(rho) else np.nan,
                "spearman_pvalue": float(pvalue) if not pd.isna(pvalue) else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    N_BONFERRONI_TESTS = len(table)
    table["bonferroni_pvalue"] = (table["spearman_pvalue"] * N_BONFERRONI_TESTS).clip(upper=1.0)
    table["significant_bonferroni_05"] = table["bonferroni_pvalue"] < 0.05
    return table.sort_values(["split", "fp_family", "loss_kind"]).reset_index(drop=True)


def compute_quantile_thresholds(series: pd.Series) -> tuple[float, float]:
    return float(series.quantile(0.1)), float(series.quantile(0.9))


def compute_plot_x_limits(
    scatter_df: pd.DataFrame,
    split: str,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.995,
) -> tuple[float, float]:
    subset = scatter_df.loc[scatter_df["split"] == split, "r2_linear"].dropna()
    if subset.empty:
        raise ValueError(f"No r2_linear values found for split={split!r}")

    x_low = float(subset.quantile(lower_quantile))
    x_high = float(subset.quantile(upper_quantile))
    span = max(x_high - x_low, 0.1)
    pad = 0.03 * span
    return x_low - pad, min(1.0, x_high + pad)


def compute_quantile_binned_trend(
    df: pd.DataFrame,
    x_col: str = "r2_linear",
    y_col: str = "auroc",
    max_bins: int = 24,
    min_bins: int = 6,
) -> pd.DataFrame:
    valid = df[[x_col, y_col]].dropna().sort_values(x_col).reset_index(drop=True)
    n_rows = len(valid)
    if n_rows < max(4, min_bins):
        return pd.DataFrame(columns=[x_col, y_col, "n_points"])

    n_bins = min(max_bins, max(min_bins, n_rows // 90))
    bin_ids = pd.qcut(np.arange(n_rows), q=n_bins, labels=False, duplicates="drop")
    valid = valid.assign(_trend_bin=bin_ids)
    trend = (
        valid.groupby("_trend_bin", observed=True)
        .agg(
            **{
                x_col: (x_col, "median"),
                y_col: (y_col, "mean"),
                "n_points": (y_col, "size"),
            }
        )
        .reset_index(drop=True)
    )
    return trend


def select_highlight_pairs(
    scatter_df: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    upper_rows: list[pd.DataFrame] = []
    disagreement_rows: list[pd.DataFrame] = []

    for (fp_family, split), group in scatter_df.groupby(["fp_family", "split"], sort=False):
        x_low, x_high = compute_quantile_thresholds(group["r2_linear"])
        y_low, y_high = compute_quantile_thresholds(group["auroc"])

        upper = group[(group["r2_linear"] >= x_high) & (group["auroc"] >= y_high)].copy()
        upper["selection_type"] = "upper_right"
        upper["ranking_score"] = (
            upper["r2_linear"].rank(pct=True) + upper["auroc"].rank(pct=True) + upper["best_abs_corr"].fillna(0.0)
        )
        upper = upper.sort_values("ranking_score", ascending=False).head(top_n)
        upper_rows.append(upper)

        high_x_low_y = group[(group["r2_linear"] >= x_high) & (group["auroc"] <= y_low)].copy()
        high_x_low_y["selection_type"] = "high_r2_low_auroc"
        high_x_low_y["ranking_score"] = (
            high_x_low_y["r2_linear"].rank(pct=True)
            + (1.0 - high_x_low_y["auroc"].rank(pct=True))
        )

        low_x_high_y = group[(group["r2_linear"] <= x_low) & (group["auroc"] >= y_high)].copy()
        low_x_high_y["selection_type"] = "low_r2_high_auroc"
        low_x_high_y["ranking_score"] = (
            low_x_high_y["auroc"].rank(pct=True)
            + (1.0 - low_x_high_y["r2_linear"].rank(pct=True))
        )

        top_high_x_low_y = high_x_low_y.sort_values("ranking_score", ascending=False).head(math.ceil(top_n / 2))
        top_low_x_high_y = low_x_high_y.sort_values("ranking_score", ascending=False).head(math.floor(top_n / 2))
        disagreement = pd.concat([top_high_x_low_y, top_low_x_high_y], ignore_index=True)
        disagreement_rows.append(disagreement)

    upper_df = pd.concat(upper_rows, ignore_index=True) if upper_rows else pd.DataFrame()
    disagreement_df = pd.concat(disagreement_rows, ignore_index=True) if disagreement_rows else pd.DataFrame()
    return upper_df, disagreement_df


def select_panel_annotation_rows(panel_df: pd.DataFrame, max_labels: int) -> pd.DataFrame:
    if panel_df.empty:
        return pd.DataFrame()

    x_low, x_high = compute_quantile_thresholds(panel_df["r2_linear"])
    y_low, y_high = compute_quantile_thresholds(panel_df["auroc"])

    upper = panel_df[(panel_df["r2_linear"] >= x_high) & (panel_df["auroc"] >= y_high)].copy()
    upper["annotation_priority"] = (
        upper["r2_linear"].rank(pct=True) + upper["auroc"].rank(pct=True) + upper["best_abs_corr"].fillna(0.0)
    )
    upper["annotation_group"] = "upper_right"
    upper = upper.sort_values("annotation_priority", ascending=False).head(2)

    high_x_low_y = panel_df[(panel_df["r2_linear"] >= x_high) & (panel_df["auroc"] <= y_low)].copy()
    high_x_low_y["annotation_priority"] = (
        high_x_low_y["r2_linear"].rank(pct=True)
        + (1.0 - high_x_low_y["auroc"].rank(pct=True))
    )
    high_x_low_y["annotation_group"] = "high_r2_low_auroc"
    high_x_low_y = high_x_low_y.sort_values("annotation_priority", ascending=False).head(2)

    low_x_high_y = panel_df[(panel_df["r2_linear"] <= x_low) & (panel_df["auroc"] >= y_high)].copy()
    low_x_high_y["annotation_priority"] = (
        low_x_high_y["auroc"].rank(pct=True)
        + (1.0 - low_x_high_y["r2_linear"].rank(pct=True))
    )
    low_x_high_y["annotation_group"] = "low_r2_high_auroc"
    low_x_high_y = low_x_high_y.sort_values("annotation_priority", ascending=False).head(1)

    label_df = pd.concat([upper, high_x_low_y, low_x_high_y], ignore_index=True)
    label_df = label_df.sort_values("annotation_priority", ascending=False)
    label_df = label_df.drop_duplicates(subset=["best_descriptor"], keep="first")
    return label_df.head(max_labels).reset_index(drop=True)


def add_panel_annotations(ax: Any, label_df: pd.DataFrame) -> None:
    if label_df.empty:
        return

    x_min, x_max = ax.get_xlim()
    x_mid = (x_min + x_max) / 2.0
    for idx, row in enumerate(label_df.itertuples(index=False)):
        dx = -14 if row.r2_linear >= x_mid else 8
        dy = -14 if row.auroc >= 0.75 else 8
        descriptor = str(row.best_descriptor)
        ax.annotate(
            descriptor,
            xy=(row.r2_linear, row.auroc),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            color=SCATTER_PALETTE[row.loss_kind],
            annotation_clip=True,
            arrowprops={"arrowstyle": "-", "color": SCATTER_PALETTE[row.loss_kind], "lw": 0.6, "alpha": 0.7},
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.92},
        )


def create_cross_axis_scatter_figure(
    scatter_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
    output_pdfs: list[Path],
    labels_per_category: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    stats_fontsize = 11
    panel_title_fontsize = 14
    axis_label_fontsize = 13
    tick_label_fontsize = 11
    legend_fontsize = 13

    ood_df = scatter_df[scatter_df["split"] == "ood"].copy()
    if ood_df.empty:
        raise ValueError("No OOD rows available for the cross-axis figure.")

    fp_order = ["ecfp4", "maccs", "map4"]
    x_min, x_max = compute_plot_x_limits(scatter_df, split="ood", lower_quantile=0.05, upper_quantile=0.995)

    fig = plt.figure(figsize=(18, 7.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 4.2], hspace=0.09, wspace=0.16)
    fig.subplots_adjust(top=0.90, bottom=0.13, left=0.07, right=0.98)

    main_axes: list[Any] = []
    hist_axes: list[Any] = []

    for col_idx, fp_family in enumerate(fp_order):
        sharey = main_axes[0] if main_axes else None
        ax = fig.add_subplot(gs[1, col_idx], sharey=sharey)
        ax_hist = fig.add_subplot(gs[0, col_idx], sharex=ax)
        main_axes.append(ax)
        hist_axes.append(ax_hist)

        panel_df = ood_df[ood_df["fp_family"] == fp_family].copy()
        plot_df = panel_df[(panel_df["r2_linear"] >= x_min) & (panel_df["r2_linear"] <= x_max)].copy()

        for loss_kind in ["bce", "cos"]:
            loss_df = plot_df[plot_df["loss_kind"] == loss_kind]
            if loss_df.empty:
                continue

            ax.scatter(
                loss_df["r2_linear"],
                loss_df["auroc"],
                s=14,
                alpha=0.18,
                color=SCATTER_PALETTE[loss_kind],
                marker=SCATTER_MARKERS[loss_kind],
                edgecolor="none",
                rasterized=True,
            )

            trend_df = compute_quantile_binned_trend(panel_df[panel_df["loss_kind"] == loss_kind])
            trend_df = trend_df[(trend_df["r2_linear"] >= x_min) & (trend_df["r2_linear"] <= x_max)]
            if len(trend_df) >= 2:
                ax.plot(
                    trend_df["r2_linear"],
                    trend_df["auroc"],
                    color=SCATTER_PALETTE[loss_kind],
                    linewidth=2.4,
                    alpha=0.95,
                )

            ax_hist.hist(
                loss_df["r2_linear"],
                bins=32,
                density=True,
                histtype="stepfilled",
                alpha=0.12,
                linewidth=0.0,
                color=SCATTER_PALETTE[loss_kind],
            )
            ax_hist.hist(
                loss_df["r2_linear"],
                bins=32,
                density=True,
                histtype="step",
                linewidth=1.0,
                color=SCATTER_PALETTE[loss_kind],
            )

        label_df = select_panel_annotation_rows(plot_df, max_labels=min(4, max(1, labels_per_category * 2)))
        add_panel_annotations(ax, label_df)

        panel_spearman = spearman_df[
            (spearman_df["fp_family"] == fp_family) & (spearman_df["split"] == "ood")
        ]
        stats_lines = []
        for loss_kind in ["bce", "cos"]:
            match = panel_spearman[panel_spearman["loss_kind"] == loss_kind]
            if match.empty:
                continue
            row = match.iloc[0]
            rho = row["spearman_rho"]
            if pd.isna(rho):
                stats_lines.append(f"{LOSS_DISPLAY[loss_kind]} rho = NA")
            else:
                stats_lines.append(f"{LOSS_DISPLAY[loss_kind]} rho = {rho:.3f}")
        if stats_lines:
            ax.text(
                0.03,
                0.97,
                "\n".join(stats_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=stats_fontsize,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d9d9d9", "alpha": 0.95},
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(FINGERPRINT_SPECS[fp_family]["display"], fontsize=panel_title_fontsize, pad=1.5)
        ax.set_xlabel(r"Axis 1 linear probe $R^2$ of best-matching descriptor", fontsize=axis_label_fontsize)
        if col_idx == 0:
            ax.set_ylabel("OOD per-bit AUROC", fontsize=axis_label_fontsize)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)
        ax.grid(alpha=0.22, linewidth=0.6)

        ax_hist.set_xlim(x_min, x_max)
        ax_hist.set_yticks([])
        ax_hist.tick_params(axis="x", labelbottom=False, labelsize=tick_label_fontsize)
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        ax_hist.spines["left"].set_visible(False)
        ax_hist.set_ylabel("")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=SCATTER_MARKERS[loss_kind],
            linestyle="",
            label=LOSS_DISPLAY[loss_kind],
            markerfacecolor=SCATTER_PALETTE[loss_kind],
            markeredgecolor="white",
            markersize=12,
        )
        for loss_kind in ["bce", "cos"]
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.975),
        fontsize=legend_fontsize,
        handletextpad=0.6,
        columnspacing=1.4,
    )

    for output_pdf in output_pdfs:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def smarts_semantic_tags(smarts: str) -> set[str]:
    if smarts == "?":
        return {"unknown"}

    tags: set[str] = set()
    smarts_without_bracket_contents = re.sub(r"\[[^\]]+\]", "[]", smarts)
    element_tags = {
        "#6": "carbon",
        "#7": "nitrogen",
        "#8": "oxygen",
        "#15": "phosphorus",
        "#16": "sulfur",
        "#9": "halogen",
        "#17": "halogen",
        "#35": "halogen",
        "#53": "halogen",
    }
    for token, tag in element_tags.items():
        if token in smarts:
            tags.add(tag)

    if "[NH2]" in smarts:
        tags.update({"nh2", "amine", "nitrogen"})
    if "]#[" in smarts:
        tags.add("triple_bond")
    if "]=" in smarts:
        tags.add("double_bond")
    if re.search(r"\d", smarts_without_bracket_contents):
        tags.add("ring")
    if "[#6]#[#7]" in smarts or "[#7]#[#6]" in smarts:
        tags.add("nitrile")
    if "[#8]~[#7](~[#8])~[#6]" in smarts or "[#7]=[#8]" in smarts:
        tags.add("nitro")
    if "[#7]~[#16]" in smarts or "[#6]~[#16]~[#7]" in smarts:
        tags.update({"sulfonamide", "nitrogen", "sulfur"})
    if "[#15]" in smarts:
        tags.add("phosphorus")
    if "!#6;!#1" in smarts and "ring" in tags:
        tags.add("heterocycle")
    return tags


def descriptor_semantic_tags(descriptor_name: str) -> set[str]:
    lower = descriptor_name.lower()
    tags: set[str] = set()

    if lower.startswith("fr_"):
        tags.add("fragment")
    if "ring" in lower:
        tags.add("ring")
    if "aromatic" in lower or "_ar_" in lower or lower.startswith("fr_ar"):
        tags.add("aromatic")
    if "nitri" in lower:
        tags.update({"nitrile", "nitrogen"})
    if "quatn" in lower:
        tags.update({"quaternary", "nitrogen"})
    if "sulfon" in lower:
        tags.update({"sulfonamide", "sulfur", "nitrogen"})
    if "nitro" in lower:
        tags.update({"nitro", "nitrogen", "oxygen"})
    if "epoxide" in lower:
        tags.update({"epoxide", "ring", "oxygen"})
    if "nh2" in lower:
        tags.update({"nh2", "amine", "nitrogen"})
    if "phos" in lower:
        tags.add("phosphorus")
    if "ester" in lower:
        tags.update({"ester", "oxygen"})
    if "n_o" in lower:
        tags.update({"nitrogen", "oxygen"})
    if "count" in lower and "ring" not in tags:
        tags.add("count")
    if lower.endswith("_n") or "nitrogen" in lower:
        tags.add("nitrogen")
    if lower.endswith("_o") or "oxygen" in lower:
        tags.add("oxygen")
    if lower.endswith("_s") or "sulfur" in lower:
        tags.add("sulfur")
    return tags


def validate_maccs_best_matches(best_matches_maccs: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top_matches = best_matches_maccs.dropna(subset=["best_descriptor"]).sort_values(
        "best_abs_corr", ascending=False
    ).head(top_n)

    rows: list[dict[str, Any]] = []
    for row in top_matches.itertuples(index=False):
        rdkit_bit_number = int(row.rdkit_bit_number)
        smarts, threshold = MACCSkeys.smartsPatts[rdkit_bit_number]
        descriptor_tags = descriptor_semantic_tags(str(row.best_descriptor))
        smarts_tags = smarts_semantic_tags(str(smarts))
        shared_tags = sorted(descriptor_tags & smarts_tags)

        if "unknown" in smarts_tags:
            status = "unclear"
            note = "MACCS SMARTS definition is unknown in RDKit."
        elif len(shared_tags) >= 2:
            status = "aligned"
            note = f"Shared semantic tags: {', '.join(shared_tags)}."
        elif len(shared_tags) == 1:
            status = "partially_aligned"
            note = f"One shared semantic tag: {shared_tags[0]}."
        else:
            status = "not_obvious"
            note = "No obvious semantic overlap between descriptor name and SMARTS pattern."

        rows.append(
            {
                "bit_index": int(row.bit_index),
                "rdkit_bit_number": rdkit_bit_number,
                "best_descriptor": row.best_descriptor,
                "best_corr": float(row.best_corr),
                "best_abs_corr": float(row.best_abs_corr),
                "maccs_smarts": smarts,
                "maccs_count_threshold": int(threshold),
                "descriptor_tags": ";".join(sorted(descriptor_tags)),
                "smarts_tags": ";".join(sorted(smarts_tags)),
                "shared_tags": ";".join(shared_tags),
                "alignment_status": status,
                "alignment_note": note,
            }
        )
    return pd.DataFrame(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    serializable = json.loads(json.dumps(payload, default=str))
    path.write_text(json.dumps(serializable, indent=2) + "\n")


def build_cross_axis_caption(spearman_df: pd.DataFrame) -> str:
    ood_df = spearman_df[spearman_df["split"] == "ood"].copy()
    if ood_df.empty:
        raise ValueError("Spearman table does not contain OOD rows for caption generation.")

    strongest_row = ood_df.sort_values("spearman_rho", ascending=False).iloc[0]
    rho_value = strongest_row["spearman_rho"]
    rho_text = "NA" if pd.isna(rho_value) else f"{rho_value:.2f}"
    loss_text = str(strongest_row["loss_display"]).lower()
    fp_text = str(strongest_row["fp_display"])

    return (
        "Each point represents a fingerprint bit, plotted by the R² of its best-matching Axis 1 descriptor "
        "(x) against its OOD per-bit AUROC (y). Trend lines show quantile-binned moving averages. "
        f"{fp_text} shows the strongest relationship (Spearman rho = {rho_text} for {loss_text} loss), "
        "consistent with its cleaner bit-to-substructure mappings. ECFP4 and MAP4 show weaker correlations, "
        "likely because hashed bit assignments dilute the bit-descriptor correspondence. Negative R² values "
        "indicate descriptors where the linear probe performed worse than the mean-baseline predictor on the "
        "test set."
    )


def build_pair_level_ood_summary(scatter_df: pd.DataFrame) -> pd.DataFrame:
    ood_df = scatter_df[scatter_df["split"] == "ood"].copy()
    if ood_df.empty:
        raise ValueError("No OOD rows available for pair-level summary.")

    meta_cols = [
        "fp_family",
        "fp_display",
        "bit_index",
        "best_descriptor_idx",
        "best_descriptor",
        "best_corr",
        "best_abs_corr",
        "bit_frequency_train",
        "r2_linear",
    ]
    if "r2_mlp" in ood_df.columns:
        meta_cols.append("r2_mlp")
    if "rdkit_bit_number" in ood_df.columns:
        meta_cols.append("rdkit_bit_number")

    meta_df = (
        ood_df[meta_cols]
        .sort_values(["fp_family", "bit_index"])
        .groupby(["fp_family", "bit_index"], as_index=False)
        .first()
    )

    auroc_wide = (
        ood_df.pivot_table(
            index=["fp_family", "bit_index"],
            columns="loss_kind",
            values="auroc",
            aggfunc="first",
        )
        .rename(columns={"bce": "auroc_ood_bce", "cos": "auroc_ood_cos"})
        .reset_index()
    )

    freq_wide = (
        ood_df.pivot_table(
            index=["fp_family", "bit_index"],
            columns="loss_kind",
            values="bit_frequency_eval",
            aggfunc="first",
        )
        .rename(columns={"bce": "bit_frequency_ood_bce", "cos": "bit_frequency_ood_cos"})
        .reset_index()
    )

    pair_df = meta_df.merge(auroc_wide, on=["fp_family", "bit_index"], how="left")
    pair_df = pair_df.merge(freq_wide, on=["fp_family", "bit_index"], how="left")

    pair_df["mean_auroc_ood"] = pair_df[["auroc_ood_bce", "auroc_ood_cos"]].mean(axis=1)
    pair_df["max_auroc_ood"] = pair_df[["auroc_ood_bce", "auroc_ood_cos"]].max(axis=1)
    pair_df["min_auroc_ood"] = pair_df[["auroc_ood_bce", "auroc_ood_cos"]].min(axis=1)
    pair_df["descriptor_repeat_count"] = (
        pair_df.groupby(["fp_family", "best_descriptor"])["bit_index"].transform("nunique")
    )
    return pair_df


def add_ood_quantile_flags(pair_df: pd.DataFrame, scatter_df: pd.DataFrame) -> pd.DataFrame:
    out = pair_df.copy()
    ood_df = scatter_df[scatter_df["split"] == "ood"].copy()

    for fp_family, group in ood_df.groupby("fp_family", sort=False):
        x_low, x_high = compute_quantile_thresholds(group["r2_linear"])
        y_low, y_high = compute_quantile_thresholds(group["auroc"])

        mask = out["fp_family"] == fp_family
        out.loc[mask, "r2_low_quantile"] = x_low
        out.loc[mask, "r2_high_quantile"] = x_high
        out.loc[mask, "auroc_low_quantile"] = y_low
        out.loc[mask, "auroc_high_quantile"] = y_high

        out.loc[mask, "high_r2_flag"] = out.loc[mask, "r2_linear"] >= x_high
        out.loc[mask, "low_r2_flag"] = out.loc[mask, "r2_linear"] <= x_low

        for loss_kind, col_name in [("bce", "auroc_ood_bce"), ("cos", "auroc_ood_cos")]:
            upper_col = f"upper_right_{loss_kind}"
            highlow_col = f"high_r2_low_auroc_{loss_kind}"
            lowhigh_col = f"low_r2_high_auroc_{loss_kind}"
            out.loc[mask, upper_col] = (
                out.loc[mask, "high_r2_flag"] & (out.loc[mask, col_name] >= y_high)
            )
            out.loc[mask, highlow_col] = (
                out.loc[mask, "high_r2_flag"] & (out.loc[mask, col_name] <= y_low)
            )
            out.loc[mask, lowhigh_col] = (
                out.loc[mask, "low_r2_flag"] & (out.loc[mask, col_name] >= y_high)
            )

    bool_cols = [
        "high_r2_flag",
        "low_r2_flag",
        "upper_right_bce",
        "upper_right_cos",
        "high_r2_low_auroc_bce",
        "high_r2_low_auroc_cos",
        "low_r2_high_auroc_bce",
        "low_r2_high_auroc_cos",
    ]
    for col in bool_cols:
        out[col] = out[col].fillna(False).astype(bool)

    out["upper_right_any_loss"] = out["upper_right_bce"] | out["upper_right_cos"]
    out["upper_right_both_losses"] = out["upper_right_bce"] & out["upper_right_cos"]
    out["high_r2_low_auroc_any_loss"] = out["high_r2_low_auroc_bce"] | out["high_r2_low_auroc_cos"]
    out["low_r2_high_auroc_any_loss"] = out["low_r2_high_auroc_bce"] | out["low_r2_high_auroc_cos"]
    return out


def add_pair_ranking_scores(pair_df: pd.DataFrame) -> pd.DataFrame:
    out = pair_df.copy()
    out["r2_rank_pct"] = out.groupby("fp_family")["r2_linear"].rank(pct=True, method="average")
    out["mean_auroc_rank_pct"] = out.groupby("fp_family")["mean_auroc_ood"].rank(pct=True, method="average")
    out["abs_corr_rank_pct"] = out.groupby("fp_family")["best_abs_corr"].rank(pct=True, method="average")

    out["reliable_score"] = (
        out["r2_rank_pct"] + out["mean_auroc_rank_pct"] + out["abs_corr_rank_pct"]
    )
    out["high_r2_low_auroc_score"] = out["r2_rank_pct"] + (1.0 - out["mean_auroc_rank_pct"])
    out["low_r2_high_auroc_score"] = (1.0 - out["r2_rank_pct"]) + out["mean_auroc_rank_pct"]
    return out


def reliable_pair_explanation(row: pd.Series) -> str:
    descriptor = str(row["best_descriptor"])
    fp_family = str(row["fp_family"])

    if fp_family == "maccs":
        return (
            "Predefined MACCS keys map more cleanly onto specific substructures, so this bit-descriptor pair "
            "is both easy to probe and robustly decoded OOD."
        )
    if fp_family == "ecfp4" and descriptor == "FractionCSP3":
        return (
            "This hashed ECFP4 bit tracks sp3-rich chemistry and still decodes well OOD; repeated FractionCSP3 "
            "matches across other ECFP4 bits suggest overlapping hashed environments."
        )
    if fp_family in {"ecfp4", "map4"} and row.get("descriptor_repeat_count", 0) > 1:
        return (
            "The same descriptor appears on multiple hashed bits, but this bit is one of the stable cases where "
            "the descriptor correspondence survives into strong OOD decoding."
        )
    if descriptor.startswith("fr_"):
        return (
            "The matched fragment descriptor likely captures a chemically specific motif that is both linearly "
            "encoded and spectrally recoverable OOD."
        )
    return (
        "The descriptor captures a broad structural property that is both present in the embedding and "
        "predictable from spectra OOD."
    )


def disagreement_pair_explanation(row: pd.Series) -> str:
    descriptor = str(row["best_descriptor"])
    fp_family = str(row["fp_family"])
    disagreement_type = str(row["disagreement_type"])

    if fp_family == "maccs" and descriptor == "fr_ether" and disagreement_type == "high_r2_low_auroc":
        return (
            "Ether chemistry is linearly encoded in the embedding, but this MACCS bit remains hard to predict "
            "OOD, likely because ether fragmentation depends strongly on the surrounding scaffold."
        )
    if fp_family == "ecfp4" and descriptor == "FractionCSP3" and disagreement_type == "high_r2_low_auroc":
        return (
            "Different ECFP4 bits share FractionCSP3 as their best descriptor; divergent AUROC values are "
            "consistent with hash collisions, where one bit tracks the intended sp3 signal and another mixes "
            "unrelated neighborhoods."
        )
    if disagreement_type == "high_r2_low_auroc" and fp_family in {"ecfp4", "map4"}:
        return (
            "The descriptor signal is present in the embedding, but the hashed fingerprint bit likely mixes "
            "multiple substructures or context effects, which hurts OOD decoding."
        )
    if disagreement_type == "low_r2_high_auroc" and fp_family in {"ecfp4", "map4"}:
        return (
            "The bit is decodable from spectra but not well summarized by any single RDKit descriptor, consistent "
            "with hashed bits capturing composite or context-dependent chemistry."
        )
    if disagreement_type == "high_r2_low_auroc":
        return (
            "The chemistry is probeable in the frozen embedding, but this specific bit is not predicted well OOD, "
            "suggesting context-dependent spectral expression."
        )
    return (
        "The bit is predicted well from spectra despite weak descriptor probeability, implying that the target "
        "captures chemistry not well summarized by a single descriptor."
    )


def select_reliable_pairs_per_fingerprint(
    pair_df: pd.DataFrame,
    min_pairs: int = 5,
    max_pairs: int = 10,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for fp_family, group in pair_df.groupby("fp_family", sort=False):
        upper = group[group["upper_right_any_loss"]].copy()
        upper["selection_basis"] = "upper_right_quantile"
        upper = upper.sort_values(
            ["upper_right_both_losses", "reliable_score", "mean_auroc_ood", "best_abs_corr"],
            ascending=[False, False, False, False],
        )

        selected = upper.head(max_pairs).copy()
        target_n = min(max_pairs, len(group))
        min_target = min(min_pairs, target_n)

        if len(selected) < min_target:
            supplement = group[~group["bit_index"].isin(selected["bit_index"])].copy()
            supplement["selection_basis"] = "near_upper_right_fallback"
            supplement = supplement.sort_values(
                ["reliable_score", "mean_auroc_ood", "best_abs_corr"],
                ascending=[False, False, False],
            )
            selected = pd.concat(
                [selected, supplement.head(min_target - len(selected))],
                ignore_index=True,
            )

        selected["possible_explanation"] = selected.apply(reliable_pair_explanation, axis=1)
        rows.append(selected)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["fp_family", "selection_basis", "reliable_score"], ascending=[True, True, False])
    return out.reset_index(drop=True)


def select_disagreement_pairs_per_fingerprint(
    pair_df: pd.DataFrame,
    min_pairs: int = 5,
    max_pairs: int = 10,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for fp_family, group in pair_df.groupby("fp_family", sort=False):
        flagged = group[
            group["high_r2_low_auroc_any_loss"] | group["low_r2_high_auroc_any_loss"]
        ].copy()
        flagged["disagreement_type"] = np.where(
            flagged["high_r2_low_auroc_score"] >= flagged["low_r2_high_auroc_score"],
            "high_r2_low_auroc",
            "low_r2_high_auroc",
        )
        flagged["selection_basis"] = "disagreement_quantile"
        flagged["disagreement_score"] = np.where(
            flagged["disagreement_type"] == "high_r2_low_auroc",
            flagged["high_r2_low_auroc_score"],
            flagged["low_r2_high_auroc_score"],
        )
        flagged = flagged.sort_values(["disagreement_score", "best_abs_corr"], ascending=[False, False])

        selected = flagged.head(max_pairs).copy()
        target_n = min(max_pairs, len(group))
        min_target = min(min_pairs, target_n)

        if len(selected) < min_target:
            supplement = group[~group["bit_index"].isin(selected["bit_index"])].copy()
            supplement["disagreement_type"] = np.where(
                supplement["high_r2_low_auroc_score"] >= supplement["low_r2_high_auroc_score"],
                "high_r2_low_auroc",
                "low_r2_high_auroc",
            )
            supplement["selection_basis"] = "near_disagreement_fallback"
            supplement["disagreement_score"] = np.where(
                supplement["disagreement_type"] == "high_r2_low_auroc",
                supplement["high_r2_low_auroc_score"],
                supplement["low_r2_high_auroc_score"],
            )
            supplement = supplement.sort_values(
                ["disagreement_score", "best_abs_corr"],
                ascending=[False, False],
            )
            selected = pd.concat(
                [selected, supplement.head(min_target - len(selected))],
                ignore_index=True,
            )

        selected["possible_explanation"] = selected.apply(disagreement_pair_explanation, axis=1)
        rows.append(selected)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["fp_family", "selection_basis", "disagreement_score"], ascending=[True, True, False])
    return out.reset_index(drop=True)


def build_cross_axis_discussion_notes(
    reliable_df: pd.DataFrame,
    disagreement_df: pd.DataFrame,
) -> str:
    lines = ["Cross-axis bridge discussion notes", ""]

    ecfp4_fraction = disagreement_df[
        (disagreement_df["fp_family"] == "ecfp4") & (disagreement_df["best_descriptor"] == "FractionCSP3")
    ]
    if len(ecfp4_fraction):
        lines.append(
            "- ECFP4: repeated FractionCSP3 matches across different bits with divergent AUROC values are "
            "consistent with hash-collision noise, where one hashed bit tracks sp3-rich chemistry well and "
            "another mixes unrelated neighborhoods."
        )

    maccs_ether = disagreement_df[
        (disagreement_df["fp_family"] == "maccs")
        & (disagreement_df["best_descriptor"] == "fr_ether")
        & (disagreement_df["disagreement_type"] == "high_r2_low_auroc")
    ]
    if len(maccs_ether):
        lines.append(
            "- MACCS: fr_ether is a clear high-R^2 / low-AUROC disagreement case, suggesting that ether-related "
            "information is linearly present in the embedding but harder to decode OOD because fragmentation "
            "around ether bonds is scaffold-dependent."
        )

    hashed_low_r2_high_auroc = disagreement_df[
        disagreement_df["fp_family"].isin(["ecfp4", "map4"])
        & (disagreement_df["disagreement_type"] == "low_r2_high_auroc")
    ]
    if len(hashed_low_r2_high_auroc):
        lines.append(
            "- Hashed fingerprints: several low-R^2 / high-AUROC cases indicate that spectra can decode bits "
            "that do not align cleanly to any single RDKit descriptor, which is consistent with mixed or "
            "composite hashed substructure assignments."
        )

    if len(reliable_df):
        lines.append(
            "- Reliable pairs: the strongest upper-right cases are dominated by MACCS and a subset of hashed "
            "ECFP4/MAP4 bits whose best-matching descriptors correspond to broad structural or fragment-level chemistry."
        )

    return "\n".join(lines) + "\n"


def run_analysis(cfg: BridgeConfig) -> dict[str, Any]:
    ensure_dir(cfg.output_dir)
    figure_dir = ensure_dir(cfg.output_dir / "figures")

    axis1_scores = load_axis1_scores(cfg)
    aligned_desc_df, descriptor_cols, alignment_summary = load_training_descriptor_frame(cfg, axis1_scores)
    x_desc = aligned_desc_df[descriptor_cols].to_numpy(dtype=np.float32)

    fingerprint_matrices, fingerprint_summary = build_fingerprint_matrices(
        aligned_desc_df["smiles"].astype(str).tolist(),
        cfg,
    )

    best_match_tables: dict[str, pd.DataFrame] = {}
    pointbiserial_checks: list[pd.DataFrame] = []

    for fp_family, x_bits in fingerprint_matrices.items():
        spec = FINGERPRINT_SPECS[fp_family]
        corr_matrix = compute_pointbiserial_matrix_vectorized(x_bits, x_desc)

        matrix_csv = cfg.output_dir / f"{spec['file_stem']}_bit_descriptor_pointbiserial.csv"
        matrix_npy = cfg.output_dir / f"{spec['file_stem']}_bit_descriptor_pointbiserial.npy"
        save_pointbiserial_matrix(corr_matrix, descriptor_cols, matrix_csv, matrix_npy)

        best_matches = build_best_match_table(corr_matrix, descriptor_cols, x_bits, fp_family)
        best_matches_path = cfg.output_dir / f"best_matches_{spec['file_stem']}.csv"
        best_matches.to_csv(best_matches_path, index=False)
        best_match_tables[fp_family] = best_matches

        check_df = validate_pointbiserial_equivalence(
            fp_family=fp_family,
            corr_matrix=corr_matrix,
            x_bits=x_bits,
            x_desc=x_desc,
            descriptor_cols=descriptor_cols,
            n_pairs=cfg.pointbiserial_validation_pairs,
            random_state=cfg.random_state,
        )
        pointbiserial_checks.append(check_df)

    pd.Series(descriptor_cols, name="descriptor").to_csv(
        cfg.output_dir / "descriptor_columns_used.csv",
        index=False,
    )
    pd.DataFrame([alignment_summary]).to_csv(cfg.output_dir / "training_alignment_summary.csv", index=False)

    pointbiserial_check_df = pd.concat(pointbiserial_checks, ignore_index=True)
    pointbiserial_check_df.to_csv(cfg.output_dir / "pointbiserial_equivalence_check.csv", index=False)

    maccs_validation_df = validate_maccs_best_matches(best_match_tables["maccs"], cfg.top_maccs_validation)
    maccs_validation_df.to_csv(cfg.output_dir / "maccs_top10_smarts_validation.csv", index=False)

    scatter_df = load_fine_tuned_scatter_table(cfg, axis1_scores, best_match_tables)
    scatter_df.to_csv(cfg.output_dir / "fine_tuned_cross_axis_scatter_data.csv", index=False)

    spearman_df = compute_spearman_table(scatter_df)
    spearman_df.to_csv(cfg.output_dir / "fine_tuned_cross_axis_spearman_table.csv", index=False)

    caption_text = build_cross_axis_caption(spearman_df)
    caption_path = cfg.output_dir / "cross_axis_bridge_linear_r2_vs_auroc_ood_caption.txt"
    caption_path.write_text(caption_text + "\n")

    upper_right_df, disagreement_df = select_highlight_pairs(scatter_df, cfg.top_pairs_per_group)
    upper_right_df.to_csv(cfg.output_dir / "fine_tuned_upper_right_pairs.csv", index=False)
    disagreement_df.to_csv(cfg.output_dir / "fine_tuned_axes_disagreement_pairs.csv", index=False)

    ood_pair_df = build_pair_level_ood_summary(scatter_df)
    ood_pair_df = add_ood_quantile_flags(ood_pair_df, scatter_df)
    ood_pair_df = add_pair_ranking_scores(ood_pair_df)
    ood_pair_df.to_csv(cfg.output_dir / "ood_pair_level_summary.csv", index=False)

    reliable_pairs_df = select_reliable_pairs_per_fingerprint(ood_pair_df, min_pairs=5, max_pairs=10)
    reliable_pairs_df.to_csv(cfg.output_dir / "ood_reliable_pairs_by_fingerprint.csv", index=False)

    disagreement_pairs_df = select_disagreement_pairs_per_fingerprint(ood_pair_df, min_pairs=5, max_pairs=10)
    disagreement_pairs_df.to_csv(cfg.output_dir / "ood_disagreement_pairs_by_fingerprint.csv", index=False)

    discussion_notes_text = build_cross_axis_discussion_notes(reliable_pairs_df, disagreement_pairs_df)
    discussion_notes_path = cfg.output_dir / "cross_axis_bridge_discussion_notes.txt"
    discussion_notes_path.write_text(discussion_notes_text)

    scatter_pdf = figure_dir / "cross_axis_bridge_linear_r2_vs_auroc_ood.pdf"
    legacy_scatter_pdf = figure_dir / "cross_axis_bridge_linear_r2_vs_auroc.pdf"
    create_cross_axis_scatter_figure(
        scatter_df,
        spearman_df,
        output_pdfs=[scatter_pdf, legacy_scatter_pdf],
        labels_per_category=cfg.labels_per_category,
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "thesis_root": str(cfg.thesis_root),
        "output_dir": str(cfg.output_dir),
        "inputs": {
            "descriptor_table_path": str(cfg.descriptor_table_path),
            "axis1_merged_path": str(cfg.axis1_merged_path),
            "axis1_linear_path": str(cfg.axis1_linear_path),
            "axis1_mlp_path": str(cfg.axis1_mlp_path),
            "finetuning_hdf5_path": str(cfg.finetuning_hdf5_path),
            "model_runs_dir": str(cfg.model_runs_dir),
        },
        "fingerprint_bits": {fp_family: spec["bits"] for fp_family, spec in FINGERPRINT_SPECS.items()},
        "fine_tuned_run_tags": list(cfg.fine_tuned_run_tags),
        "all_run_tags": list(cfg.all_run_tags),
        "training_alignment_summary": alignment_summary,
        "fingerprint_summary": fingerprint_summary,
        "axis1_descriptor_count": int(axis1_scores["descriptor"].nunique()),
        "scatter_rows": int(len(scatter_df)),
    }
    save_json(cfg.output_dir / "analysis_manifest.json", manifest)

    return {
        "axis1_scores": axis1_scores,
        "aligned_desc_df": aligned_desc_df,
        "best_match_tables": best_match_tables,
        "pointbiserial_check_df": pointbiserial_check_df,
        "maccs_validation_df": maccs_validation_df,
        "scatter_df": scatter_df,
        "spearman_df": spearman_df,
        "caption_text": caption_text,
        "caption_path": caption_path,
        "ood_pair_df": ood_pair_df,
        "reliable_pairs_df": reliable_pairs_df,
        "disagreement_pairs_df": disagreement_pairs_df,
        "discussion_notes_text": discussion_notes_text,
        "discussion_notes_path": discussion_notes_path,
        "upper_right_df": upper_right_df,
        "disagreement_df": disagreement_df,
        "output_dir": cfg.output_dir,
        "scatter_pdf": scatter_pdf,
    }


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)

    print("Cross-axis bridge configuration:")
    print(f"  Thesis root:      {cfg.thesis_root}")
    print(f"  Output dir:       {cfg.output_dir}")
    print(f"  Descriptor table: {cfg.descriptor_table_path}")
    print(f"  Axis 1 scores:    {cfg.axis1_merged_path}")
    print(f"  Fine-tune HDF5:   {cfg.finetuning_hdf5_path}")
    print(f"  Model runs dir:   {cfg.model_runs_dir}")
    print(f"  MAP4 available:   {MAP4_AVAILABLE}")
    print(f"  MAP4 bits:        {cfg.map4_bits}")
    print(f"  Fine-tuned runs:  {len(cfg.fine_tuned_run_tags)}")

    results = run_analysis(cfg)

    print("\nSaved outputs:")
    print(f"  Output dir:    {results['output_dir']}")
    print(f"  Scatter PDF:   {results['scatter_pdf']}")
    print(f"  Caption text:  {results['caption_path']}")
    print("  Key tables:")
    print("    - training_alignment_summary.csv")
    print("    - pointbiserial_equivalence_check.csv")
    print("    - maccs_top10_smarts_validation.csv")
    print("    - fine_tuned_cross_axis_scatter_data.csv")
    print("    - fine_tuned_cross_axis_spearman_table.csv")
    print("    - cross_axis_bridge_linear_r2_vs_auroc_ood_caption.txt")
    print("    - ood_pair_level_summary.csv")
    print("    - ood_reliable_pairs_by_fingerprint.csv")
    print("    - ood_disagreement_pairs_by_fingerprint.csv")
    print("    - cross_axis_bridge_discussion_notes.txt")
    print("    - fine_tuned_upper_right_pairs.csv")
    print("    - fine_tuned_axes_disagreement_pairs.csv")

    spearman_df = results["spearman_df"]
    significant_count = int(spearman_df["significant_bonferroni_05"].sum())
    total_correlations = int(len(spearman_df))
    print(
        f"{significant_count} of {total_correlations} correlations significant "
        "under Bonferroni at alpha = 0.05"
    )


if __name__ == "__main__":
    main()
