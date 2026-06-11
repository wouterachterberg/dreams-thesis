"""
Aggregate all Axis 2 metrics across model runs into a single comparison table.
Run from the DreaMS project root.

Usage:
    python dreams-thesis-wa/scripts/aggregate_all_metrics.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ---- Config ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_OUTPUT = PROJECT_ROOT / "dreams-thesis-wa" / "results" / "model_runs"
OUT_DIR = PROJECT_ROOT / "dreams-thesis-wa" / "results" / "cross_run_integration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_TAGS = [
    # Fine-tuned
    "morgan_2048_cos",
    "morgan_2048_bce",
    "maccs_166_cos",
    "maccs_166_bce",
    "map4_2048_cos",
    "map4_2048_bce",
    # Frozen baselines
    "morgan_2048_cos_frozen",
    "morgan_2048_bce_frozen",
    "maccs_166_cos_frozen",
    "maccs_166_bce_frozen",
    "map4_2048_cos_frozen",
    "map4_2048_bce_frozen",
]

# Keep retrieval runtime reasonable for larger splits.
MAX_RETRIEVAL_QUERIES = 5000
RNG_SEED = 3407


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def cosine_sim_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two matrices."""
    num = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(denom, 1e-8)


def tanimoto_sim_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Tanimoto similarity between two binary matrices."""
    intersection = np.sum(a * b, axis=1)
    union = np.sum(a, axis=1) + np.sum(b, axis=1) - intersection
    return intersection / np.maximum(union, 1e-8)


def infer_fp_type(tag: str) -> str:
    if "morgan" in tag:
        return "morgan_2048"
    if "maccs" in tag:
        return "maccs_166"
    if "map4" in tag:
        return "map4_2048"
    return "unknown"


def infer_loss(tag: str) -> str:
    return "bce" if "_bce" in tag else ("cos" if "_cos" in tag else "unknown")


def read_apply_sigmoid(tag: str, run_dir: Path) -> bool:
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            if "apply_sigmoid_to_pred" in cfg:
                return bool(cfg["apply_sigmoid_to_pred"])
        except Exception:
            pass
    return infer_loss(tag) == "bce"


def compute_retrieval_metrics(y_pred_prob: np.ndarray, y_true: np.ndarray, best_tau: float) -> dict:
    y_bin_best = (y_pred_prob >= best_tau).astype(np.uint8)
    n_samples = int(y_true.shape[0])

    if n_samples > MAX_RETRIEVAL_QUERIES:
        rng = np.random.RandomState(RNG_SEED)
        idx = rng.choice(n_samples, size=MAX_RETRIEVAL_QUERIES, replace=False)
        y_bin_query = y_bin_best[idx]
        query_true = y_true[idx]
    else:
        idx = np.arange(n_samples)
        y_bin_query = y_bin_best
        query_true = y_true

    # Search against full true library.
    y_true_lib = y_true

    # Jaccard distance for binary vectors: distance = 1 - Tanimoto similarity.
    tan_dist = cdist(y_bin_query, y_true_lib, metric="jaccard")

    ranks = []
    for i in range(len(y_bin_query)):
        sorted_idx = np.argsort(tan_dist[i])
        true_fp = query_true[i]
        rank_val = n_samples
        for rank, j in enumerate(sorted_idx, start=1):
            if np.array_equal(y_true_lib[j], true_fp):
                rank_val = rank
                break
        ranks.append(rank_val)

    ranks = np.asarray(ranks, dtype=np.int32)
    return {
        "acc_at_1": float(np.mean(ranks == 1)),
        "acc_at_5": float(np.mean(ranks <= 5)),
        "acc_at_10": float(np.mean(ranks <= 10)),
        "acc_at_20": float(np.mean(ranks <= 20)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "retrieval_queries": int(len(y_bin_query)),
        "retrieval_library_size": int(len(y_true_lib)),
    }


def compute_metrics_for_split(y_pred: np.ndarray, y_true: np.ndarray, apply_sigmoid: bool = False) -> dict:
    """Compute all metrics for one split (val or OOD)."""
    metrics = {}

    y_pred_prob = sigmoid(y_pred) if apply_sigmoid else y_pred

    # 1. Cosine similarity (continuous predictions vs true)
    cos_sims = cosine_sim_rows(y_pred_prob, y_true)
    metrics["cosine_sim_mean"] = float(np.mean(cos_sims))
    metrics["cosine_sim_median"] = float(np.median(cos_sims))

    # 2. Per-bit AUROC with constant-bit filtering
    aurocs = []
    for bit in range(y_true.shape[1]):
        true_col = y_true[:, bit]
        if true_col.sum() == 0 or true_col.sum() == len(true_col):
            continue
        try:
            aurocs.append(float(roc_auc_score(true_col, y_pred_prob[:, bit])))
        except Exception:
            continue

    metrics["mean_auroc"] = float(np.mean(aurocs)) if aurocs else np.nan
    metrics["median_auroc"] = float(np.median(aurocs)) if aurocs else np.nan
    metrics["valid_bits"] = int(len(aurocs))

    # 3. Threshold sweep for mean Tanimoto
    best_tanimoto = -1.0
    best_tau = 0.5
    for tau in np.arange(0.05, 1.00, 0.05):
        y_bin = (y_pred_prob >= tau).astype(np.uint8)
        tan_sims = tanimoto_sim_rows(y_bin, y_true)
        mean_tan = float(np.mean(tan_sims))
        if mean_tan > best_tanimoto:
            best_tanimoto = mean_tan
            best_tau = float(tau)

    metrics["best_tanimoto"] = float(best_tanimoto)
    metrics["optimal_tau"] = float(best_tau)

    y_bin_05 = (y_pred_prob >= 0.5).astype(np.uint8)
    tan_05 = tanimoto_sim_rows(y_bin_05, y_true)
    metrics["tanimoto_at_0.5"] = float(np.mean(tan_05))

    # 4. Retrieval metrics
    retrieval = compute_retrieval_metrics(y_pred_prob, y_true, best_tau=best_tau)
    metrics.update(retrieval)

    return metrics


def safe_gap(a: float, b: float) -> float:
    if pd.notna(a) and pd.notna(b):
        return float(a - b)
    return np.nan


def main() -> None:
    all_results = []

    for tag in RUN_TAGS:
        run_dir = BASE_OUTPUT / tag / "axis2_artifacts"
        print(f"\n{'=' * 60}")
        print(f"Processing: {tag}")
        print(f"{'=' * 60}")

        row = {
            "run_tag": tag,
            "fp_type": infer_fp_type(tag),
            "loss": infer_loss(tag),
            "frozen": bool("_frozen" in tag),
        }

        files_needed = {
            "y_pred": run_dir / "y_pred.npy",
            "y_true": run_dir / "y_true.npy",
            "y_pred_val": run_dir / "y_pred_val.npy",
            "y_true_val": run_dir / "y_true_val.npy",
        }

        missing = [k for k, v in files_needed.items() if not v.exists()]
        if missing:
            print(f"  SKIPPING - missing: {missing}")
            row["status"] = f"missing: {', '.join(missing)}"
            all_results.append(row)
            continue

        row["status"] = "ok"

        y_pred_ood = np.load(files_needed["y_pred"])
        y_true_ood = np.load(files_needed["y_true"])
        y_pred_val = np.load(files_needed["y_pred_val"])
        y_true_val = np.load(files_needed["y_true_val"])

        print(f"  OOD shape: pred={y_pred_ood.shape}, true={y_true_ood.shape}")
        print(f"  Val shape: pred={y_pred_val.shape}, true={y_true_val.shape}")

        apply_sig = read_apply_sigmoid(tag, run_dir)
        row["apply_sigmoid"] = bool(apply_sig)
        print(f"  Apply sigmoid: {apply_sig}")

        print("  Computing VAL metrics...")
        val_metrics = compute_metrics_for_split(y_pred_val, y_true_val, apply_sigmoid=apply_sig)
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v

        print("  Computing OOD metrics...")
        ood_metrics = compute_metrics_for_split(y_pred_ood, y_true_ood, apply_sigmoid=apply_sig)
        for k, v in ood_metrics.items():
            row[f"ood_{k}"] = v

        row["auroc_gap"] = safe_gap(row.get("val_mean_auroc"), row.get("ood_mean_auroc"))
        row["cosine_gap"] = safe_gap(row.get("val_cosine_sim_mean"), row.get("ood_cosine_sim_mean"))

        all_results.append(row)

        val_auc = row.get("val_mean_auroc", np.nan)
        ood_auc = row.get("ood_mean_auroc", np.nan)
        val_auc_txt = f"{val_auc:.4f}" if pd.notna(val_auc) else "N/A"
        ood_auc_txt = f"{ood_auc:.4f}" if pd.notna(ood_auc) else "N/A"
        print(f"  Done. Val AUROC={val_auc_txt}, OOD AUROC={ood_auc_txt}")

    df = pd.DataFrame(all_results)

    if len(df) == 0:
        raise RuntimeError("No runs processed.")

    df = df.sort_values(["fp_type", "frozen", "loss", "run_tag"]).reset_index(drop=True)

    out_path_root = BASE_OUTPUT / "full_comparison_table.csv"
    out_path_axis = OUT_DIR / "full_comparison_table.csv"
    df.to_csv(out_path_root, index=False)
    df.to_csv(out_path_axis, index=False)
    print(f"\nFull table saved to: {out_path_root}")
    print(f"Full table saved to: {out_path_axis}")

    summary_cols = [
        "run_tag",
        "fp_type",
        "loss",
        "frozen",
        "apply_sigmoid",
        "status",
        "val_mean_auroc",
        "ood_mean_auroc",
        "auroc_gap",
        "val_cosine_sim_mean",
        "ood_cosine_sim_mean",
        "cosine_gap",
        "val_best_tanimoto",
        "ood_best_tanimoto",
        "val_optimal_tau",
        "ood_optimal_tau",
        "val_tanimoto_at_0.5",
        "ood_tanimoto_at_0.5",
        "val_acc_at_1",
        "ood_acc_at_1",
        "val_acc_at_10",
        "ood_acc_at_10",
        "val_median_rank",
        "ood_median_rank",
        "val_retrieval_queries",
        "ood_retrieval_queries",
    ]

    existing_cols = [c for c in summary_cols if c in df.columns]
    summary = df[existing_cols].copy()

    summary_out_root = BASE_OUTPUT / "full_comparison_summary.csv"
    summary_out_axis = OUT_DIR / "full_comparison_summary.csv"
    summary.to_csv(summary_out_root, index=False)
    summary.to_csv(summary_out_axis, index=False)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("FROZEN vs FINE-TUNED COMPARISON")
    print("=" * 80)

    for fp in ["morgan_2048", "maccs_166", "map4_2048"]:
        for loss in ["bce", "cos"]:
            ft_tag = f"{fp}_{loss}"
            fr_tag = f"{fp}_{loss}_frozen"
            ft_row = df[df["run_tag"] == ft_tag]
            fr_row = df[df["run_tag"] == fr_tag]
            if ft_row.empty or fr_row.empty:
                continue

            print(f"\n--- {fp} / {loss} ---")
            for metric in [
                "val_mean_auroc",
                "ood_mean_auroc",
                "val_cosine_sim_mean",
                "ood_cosine_sim_mean",
                "val_best_tanimoto",
                "ood_best_tanimoto",
                "val_acc_at_1",
                "ood_acc_at_1",
                "val_median_rank",
                "ood_median_rank",
            ]:
                if metric not in ft_row.columns or metric not in fr_row.columns:
                    continue

                ft_val = ft_row[metric].values[0]
                fr_val = fr_row[metric].values[0]
                if pd.isna(ft_val) or pd.isna(fr_val):
                    continue

                diff = float(ft_val - fr_val)
                winner = "FT" if diff > 0 else "FROZEN"
                if "rank" in metric:
                    winner = "FT" if diff < 0 else "FROZEN"

                print(
                    f"  {metric:30s}  FT={ft_val:8.4f}  FROZEN={fr_val:8.4f}  diff={diff:+.4f}  -> {winner}"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
