#!/usr/bin/env python3
"""
Helpers for building publication-quality Axis 2 figures from saved model artifacts.

The companion notebook `Axis2_publication_figures.ipynb` imports this module so that:
- each output cell can stay focused on a single figure or table
- the notebook reuses a single, reproducible data-loading layer
- figures are saved consistently to results/axis2/figures
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "axis2_publication_figures_mplconfig"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FP_ORDER = ["morgan_2048", "maccs_166", "map4_2048"]
FP_DISPLAY = {
    "morgan_2048": "ECFP4",
    "maccs_166": "MACCS",
    "map4_2048": "MAP4",
}
LOSS_ORDER = ["bce", "cos"]
LOSS_DISPLAY = {
    "bce": "BCE",
    "cos": "Cosine",
}
LOSS_COLORS = {
    "bce": "#0b6e4f",
    "cos": "#b85c38",
}
SPLIT_ORDER = ["val", "ood"]
SPLIT_DISPLAY = {
    "val": "ID-Val",
    "ood": "OOD",
}
SPLIT_COLORS = {
    "val": "#3b82f6",
    "ood": "#ef4444",
}
SPLIT_LINESTYLES = {
    "val": "--",
    "ood": "-",
}
MODEL_VARIANT_ORDER = ["fine_tuned", "frozen"]
MODEL_VARIANT_DISPLAY = {
    "fine_tuned": "Fine-tuned",
    "frozen": "Frozen",
}
PUBLICATION_DPI = 300


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": PUBLICATION_DPI,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def default_paths(thesis_root: Path | None = None) -> dict[str, Path]:
    if thesis_root is None:
        thesis_root = Path(__file__).resolve().parents[1]
    thesis_root = thesis_root.resolve()
    results_dir = thesis_root / "results"
    return {
        "thesis_root": thesis_root,
        "results_dir": results_dir,
        "model_runs_dir": results_dir / "model_runs",
        "cross_axis_dir": results_dir / "cross_axis",
        "output_dir": results_dir / "axis2" / "figures",
    }


def parse_run_tag(run_tag: str) -> dict[str, Any]:
    frozen = run_tag.endswith("_frozen")
    base = run_tag[:-7] if frozen else run_tag

    if base.endswith("_bce"):
        loss_kind = "bce"
        fp_type = base[:-4]
    elif base.endswith("_cos"):
        loss_kind = "cos"
        fp_type = base[:-4]
    else:
        raise ValueError(f"Unsupported run tag: {run_tag}")

    return {
        "run_tag": run_tag,
        "fp_type": fp_type,
        "fp_display": FP_DISPLAY[fp_type],
        "loss_kind": loss_kind,
        "loss_display": LOSS_DISPLAY[loss_kind],
        "frozen": frozen,
        "model_variant": "frozen" if frozen else "fine_tuned",
        "model_variant_display": MODEL_VARIANT_DISPLAY["frozen" if frozen else "fine_tuned"],
    }


def pivot_metric_table(df: pd.DataFrame, key_col: str, value_col: str) -> dict[str, Any]:
    if df.empty:
        return {}
    return dict(zip(df[key_col].astype(str), df[value_col], strict=False))


def pivot_split_table(
    df: pd.DataFrame,
    split_col: str,
    value_cols: list[str],
    prefixes: list[str] | None = None,
) -> dict[str, Any]:
    if df.empty:
        return {}

    if prefixes is None:
        prefixes = value_cols

    out: dict[str, Any] = {}
    for _, row in df.iterrows():
        split = str(row[split_col]).strip().lower()
        for value_col, prefix in zip(value_cols, prefixes, strict=False):
            out[f"{prefix}_{split}"] = row[value_col]
    return out


def pivot_retrieval_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or not {"Metric", "Validation", "OOD"}.issubset(df.columns):
        return {}

    out: dict[str, Any] = {}
    for _, row in df.iterrows():
        metric = str(row["Metric"]).strip().lower()
        metric = metric.replace(" ", "_").replace("@", "at").replace("-", "_")
        out[f"{metric}_val"] = row["Validation"]
        out[f"{metric}_ood"] = row["OOD"]
    return out


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, Path]:
    output_dir = ensure_output_dir(output_dir)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=PUBLICATION_DPI)
    fig.savefig(pdf_path, bbox_inches="tight")
    return {"png": png_path, "pdf": pdf_path}


def enrich_threshold_table(
    threshold_table: pd.DataFrame,
    sweep_val: pd.DataFrame,
    sweep_ood: pd.DataFrame,
) -> pd.DataFrame:
    sweep_tables = {
        "val": sweep_val.copy(),
        "ood": sweep_ood.copy(),
    }
    out = threshold_table.copy()

    for split, sweep_df in sweep_tables.items():
        sweep_df["threshold"] = pd.to_numeric(sweep_df["threshold"], errors="coerce")
        for col in ["precision", "recall", "f1"]:
            sweep_df[col] = pd.to_numeric(sweep_df[col], errors="coerce")

        mask = out["split"].astype(str).str.lower() == split
        for row_idx in out.index[mask]:
            tau = pd.to_numeric(out.at[row_idx, "best_tau_tanimoto"], errors="coerce")
            if pd.isna(tau):
                continue

            matches = sweep_df[np.isclose(sweep_df["threshold"], float(tau), atol=1e-9)]
            if matches.empty:
                nearest_idx = (sweep_df["threshold"] - float(tau)).abs().idxmin()
                match = sweep_df.loc[nearest_idx]
            else:
                match = matches.iloc[0]

            out.at[row_idx, "precision_at_best_tau"] = float(match["precision"])
            out.at[row_idx, "recall_at_best_tau"] = float(match["recall"])
            out.at[row_idx, "f1_at_best_tau"] = float(match["f1"])

    return out


def load_run_artifact_data(run_dir: Path) -> dict[str, Any]:
    artifacts_dir = run_dir / "axis2_artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Missing axis2_artifacts for {run_dir}")

    auroc_comparison = pd.read_csv(artifacts_dir / "per_bit_auroc" / "auroc_comparison.csv")
    auroc_summary = pd.read_csv(artifacts_dir / "per_bit_auroc" / "auroc_summary_table.csv")
    threshold_table = pd.read_csv(artifacts_dir / "threshold_sweep" / "threshold_comparison_table.csv")
    sweep_val = pd.read_csv(artifacts_dir / "threshold_sweep" / "sweep_results_val.csv")
    sweep_ood = pd.read_csv(artifacts_dir / "threshold_sweep" / "sweep_results_ood.csv")
    threshold_table = enrich_threshold_table(threshold_table, sweep_val, sweep_ood)
    retrieval_metrics = pd.read_csv(artifacts_dir / "retrieval" / "retrieval_metrics.csv")
    ranks_val = np.load(artifacts_dir / "retrieval" / "ranks_val.npy")
    ranks_ood = np.load(artifacts_dir / "retrieval" / "ranks_ood.npy")

    return {
        "artifacts_dir": artifacts_dir,
        "auroc_comparison": auroc_comparison,
        "auroc_summary": auroc_summary,
        "threshold_table": threshold_table,
        "sweep_val": sweep_val,
        "sweep_ood": sweep_ood,
        "retrieval_metrics": retrieval_metrics,
        "ranks_val": ranks_val,
        "ranks_ood": ranks_ood,
    }


def build_master_comparison_table(model_runs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(model_runs_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "axis2_artifacts").exists():
            continue

        info = parse_run_tag(run_dir.name)
        artifacts = load_run_artifact_data(run_dir)
        auroc_summary = pivot_metric_table(artifacts["auroc_summary"], "metric", "value")
        threshold_summary = pivot_split_table(
            artifacts["threshold_table"],
            split_col="split",
            value_cols=[
                "best_tau_tanimoto",
                "best_tanimoto",
                "precision_at_best_tau",
                "recall_at_best_tau",
                "f1_at_best_tau",
            ],
            prefixes=[
                "optimal_tau",
                "best_tanimoto",
                "precision_at_optimal_tau",
                "recall_at_optimal_tau",
                "f1_at_optimal_tau",
            ],
        )
        retrieval_summary = pivot_retrieval_metrics(artifacts["retrieval_metrics"])

        bit_density_val = float(artifacts["auroc_comparison"]["freq_val"].mean())
        bit_density_ood = float(artifacts["auroc_comparison"]["freq_ood"].mean())
        n_bits = int(len(artifacts["auroc_comparison"]))

        row = {
            **info,
            "n_bits": n_bits,
            "bit_density_val": bit_density_val,
            "bit_density_ood": bit_density_ood,
            "mean_auroc_val": float(auroc_summary.get("mean_per_bit_auroc_val", np.nan)),
            "mean_auroc_ood": float(auroc_summary.get("mean_per_bit_auroc_ood", np.nan)),
            "mean_auroc_drop": float(auroc_summary.get("mean_per_bit_auroc_drop", np.nan)),
            "artifacts_dir": str(artifacts["artifacts_dir"]),
        }
        row.update(threshold_summary)
        row.update(retrieval_summary)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    summary_df["fp_display"] = pd.Categorical(
        summary_df["fp_display"], categories=[FP_DISPLAY[x] for x in FP_ORDER], ordered=True
    )
    summary_df["loss_display"] = pd.Categorical(
        summary_df["loss_display"], categories=[LOSS_DISPLAY[x] for x in LOSS_ORDER], ordered=True
    )
    summary_df["model_variant_display"] = pd.Categorical(
        summary_df["model_variant_display"],
        categories=[MODEL_VARIANT_DISPLAY[x] for x in MODEL_VARIANT_ORDER],
        ordered=True,
    )
    return summary_df.sort_values(["fp_display", "loss_display", "model_variant_display"]).reset_index(drop=True)


def load_publication_bundle(thesis_root: Path | None = None) -> dict[str, Any]:
    paths = default_paths(thesis_root)
    summary_df = build_master_comparison_table(paths["model_runs_dir"])
    return {
        **paths,
        "summary_df": summary_df,
        "fine_tuned_df": summary_df[~summary_df["frozen"]].copy(),
        "frozen_df": summary_df[summary_df["frozen"]].copy(),
    }


def iter_fine_tuned_rows(bundle: dict[str, Any]) -> pd.DataFrame:
    df = bundle["fine_tuned_df"].copy()
    return df.sort_values(["fp_display", "loss_display"]).reset_index(drop=True)


def load_artifacts_for_run(bundle: dict[str, Any], run_tag: str) -> dict[str, Any]:
    return load_run_artifact_data(bundle["model_runs_dir"] / run_tag)


def plot_per_bit_auroc_distributions(bundle: dict[str, Any], output_dir: Path | None = None) -> tuple[plt.Figure, dict[str, Path]]:
    configure_plot_style()
    from matplotlib.lines import Line2D

    summary_df = bundle["summary_df"].copy()
    fig, axes = plt.subplots(1, len(FP_ORDER), figsize=(16.8, 5.2), sharex=True, sharey=True)
    bins = np.linspace(0.0, 1.0, 31)
    legend_fontsize = 13
    variant_linestyles = {
        "fine_tuned": "-",
        "frozen": "--",
    }
    variant_alpha = {
        "fine_tuned": 0.95,
        "frozen": 0.75,
    }

    for ax, fp_type in zip(np.atleast_1d(axes), FP_ORDER, strict=False):
        for loss_kind in LOSS_ORDER:
            for model_variant in MODEL_VARIANT_ORDER:
                row = summary_df[
                    (summary_df["fp_type"] == fp_type)
                    & (summary_df["loss_kind"] == loss_kind)
                    & (summary_df["model_variant"] == model_variant)
                ].iloc[0]
                artifacts = load_artifacts_for_run(bundle, row["run_tag"])
                values = artifacts["auroc_comparison"]["auroc_ood"].dropna().to_numpy()

                ax.hist(
                    values,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=2.0,
                    linestyle=variant_linestyles[model_variant],
                    color=LOSS_COLORS[loss_kind],
                    alpha=variant_alpha[model_variant],
                )

        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.set_title(FP_DISPLAY[fp_type])
        ax.set_xlabel("OOD per-bit AUROC")

    axes[0].set_ylabel("Density")

    loss_handles = [
        Line2D([0], [0], color=LOSS_COLORS[loss_kind], linewidth=2.4, label=LOSS_DISPLAY[loss_kind])
        for loss_kind in LOSS_ORDER
    ]
    variant_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=2.4,
            linestyle=variant_linestyles[model_variant],
            label=MODEL_VARIANT_DISPLAY[model_variant],
        )
        for model_variant in MODEL_VARIANT_ORDER
    ]
    loss_legend = fig.legend(
        handles=loss_handles,
        loc="upper center",
        bbox_to_anchor=(0.34, 0.995),
        ncol=2,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.0,
        columnspacing=1.6,
    )
    fig.add_artist(loss_legend)
    fig.legend(
        handles=variant_handles,
        loc="upper center",
        bbox_to_anchor=(0.73, 0.995),
        ncol=2,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.0,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])

    saved: dict[str, Path] = {}
    if output_dir is not None:
        saved = save_figure(fig, output_dir, "axis2_per_bit_auroc_distributions")
    return fig, saved


def plot_generalisation_gap(bundle: dict[str, Any], output_dir: Path | None = None) -> tuple[plt.Figure, dict[str, Path]]:
    configure_plot_style()
    from matplotlib.lines import Line2D

    summary_df = bundle["summary_df"].copy()
    condition_order = [
        ("fine_tuned", "bce"),
        ("fine_tuned", "cos"),
        ("frozen", "bce"),
        ("frozen", "cos"),
    ]
    row_labels = [
        f"{MODEL_VARIANT_DISPLAY[model_variant]} · {LOSS_DISPLAY[loss_kind]}"
        for model_variant, loss_kind in condition_order
    ]
    y_positions = np.arange(len(condition_order))

    min_score = float(
        min(summary_df["mean_auroc_val"].min(), summary_df["mean_auroc_ood"].min())
    )
    max_score = float(
        max(summary_df["mean_auroc_val"].max(), summary_df["mean_auroc_ood"].max())
    )
    x_min = max(0.0, min_score - 0.03)
    x_max = min(1.0, max_score + 0.03)

    fig, axes = plt.subplots(1, len(FP_ORDER), figsize=(14.8, 5.0), sharex=True, sharey=True)

    for panel_idx, (ax, fp_type) in enumerate(zip(np.atleast_1d(axes), FP_ORDER, strict=False)):
        for y, (model_variant, loss_kind) in zip(y_positions, condition_order, strict=False):
            row = summary_df[
                (summary_df["fp_type"] == fp_type)
                & (summary_df["model_variant"] == model_variant)
                & (summary_df["loss_kind"] == loss_kind)
            ].iloc[0]

            ax.plot(
                [row["mean_auroc_ood"], row["mean_auroc_val"]],
                [y, y],
                color="#9ca3af",
                linewidth=2.0,
                solid_capstyle="round",
                zorder=1,
            )
            ax.scatter(
                row["mean_auroc_val"],
                y,
                s=70,
                color=SPLIT_COLORS["val"],
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
            ax.scatter(
                row["mean_auroc_ood"],
                y,
                s=70,
                color=SPLIT_COLORS["ood"],
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )

        ax.set_title(FP_DISPLAY[fp_type])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, len(condition_order) - 0.5)
        ax.set_yticks(y_positions)
        ax.grid(axis="x", alpha=0.18, linewidth=0.6)
        ax.invert_yaxis()
        ax.set_xlabel("Mean per-bit AUROC")

        if panel_idx == 0:
            ax.set_yticklabels(row_labels)
        else:
            ax.tick_params(axis="y", labelleft=False)

    legend_fontsize = 13
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=SPLIT_COLORS["val"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=12,
            label=SPLIT_DISPLAY["val"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=SPLIT_COLORS["ood"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=12,
            label=SPLIT_DISPLAY["ood"],
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        fontsize=legend_fontsize,
        handletextpad=0.6,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])

    saved: dict[str, Path] = {}
    if output_dir is not None:
        saved = save_figure(fig, output_dir, "axis2_generalisation_gap")
    return fig, saved


def plot_threshold_sweeps(bundle: dict[str, Any], output_dir: Path | None = None) -> tuple[plt.Figure, dict[str, Path]]:
    configure_plot_style()
    fine_df = iter_fine_tuned_rows(bundle)
    fig, axes = plt.subplots(1, len(FP_ORDER), figsize=(17.0, 4.8), sharey=True)

    for ax, fp_type in zip(axes, FP_ORDER, strict=False):
        fp_rows = fine_df[fine_df["fp_type"] == fp_type].sort_values("loss_display")
        for _, row in fp_rows.iterrows():
            artifacts = load_artifacts_for_run(bundle, row["run_tag"])
            for split, split_df in [("val", artifacts["sweep_val"]), ("ood", artifacts["sweep_ood"])]:
                ax.plot(
                    split_df["threshold"],
                    split_df["tanimoto_mean"],
                    color=LOSS_COLORS[row["loss_kind"]],
                    linestyle=SPLIT_LINESTYLES[split],
                    linewidth=2.2,
                    alpha=0.95,
                    label=f"{row['loss_display']} · {SPLIT_DISPLAY[split]}",
                )
                best_row = artifacts["threshold_table"]
                best_match = best_row[best_row["split"].str.lower() == split]
                if len(best_match):
                    tau = float(best_match["best_tau_tanimoto"].iloc[0])
                    best_value = float(best_match["best_tanimoto"].iloc[0])
                    ax.scatter(
                        [tau],
                        [best_value],
                        color=LOSS_COLORS[row["loss_kind"]],
                        edgecolor="white",
                        linewidth=0.7,
                        s=42,
                        zorder=5,
                    )

        ax.set_title(FP_DISPLAY[fp_type])
        ax.set_xlabel("Binarization threshold")
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.2, linewidth=0.6)

    axes[0].set_ylabel("Mean Tanimoto")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Axis 2 Threshold Sweeps (Validation dashed, OOD solid)", y=1.08, fontsize=18)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    saved: dict[str, Path] = {}
    if output_dir is not None:
        saved = save_figure(fig, output_dir, "axis2_threshold_sweeps")
    return fig, saved


def compute_accuracy_at_k_curve(ranks: np.ndarray, ks: np.ndarray) -> np.ndarray:
    return np.array([(ranks <= k).mean() for k in ks], dtype=float)


def plot_retrieval_acc_at_k(bundle: dict[str, Any], output_dir: Path | None = None) -> tuple[plt.Figure, dict[str, Path]]:
    configure_plot_style()
    from matplotlib.lines import Line2D

    summary_df = bundle["summary_df"].copy()
    fig, axes = plt.subplots(1, len(FP_ORDER), figsize=(16.8, 5.2), sharey=True)
    ks = np.unique(np.round(np.geomspace(1, 1000, 70)).astype(int))

    variant_linewidths = {
        "fine_tuned": 2.6,
        "frozen": 1.6,
    }
    variant_alphas = {
        "fine_tuned": 0.98,
        "frozen": 0.72,
    }
    legend_fontsize = 13

    for ax, fp_type in zip(axes, FP_ORDER, strict=False):
        fp_rows = (
            summary_df[summary_df["fp_type"] == fp_type]
            .assign(
                loss_order=lambda df: df["loss_kind"].map({loss: idx for idx, loss in enumerate(LOSS_ORDER)}),
                variant_order=lambda df: df["model_variant"].map(
                    {variant: idx for idx, variant in enumerate(MODEL_VARIANT_ORDER)}
                ),
            )
            .sort_values(["loss_order", "variant_order"])
        )
        for _, row in fp_rows.iterrows():
            artifacts = load_artifacts_for_run(bundle, row["run_tag"])
            for split, ranks in [("val", artifacts["ranks_val"]), ("ood", artifacts["ranks_ood"])]:
                acc_curve = compute_accuracy_at_k_curve(ranks, ks)
                ax.plot(
                    ks,
                    acc_curve,
                    color=LOSS_COLORS[row["loss_kind"]],
                    linestyle=SPLIT_LINESTYLES[split],
                    linewidth=variant_linewidths[row["model_variant"]],
                    alpha=variant_alphas[row["model_variant"]],
                )

        ax.set_xscale("log")
        ax.set_title(FP_DISPLAY[fp_type])
        ax.set_xlabel("k")
        ax.grid(alpha=0.2, linewidth=0.6)

    axes[0].set_ylabel("Accuracy@k")
    loss_handles = [
        Line2D([0], [0], color=LOSS_COLORS[loss_kind], linewidth=2.8, label=LOSS_DISPLAY[loss_kind])
        for loss_kind in LOSS_ORDER
    ]
    split_handles = [
        Line2D([0], [0], color="#444444", linewidth=2.4, linestyle=SPLIT_LINESTYLES[split], label=SPLIT_DISPLAY[split])
        for split in SPLIT_ORDER
    ]
    variant_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=variant_linewidths[model_variant],
            alpha=variant_alphas[model_variant],
            label=MODEL_VARIANT_DISPLAY[model_variant],
        )
        for model_variant in MODEL_VARIANT_ORDER
    ]
    loss_legend = fig.legend(
        handles=loss_handles,
        loc="upper center",
        bbox_to_anchor=(0.23, 0.995),
        ncol=2,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.0,
        columnspacing=1.6,
    )
    fig.add_artist(loss_legend)
    split_legend = fig.legend(
        handles=split_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=2,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.0,
        columnspacing=1.6,
    )
    fig.add_artist(split_legend)
    fig.legend(
        handles=variant_handles,
        loc="upper center",
        bbox_to_anchor=(0.82, 0.995),
        ncol=2,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.0,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])

    saved: dict[str, Path] = {}
    if output_dir is not None:
        saved = save_figure(fig, output_dir, "axis2_retrieval_accuracy_at_k")
    return fig, saved


def create_cross_axis_scatter(bundle: dict[str, Any], output_dir: Path | None = None) -> dict[str, Path]:
    output_dir = ensure_output_dir(output_dir or bundle["output_dir"])
    script_path = Path(__file__).resolve().parent / "cross_axis_bridge.py"
    spec = importlib.util.spec_from_file_location("cross_axis_bridge_publication", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load cross_axis_bridge from {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    scatter_df = pd.read_csv(bundle["cross_axis_dir"] / "fine_tuned_cross_axis_scatter_data.csv")
    spearman_df = pd.read_csv(bundle["cross_axis_dir"] / "fine_tuned_cross_axis_spearman_table.csv")
    outputs = [
        output_dir / "axis2_cross_axis_bridge_axis1_r2_vs_axis2_auroc.png",
        output_dir / "axis2_cross_axis_bridge_axis1_r2_vs_axis2_auroc.pdf",
    ]
    module.create_cross_axis_scatter_figure(scatter_df, spearman_df, outputs, labels_per_category=2)
    return {"png": outputs[0], "pdf": outputs[1]}


def select_best_fine_tuned_run(bundle: dict[str, Any], metric: str = "mean_auroc_ood") -> pd.Series:
    fine_df = bundle["fine_tuned_df"].copy()
    if fine_df.empty:
        raise ValueError("No fine-tuned runs found.")
    return fine_df.sort_values(metric, ascending=False).iloc[0]


def plot_best_model_auroc_val_vs_ood(bundle: dict[str, Any], output_dir: Path | None = None) -> tuple[plt.Figure, dict[str, Path], pd.Series]:
    configure_plot_style()
    best_row = select_best_fine_tuned_run(bundle)
    artifacts = load_artifacts_for_run(bundle, best_row["run_tag"])
    auroc_df = artifacts["auroc_comparison"].copy()

    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    hb = ax.hexbin(
        auroc_df["auroc_val"],
        auroc_df["auroc_ood"],
        gridsize=32,
        cmap="viridis",
        mincnt=1,
        linewidths=0.0,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1.2, alpha=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Validation per-bit AUROC")
    ax.set_ylabel("OOD per-bit AUROC")
    ax.set_title(
        f"Per-bit AUROC Shift for Best Fine-tuned Model\n"
        f"{best_row['fp_display']} · {best_row['loss_display']} ({best_row['run_tag']})"
    )
    ax.grid(alpha=0.18, linewidth=0.6)

    stats_text = (
        f"Mean val = {auroc_df['auroc_val'].mean():.3f}\n"
        f"Mean OOD = {auroc_df['auroc_ood'].mean():.3f}\n"
        f"Mean drop = {(auroc_df['auroc_val'] - auroc_df['auroc_ood']).mean():.3f}"
    )
    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#d9d9d9", "alpha": 0.92},
    )
    fig.colorbar(hb, ax=ax, label="Bit count")
    fig.tight_layout()

    saved: dict[str, Path] = {}
    if output_dir is not None:
        saved = save_figure(fig, output_dir, "axis2_best_model_auroc_val_vs_ood")
    return fig, saved, best_row


def plot_bit_density_vs_optimal_threshold(bundle: dict[str, Any], output_dir: Path | None = None) -> tuple[plt.Figure, dict[str, Path], pd.DataFrame]:
    configure_plot_style()
    fine_df = iter_fine_tuned_rows(bundle)
    rows: list[dict[str, Any]] = []
    for _, row in fine_df.iterrows():
        for split in SPLIT_ORDER:
            rows.append(
                {
                    "run_tag": row["run_tag"],
                    "fp_type": row["fp_type"],
                    "fp_display": row["fp_display"],
                    "loss_kind": row["loss_kind"],
                    "loss_display": row["loss_display"],
                    "split": split,
                    "split_display": SPLIT_DISPLAY[split],
                    "bit_density": row[f"bit_density_{split}"],
                    "optimal_tau": row[f"optimal_tau_{split}"],
                    "best_tanimoto": row[f"best_tanimoto_{split}"],
                    "label": f"{row['fp_display']} {row['loss_display']}\n{SPLIT_DISPLAY[split]}",
                }
            )
    density_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    sns.scatterplot(
        data=density_df,
        x="bit_density",
        y="optimal_tau",
        hue="loss_display",
        style="split_display",
        palette={LOSS_DISPLAY[k]: v for k, v in LOSS_COLORS.items()},
        s=130,
        ax=ax,
    )

    if density_df["bit_density"].nunique() >= 2:
        x = density_df["bit_density"].to_numpy(dtype=float)
        y = density_df["optimal_tau"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        x_fit = np.linspace(x.min() * 0.95, x.max() * 1.05, 200)
        ax.plot(x_fit, intercept + slope * x_fit, color="#444444", linewidth=1.5, alpha=0.75)

    for _, row in density_df.iterrows():
        ax.annotate(
            row["label"],
            xy=(row["bit_density"], row["optimal_tau"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.set_xlabel("Observed bit density")
    ax.set_ylabel("Optimal threshold (best Tanimoto)")
    ax.set_title("Bit Density vs Optimal Threshold Across Fine-tuned Conditions")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    saved: dict[str, Path] = {}
    if output_dir is not None:
        saved = save_figure(fig, output_dir, "axis2_bit_density_vs_optimal_threshold")
    return fig, saved, density_df


def build_frozen_vs_finetuned_delta_table(bundle: dict[str, Any]) -> pd.DataFrame:
    summary_df = bundle["summary_df"].copy()
    rows: list[dict[str, Any]] = []
    for fp_type in FP_ORDER:
        for loss_kind in LOSS_ORDER:
            fine = summary_df[
                (summary_df["fp_type"] == fp_type)
                & (summary_df["loss_kind"] == loss_kind)
                & (~summary_df["frozen"])
            ]
            frozen = summary_df[
                (summary_df["fp_type"] == fp_type)
                & (summary_df["loss_kind"] == loss_kind)
                & (summary_df["frozen"])
            ]
            if fine.empty or frozen.empty:
                continue

            fine_row = fine.iloc[0]
            frozen_row = frozen.iloc[0]
            rows.append(
                {
                    "fp_type": fp_type,
                    "fp_display": FP_DISPLAY[fp_type],
                    "loss_kind": loss_kind,
                    "loss_display": LOSS_DISPLAY[loss_kind],
                    "fine_tuned_run_tag": fine_row["run_tag"],
                    "frozen_run_tag": frozen_row["run_tag"],
                    "delta_mean_auroc_ood": fine_row["mean_auroc_ood"] - frozen_row["mean_auroc_ood"],
                    "delta_best_tanimoto_ood": fine_row["best_tanimoto_ood"] - frozen_row["best_tanimoto_ood"],
                    "delta_acc_at_10_ood": fine_row["accat10_ood"] - frozen_row["accat10_ood"],
                    "delta_cosine_sim_mean_ood": fine_row["cosine_sim_mean_ood"] - frozen_row["cosine_sim_mean_ood"],
                }
            )
    return pd.DataFrame(rows).sort_values(["fp_display", "loss_display"]).reset_index(drop=True)


def plot_frozen_vs_finetuned_delta(bundle: dict[str, Any], output_dir: Path | None = None) -> tuple[plt.Figure, dict[str, Path], pd.DataFrame]:
    configure_plot_style()
    delta_df = build_frozen_vs_finetuned_delta_table(bundle)
    metric_specs = [
        ("delta_mean_auroc_ood", "AUROC", "#2563eb"),
        ("delta_cosine_sim_mean_ood", "Cosine sim", "#0f766e"),
        ("delta_best_tanimoto_ood", "Tanimoto", "#d97706"),
    ]
    legend_fontsize = 13
    bar_width = 0.22
    x_positions = np.arange(len(LOSS_ORDER))
    offsets = np.linspace(-bar_width, bar_width, len(metric_specs))
    max_abs_delta = float(max(delta_df[metric_col].abs().max() for metric_col, _, _ in metric_specs))
    y_limit = min(0.20, max_abs_delta + 0.03)

    fig, axes = plt.subplots(1, len(FP_ORDER), figsize=(14.8, 5.4), sharey=True)

    for panel_idx, (ax, fp_type) in enumerate(zip(np.atleast_1d(axes), FP_ORDER, strict=False)):
        fp_df = (
            delta_df[delta_df["fp_type"] == fp_type]
            .assign(loss_order=lambda df: df["loss_kind"].map({loss: idx for idx, loss in enumerate(LOSS_ORDER)}))
            .sort_values("loss_order")
            .reset_index(drop=True)
        )

        for offset, (metric_col, metric_label, metric_color) in zip(offsets, metric_specs, strict=False):
            values = fp_df[metric_col].to_numpy(dtype=float)
            ax.bar(
                x_positions + offset,
                values,
                width=bar_width,
                color=metric_color,
                alpha=0.92,
                label=metric_label if panel_idx == 0 else None,
            )

        ax.axhline(0.0, color="#444444", linewidth=1.1)
        ax.set_title(FP_DISPLAY[fp_type])
        ax.set_xticks(x_positions)
        ax.set_xticklabels([LOSS_DISPLAY[loss_kind] for loss_kind in LOSS_ORDER])
        ax.set_ylim(-y_limit, y_limit)
        ax.grid(alpha=0.18, linewidth=0.6, axis="y")

    axes[0].set_ylabel("OOD delta (Fine-tuned - Frozen)")
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=1.8,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])

    saved: dict[str, Path] = {}
    if output_dir is not None:
        saved = save_figure(fig, output_dir, "axis2_frozen_vs_finetuned_delta")
        delta_df.to_csv(ensure_output_dir(output_dir) / "axis2_frozen_vs_finetuned_delta_table.csv", index=False)
    return fig, saved, delta_df


def save_master_comparison_table(bundle: dict[str, Any], output_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, Path]]:
    output_dir = ensure_output_dir(output_dir or bundle["output_dir"])
    summary_df = bundle["summary_df"].copy()
    table_df = summary_df[
        [
            "run_tag",
            "model_variant_display",
            "fp_display",
            "loss_display",
            "n_bits",
            "bit_density_val",
            "bit_density_ood",
            "mean_auroc_val",
            "mean_auroc_ood",
            "optimal_tau_val",
            "optimal_tau_ood",
            "best_tanimoto_val",
            "best_tanimoto_ood",
            "precision_at_optimal_tau_val",
            "precision_at_optimal_tau_ood",
            "recall_at_optimal_tau_val",
            "recall_at_optimal_tau_ood",
            "f1_at_optimal_tau_val",
            "f1_at_optimal_tau_ood",
            "cosine_sim_mean_val",
            "cosine_sim_mean_ood",
            "accat1_val",
            "accat1_ood",
            "accat10_val",
            "accat10_ood",
        ]
    ].copy()
    table_df = table_df.rename(
        columns={
            "model_variant_display": "model_variant",
            "fp_display": "fingerprint",
            "loss_display": "loss",
            "bit_density_val": "bit_density_val",
            "bit_density_ood": "bit_density_ood",
            "mean_auroc_val": "mean_auroc_val",
            "mean_auroc_ood": "mean_auroc_ood",
            "optimal_tau_val": "optimal_tau_val",
            "optimal_tau_ood": "optimal_tau_ood",
            "best_tanimoto_val": "best_tanimoto_val",
            "best_tanimoto_ood": "best_tanimoto_ood",
            "precision_at_optimal_tau_val": "precision_at_optimal_tau_val",
            "precision_at_optimal_tau_ood": "precision_at_optimal_tau_ood",
            "recall_at_optimal_tau_val": "recall_at_optimal_tau_val",
            "recall_at_optimal_tau_ood": "recall_at_optimal_tau_ood",
            "f1_at_optimal_tau_val": "f1_at_optimal_tau_val",
            "f1_at_optimal_tau_ood": "f1_at_optimal_tau_ood",
            "cosine_sim_mean_val": "cosine_sim_mean_val",
            "cosine_sim_mean_ood": "cosine_sim_mean_ood",
            "accat1_val": "acc_at_1_val",
            "accat1_ood": "acc_at_1_ood",
            "accat10_val": "acc_at_10_val",
            "accat10_ood": "acc_at_10_ood",
        }
    )
    table_df = table_df.round(4)

    csv_path = output_dir / "axis2_master_comparison_table.csv"
    html_path = output_dir / "axis2_master_comparison_table.html"
    table_df.to_csv(csv_path, index=False)
    html_path.write_text(table_df.to_html(index=False))
    return table_df, {"csv": csv_path, "html": html_path}


def format_table_value(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def save_thesis_comparison_table(bundle: dict[str, Any], output_dir: Path | None = None) -> dict[str, Path]:
    output_dir = ensure_output_dir(output_dir or bundle["output_dir"])
    summary_df = bundle["summary_df"].copy()
    table_df = summary_df[
        [
            "model_variant_display",
            "fp_display",
            "loss_display",
            "mean_auroc_val",
            "mean_auroc_ood",
            "best_tanimoto_val",
            "best_tanimoto_ood",
            "cosine_sim_mean_val",
            "cosine_sim_mean_ood",
            "accat1_val",
            "accat1_ood",
            "accat10_val",
            "accat10_ood",
        ]
    ].copy()
    table_df = table_df.rename(
        columns={
            "model_variant_display": "Model",
            "fp_display": "Fingerprint",
            "loss_display": "Loss",
            "mean_auroc_val": "AUROC (Val)",
            "mean_auroc_ood": "AUROC (OOD)",
            "best_tanimoto_val": "Tanimoto (Val)",
            "best_tanimoto_ood": "Tanimoto (OOD)",
            "cosine_sim_mean_val": "Cosine (Val)",
            "cosine_sim_mean_ood": "Cosine (OOD)",
            "accat1_val": "Acc@1 (Val)",
            "accat1_ood": "Acc@1 (OOD)",
            "accat10_val": "Acc@10 (Val)",
            "accat10_ood": "Acc@10 (OOD)",
        }
    ).round(4)

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Axis 2 comparison across all 12 fingerprint-prediction conditions on the "
            r"validation and OOD splits. Retrieval quality is reported as accuracy at rank 1 and "
            r"rank 10.}"
        ),
        r"\label{tab:axis2_condition_metrics}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllcccccccccc}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{FP} & \textbf{Loss} & "
            r"\multicolumn{2}{c}{\textbf{AUROC}} & "
            r"\multicolumn{2}{c}{\textbf{Tanimoto}} & "
            r"\multicolumn{2}{c}{\textbf{Cosine}} & "
            r"\multicolumn{2}{c}{\textbf{Acc@1}} & "
            r"\multicolumn{2}{c}{\textbf{Acc@10}} \\"
        ),
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
        r" & & & \textbf{Val} & \textbf{OOD} & \textbf{Val} & \textbf{OOD} & \textbf{Val} & \textbf{OOD} & \textbf{Val} & \textbf{OOD} & \textbf{Val} & \textbf{OOD} \\",
        r"\midrule",
    ]

    previous_fp = None
    for _, row in table_df.iterrows():
        current_fp = str(row["Fingerprint"])
        if previous_fp is not None and current_fp != previous_fp:
            latex_lines.append(r"\midrule")
        previous_fp = current_fp

        latex_lines.append(
            " & ".join(
                [
                    str(row["Model"]),
                    current_fp,
                    str(row["Loss"]),
                    format_table_value(row["AUROC (Val)"]),
                    format_table_value(row["AUROC (OOD)"]),
                    format_table_value(row["Tanimoto (Val)"]),
                    format_table_value(row["Tanimoto (OOD)"]),
                    format_table_value(row["Cosine (Val)"]),
                    format_table_value(row["Cosine (OOD)"]),
                    format_table_value(row["Acc@1 (Val)"]),
                    format_table_value(row["Acc@1 (OOD)"]),
                    format_table_value(row["Acc@10 (Val)"]),
                    format_table_value(row["Acc@10 (OOD)"]),
                ]
            )
            + r" \\"
        )

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )

    csv_path = output_dir / "axis2_thesis_comparison_table.csv"
    tex_path = output_dir / "axis2_thesis_comparison_table.tex"
    table_df.to_csv(csv_path, index=False)
    tex_path.write_text("\n".join(latex_lines))
    return {"csv": csv_path, "tex": tex_path}


def save_optimal_tau_prf_table(bundle: dict[str, Any], output_dir: Path | None = None) -> Path:
    output_dir = ensure_output_dir(output_dir or bundle["output_dir"])
    summary_df = bundle["summary_df"].copy()
    table_df = summary_df[
        [
            "run_tag",
            "model_variant_display",
            "fp_display",
            "loss_display",
            "optimal_tau_val",
            "precision_at_optimal_tau_val",
            "recall_at_optimal_tau_val",
            "f1_at_optimal_tau_val",
            "optimal_tau_ood",
            "precision_at_optimal_tau_ood",
            "recall_at_optimal_tau_ood",
            "f1_at_optimal_tau_ood",
        ]
    ].copy()
    table_df = table_df.rename(
        columns={
            "model_variant_display": "model_variant",
            "fp_display": "fingerprint",
            "loss_display": "loss",
        }
    ).round(4)

    csv_path = output_dir / "axis2_optimal_tau_precision_recall_table.csv"
    table_df.to_csv(csv_path, index=False)
    return csv_path


def generate_all_axis2_publication_outputs(thesis_root: Path | None = None) -> dict[str, Any]:
    bundle = load_publication_bundle(thesis_root)
    output_dir = ensure_output_dir(bundle["output_dir"])

    outputs: dict[str, Any] = {
        "output_dir": output_dir,
    }
    fig, saved = plot_generalisation_gap(bundle, output_dir)
    plt.close(fig)
    outputs["generalisation_gap"] = saved

    fig, saved = plot_per_bit_auroc_distributions(bundle, output_dir)
    plt.close(fig)
    outputs["per_bit_auroc_distributions"] = saved

    fig, saved = plot_threshold_sweeps(bundle, output_dir)
    plt.close(fig)
    outputs["threshold_sweeps"] = saved

    fig, saved = plot_retrieval_acc_at_k(bundle, output_dir)
    plt.close(fig)
    outputs["retrieval_acc_at_k"] = saved

    outputs["cross_axis_scatter"] = create_cross_axis_scatter(bundle, output_dir)

    fig, saved, best_row = plot_best_model_auroc_val_vs_ood(bundle, output_dir)
    plt.close(fig)
    outputs["best_model_auroc_val_vs_ood"] = saved
    outputs["best_model_run_tag"] = str(best_row["run_tag"])

    fig, saved, density_df = plot_bit_density_vs_optimal_threshold(bundle, output_dir)
    plt.close(fig)
    density_csv = output_dir / "axis2_bit_density_vs_optimal_threshold_data.csv"
    density_df.to_csv(density_csv, index=False)
    outputs["bit_density_vs_optimal_threshold"] = saved
    outputs["bit_density_vs_optimal_threshold_csv"] = density_csv

    fig, saved, delta_df = plot_frozen_vs_finetuned_delta(bundle, output_dir)
    plt.close(fig)
    outputs["frozen_vs_finetuned_delta"] = saved
    outputs["frozen_vs_finetuned_delta_csv"] = output_dir / "axis2_frozen_vs_finetuned_delta_table.csv"

    master_df, table_paths = save_master_comparison_table(bundle, output_dir)
    outputs["master_comparison_table"] = table_paths
    outputs["thesis_comparison_table"] = save_thesis_comparison_table(bundle, output_dir)
    outputs["optimal_tau_precision_recall_table_csv"] = save_optimal_tau_prf_table(bundle, output_dir)
    outputs["n_master_rows"] = int(len(master_df))
    return outputs


def main() -> None:
    bundle = load_publication_bundle()
    outputs = generate_all_axis2_publication_outputs(bundle["thesis_root"])
    print("Axis 2 publication outputs:")
    print(f"  Output dir: {outputs['output_dir']}")
    print(f"  Best model for AUROC val-vs-OOD scatter: {outputs['best_model_run_tag']}")
    for key, value in outputs.items():
        if key in {"output_dir", "best_model_run_tag", "n_master_rows"}:
            continue
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
