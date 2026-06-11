#!/usr/bin/env python3
"""Build final thesis Axis 3 figures from exported Axis 3 CSVs."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "axis3_publication_figures_mplconfig"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "dreams-thesis-wa" / "results" / "axis3" / "specific"
OUTPUT_DIR = ROOT / "dreams-thesis-wa" / "results" / "axis3" / "figures"
PER_SPECTRUM_CSV = INPUT_DIR / "axis3_specific_per_spectrum_ranks.csv"
PER_BIT_CSV = INPUT_DIR / "axis3_specific_per_bit_auroc.csv"
FIG_DECOMPOSITION = OUTPUT_DIR / "axis3_decomposition.pdf"
FIG_TRANSFER = OUTPUT_DIR / "axis3_substructure_transfer.pdf"

PUBLICATION_DPI = 300
SEEN_COLOR = "#0b6e4f"
NOVEL_COLOR = "#bdbdbd"
BAR_COLOR = SEEN_COLOR
POINT_COLOR = "#3b82f6"
ADDUCT_ORDER = [
    "[M+H]+",
    "[M-H2O+H]+",
    "[M+NH4]+",
    "[M+Na]+",
    "[M-2H2O+H]+",
    "[M-NH3+H]+",
]
PEAK_ORDER = ["1", "2", "3", "4-5", "6-10", "11+"]


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")   # paper, NOT talk
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": PUBLICATION_DPI,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,   # was missing -> inherited the huge talk size
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def peak_bin(value: int) -> str:
    if value <= 3:
        return str(value)
    if value <= 5:
        return "4-5"
    if value <= 10:
        return "6-10"
    return "11+"


def add_bar_labels(ax: plt.Axes, xs: np.ndarray, values: np.ndarray) -> None:
    for x, value in zip(xs, values):
        ax.text(
            float(x),
            float(value) + 0.006,
            f"{float(value):.2f}",
            ha="center",
            va="bottom",
            fontsize=plt.rcParams["xtick.labelsize"],
        )


def summarize_by_category(data: pd.DataFrame, category_col: str, categories: list[str]) -> pd.DataFrame:
    rows = []
    for category in categories:
        category_mask = data[category_col].astype(str).eq(category)
        for subset, seen_value in [("seen", True), ("novel", False)]:
            mask = category_mask & data["seen_in_axis2"].astype(bool).eq(seen_value)
            n = int(mask.sum())
            hits = int(data.loc[mask, "hit_at_1"].sum())
            rows.append(
                {
                    "category": category,
                    "subset": subset,
                    "acc_at_1": hits / n if n else 0.0,
                    "hits": hits,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def print_category_summary(title: str, summary: pd.DataFrame) -> None:
    print(f"\n{title}")
    for category in summary["category"].drop_duplicates():
        parts = []
        for subset in ["seen", "novel"]:
            row = summary.loc[
                (summary["category"] == category) & (summary["subset"] == subset)
            ].iloc[0]
            parts.append(
                f"{subset}: acc@1={row.acc_at_1:.4f}, n={int(row.n)}, hits={int(row.hits)}"
            )
        print(f"{category}: " + "; ".join(parts))


def draw_grouped_bars(
    ax: plt.Axes,
    summary: pd.DataFrame,
    categories: list[str],
    xlabel: str,
    rotate_labels: bool,
    show_legend: bool = False,
) -> None:
    x = np.arange(len(categories))
    width = 0.38
    seen_values = (
        summary.loc[summary["subset"] == "seen"]
        .set_index("category")
        .reindex(categories)["acc_at_1"]
        .to_numpy(dtype=float)
    )
    novel_values = (
        summary.loc[summary["subset"] == "novel"]
        .set_index("category")
        .reindex(categories)["acc_at_1"]
        .to_numpy(dtype=float)
    )

    seen_x = x - width / 2
    novel_x = x + width / 2
    ax.bar(
        seen_x,
        seen_values,
        width,
        color=SEEN_COLOR,
        edgecolor="white",
        linewidth=0.4,
        label="seen",
    )
    ax.bar(
        novel_x,
        novel_values,
        width,
        color=NOVEL_COLOR,
        edgecolor="white",
        linewidth=0.4,
        label="novel",
    )
    add_bar_labels(ax, seen_x, seen_values)
    add_bar_labels(ax, novel_x, novel_values)

    ax.set_xticks(x)
    ax.set_xticklabels(
        categories,
        rotation=30 if rotate_labels else 0,
        ha="right" if rotate_labels else "center",
    )
    ax.set_ylim(0, 0.35)
    ax.set_ylabel("acc@1")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    if show_legend:
        ax.legend(loc="upper right", frameon=True)


def plot_decomposition() -> Path:
    ranks = pd.read_csv(PER_SPECTRUM_CSV)
    data = ranks.loc[(ranks["model"] == "fine_tuned") & (ranks["pool"] == "closed")].copy()
    data["hit_at_1"] = data["hit_at_1"].astype(float)
    data["n_peaks"] = data["n_peaks"].astype(int)

    data["pkbin"] = data["n_peaks"].map(peak_bin)
    adduct_summary = summarize_by_category(data, "adduct", ADDUCT_ORDER)
    peak_summary = summarize_by_category(data, "pkbin", PEAK_ORDER)

    print_category_summary("Adduct acc@1 by seen/novel", adduct_summary)
    print_category_summary("Retained-peak acc@1 by seen/novel", peak_summary)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

    draw_grouped_bars(axes[0], adduct_summary, ADDUCT_ORDER, "Adduct", rotate_labels=True, show_legend=True)
    draw_grouped_bars(axes[1], peak_summary, PEAK_ORDER, "Retained peaks", rotate_labels=False)

    fig.savefig(FIG_DECOMPOSITION, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return FIG_DECOMPOSITION


def prevalence_to_size(prevalence: float) -> float:
    return 12.0 + 110.0 * float(prevalence)


def plot_transfer() -> Path:
    per_bit = pd.read_csv(PER_BIT_CSV)
    data = per_bit.dropna(subset=["axis2_ood_auroc", "mac_auroc_all"]).copy()
    novel = per_bit.dropna(subset=["axis2_ood_auroc", "mac_auroc_novel"]).copy()

    rho_all, p_all = spearmanr(data["axis2_ood_auroc"], data["mac_auroc_all"])
    rho_novel, p_novel = spearmanr(novel["axis2_ood_auroc"], novel["mac_auroc_novel"])

    fig, ax = plt.subplots(figsize=(3.3, 3.0), constrained_layout=True)
    sizes = data["mac_bit_prevalence"].map(prevalence_to_size)
    ax.scatter(
        data["axis2_ood_auroc"],
        data["mac_auroc_all"],
        s=sizes,
        color=POINT_COLOR,
        alpha=0.68,
        edgecolor="white",
        linewidth=0.35,
    )

    ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.axhline(0.5, color="#999999", linestyle=":", linewidth=0.8, alpha=0.65)
    ax.axvline(0.5, color="#999999", linestyle=":", linewidth=0.8, alpha=0.65)
    ax.set_xlim(0.25, 1)
    ax.set_ylim(0.25, 1)
    ax.set_xlabel("Axis 2 OOD AUROC")
    ax.set_ylabel("Axis 3 MAC AUROC")

    stats_text = (
        rf"$\mathbf{{all}}$: $\rho={rho_all:.3f}$, $p={p_all:.3f}$"
        "\n"
        rf"$\mathbf{{novel}}$: $\rho={rho_novel:.3f}$, $p={p_novel:.3f}$"
    )
    ax.text(
        0.05,
        0.06,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=plt.rcParams["xtick.labelsize"],
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.92,
        },
    )

    tick_size = plt.rcParams["xtick.labelsize"]
    legend_levels = [0.05, 0.25, 0.90]
    legend_handles = [
        ax.scatter(
            [],
            [],
            s=prevalence_to_size(level),
            color=POINT_COLOR,
            alpha=0.68,
            edgecolor="white",
            linewidth=0.35,
        )
        for level in legend_levels
    ]
    legend = ax.legend(
        legend_handles,
        [f"{level:.2f}" for level in legend_levels],
        title=r"$P(\mathrm{bit}=1)$",
        loc="upper left",
        bbox_to_anchor=(0.03, 0.98),
        frameon=True,
        fontsize=tick_size,
        title_fontsize=tick_size,
        markerscale=1.0,
        borderpad=0.25,
        labelspacing=0.25,
        handletextpad=0.45,
        handlelength=1.0,
    )
    legend.get_frame().set_alpha(0.92)
    legend.get_frame().set_linewidth(0.6)

    label_specs = {
        28: ("[#15]", (-42, -9), "right", "center"),
        102: ("Cl", (12, 13), "left", "bottom"),
        16: ("[#6]#[#6]", (30, 6), "right", "bottom"),
    }
    arrowprops = {
        "arrowstyle": "-",
        "color": "#333333",
        "linewidth": 0.7,
        "shrinkA": 2,
        "shrinkB": 2,
    }
    for bit_index, (label, offset, ha, va) in label_specs.items():
        row = data.loc[data["bit_index"] == bit_index].iloc[0]
        ax.annotate(
            label,
            xy=(row["axis2_ood_auroc"], row["mac_auroc_all"]),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=tick_size,
            arrowprops=arrowprops,
        )

    fig.savefig(FIG_TRANSFER, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return FIG_TRANSFER


def main() -> int:
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [plot_decomposition(), plot_transfer()]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
