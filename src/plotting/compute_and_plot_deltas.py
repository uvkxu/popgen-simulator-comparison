#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl
import numpy as np
from tqdm import tqdm

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt




ORDER = [
    "one_population",
    "zigzag",
    "locke_orangutan",
    "gutenkunst_ooa",
    "browning_america",
    "pawar_gorilla",
    "freedman_wolf",
    "kamm_westerneurasia",
    "jacobs_papuans",
]


# ==========================================================
# matplotlib
# ==========================================================

def setup_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })

def _shorten(s: str, maxlen: int = 24) -> str:
    return s if len(s) <= maxlen else s[:maxlen - 1] + "…"


# ==========================================================
# CSV loading (lazy + parallel)
# ==========================================================

def load_all_configs(root: Path):

    csv_files = sorted(root.glob("*/simulator_feature_values.csv"))
    print(f"Loading {len(csv_files)} CSV files")

    lazy_frames = []

    for csv in csv_files:

        lf = (
            pl.scan_csv(csv)
            .with_columns(pl.lit(csv.parent.name).alias("config"))
        )

        lazy_frames.append(lf)

    df = pl.concat(lazy_frames)

    return df.collect()


# ==========================================================
# Delta computation
# ==========================================================

def compute_deltas(df: pl.DataFrame) -> pl.DataFrame:

    group_cols = [
        "config",
        "sample_size_key",
        "seq_len",
        "feature",
    ]

    print("Computing simulator medians")

    sim_medians = (
        df
        .group_by(group_cols + ["simulator"])
        .agg(pl.median("value").alias("value"))
    )

    print("Computing baseline medians")

    baseline = (
        sim_medians
        .group_by(group_cols)
        .agg(pl.median("value").alias("baseline"))
    )

    print("Joining baseline")

    df = df.join(baseline, on=group_cols)

    print("Computing deltas")

    df = df.with_columns(
        (pl.col("value") - pl.col("baseline")).alias("delta")
    )

    return df


# ==========================================================
# Compute boxplot statistics
# ==========================================================

def compute_box_stats(df: pl.DataFrame):

    print("Computing boxplot statistics")

    stats = (
        df
        .group_by(["feature", "config", "simulator"])
        .agg([
            pl.col("delta").quantile(0.0).alias("min"),
            pl.col("delta").quantile(0.25).alias("q1"),
            pl.col("delta").quantile(0.5).alias("median"),
            pl.col("delta").quantile(0.75).alias("q3"),
            pl.col("delta").quantile(1.0).alias("max"),
        ])
    )

    return stats.to_pandas()


# ==========================================================
# Plotting using precomputed stats
# ==========================================================


def plot_feature(stats_df, feature, out_path, scale="linear"):

    df_f = stats_df[stats_df["feature"] == feature]

    configs = [c for c in ORDER if c in df_f["config"].unique()]
    simulators = sorted(df_f["simulator"].unique())

    n_cfg = len(configs)
    n_sim = len(simulators)

    config_gap = 1.8
    within_gap = 0.22
    box_width = 0.18

    base = np.arange(n_cfg) * config_gap

    fig_w = max(12, min(26, 0.8 * n_cfg + 6))
    fig, ax = plt.subplots(figsize=(fig_w, 6.2), constrained_layout=True)

    if scale == "arcsinh":
        ax.set_yscale("function", functions=(np.arcsinh, np.sinh))
        ax.set_ylabel("Feature deviation from global median baseline (Δ, arcsinh scale)")
    elif scale == "symlog":
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylabel("Feature deviation from global median baseline (Δ, symlog scale)")  
    else:
        ax.set_ylabel("Feature deviation from global median baseline (Δ)")

    ax.set_xlabel("Demographic models (ordered by number of populations)")
    

    cmap = plt.get_cmap("tab10")
    colors = {sim: cmap(i) for i, sim in enumerate(simulators)}

    for i in range(n_cfg - 1):
        ax.axvline(
            (base[i] + base[i + 1]) / 2,
            color="0.2",
            linewidth=0.8,
            alpha=0.12,
            zorder=0
        )

    for s_idx, sim in enumerate(simulators):

        offset = (s_idx - (n_sim - 1) / 2) * within_gap

        positions = []
        stats = []

        for c_idx, cfg in enumerate(configs):

            row = df_f[
                (df_f["config"] == cfg) &
                (df_f["simulator"] == sim)
            ]

            if len(row) == 0:
                continue

            r = row.iloc[0]

            positions.append(base[c_idx] + offset)

            stats.append({
                "med": r["median"],
                "q1": r["q1"],
                "q3": r["q3"],
                "whislo": r["min"],
                "whishi": r["max"],
            })

        if not stats:
            continue

        bp = ax.bxp(
            stats,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
        )

        for b in bp["boxes"]:
            b.set_facecolor(colors[sim])
            b.set_alpha(0.35)
            b.set_edgecolor(colors[sim])

        for w in bp["whiskers"]:
            w.set_color(colors[sim])
        for c in bp["caps"]:
            c.set_color(colors[sim])

    ax.axhline(0, linestyle="--")

    ax.yaxis.grid(True, alpha=0.22)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(base)
    ax.set_xticklabels(
        [_shorten(c) for c in configs],
        rotation=35,
        ha="right"
    )

    legend_handles = [
        Patch(facecolor=colors[sim], edgecolor=colors[sim],
              alpha=0.35, label=sim)
        for sim in simulators
    ]

    legend_handles.append(
        Line2D([], [], linestyle="--",
               color="0.1", linewidth=1.4,
               label="baseline (0)")
    )

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_feature_split(stats_df, feature, out_path, scale="linear"):

    df_f = stats_df[stats_df["feature"] == feature]

    simulators = sorted(df_f["simulator"].unique())

    top_cfgs = ["one_population", "zigzag", "locke_orangutan"]
    bottom_cfgs = [c for c in ORDER if c not in top_cfgs]

    config_gap = 1.8
    within_gap = 0.22
    box_width = 0.18

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        sharex=False,
        gridspec_kw={"height_ratios": [1.1, 1.6]},
        constrained_layout=True
    )

    axes = [(ax_top, top_cfgs), (ax_bottom, bottom_cfgs)]

    cmap = plt.get_cmap("tab10")
    colors = {sim: cmap(i) for i, sim in enumerate(simulators)}

    for ax, configs in axes:

        configs = [c for c in configs if c in df_f["config"].unique()]

        n_cfg = len(configs)
        n_sim = len(simulators)

        base = np.arange(n_cfg) * config_gap

        if scale == "arcsinh":
            ax.set_yscale("function", functions=(np.arcsinh, np.sinh))
        elif scale == "symlog":
            ax.set_yscale("symlog", linthresh=1)

        for i in range(n_cfg - 1):
            ax.axvline(
                (base[i] + base[i + 1]) / 2,
                color="0.2",
                linewidth=0.8,
                alpha=0.12,
                zorder=0
            )

        for s_idx, sim in enumerate(simulators):

            offset = (s_idx - (n_sim - 1) / 2) * within_gap

            positions = []
            stats = []

            for c_idx, cfg in enumerate(configs):

                row = df_f[
                    (df_f["config"] == cfg) &
                    (df_f["simulator"] == sim)
                ]

                if len(row) == 0:
                    continue

                r = row.iloc[0]

                positions.append(base[c_idx] + offset)

                stats.append({
                    "med": r["median"],
                    "q1": r["q1"],
                    "q3": r["q3"],
                    "whislo": r["min"],
                    "whishi": r["max"],
                })

            if not stats:
                continue

            bp = ax.bxp(
                stats,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
            )

            for b in bp["boxes"]:
                b.set_facecolor(colors[sim])
                b.set_alpha(0.35)
                b.set_edgecolor(colors[sim])

        ax.axhline(0, linestyle="--")

        ax.yaxis.grid(True, alpha=0.22)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xticks(base)
        ax.set_xticklabels(
            [_shorten(c) for c in configs],
            rotation=35,
            ha="right"
        )

    ax_bottom.set_xlabel("Demographic models (ordered by number of populations)")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_per_config(stats_df, feature, config, out_path):

    df_f = stats_df[
        (stats_df["feature"] == feature) &
        (stats_df["config"] == config)
    ]

    simulators = sorted(df_f["simulator"].unique())

    n_sim = len(simulators)

    base = np.arange(n_sim)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    cmap = plt.get_cmap("tab10")
    colors = {sim: cmap(i) for i, sim in enumerate(simulators)}

    stats = []

    for sim in simulators:

        row = df_f[df_f["simulator"] == sim]

        if len(row) == 0:
            continue

        r = row.iloc[0]

        stats.append({
            "med": r["median"],
            "q1": r["q1"],
            "q3": r["q3"],
            "whislo": r["min"],
            "whishi": r["max"],
        })

    bp = ax.bxp(
        stats,
        positions=base,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )

    for i, b in enumerate(bp["boxes"]):
        sim = simulators[i]
        b.set_facecolor(colors[sim])
        b.set_alpha(0.35)
        b.set_edgecolor(colors[sim])

    for w in bp["whiskers"]:
        w.set_linewidth(1.1)

    for c in bp["caps"]:
        c.set_linewidth(1.1)

    ax.axhline(0, linestyle="--")

    ax.set_ylabel("Feature deviation from global median baseline (Δ)")
    ax.set_xlabel("Simulator")

    ax.set_xticks(base)
    ax.set_xticklabels(simulators, rotation=35, ha="right")

    ax.yaxis.grid(True, alpha=0.22)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(f"{feature} — {config}")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)

    args = parser.parse_args()

    root = args.input_dir.resolve()
    outdir = args.outdir.resolve()

    outdir.mkdir(parents=True, exist_ok=True)
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib()

    # ------------------------------------------------------

    print("Loading data")
    df = load_all_configs(root)

    # ------------------------------------------------------

    print("Computing deltas")
    df = compute_deltas(df)

    # ------------------------------------------------------

    stats_df = compute_box_stats(df)

    # ------------------------------------------------------

    print("Generating plots")

    SCALED_FEATURES = {"min_laplace", "min_adj", "num_trees"}

    for feature in tqdm(stats_df["feature"].unique(), desc="Plotting"):

        # normal
        plot_feature(
            stats_df,
            feature,
            plotdir / f"delta_{feature}.png",
            scale="linear"
        )

        if feature in SCALED_FEATURES:

            plot_feature(
                stats_df,
                feature,
                plotdir / f"delta_{feature}_arcsinh.png",
                scale="arcsinh"
            )

            plot_feature(
                stats_df,
                feature,
                plotdir / f"delta_{feature}_symlog.png",
                scale="symlog"
            )

            # split plots
            plot_feature_split(
                stats_df,
                feature,
                plotdir / f"delta_{feature}_split.png",
                scale="linear"
            )

            plot_feature_split(
                stats_df,
                feature,
                plotdir / f"delta_{feature}_split_arcsinh.png",
                scale="arcsinh"
            )

            plot_feature_split(
                stats_df,
                feature,
                plotdir / f"delta_{feature}_split_symlog.png",
                scale="symlog"
            )

            for cfg in ORDER:

                if cfg not in stats_df["config"].unique():
                    continue

                plot_feature_per_config(
                    stats_df,
                    feature,
                    cfg,
                    plotdir / f"delta_{feature}_{cfg}.png"
                )

    print("All plots generated.")


if __name__ == "__main__":
    main()