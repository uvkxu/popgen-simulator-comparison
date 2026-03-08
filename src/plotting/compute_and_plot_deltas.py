#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl
import numpy as np
from tqdm import tqdm

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

def setup_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 11,
    })


def _shorten(s: str, maxlen: int = 24):
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

def plot_feature(stats_df, feature, out_path):

    df_f = stats_df[stats_df["feature"] == feature]

    configs = [c for c in ORDER if c in df_f["config"].unique()]
    simulators = sorted(df_f["simulator"].unique())

    base = np.arange(len(configs)) * 1.8

    fig, ax = plt.subplots(figsize=(14, 6))

    cmap = plt.get_cmap("tab10")
    colors = {sim: cmap(i) for i, sim in enumerate(simulators)}

    for s_idx, sim in enumerate(simulators):

        offset = (s_idx - (len(simulators) - 1) / 2) * 0.22

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
            widths=0.18,
            patch_artist=True,
            showfliers=False,
        )

        for b in bp["boxes"]:
            b.set_facecolor(colors[sim])
            b.set_alpha(0.35)

    ax.axhline(0, linestyle="--")

    ax.set_xticks(base)
    ax.set_xticklabels(
        [_shorten(c) for c in configs],
        rotation=35,
        ha="right"
    )

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

    for feature in tqdm(stats_df["feature"].unique(), desc="Plotting"):

        plot_feature(
            stats_df,
            feature,
            plotdir / f"delta_{feature}.png"
        )

    print("All plots generated.")


if __name__ == "__main__":
    main()