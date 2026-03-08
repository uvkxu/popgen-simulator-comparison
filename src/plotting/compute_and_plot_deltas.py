#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing

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
# CSV loading (polars parallel)
# ==========================================================

def load_all_configs(root: Path):

    csv_files = sorted(root.glob("*/simulator_feature_values.csv"))

    print(f"Loading {len(csv_files)} CSV files")

    dfs = []

    for csv in tqdm(csv_files, desc="Reading CSVs"):

        df = pl.read_csv(
            csv,
            try_parse_dates=False
        )

        if "config" not in df.columns:
            df = df.with_columns(pl.lit(csv.parent.name).alias("config"))

        if "sample_size_key" not in df.columns:
            df = df.with_columns(pl.lit(None).alias("sample_size_key"))

        if "seq_len" not in df.columns:
            df = df.with_columns(pl.lit(None).alias("seq_len"))

        dfs.append(df)

    return pl.concat(dfs)


# ==========================================================
# Delta computation (fully parallel)
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
        .groupby(group_cols + ["simulator"])
        .agg(pl.median("value").alias("value"))
    )

    print("Computing baseline medians")

    baseline = (
        sim_medians
        .groupby(group_cols)
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
# Plotting
# ==========================================================

def plot_feature(feature_df: pd.DataFrame, feature: str, out_path: Path):

    configs = [c for c in ORDER if c in feature_df["config"].unique()]
    simulators = sorted(feature_df["simulator"].unique())

    base = np.arange(len(configs)) * 1.8

    fig, ax = plt.subplots(figsize=(14, 6))

    cmap = plt.get_cmap("tab10")
    colors = {sim: cmap(i) for i, sim in enumerate(simulators)}

    for s_idx, sim in enumerate(simulators):

        offset = (s_idx - (len(simulators) - 1) / 2) * 0.22

        positions = []
        data = []

        for c_idx, cfg in enumerate(configs):

            vals = (
                feature_df[
                    (feature_df["config"] == cfg) &
                    (feature_df["simulator"] == sim)
                ]["delta"]
                .dropna()
                .values
            )

            if len(vals) > 0:
                positions.append(base[c_idx] + offset)
                data.append(vals)

        if not data:
            continue

        bp = ax.boxplot(
            data,
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

    print("Converting to pandas for plotting")
    df_pd = df.to_pandas()

    delta_path = outdir / "deltas.csv"
    df_pd.to_csv(delta_path, index=False)

    print(f"Wrote {delta_path}")

    # ------------------------------------------------------

    print("Generating plots")

    for feature in tqdm(df_pd["feature"].unique(), desc="Plotting"):

        df_f = df_pd[df_pd["feature"] == feature]

        plot_feature(
            df_f,
            feature,
            plotdir / f"delta_{feature}.png"
        )

    print("All plots generated.")


if __name__ == "__main__":
    main()