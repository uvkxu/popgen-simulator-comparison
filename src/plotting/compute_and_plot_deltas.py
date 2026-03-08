#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

# prevent BLAS oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
# Matplotlib setup
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
# Efficient delta computation
# ==========================================================

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:

    group_cols = [
        "config",
        "sample_size_key",
        "seq_len",
        "feature",
    ]

    sim_medians = (
        df.groupby(group_cols + ["simulator"])["value"]
        .median()
        .reset_index()
    )

    baseline = (
        sim_medians
        .groupby(group_cols)["value"]
        .median()
        .rename("baseline")
        .reset_index()
    )

    df = df.merge(baseline, on=group_cols, how="left")

    df["delta"] = df["value"] - df["baseline"]

    return df


# ==========================================================
# Parallel CSV loading
# ==========================================================

def load_single_csv(csv: Path) -> pd.DataFrame:

    print(f"Reading {csv}")

    df = pd.read_csv(csv)

    mask = (df["feature"] == "difficulty") & (df["value"] == -1)
    df.loc[mask, "value"] = np.nan

    if "config" not in df.columns:
        df["config"] = csv.parent.name

    if "sample_size_key" not in df.columns:
        df["sample_size_key"] = None

    if "seq_len" not in df.columns:
        df["seq_len"] = None

    return df


def load_all_configs(root: Path, threads: int):

    csv_files = sorted(root.glob("*/simulator_feature_values.csv"))

    if not csv_files:
        raise RuntimeError("No simulator_feature_values.csv found.")

    print(f"Found {len(csv_files)} CSV files")
    print(f"Reading CSVs with {threads} parallel workers")

    with ProcessPoolExecutor(max_workers=threads) as pool:
        dfs = list(pool.map(load_single_csv, csv_files))

    print("Concatenating tables...")

    return pd.concat(dfs, ignore_index=True)


# ==========================================================
# Plotting
# ==========================================================

def plot_feature(
    df: pd.DataFrame,
    feature: str,
    out_path: Path,
):

    df_f = df[df["feature"] == feature]

    if df_f.empty or df_f["delta"].notna().sum() == 0:
        return

    configs = [c for c in ORDER if c in df_f["config"].unique()]
    simulators = sorted(df_f["simulator"].unique())

    base = np.arange(len(configs)) * 1.8

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

    cmap = plt.get_cmap("tab10")
    colors = {sim: cmap(i) for i, sim in enumerate(simulators)}

    for s_idx, sim in enumerate(simulators):

        offset = (s_idx - (len(simulators) - 1) / 2) * 0.22

        positions = []
        data = []

        for c_idx, cfg in enumerate(configs):

            vals = (
                df_f[(df_f["config"] == cfg) &
                     (df_f["simulator"] == sim)]["delta"]
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
            whis=(0, 100),
            showfliers=False,
            medianprops=dict(linewidth=1.5, color="black"),
        )

        for b in bp["boxes"]:
            b.set_facecolor(colors[sim])
            b.set_alpha(0.35)
            b.set_edgecolor(colors[sim])

    ax.axhline(0, linestyle="--", linewidth=1.4, color="0.1")

    ax.set_xticks(base)
    ax.set_xticklabels(
        [_shorten(c) for c in configs],
        rotation=35,
        ha="right"
    )

    ax.yaxis.grid(True, alpha=0.22)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--outdir",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2)
    )

    args = parser.parse_args()

    root = args.input_dir.resolve()
    outdir = args.outdir.resolve()

    outdir.mkdir(parents=True, exist_ok=True)
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib()

    print("Loading feature tables (parallel)...")

    df = load_all_configs(root, args.threads)

    print("Computing deltas...")

    df_delta = compute_deltas(df)

    delta_path = outdir / "deltas.csv"
    df_delta.to_csv(delta_path, index=False)

    print(f"Wrote {delta_path}")

    print("Generating plots...")

    features = sorted(df_delta["feature"].unique())

    for feature in features:

        plot_feature(
            df_delta,
            feature,
            plotdir / f"delta_{feature}.png"
        )

    print("All plots generated.")


if __name__ == "__main__":
    raise SystemExit(main())