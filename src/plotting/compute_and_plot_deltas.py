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

def setup_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })


def _shorten(s: str, maxlen: int = 24):
    return s if len(s) <= maxlen else s[:maxlen - 1] + "…"


# ==========================================================
# Parallel CSV loading
# ==========================================================

def load_single_csv(csv: Path):

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

    print(f"Loading {len(csv_files)} CSV files with {threads} workers")

    with ProcessPoolExecutor(max_workers=threads) as pool:
        dfs = list(pool.map(load_single_csv, csv_files))

    return pd.concat(dfs, ignore_index=True)


# ==========================================================
# Parallel delta computation
# ==========================================================

def compute_group_delta(args):

    key, group = args

    sim_medians = (
        group.groupby("simulator")["value"]
        .median()
        .reset_index()
    )

    baseline = sim_medians["value"].median()

    group = group.copy()
    group["baseline"] = baseline
    group["delta"] = group["value"] - baseline

    return group


def compute_deltas_parallel(df: pd.DataFrame, threads: int):

    group_cols = [
        "config",
        "sample_size_key",
        "seq_len",
        "feature",
    ]

    groups = list(df.groupby(group_cols))

    print(f"Computing deltas for {len(groups)} groups")

    with ProcessPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(compute_group_delta, groups))

    return pd.concat(results, ignore_index=True)


# ==========================================================
# Plotting
# ==========================================================

def plot_feature(args):

    feature, df_f, out_path = args

    configs = [c for c in ORDER if c in df_f["config"].unique()]
    simulators = sorted(df_f["simulator"].unique())

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

    # ------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------

    print("Loading CSV tables...")
    df = load_all_configs(root, args.threads)

    # ------------------------------------------------------
    # Delta computation
    # ------------------------------------------------------

    print("Computing deltas...")
    df_delta = compute_deltas_parallel(df, args.threads)

    df_delta.to_csv(outdir / "deltas.csv", index=False)

    # ------------------------------------------------------
    # Plotting
    # ------------------------------------------------------

    print("Generating plots...")

    tasks = []

    for feature in df_delta["feature"].unique():

        df_f = df_delta[df_delta["feature"] == feature]

        tasks.append(
            (feature, df_f, plotdir / f"delta_{feature}.png")
        )

    with ProcessPoolExecutor(max_workers=args.threads) as pool:
        pool.map(plot_feature, tasks)

    print("All plots generated.")


if __name__ == "__main__":
    raise SystemExit(main())