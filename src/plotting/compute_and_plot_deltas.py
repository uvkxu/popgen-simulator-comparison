#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

# prevent numpy / BLAS oversubscription
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
# Plotting
# ==========================================================

def plot_feature_split(
    df: pd.DataFrame,
    feature: str,
    out_path: Path,
    threshold_factor: float = 5.0,
):

    df_f = df[df["feature"] == feature]

    if df_f.empty or df_f["delta"].notna().sum() == 0:
        return

    configs = [c for c in ORDER if c in df_f["config"].unique()]
    simulators = sorted(df_f["simulator"].unique())

    medians = (
        df_f.groupby("config")["delta"]
        .apply(lambda x: np.nanmedian(np.abs(x)))
    )

    global_med = np.nanmedian(medians.values)

    if not np.isfinite(global_med):
        return

    extreme = medians[medians > threshold_factor * global_med].index.tolist()
    regular = [c for c in configs if c not in extreme]

    if len(extreme) == 0:
        return

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        sharex=False,
        gridspec_kw={"height_ratios": [2.2, 1.3]},
        constrained_layout=True
    )

    def draw(ax, cfg_subset):

        base = np.arange(len(cfg_subset)) * 1.8
        cmap = plt.get_cmap("tab10")
        colors = {sim: cmap(i) for i, sim in enumerate(simulators)}

        for s_idx, sim in enumerate(simulators):
            offset = (s_idx - (len(simulators) - 1) / 2) * 0.22

            positions = []
            data = []

            for c_idx, cfg in enumerate(cfg_subset):

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
        ax.yaxis.grid(True, alpha=0.22)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xticks(base)
        ax.set_xticklabels(
            [_shorten(c) for c in cfg_subset],
            rotation=35,
            ha="right"
        )

    draw(ax_top, regular)
    draw(ax_bottom, extreme)

    ax_top.set_ylabel("Δ (regular models)")
    ax_bottom.set_ylabel("Δ (extreme models)")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


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
# Load configs
# ==========================================================

def load_all_configs(root: Path):

    csv_files = sorted(root.glob("*/simulator_feature_values.csv"))

    dfs = []

    for csv in csv_files:

        df = pd.read_csv(csv)

        mask = (df["feature"] == "difficulty") & (df["value"] == -1)
        df.loc[mask, "value"] = np.nan

        if "config" not in df.columns:
            df["config"] = csv.parent.name

        if "sample_size_key" not in df.columns:
            df["sample_size_key"] = None

        if "seq_len" not in df.columns:
            df["seq_len"] = None

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ==========================================================
# Parallel worker
# ==========================================================

def prepare_group(args):

    feature, config, sample_size, seq_len, df_g = args

    rows = []

    for sim in df_g["simulator"].unique():

        vals = (
            df_g[df_g["simulator"] == sim]["delta"]
            .dropna()
            .values
        )

        for v in vals:
            rows.append({
                "feature": feature,
                "config": config,
                "sample_size_key": sample_size,
                "seq_len": seq_len,
                "simulator": sim,
                "delta": v
            })

    return rows


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

    print("Loading feature tables...")
    df = load_all_configs(root)

    print("Computing deltas...")
    df_delta = compute_deltas(df)

    delta_path = outdir / "deltas.csv"
    df_delta.to_csv(delta_path, index=False)

    print("Preparing parallel tasks...")

    group_cols = [
        "feature",
        "config",
        "sample_size_key",
        "seq_len",
    ]

    groups = list(df_delta.groupby(group_cols))

    tasks = [
        (feature, config, sample_size, seq_len, gdf)
        for (feature, config, sample_size, seq_len), gdf in groups
    ]

    print(f"Running {len(tasks)} tasks on {args.threads} workers")

    with ProcessPoolExecutor(max_workers=args.threads) as pool:
        results = list(pool.map(prepare_group, tasks))

    rows = [r for sub in results for r in sub]

    df_prepared = pd.DataFrame(rows)

    print("Generating plots...")

    for feature in df_prepared["feature"].unique():

        plot_feature(
            df_prepared,
            feature,
            plotdir / f"delta_{feature}.png"
        )

        plot_feature_split(
            df_prepared,
            feature,
            plotdir / f"delta_{feature}_split.png"
        )

    print("All plots generated.")


if __name__ == "__main__":
    raise SystemExit(main())