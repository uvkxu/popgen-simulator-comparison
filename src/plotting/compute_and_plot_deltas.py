#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

ORDER = [
    "one_population",
    "zigzag",
    "li_african3epoch",
    "lavretsky_ducks",
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


def plot_feature_split(
    df: pd.DataFrame,
    feature: str,
    out_path: Path,
    threshold_factor: float = 5.0,
):

    df_f = df[df["feature"] == feature]

    # nothing at all
    if df_f.empty or df_f["delta"].notna().sum() == 0:
        return

    configs = [c for c in ORDER if c in df_f["config"].unique()]
    simulators = sorted(df_f["simulator"].unique())

    # ---- NaN-robust medians ----
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
        return  # nothing to split

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
    config_gap: float = 1.8,
    within_gap: float = 0.22,
    box_width: float = 0.18,
):

    df_f = df[df["feature"] == feature]

    if df_f.empty or df_f["delta"].notna().sum() == 0:
        return

    configs = [c for c in ORDER if c in df_f["config"].unique()]
    simulators = sorted(df_f["simulator"].unique())

    n_cfg = len(configs)
    n_sim = len(simulators)

    base = np.arange(n_cfg) * config_gap

    fig_w = max(12, min(26, 0.8 * n_cfg + 6))
    fig, ax = plt.subplots(figsize=(fig_w, 6.2), constrained_layout=True)

    problematic_features = {
        "phylogenetic_div",
        "mean_branch_length",
        "tree_height",
        "bollback_multinomial"
    }

    if feature in problematic_features:
        ax.set_ylabel("Feature deviation from global median (Δ, arcsinh scale)")
        ax.set_yscale("function", functions=(np.arcsinh, np.sinh))
    else: 
        ax.set_ylabel("Feature deviation from global median baseline (Δ)")


    for i in range(n_cfg - 1):
        ax.axvline(
            (base[i] + base[i + 1]) / 2,
            color="0.2",
            linewidth=0.8,
            alpha=0.12,
            zorder=0
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.4,
        color="0.1",
        alpha=0.9,
        zorder=1
    )

    cmap = plt.get_cmap("tab10")
    colors = {sim: cmap(i) for i, sim in enumerate(simulators)}

    for s_idx, sim in enumerate(simulators):

        offset = (s_idx - (n_sim - 1) / 2) * within_gap

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
            widths=box_width,
            patch_artist=True,
            whis=(0, 100),
            showfliers=False,
            medianprops=dict(linewidth=1.5, color="black"),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
            boxprops=dict(linewidth=1.0),
        )

        for b in bp["boxes"]:
            b.set_facecolor(colors[sim])
            b.set_alpha(0.35)
            b.set_edgecolor(colors[sim])

        for w in bp["whiskers"]:
            w.set_color(colors[sim])
        for c in bp["caps"]:
            c.set_color(colors[sim])

    ax.set_xlabel("Demographic models (ordered by number of populations)")

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

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

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

    ax.legend(handles=legend_handles, 
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True
    )

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

    # median per simulator inside subgroup
    sim_medians = (
        df.groupby(group_cols + ["simulator"])["value"]
        .median()
        .reset_index()
    )

    # baseline = median over simulator medians
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
# Load all configs
# ==========================================================

def load_all_configs(root: Path):

    csv_files = sorted(root.glob("*/simulator_feature_values.csv"))

    if not csv_files:
        raise RuntimeError("No simulator_feature_values.csv found.")

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
# Parallel plotting worker
# ==========================================================

def plot_worker(args):

    feature, df, outdir = args
    plot_dir = "plots"

    plot_feature(
        df,
        feature,
        Path(outdir / plot_dir / f"delta_{feature}.png")
    )

    plot_feature_split(
        df,
        feature,
        Path(outdir / plot_dir / f"delta_{feature}_split.png")
    )


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
        default=max(1, multiprocessing.cpu_count() - 1)
    )

    args = parser.parse_args()

    root = args.input_dir.resolve()
    outdir = args.outdir.resolve()

    outdir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib()

    print("Loading feature tables...")
    df = load_all_configs(root)

    print("Computing deltas (vectorized)...")
    df_delta = compute_deltas(df)

    delta_path = outdir / "deltas.csv"
    df_delta.to_csv(delta_path, index=False)

    print(f"Wrote {delta_path}")

    stats.to_csv(
        outdir / "delta_feature_statistics.csv",
        index=False
    )

    features = sorted(df_delta["feature"].unique())

    print(f"Generating plots for {len(features)} features "
          f"using {args.threads} threads")

    tasks = [
        (feature, df_delta, outdir)
        for feature in features
    ]

    with ProcessPoolExecutor(max_workers=args.threads) as pool:
        list(pool.map(plot_worker, tasks))

    print("All plots generated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())