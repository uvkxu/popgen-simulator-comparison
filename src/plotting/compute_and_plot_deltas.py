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

from plot_functions import (
    plot_feature,
    plot_feature_split,
    print_feature_statistics,
    setup_matplotlib
)


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

    plot_feature(
        df,
        feature,
        outdir / f"delta_{feature}.png"
    )

    plot_feature_split(
        df,
        feature,
        outdir / f"delta_{feature}_split.png"
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

    print("Computing statistics...")
    stats = print_feature_statistics(df_delta)

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