#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import SIMULATORS, FILE_FORMATS
from utils.io_helpers import collect_csv_files

CSV_TYPES = [FILE_FORMATS["stats"].format(SIM=sim) for sim in SIMULATORS]


def load_and_merge(files):
    """Load multiple CSV files and merge them row-wise."""
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def filter_invalid_statistics(data_dict, log_file):
    all_stats = set.intersection(*(set(df.columns) for df in data_dict.values() if df is not None))
    removed_stats, kept_stats = [], []

    for stat in all_stats:
        has_invalid = any(df[stat].isna().any() or df[stat].isin([np.inf, -np.inf]).any()
                          for df in data_dict.values() if df is not None)
        if has_invalid:
            removed_stats.append(stat)
            for df in data_dict.values():
                if df is not None and stat in df.columns:
                    df.drop(columns=[stat], inplace=True)
        else:
            kept_stats.append(stat)

    with open(log_file, "w") as f:
        f.write("=== Removed statistics due to NaN/Inf ===\n")
        for stat in removed_stats:
            f.write(stat + "\n")
        f.write("\n=== Kept statistics ===\n")
        for stat in kept_stats:
            f.write(stat + "\n")

    return data_dict, kept_stats


def filter_zero_branch_rows(data_dict):
    """Filter out rows where zero_branch_count != 0 or tree_type == 0."""
    filtered_data = {}
    for sim_name, df in data_dict.items():
        if df is not None:
            missing_cols = [c for c in ["zero_branch_count", "tree_type"] if c not in df.columns]
            if missing_cols:
                print(f"⚠️  Warning: Columns {missing_cols} not found in {sim_name} data, skipping those filters.")
                filtered_df = df.copy()
            else:
                before = len(df)
                filtered_df = df[(df["zero_branch_count"] == 0) & (df["tree_type"] != 0) & (df["cut_edge_weight"] != -1)].copy()
                removed_count = before - len(filtered_df)
                print(f"{sim_name}: removed {removed_count} rows (zero_branch_count != 0 or tree_type == 0).")

            if len(filtered_df) == 0:
                print(f"⚠️  No rows left for {sim_name} after filtering.")
                filtered_data[sim_name] = None
            else:
                filtered_data[sim_name] = filtered_df
        else:
            filtered_data[sim_name] = None

    return filtered_data


def melt_data_for_plotting(data_dict):
    dfs = []
    for sim_name, df in data_dict.items():
        if df is not None:
            df_long = df.melt(var_name="Statistic", value_name="Value")
            df_long["Simulation"] = sim_name
            dfs.append(df_long)
    combined = pd.concat(dfs, ignore_index=True) if dfs else None
    return combined


def plot_boxplots(combined_df, output_dir):
    if combined_df is None or combined_df.empty:
        print("No data available for plotting.")
        return

    stats = combined_df["Statistic"].unique()
    output_dir.mkdir(parents=True, exist_ok=True)

    for stat in stats:
        df_stat = combined_df[combined_df["Statistic"] == stat]

        # Skip if there is not enough data for a proper boxplot
        valid_sims = df_stat["Simulation"].nunique()
        if len(df_stat) < 5 or valid_sims < 2:
            print(f"Skipping {stat}: not enough data (rows={len(df_stat)}, sims={valid_sims})")
            continue

        plt.figure(figsize=(6, 4))
        sns.boxplot(x="Simulation", y="Value", data=df_stat)
        plt.title(stat)
        plt.tight_layout()
        out_file = output_dir / f"{stat}_boxplot.png"
        plt.savefig(out_file)
        plt.close()
        print(f"Saved boxplot: {out_file}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_boxplots.py <input_folder> <output_folder> [--filter_zero_branches]")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    filter_zero_branches = "--filter_zero_branches" in sys.argv

    if not input_folder.exists():
        print(f"Input folder does not exist: {input_folder}")
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)
    log_file = output_folder / "removed_statistics.log"

    # Collect all CSV files (list of Paths)
    csv_files = collect_csv_files(input_folder, CSV_TYPES)

    if not csv_files:
        print("No CSV files found in input folder.")
        return

    # Organize files by simulator type
    data_dict = {sim: [] for sim in SIMULATORS}
    for f in csv_files:
        fname = f.name.lower()
        if "msprime" in fname:
            data_dict["msprime"].append(f)
        elif "msms" in fname:
            data_dict["msms"].append(f)
        elif "ms_" in fname or fname == "ms.csv":
            data_dict["ms"].append(f)
        else:
            print(f"Skipping unknown file: {f}")

    # Load and merge CSVs per simulator
    data_dict = {k: load_and_merge(v) for k, v in data_dict.items() if v}

    # Optional filtering by zero_branch_count
    if filter_zero_branches:
        print("Applying zero_branch_count == 0 filter...")
        data_dict = filter_zero_branch_rows(data_dict)

    # Filter invalid statistics
    data_dict, kept_stats = filter_invalid_statistics(data_dict, log_file)

    if not kept_stats:
        print("No valid statistics to plot after filtering NaN/Inf. See log for details.")
        return

    combined_df = melt_data_for_plotting(data_dict)
    plot_boxplots(combined_df, output_folder)
    print(f"All boxplots saved to {output_folder}, removed/kept statistics logged in {log_file}")


if __name__ == "__main__":
    main()
