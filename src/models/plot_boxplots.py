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
                if df is not None:
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
    stats = combined_df["Statistic"].unique()
    output_dir.mkdir(parents=True, exist_ok=True)

    for stat in stats:
        df_stat = combined_df[combined_df["Statistic"] == stat]
        plt.figure(figsize=(6,4))
        sns.boxplot(x="Simulation", y="Value", data=df_stat)
        plt.title(stat)
        plt.tight_layout()
        out_file = output_dir / f"{stat}_boxplot.png"
        plt.savefig(out_file)
        plt.close()
        print(f"Saved boxplot: {out_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_boxplots.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])

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
