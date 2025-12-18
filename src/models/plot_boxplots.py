#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
import os
import yaml
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import SIMULATORS, VALID_GROUPING_MODES
from utils.io_helpers import collect_csv_files, detect_simulator_from_yaml


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


def melt_data_for_plotting(data_dict):
    dfs = []
    for sim_name, df in data_dict.items():
        if df is not None:
            df_long = df.melt(var_name="Statistic", value_name="Value")
            df_long["Simulation"] = sim_name
            dfs.append(df_long)
    combined = pd.concat(dfs, ignore_index=True) if dfs else None
    return combined


def get_features_by_simulator(csv_files, output_folder, selected_simulators):
    output_folder.mkdir(parents=True, exist_ok=True)
    log_file = output_folder / "removed_statistics.log"

    if not csv_files:
        print("No CSV files found in input folder.")
        return

    data_dict = {sim: [] for sim in selected_simulators}

    for f in csv_files:
        sim = detect_simulator_from_yaml(
            input_file=f
        )
        if sim is None:
            print(f"Skipping unknown file: {f}")
            continue
        data_dict[sim].append(f)

    # Load and merge CSVs per simulator
    data_dict = {k: load_and_merge(v) for k, v in data_dict.items() if v}

    # Filter invalid statistics
    data_dict, kept_stats = filter_invalid_statistics(data_dict, log_file)

    if not kept_stats:
        print("No valid statistics to plot after filtering NaN/Inf. See log for details.")
        return
    
    return data_dict


def plot_boxplots(csv_files, output_folder, selected_simulators):
    data_dict = get_features_by_simulator(
        csv_files=csv_files,
        output_folder=output_folder,
        selected_simulators=selected_simulators
    )

    combined_df = melt_data_for_plotting(data_dict)
    if combined_df is None or combined_df.empty:
        print("No data available for plotting.")
        return

    stats = combined_df["Statistic"].unique()

    for stat in stats:
        df_stat = combined_df[combined_df["Statistic"] == stat]

        # ---- NEW: Print unique values for zero_branch_count ----
        if stat == "zero_branch_count":
            print("\n=== Unique zero_branch_count values per simulator ===")
            for sim in df_stat["Simulation"].unique():
                vals = sorted(df_stat[df_stat["Simulation"] == sim]["Value"].unique().tolist())
                print(f"  Simulator {sim}: unique values = {vals}")
            print("=" * 60 + "\n")
        # ----------------------------------------------------------
        
        # Skip if there is not enough data for a proper boxplot
        valid_sims = df_stat["Simulation"].nunique()
        if len(df_stat) < 5 or valid_sims < 2:
            print(f"Skipping {stat}: not enough data (rows={len(df_stat)}, sims={valid_sims})")
            continue

        plt.figure(figsize=(6, 4))
        sns.boxplot(x="Simulation", y="Value", data=df_stat)
        plt.title(stat)
        plt.tight_layout()
        out_file = output_folder / f"{stat}_boxplot.png"
        plt.savefig(out_file)
        plt.close()
        # print(f"Saved boxplot: {out_file}")


def plot_histograms(csv_files, output_folder, selected_simulators):
    data_dict = get_features_by_simulator(
        csv_files=csv_files,
        output_folder=output_folder,
        selected_simulators=selected_simulators
    )

    shared_columns = set.intersection(*(set(df.columns) for df in data_dict.values() if df is not None))
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Plotting distributions for {len(shared_columns)} shared statistics...")

    for stat in shared_columns:
        plt.figure(figsize=(7, 5))

        for sim_name, df in data_dict.items():
            if df is not None and stat in df.columns:
                sns.kdeplot(df[stat], label=sim_name, fill=True, alpha=0.3)
        
        if stat == "cut_edge_weight":
            plt.xscale("log")

        plt.title(f"Distribution of {stat}")
        plt.xlabel(stat)
        plt.ylabel("Density")
        plt.legend(title="Simulator")
        plt.tight_layout()
        out_file = output_folder / f"{stat}_histogram.png"
        plt.savefig(out_file)
        plt.close()
        print(f"Saved: {out_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract top features from simulation results."
    )

    parser.add_argument(
        "--input_folder",
        type=Path,
        required=True,
        help="Input folder containing result files",
    )

    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Output folder path",
    )

    parser.add_argument(
        "--simulators",
        type=lambda s: s.split(","),
        default=SIMULATORS,
        help="Comma-separated list of simulators (e.g. sim1,sim2)",
    )

    parser.add_argument(
        "--grouped_by",
        type=str,
        choices=VALID_GROUPING_MODES,
        default="single",
        help=f"Grouping mode. Choices: {', '.join(VALID_GROUPING_MODES)} (default: single)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    selected_simulators = args.simulators
    grouped_by = args.grouped_by

    output_folder.mkdir(parents=True, exist_ok=True)

    csv_with_precision = []
    csv_without_precision = []

    csv_with_recombination = []
    csv_without_recombination = []

    if "single" == grouped_by or  "overall" == grouped_by:
        csv_files = collect_csv_files(input_folder)
        if not csv_files:
            print(f"No CSV files found in input folder {input_folder} with mode {grouped_by}.")
            return
        plot_boxplots(csv_files, output_folder, selected_simulators)
        plot_histograms(csv_files, output_folder, selected_simulators)
        return
        
    for root, _, files in os.walk(input_folder):
        root_path = Path(root)

        combo_yamls = [f for f in files if f.endswith(".yaml") or f.endswith(".yml")]
        if not combo_yamls:
            continue
        
        combo_yaml_path = root_path / combo_yamls[0]
        with open(combo_yaml_path, "r") as f:
            combo_params = yaml.safe_load(f)

        usePrecision = bool(combo_params.get("usePrecision"))
        recombination_rate = float(combo_params.get("recombination_rate") or 0)

        csv_files = collect_csv_files(root_path)

        if usePrecision:
            csv_with_precision += csv_files
            if recombination_rate:
                csv_with_recombination += csv_files
            else:
                csv_without_recombination += csv_files
        else:
            csv_without_precision += csv_files

    if "precision_applied" == grouped_by:
        if not csv_with_precision:
            print(f"No CSV files found in input folder {input_folder} with mode {grouped_by}.")
            return
        plot_boxplots(csv_with_precision, output_folder, selected_simulators)
        plot_histograms(csv_with_precision, output_folder, selected_simulators)
    elif "precision_skipped" == grouped_by:
        if not csv_without_precision:
            print(f"No CSV files found in input folder {input_folder} with mode {grouped_by}.")
            return
        plot_boxplots(csv_without_precision, output_folder, selected_simulators)
        plot_histograms(csv_without_precision, output_folder, selected_simulators)
    elif "recombination_applied" == grouped_by:
        if not csv_with_recombination:
            print(f"No CSV files found in input folder {input_folder} with mode {grouped_by}.")
            return
        plot_boxplots(csv_with_recombination, output_folder, selected_simulators)
        plot_histograms(csv_with_recombination, output_folder, selected_simulators)
    elif "recombination_skipped" == grouped_by:
        if not csv_without_recombination:
            print(f"No CSV files found in input folder {input_folder} with mode {grouped_by}.")
            return
        plot_boxplots(csv_without_recombination, output_folder, selected_simulators)
        plot_histograms(csv_without_recombination, output_folder, selected_simulators)



if __name__ == "__main__":
    main()
