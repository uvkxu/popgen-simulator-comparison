import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse
import os
import sys


UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from utils.io_helpers import collect_csv_files, detect_simulator_from_yaml

def plot_pigot_rho_relationships(csv_files, output_folder):
    """
    Generate plots for Pigot's rho vs. different tree statistics.
    
    Parameters
    ----------
    csv_file : str or Path
        Path to the CSV file with columns:
        ['pigot_rho', 'is_ultrametric', 'num_branches', 'sample_size', 'effective_pop_size']
    output_folder : str or Path
        Folder to save the plots.
    """
    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = f.name  # optional: track origin
            all_dfs.append(df)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    
    if not all_dfs:
        print("No valid CSVs loaded.")
        return
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    # --- 1. Boxplots by ultrametric ---
    plt.figure(figsize=(6, 4))
    df_box = [df.loc[df['is_ultrametric'], 'pigot_rho'],
              df.loc[~df['is_ultrametric'], 'pigot_rho']]
    plt.boxplot(df_box, labels=['Ultrametric', 'Non-Ultrametric'])
    plt.ylabel("Pigot's ρ")
    plt.title("Pigot's ρ by Ultrametricity")
    plt.tight_layout()
    plt.savefig(output_folder / "pigot_rho_ultrametric.png")
    plt.close()
    
    # --- 2. Boxplots by sample_size and effective_pop_size ---
    for col in ['sample_size', 'effective_pop_size']:
        if col not in df.columns:
            continue
        unique_values = sorted(df[col].dropna().unique())
        plt.figure(figsize=(max(6, len(unique_values)), 4))
        data = [df.loc[df[col] == val, 'pigot_rho'] for val in unique_values]
        plt.boxplot(data, labels=[str(int(v)) for v in unique_values])
        plt.ylabel("Pigot's ρ")
        plt.xlabel(col)
        plt.title(f"Pigot's ρ by {col}")
        plt.tight_layout()
        plt.savefig(output_folder / f"pigot_rho_by_{col}.png")
        plt.close()
    
    # Optional: ratio of population / sample size
    if 'sample_size' in df.columns and 'effective_pop_size' in df.columns:
        df_ratio = df.copy()
        df_ratio['pop_sample_ratio'] = df_ratio['effective_pop_size'] / df_ratio['sample_size']
        unique_values = sorted(df_ratio['pop_sample_ratio'].dropna().unique())
        plt.figure(figsize=(max(6, len(unique_values)), 4))
        data = [df_ratio.loc[df_ratio['pop_sample_ratio'] == val, 'pigot_rho'] for val in unique_values]
        plt.boxplot(data, labels=[f"{v:.1f}" for v in unique_values])
        plt.ylabel("Pigot's ρ")
        plt.xlabel("Population / Sample size ratio")
        plt.title("Pigot's ρ by pop/sample ratio")
        plt.tight_layout()
        plt.savefig(output_folder / "pigot_rho_by_pop_sample_ratio.png")
        plt.close()
    
    # --- 3. Scatter plot Pigot's ρ vs num_branches ---
    if 'num_branches' in df.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df['num_branches'], df['pigot_rho'], alpha=0.6)
        plt.xlabel("Number of branches (unweighted)")
        plt.ylabel("Pigot's ρ")
        plt.title("Pigot's ρ vs Number of branches")
        plt.tight_layout()
        plt.savefig(output_folder / "pigot_rho_vs_num_branches.png")
        plt.close()
    
    print(f"Plots saved in {output_folder}")


def filter_csvs_by_simulator(csv_files, simulators):
    if simulators is None:
        return csv_files

    filtered = []
    for csv in csv_files:
        sim = detect_simulator_from_yaml(csv)
        if sim in simulators:
            filtered.append(csv)

    return filtered


def parse_args():
    parser = argparse.ArgumentParser(description="Extract top features from simulation results.")
    parser.add_argument("--input_folder", type=Path, required=True, help="Input folder containing result files")
    parser.add_argument("--output_folder", type=Path, required=True, help="Output folder path")
    parser.add_argument(
        "--simulators",
        type=str,
        default=None,
        help="Comma-separated list of simulators (e.g. ms,msms). If omitted, all are used."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    if args.simulators is not None:
        simulators = {s.strip() for s in args.simulators.split(",") if s.strip()}
    else:
        simulators = "ms"

    csv_files = collect_csv_files(input_folder)
    csv_files = filter_csvs_by_simulator(csv_files, simulators)

    output_folder.mkdir(parents=True, exist_ok=True)
    plot_pigot_rho_relationships(csv_files, output_folder)

if __name__ == "__main__":
    main()