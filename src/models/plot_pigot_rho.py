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
    nan_cols = []

    for col in df.columns:
        if df[col].isna().any():
            nan_cols.append(col)
    
    if nan_cols:
        raise ValueError(f"Columns with NaN values found: {nan_cols}. Please clean the data before plotting.")
    
    bool_stats = {
            "is_ultrametric": "Ultrametricity",
            "right_root_node": "Root Node Correctness",
            "right_tip_nodes": "Tip Node ID Correctness",
            "is_matrix": "Format as Matrix"
        }

    # --- 1. Boolean stats ---
    for col, title_suffix in bool_stats.items():
        plot_path = output_folder / f"pigot_rho_{col}.png"

        plt.figure(figsize=(6, 4))
        df_box = [df.loc[df[col], 'pigot_rho'], df.loc[~df[col], 'pigot_rho']]
        plt.boxplot(df_box, labels=[f"{title_suffix} = True", f"{title_suffix} = False"])
        plt.ylabel("Pigot's ρ")
        plt.title(f"Pigot's ρ by {title_suffix}")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    
    # --- 2. Boxplots by sample_size and effective_pop_size ---
    for col in ['sample_size', 'effective_pop_size', 'overall_sample_size']:
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

    if 'n2' in df.columns and 'n3' in df.columns:
        df_ratio = df.copy()
        df_ratio['n2_n3_ratio'] = df_ratio['n2'] / df_ratio['n3']

        plt.figure(figsize=(6, 4))
        plt.scatter(df_ratio['n2_n3_ratio'], df_ratio['pigot_rho'], alpha=0.6)
        plt.xlabel("n2 / n3 ratio")
        plt.ylabel("Pigot's ρ")
        plt.title("Pigot's ρ vs n2/n3 ratio")
        plt.tight_layout()
        plt.savefig(output_folder / "pigot_rho_vs_n2_n3_ratio.png")
        plt.close()

    if 'diag_r1' in df.columns and 'diag_r2' in df.columns:
        df_ratio = df.copy()
        df_ratio['r1_r2_ratio'] = df_ratio['diag_r1'] / df_ratio['diag_r2']

        plt.figure(figsize=(6, 4))
        plt.scatter(df_ratio['r1_r2_ratio'], df_ratio['pigot_rho'], alpha=0.6)
        plt.xlabel("r1 / r2 ratio")
        plt.ylabel("Pigot's ρ")
        plt.title("Pigot's ρ vs r1/r2 ratio")
        plt.tight_layout()
        plt.savefig(output_folder / "pigot_rho_vs_r1_r2_ratio.png")
        plt.close()

    
    # --- 3. Scatter plot Pigot's ρ vs num_branches ---
    for stat in ['num_branches', 'tree_height', 'mean_branch_length', 'crown_age', 
                    "n2", "n3", "diag_rho", "diag_n2", "diag_n3", "diag_r1", "diag_r2"]:
        if stat in df.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(df[stat], df['pigot_rho'], alpha=0.6)
            plt.xlabel(stat.replace('_', ' ').title())
            plt.ylabel("Pigot's ρ")
            plt.title(f"Pigot's ρ vs {stat.replace('_', ' ').title()}")
            plt.tight_layout()
            plt.savefig(output_folder / f"pigot_rho_vs_{stat}.png")
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