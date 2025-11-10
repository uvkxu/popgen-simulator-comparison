#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
import os

# Projektpfade
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import SIMULATORS, FILE_FORMATS
from utils.io_helpers import collect_csv_files

CSV_TYPES = [FILE_FORMATS["stats"].format(SIM=sim) for sim in SIMULATORS]


def load_and_merge(files):
    """Lädt mehrere CSV-Dateien und merged sie zeilenweise."""
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def filter_invalid_statistics(data_dict, log_file):
    """Entfernt Statistiken mit NaN- oder Inf-Werten."""
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
    """Filtert Zeilen heraus, bei denen zero_branch_count != 0, tree_type == 0 oder cut_edge_weight == -1."""
    filtered_data = {}
    for sim_name, df in data_dict.items():
        if df is not None:
            missing_cols = [c for c in ["zero_branch_count", "tree_type", "cut_edge_weight"] if c not in df.columns]
            if missing_cols:
                print(f"⚠️  Warning: Columns {missing_cols} not found in {sim_name} data, skipping those filters.")
                filtered_df = df.copy()
            else:
                before = len(df)
                filtered_df = df[
                    (df["zero_branch_count"] == 0)
                    & (df["tree_type"] != 0)
                    & (df["cut_edge_weight"] != -1)
                ].copy()
                removed_count = before - len(filtered_df)
                print(f"{sim_name}: removed {removed_count} rows (invalid branches/edges filtered).")

            filtered_data[sim_name] = filtered_df if len(filtered_df) > 0 else None
        else:
            filtered_data[sim_name] = None

    return filtered_data


def plot_distributions(data_dict, output_dir):
    """Erstellt Verteilungsplots (Histogramme/Dichtekurven) pro Statistik mit allen Simulatoren im selben Plot."""
    # gemeinsame Spalten (nur die behalten, die alle Simulatoren haben)
    shared_columns = set.intersection(*(set(df.columns) for df in data_dict.values() if df is not None))
    output_dir.mkdir(parents=True, exist_ok=True)

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
        out_file = output_dir / f"{stat}_distribution.png"
        plt.savefig(out_file)
        plt.close()
        print(f"Saved: {out_file}")


def sanity_check_cut_edge_weights(data_dict):
    """Überprüft, ob nach dem Filtern noch ungültige cut_edge_weight-Werte (<= 0) vorhanden sind."""
    print("\n=== 🧪 Sanity Check: cut_edge_weight values ===")
    for sim_name, df in data_dict.items():
        if df is None:
            print(f"{sim_name}: ⚠️  Keine Daten.")
            continue
        elif "cut_edge_weight" not in df.columns:
            print(f"{sim_name}: ⚠️  Keine Daten oder Spalte 'cut_edge_weight' fehlt.")
            continue

        invalid_mask = df["cut_edge_weight"] <= 0
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            min_val = df["cut_edge_weight"].min()
            print(f"{sim_name}: ⚠️  {invalid_count} ungültige Werte gefunden (min={min_val}).")
        else:
            print(f"{sim_name}: ✅ alle cut_edge_weight-Werte > 0.")
    print("=== End of check ===\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_distributions.py <input_folder> <output_folder> [--filter_zero_branches]")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    filter_zero_branches = "--filter_zero_branches" in sys.argv

    if not input_folder.exists():
        print(f"Input folder does not exist: {input_folder}")
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)
    log_file = output_folder / "removed_statistics.log"

    # CSV-Dateien sammeln
    csv_files = collect_csv_files(input_folder, CSV_TYPES)
    if not csv_files:
        print("No CSV files found.")
        return

    # Nach Simulatoren gruppieren
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

    # Zusammenführen
    data_dict = {k: load_and_merge(v) for k, v in data_dict.items() if v}

    # Optional: Filter
    if filter_zero_branches:
        print("Applying zero-branch filtering...")
        data_dict = filter_zero_branch_rows(data_dict)
        sanity_check_cut_edge_weights(data_dict)

    # Ungültige Statistiken entfernen
    data_dict, kept_stats = filter_invalid_statistics(data_dict, log_file)
    if not kept_stats:
        print("No valid statistics to plot after filtering.")
        return

    # Plotten
    plot_distributions(data_dict, output_folder)
    print(f"✅ All distribution plots saved to {output_folder}")


if __name__ == "__main__":
    main()