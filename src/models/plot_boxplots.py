#!/usr/bin/env python3
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
import os
import yaml
import argparse
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import SIMULATORS, VALID_GROUPING_MODES
from utils.io_helpers import collect_csv_files, detect_simulator_from_yaml


def load_csv_float32(file):
    """Load a single CSV with float32 dtype to save memory."""
    return pd.read_csv(file, dtype=np.float32)


def filter_invalid_statistics(
    data_dict,
    log_file,
    max_nan_fraction=0.05,
    min_remaining_rows=10
):
    all_stats = set.intersection(
        *(set(df.columns) for df in data_dict.values() if df is not None)
    )

    removed_stats = []
    kept_stats = []
    log_lines = []

    for stat in sorted(all_stats):
        drop_stat = False
        stat_logs = []

        for sim, df in data_dict.items():
            if df is None or stat not in df.columns:
                continue

            total_rows = len(df)
            invalid_mask = df[stat].isna() | ~np.isfinite(df[stat])
            n_invalid = invalid_mask.sum()
            frac_invalid = n_invalid / total_rows if total_rows > 0 else 0

            if n_invalid == 0:
                continue

            stat_logs.append(
                f"[{sim}] {stat}: found {n_invalid}/{total_rows} invalid rows ({frac_invalid:.2%})"
            )

            # Entscheidung: Spalte dropen oder nur Zeilen
            if frac_invalid >= max_nan_fraction or (total_rows - n_invalid) < min_remaining_rows:
                drop_stat = True
                stat_logs.append(f"[{sim}] {stat}: exceeds thresholds → drop column")
                break

        if drop_stat:
            removed_stats.append(stat)
            for df in data_dict.values():
                if df is not None and stat in df.columns:
                    df.drop(columns=[stat], inplace=True)
            log_lines.extend(stat_logs)
        else:
            kept_stats.append(stat)
            for sim, df in data_dict.items():
                if df is not None and stat in df.columns:
                    before = len(df)
                    # nur die Zeilen mit NaN/Inf löschen
                    df.drop(index=df[~np.isfinite(df[stat]) | df[stat].isna()].index, inplace=True)
                    after = len(df)
                    if before != after:
                        log_lines.append(f"[{sim}] {stat}: kept column, dropped {before-after} rows")

    # --- Write log ---
    with open(log_file, "w") as f:
        f.write("=== NaN / Inf cleanup log ===\n\n")
        f.write("\n".join(log_lines))
        f.write("\n\n=== Removed statistics ===\n")
        f.write("\n".join(removed_stats))
        f.write("\n\n=== Kept statistics ===\n")
        f.write("\n".join(kept_stats))

    return data_dict, kept_stats


def get_features_by_simulator(csv_files, output_folder, selected_simulators):
    """Load CSVs per simulator and filter invalid statistics."""
    output_folder.mkdir(parents=True, exist_ok=True)
    log_file = output_folder / "removed_statistics.log"

    if not csv_files:
        print("No CSV files found in input folder.")
        return None

    data_dict = {sim: [] for sim in selected_simulators}

    for f in csv_files:
        sim = detect_simulator_from_yaml(input_file=f)
        if sim is None:
            print(f"Skipping unknown file: {f}")
            continue
        data_dict[sim].append(f)

    # Load CSVs one by one and concatenate
    for sim in data_dict:
        files = data_dict[sim]
        if files:
            dfs = [load_csv_float32(f) for f in files]
            data_dict[sim] = pd.concat(dfs, ignore_index=True)
        else:
            data_dict[sim] = None

    # Filter invalid statistics
    data_dict, kept_stats = filter_invalid_statistics(data_dict, log_file)
    if not kept_stats:
        print("No valid statistics to plot after filtering NaN/Inf. See log for details.")
        return None

    return data_dict


def plot_boxplots(csv_files, output_folder, selected_simulators):
    data_dict = get_features_by_simulator(csv_files, output_folder, selected_simulators)
    if not data_dict:
        return

    kept_stats = next(iter(data_dict.values())).columns.tolist()

    for stat in kept_stats:
        plt.figure(figsize=(6, 4))
        data_to_plot = []
        labels = []

        for sim_name, df in data_dict.items():
            if df is not None and stat in df.columns:
                values = df[stat].dropna()
                if len(values) > 0:
                    data_to_plot.append(values)
                    labels.append(sim_name)

        if len(data_to_plot) < 2 or sum(len(d) for d in data_to_plot) < 5:
            plt.close()
            continue

        plt.boxplot(data_to_plot, labels=labels)
        plt.title(stat)
        plt.tight_layout()
        out_file = output_folder / f"{stat}_boxplot.png"
        plt.savefig(out_file)
        plt.close()
        del data_to_plot


def plot_histograms(csv_files, output_folder, selected_simulators):
    data_dict = get_features_by_simulator(csv_files, output_folder, selected_simulators)
    if not data_dict:
        return

    kept_stats = next(iter(data_dict.values())).columns.tolist()

    for stat in kept_stats:
        plt.figure(figsize=(7, 5))
        plotted = False
        for sim_name, df in data_dict.items():
            if df is not None and stat in df.columns:
                values = df[stat].dropna()
                if len(values) > 0:
                    plt.hist(values, bins=50, alpha=0.3, label=sim_name, density=True)
                    plotted = True

        if not plotted:
            plt.close()
            continue

        plt.title(f"Distribution of {stat}")
        plt.xlabel(stat)
        plt.ylabel("Density")
        plt.legend(title="Simulator")
        plt.tight_layout()
        out_file = output_folder / f"{stat}_histogram.png"
        plt.savefig(out_file)
        plt.close()


def csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract top features from simulation results.")
    parser.add_argument("--input_folder", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("--simulators", type=lambda s: s.split(","), default=SIMULATORS)

    parser.add_argument(
        "--grouped_by",
        type=csv_list,
        default=["single"],
        help=(
            "Comma-separated list of grouping modes, e.g. "
            "'precision_applied,recombination_skipped,sample_size500'. "
            f"Base choices: {', '.join(VALID_GROUPING_MODES)} "
            "(sample_sizeN / seq_lenN allowed)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    selected_simulators = args.simulators
    grouped_by = args.grouped_by
    normalized_grouped_by = []
    sample_size_group = None
    seq_len_group = None

    for mode in grouped_by:
        if mode.startswith("sample_size"):
            m = re.search(r"\d+", mode)
            if not m:
                raise ValueError("sample_size mode needs a number")
            sample_size_group = int(m.group())
            normalized_grouped_by.append("sample_size")

        elif mode.startswith("seq_len"):
            m = re.search(r"\d+", mode)
            if not m:
                raise ValueError("seq_len mode needs a number")
            seq_len_group = int(m.group())
            normalized_grouped_by.append("seq_len")

        else:
            normalized_grouped_by.append(mode)

    grouped_by = set(normalized_grouped_by)
    output_folder.mkdir(parents=True, exist_ok=True)

    csv_my_mode = []

    if "single" in grouped_by or "overall" in grouped_by:
        csv_files = collect_csv_files(input_folder)
        plot_boxplots(csv_files, output_folder, selected_simulators)
        plot_histograms(csv_files, output_folder, selected_simulators)
        return

    # group CSV files based on YAML params
    for root, _, files in os.walk(input_folder):
        root_path = Path(root)
        combo_yamls = [f for f in files if f.endswith((".yaml", ".yml"))]
        if not combo_yamls:
            continue

        combo_yaml_path = root_path / combo_yamls[0]
        with open(combo_yaml_path, "r") as f:
            combo_params = yaml.safe_load(f)

        usePrecision = bool(combo_params.get("usePrecision"))
        recombination_rate = float(combo_params.get("recombination_rate") or 0)
        sample_size = int(combo_params["sample_structure"][0]) 
        seq_len = int(combo_params["sequence_length"])

        if (
            (("precision_applied" not in grouped_by) or usePrecision) and
            (("precision_skipped" not in grouped_by) or (not usePrecision)) and
            (("recombination_applied" not in grouped_by) or (recombination_rate > 0)) and
            (("recombination_skipped" not in grouped_by) or (recombination_rate == 0)) and
            (("sample_size" not in grouped_by) or (sample_size == sample_size_group and usePrecision)) and
            (("seq_len" not in grouped_by) or (seq_len == seq_len_group and usePrecision))
        ):
            csv_my_mode += collect_csv_files(root_path)

        

    if not csv_my_mode:
        print(f"No CSV files found for mode {grouped_by}")
        return

    plot_boxplots(csv_my_mode, output_folder, selected_simulators)


if __name__ == "__main__":
    main()