from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import pandas as pd

import sys
import os
import yaml
import argparse
import re
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import (
    is_valid_grouping_mode,
    format_feature_ranking_file
)
from utils.io_helpers import collect_csv_files, detect_simulator_from_yaml


def _relative_after_dir(p: Path, anchor: str) -> Path:
    """
    Return path relative to the directory named `anchor`
    e.g. .../tree/configX/rep1/file.csv -> configX/rep1/file.csv
    """
    parts = p.parts
    if anchor not in parts:
        raise ValueError(f"'{anchor}' not found in path: {p}")

    idx = parts.index(anchor)
    return Path(*parts[idx + 1:])


def aggregate_tree_block(block: pd.DataFrame) -> dict:
    """
    Aggregate all tree features in one replicate block.
    """
    agg = {}
    for col in block.columns:
        agg[f"{col}_mean"] = block[col].mean()
        agg[f"{col}_std"] = block[col].std()
        agg[f"{col}_min"] = block[col].min()
        agg[f"{col}_max"] = block[col].max()
    agg["num_trees"] = len(block)
    return agg


def count_trees_per_replicate(cleaned_output):
    counts = []
    current = 0

    with open(cleaned_output) as f:
        for line in f:
            line = line.strip()

            if is_tree_line(line, cleaned_output):
                current += 1
            elif current > 0:
                counts.append(current)
                current = 0

    if current > 0:
        counts.append(current)

    return counts


def is_tree_line(line: str, file_path: str | Path | None = None) -> bool:
    line = line.strip()

    return (
        "(" in line
        and ";" in line
        and not line.startswith("//")
        and not line.lower().startswith("segsites")
        and not line.lower().startswith("positions")
    )


def collect_and_merge_tree_allele_features(
    tree_csv_files: list[Path],
    allele_csv_files: list[Path],
    simulators: list[str],
    cleaned_csv_files: list[Path] | None = None,
    aggregate_trees: bool = False,
    suffix_tree: str = "_tree_features.csv",
    suffix_allele: str = "_allel_features.csv",
    suffix_cleaned: str = "_cleaned_output.txt",
) -> list[pd.DataFrame]:
    """
    Merge tree and allele feature CSVs.

    If aggregate_trees=True, tree features are aggregated at replicate
    level using the number of marginal trees reconstructed from the
    cleaned simulator output.

    If aggregate_trees=False, tree and allele features are merged
    directly without aggregation.
    """

    if aggregate_trees and cleaned_csv_files is None:
        raise ValueError(
            "cleaned_csv_files must be provided when aggregate_trees=True."
        )

    # --- Build allele lookup ---
    allele_lookup: dict[Path, Path] = {}

    for a in allele_csv_files:
        rel = _relative_after_dir(a, "allele")

        if aggregate_trees:
            key = rel.with_name(rel.name.replace(suffix_allele, ""))
        else:
            key = rel

        allele_lookup[key] = a

    # --- Build cleaned output lookup if aggregation is needed ---
    cleaned_lookup: dict[Path, Path] = {}

    if aggregate_trees:
        for r in cleaned_csv_files:
            rel = _relative_after_dir(r, "cleaned")
            key = rel.with_name(rel.name.replace(suffix_cleaned, ""))
            cleaned_lookup[key] = r

    merged_dfs: list[pd.DataFrame] = []

    for tree_csv in tree_csv_files:

        # --- Identify simulator ---
        sim_name = detect_simulator_from_yaml(tree_csv)

        if sim_name not in simulators:
            continue

        # --- Match corresponding allele file ---
        rel_tree = _relative_after_dir(tree_csv, "tree")

        if aggregate_trees:
            key = rel_tree.with_name(
                rel_tree.name.replace(suffix_tree, "")
            )
        else:
            key = rel_tree.with_name(
                tree_csv.name.replace(suffix_tree, suffix_allele)
            )

        if key not in allele_lookup:
            raise FileNotFoundError(
                f"No matching allele CSV for:\n{tree_csv}\n"
                f"Looking for key: {key}"
            )

        # --- Read feature files ---
        df_tree = pd.read_csv(tree_csv)
        df_allele = pd.read_csv(allele_lookup[key])

        # --- Aggregate tree features if required ---
        if aggregate_trees:

            if key not in cleaned_lookup:
                raise FileNotFoundError(
                    f"No matching cleaned output for:\n{tree_csv}\n"
                    f"Looking for key: {key}"
                )

            cleaned_file = cleaned_lookup[key]

            counts = count_trees_per_replicate(cleaned_file)

            if sum(counts) != len(df_tree):
                raise ValueError(
                    f"Tree count mismatch:\n"
                    f"sum(counts) = {sum(counts)}\n"
                    f"tree rows = {len(df_tree)}\n"
                    f"Tree file: {tree_csv}\n"
                    f"Cleaned file: {cleaned_file}"
                )

            # Aggregate marginal trees per replicate
            offset = 0
            aggregated = []

            for k in counts:
                block = df_tree.iloc[offset:offset + k]
                block = block.drop(columns=["simulator"], errors="ignore")

                aggregated.append(aggregate_tree_block(block))
                offset += k

            df_tree = pd.DataFrame(aggregated)

            # Save aggregated tree features
            parts = list(tree_csv.parts)
            tree_idx = parts.index("tree")
            parts[tree_idx] = "tree_block"

            block_path = Path(*parts).with_name(
                tree_csv.name.replace(
                    suffix_tree,
                    "_tree_block_features.csv"
                )
            )

            block_path.parent.mkdir(parents=True, exist_ok=True)
            df_tree.to_csv(block_path, index=False)

        else:
            df_tree = df_tree.drop(
                columns=["simulator"],
                errors="ignore"
            )

        # --- Remove simulator column from allele features ---
        df_allele = df_allele.drop(
            columns=["simulator"],
            errors="ignore"
        )

        # --- Check row correspondence ---
        if len(df_tree) != len(df_allele):
            raise ValueError(
                f"Row mismatch between tree and allele features:\n"
                f"Tree file: {tree_csv}\n"
                f"Allele file: {allele_lookup[key]}\n"
                f"Tree rows: {len(df_tree)}\n"
                f"Allele rows: {len(df_allele)}"
            )

        # --- Merge features ---
        sim_col = pd.Series(
            [sim_name] * len(df_tree),
            name="simulator"
        )

        df_merged = pd.concat(
            [sim_col, df_tree, df_allele],
            axis=1
        )

        merged_dfs.append(df_merged)

    if not merged_dfs:
        raise ValueError(
            "No CSV pairs were merged.\n"
            f"Number of tree CSVs: {len(tree_csv_files)}\n"
            f"Number of allele CSVs: {len(allele_csv_files)}\n"
            f"Simulators: {simulators}"
        )

    return merged_dfs


def comparison(
    tree_csv_files,
    allel_csv_files,
    cleaned_csv_files,
    aggregate_trees,
    output_file, 
    simulator_a, 
    simulator_b):
    results = []

    # --- Load and label data ---
    all_dfs = collect_and_merge_tree_allele_features(
        tree_csv_files=tree_csv_files,
        allele_csv_files=allel_csv_files,
        cleaned_csv_files=cleaned_csv_files,
        aggregate_trees=aggregate_trees,
        simulators=[simulator_a, simulator_b]
    )
    data = pd.concat(all_dfs, ignore_index=True)
    del all_dfs

    MAX_NAN_FRAC = 0.05

    log_path = output_file.parent / f"nan_cleanup_{output_file.stem}.log"
    log_lines: list[str] = []

    n_rows_initial = len(data)

    for col in list(data.columns):
        if col == "simulator":
            continue

        n_nan = int(data[col].isna().sum())
        frac_nan = n_nan / n_rows_initial if n_rows_initial > 0 else 0

        if frac_nan >= MAX_NAN_FRAC:
            log_lines.append(
                f"Dropped feature '{col}' "
                f"({n_nan} NaNs, {frac_nan:.2%} >= {MAX_NAN_FRAC:.2%})."
            )
            data = data.drop(columns=col)

    if log_lines:
        with open(log_path, "w", encoding="utf-8") as f:
            for line in log_lines:
                f.write(line + "\n")

    simA = data[data["simulator"] == simulator_a]
    simB = data[data["simulator"] == simulator_b]

    feature_names = [
        col for col in data.columns
        if col != "simulator"
    ]

    for feature in feature_names:

        x = simA[feature].dropna()
        y = simB[feature].dropna()

        stat, p = mannwhitneyu(
            x,
            y,
            alternative="two-sided"
        )

        delta = (2 * stat) / (len(x) * len(y)) - 1

        results.append({
            "feature": feature,
            "p_value": p,
            "effect_size": delta,
            "median_A": x.median(),
            "median_B": y.median(),
        })

    df = pd.DataFrame(results)

    # multiple testing correction
    df["p_corrected"] = multipletests(
        df["p_value"],
        method="fdr_bh"
    )[1]

    df = df.sort_values("p_corrected")

    df.to_csv(output_file, index=False)


def csv_list(value):
    modes = [item.strip() for item in value.split(",")]

    invalid_modes = [
        mode for mode in modes
        if not is_valid_grouping_mode(mode)
    ]

    if invalid_modes:
        raise argparse.ArgumentTypeError(
            f"Invalid grouping mode(s): {', '.join(invalid_modes)}"
        )

    return modes


def get_csv_files_mode(input_folder, grouped_by, seq_len_group=None, sample_size_group=None, simulator_group=None, file_type="csv"):
    csv_my_mode = []
    has_recombination = False

    for root, _, files in os.walk(input_folder):
        root_path = Path(root)

        combo_yamls = [f for f in files if f.endswith(".yaml") or f.endswith(".yml")]
        if not combo_yamls:
            continue

        combo_yaml_path = root_path / combo_yamls[0]
        with open(combo_yaml_path, "r") as f:
            combo_params = yaml.safe_load(f)

        recombination_rate = float(combo_params.get("recombination_rate") or 0)
        has_recombination = (recombination_rate != 0)
        sample_size = int(combo_params["sample_structure"][0])
        seq_len = int(combo_params["sequence_length"])

        if ((("sample_size" not in grouped_by) or (sample_size == sample_size_group))
            and (("seq_len" not in grouped_by) or (seq_len == seq_len_group))
        ):
            if "cleaned" in input_folder.parts:
                csv_files = list(root_path.rglob("*_output.txt"))
            else:
                csv_files = collect_csv_files(root_path)

            if simulator_group:
                    csv_files = [
                        csv_file
                        for csv_file in csv_files
                        if detect_simulator_from_yaml(csv_file) in simulator_group
                    ]

            csv_my_mode += csv_files

    return has_recombination, csv_my_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Extract top features from simulation results.")

    parser.add_argument("--input_folder_tree", type=Path, required=True, help="Input folder containing result files")
    parser.add_argument("--input_folder_allel", type=Path, required=True, help="Input folder containing result files")
    parser.add_argument(
        "--input_folder_cleaned",
        type=Path,
        required=False,
        help="Cleaned input folder",
    )
    parser.add_argument("--output_file", type=Path, required=True, help="Output file path")

    parser.add_argument(
        "--simulators",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated list of simulators (e.g. sim1,sim2)",
    )
    parser.add_argument(
    "--grouped_by",
        type=csv_list,
        default=["single"],
        help="Comma-separated list of grouping modes."
    )

    return parser.parse_args()


def assert_non_empty(csv_files, mode):
    if not csv_files:
        raise ValueError(
            f"No CSV files found for mode '{mode}'. "
            "This indicates an inconsistent input directory or configuration."
        )


def main():
    args = parse_args()

    input_folder_tree = args.input_folder_tree
    input_folder_allel = args.input_folder_allel
    input_folder_cleaned = args.input_folder_cleaned
    output_file = args.output_file
    simulator_group = args.simulators
    grouped_by = args.grouped_by

    normalized_grouped_by = []
    sample_size_group = None
    seq_len_group = None

    if not simulator_group or len(simulator_group) != 2:
        raise ValueError("Exactly two simulators must be specified for comparison.")
    
    simulator_a, simulator_b = simulator_group[0], simulator_group[1]

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

    output_file.parent.mkdir(parents=True, exist_ok=True)
    has_recombination, tree_csv_files = get_csv_files_mode(input_folder_tree, grouped_by, seq_len_group, sample_size_group, simulator_group)
    _, allel_csv_files = get_csv_files_mode(input_folder_allel, grouped_by, seq_len_group, sample_size_group, simulator_group)
    _, cleaned_csv_files = None if not input_folder_cleaned else get_csv_files_mode(input_folder_cleaned, grouped_by, seq_len_group, sample_size_group, simulator_group, "txt")

    if len(tree_csv_files) != len(allel_csv_files):
        tree_keys = {
            _relative_after_dir(p, "tree").with_name(
                p.name.replace("_tree_features.csv", "_allel_features.csv")
            )
            for p in tree_csv_files
        }

        allele_lookup = {
            _relative_after_dir(p, "allele"): p
            for p in allel_csv_files
        }

        extra_allele_keys = allele_lookup.keys() - tree_keys

        extra_allele_files = [
            str(allele_lookup[key])
            for key in sorted(extra_allele_keys)
        ]

        raise AssertionError(
            f"Tree/allele file count mismatch: "
            f"tree={len(tree_csv_files)}, allele={len(allel_csv_files)}\n"
            f"Allele files without matching tree file:\n"
            + "\n".join(extra_allele_files)
        )
    
    if has_recombination:
        assert len(cleaned_csv_files) != 0
        assert len(tree_csv_files) == len(cleaned_csv_files)

    comparison(
        tree_csv_files,
        allel_csv_files,
        cleaned_csv_files,
        has_recombination,
        output_file, 
        simulator_a, 
        simulator_b
    )


if __name__ == "__main__":
    main()