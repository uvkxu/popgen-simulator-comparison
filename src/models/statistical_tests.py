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
    VALID_GROUPING_MODES,
    format_feature_ranking_file,
    format_feature_ranking_barplot,
    format_classifier_importance_plot,
    XGB_STR,
)
from utils.io_helpers import collect_csv_files, detect_simulator_from_yaml

EXCLUDE_FEATURES = {
    "phylogenetic_div", "lap_max", "lap_fiedler", "lap_var", "lap_skew", "lap_mean",
        "lap_kurt", "lap_second_min", "lap_entropy", "lap_sum",
        "cut_weight", "conductance", "normalized_cut", "bottleneck_strength",
        "density_pos", "density_neg", "modularity", "fiedler_sd"
}


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

def collect_and_merge_tree_allele_features(
    tree_csv_files: list[Path],
    allele_csv_files: list[Path],
    simulators: list[str],
    suffix_tree: str = "_tree_features.csv",
    suffix_allele: str = "_allel_features.csv",
) -> list[pd.DataFrame]:
    """
    Merge tree and allele feature CSVs.
    Paths are assumed identical below `tree/` vs `allele/`.
    Feature names are NOT prefixed.
    """

    # --- build allele lookup: relative path below allele/ ---
    allele_lookup: dict[Path, Path] = {}

    for a in allele_csv_files:
        key = _relative_after_dir(a, "allele")
        allele_lookup[key] = a

    merged_dfs: list[pd.DataFrame] = []

    for tree_csv in tree_csv_files:
        tree_key = _relative_after_dir(tree_csv, "tree")

        allele_key = tree_key.with_name(
            tree_csv.name.replace(suffix_tree, suffix_allele)
        )

        if allele_key not in allele_lookup:
            raise FileNotFoundError(
                f"No matching allele CSV for:\n{tree_csv}"
            )

        allele_csv = allele_lookup[allele_key]

        df_tree = pd.read_csv(tree_csv)
        df_allele = pd.read_csv(allele_csv)

        if len(df_tree) != len(df_allele):
            raise ValueError(
                f"Row mismatch:\n{tree_csv}\n{allele_csv}"
            )

        # --- infer simulator from filename ---
        simulator_name = detect_simulator_from_yaml(
                input_file=tree_csv)

        if simulator_name not in simulators:
            continue

        simulator_col = pd.Series(
            [simulator_name] * len(df_tree),
            name="simulator"
        )

        tree_feats = df_tree.drop(columns=["simulator"], errors="ignore")
        allele_feats = df_allele.drop(columns=["simulator"], errors="ignore")

        df_merged = pd.concat(
            [simulator_col, tree_feats, allele_feats],
            axis=1
        )

        merged_dfs.append(df_merged)

    if not merged_dfs:
        raise ValueError("No CSV pairs were merged.")

    return merged_dfs


def comparison(
    tree_csv_files,
    allel_csv_files,
    output_file, 
    simulator_a, 
    simulator_b):
    results = []

    # --- Load and label data ---
    all_dfs =  collect_and_merge_tree_allele_features(
        tree_csv_files=tree_csv_files,
        allele_csv_files=allel_csv_files,
        simulators=[simulator_a, simulator_b]
    )
    data = pd.concat(all_dfs, ignore_index=True)
    del all_dfs

    MAX_NAN_FRAC = 0.05

    for col in list(data.columns):
        if col == "simulator":
            continue

        frac_nan = data[col].isna().mean()

        if frac_nan >= MAX_NAN_FRAC:
            print(
                f"Dropped feature '{col}' "
                f"({frac_nan:.2%} NaNs >= {MAX_NAN_FRAC:.2%})."
            )
            data = data.drop(columns=col)

    simA = data[data["simulator"] == simulator_a]
    simB = data[data["simulator"] == simulator_b]

    FEATURES_TO_EXCLUDE = [
        # --- stats_laplace (weighted = TRUE) ---
        "lap_max",
        "lap_fiedler",
        "lap_var",
        "lap_skew",
        "lap_mean",
        "lap_kurt",
        "lap_second_min",
        "lap_entropy",
        "lap_sum",
        "tree_type",
        "cut_edge_weight",
        "share_pos",
        "share_neg",
        "share_zero",
        "ratio_neg_pos",
        "dist_char",

        # --- stats_laplace (weighted = FALSE / unweighted) ---
        "lap_max_uw",
        "lap_fiedler_uw",
        "lap_var_uw",
        "lap_skew_uw",
        "lap_mean_uw",
        "lap_kurt_uw",
        "lap_second_min_uw",
        "lap_entropy_uw",
        "lap_sum_uw",
        "tree_type_uw",
        "laplace_spectrum_a_uw",
        "laplace_spectrum_p_uw",
        "laplace_spectrum_e_uw",
        "laplace_spectrum_g_uw",
        "min_laplace_uw",
        "max_laplace_uw",
        "min_adj_uw",
        "max_adj_uw",
        "mpd_uw",
        "psv_uw",
        "vpd_uw",
        "mntd_uw",
        "j_stat_uw",
        "wiener_uw",
        "max_closenessW_uw",
        "eigen_centralityW_uw",
        "treeness_uw"

        # --- compute_characteristic_features ---
        "cut_weight",
        "conductance",
        "normalized_cut",
        "bottleneck_strength",
        "density_pos",
        "density_neg",
        "modularity",
        "fiedler_sd",
    ]

    feature_names = [
        col for col in data.columns
        if col != "simulator" and col not in FEATURES_TO_EXCLUDE
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


def csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def get_csv_files_mode(input_folder, grouped_by, seq_len_group=None, sample_size_group=None):
    csv_my_mode = []

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
            sample_size = int(combo_params["sample_structure"][0])
            seq_len = int(combo_params["sequence_length"])

            if (
                (("precision_applied" not in grouped_by) or usePrecision)
                and (("precision_skipped" not in grouped_by) or (not usePrecision))
                and (("recombination_applied" not in grouped_by) or (recombination_rate > 0))
                and (("recombination_skipped" not in grouped_by) or (recombination_rate == 0))
                and (("sample_size" not in grouped_by) or (sample_size == sample_size_group and usePrecision))
                and (("seq_len" not in grouped_by) or (seq_len == seq_len_group and usePrecision))
            ):
                csv_my_mode += collect_csv_files(root_path)

    return csv_my_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Extract top features from simulation results.")

    parser.add_argument("--input_folder_tree", type=Path, required=True, help="Input folder containing result files")
    parser.add_argument("--input_folder_allel", type=Path, required=True, help="Input folder containing result files")
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
        help=(
            "Comma-separated list of grouping modes, e.g. "
            "'precision_applied,recombination_skipped,sample_size500'. "
            f"Base choices: {', '.join(VALID_GROUPING_MODES)} "
            "(sample_sizeN / seq_lenN allowed)."
        ),
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
    output_file = args.output_file
    simulator_group = args.simulators
    grouped_by = args.grouped_by

    normalized_grouped_by = []
    sample_size_group = None
    seq_len_group = None

    if len(simulator_group) != 2:
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
    tree_csv_files = get_csv_files_mode(input_folder_tree, grouped_by, seq_len_group, sample_size_group)
    allel_csv_files = get_csv_files_mode(input_folder_allel, grouped_by, seq_len_group, sample_size_group)

    assert len(tree_csv_files) == len(allel_csv_files)

    comparison(
        tree_csv_files,
        allel_csv_files,
        output_file, 
        simulator_a, 
        simulator_b
    )


if __name__ == "__main__":
    main()