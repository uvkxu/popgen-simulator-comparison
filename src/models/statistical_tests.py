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


def comparison(
    tree_csv_files,
    allel_csv_files,
    output_file,
    simulator_a,
    simulator_b,
):
    results = []

    # --------------------------------------------------
    # TREE FEATURES
    # --------------------------------------------------
    tree_dfs = []

    for tree_csv in tree_csv_files:
        sim_name = detect_simulator_from_yaml(tree_csv)

        if sim_name not in [simulator_a, simulator_b]:
            continue

        df = pd.read_csv(tree_csv)
        df["simulator"] = sim_name

        tree_dfs.append(df)

    if tree_dfs:
        tree_data = pd.concat(tree_dfs, ignore_index=True)

        simA = tree_data[tree_data["simulator"] == simulator_a]
        simB = tree_data[tree_data["simulator"] == simulator_b]

        for feature in tree_data.columns:

            if feature == "simulator":
                continue

            if feature in FEATURES_TO_EXCLUDE:
                continue

            x = simA[feature].dropna()
            y = simB[feature].dropna()

            if len(x) == 0 or len(y) == 0:
                continue

            stat, p = mannwhitneyu(
                x,
                y,
                alternative="two-sided"
            )

            delta = (2 * stat) / (len(x) * len(y)) - 1

            results.append({
                "feature": f"tree::{feature}",
                "p_value": p,
                "effect_size": delta,
                "median_A": x.median(),
                "median_B": y.median(),
            })

    # --------------------------------------------------
    # ALLELE FEATURES
    # --------------------------------------------------
    allele_dfs = []

    for allele_csv in allel_csv_files:
        sim_name = detect_simulator_from_yaml(allele_csv)

        if sim_name not in [simulator_a, simulator_b]:
            continue

        df = pd.read_csv(allele_csv)
        df["simulator"] = sim_name

        allele_dfs.append(df)

    if allele_dfs:
        allele_data = pd.concat(allele_dfs, ignore_index=True)

        simA = allele_data[allele_data["simulator"] == simulator_a]
        simB = allele_data[allele_data["simulator"] == simulator_b]

        for feature in allele_data.columns:

            if feature == "simulator":
                continue

            if any(feature.startswith(excluded) for excluded in FEATURES_TO_EXCLUDE):
                continue

            x = simA[feature].dropna()
            y = simB[feature].dropna()

            if len(x) == 0 or len(y) == 0:
                continue

            stat, p = mannwhitneyu(
                x,
                y,
                alternative="two-sided"
            )

            delta = (2 * stat) / (len(x) * len(y)) - 1

            results.append({
                "feature": f"allele::{feature}",
                "p_value": p,
                "effect_size": delta,
                "median_A": x.median(),
                "median_B": y.median(),
            })

    if not results:
        raise ValueError(
            f"""
            No comparable features found for {simulator_a} vs {simulator_b}

            tree_dfs={len(tree_dfs)}
            allele_dfs={len(allele_dfs)}

            tree_simulators=
            {tree_data['simulator'].value_counts() if tree_dfs else 'NO TREE DATA'}

            allele_simulators=
            {allele_data['simulator'].value_counts() if allele_dfs else 'NO ALLELE DATA'}

            simulator_a={simulator_a}
            simulator_b={simulator_b}
        """
        )

    df = pd.DataFrame(results)

    df["p_corrected"] = multipletests(
        df["p_value"],
        method="fdr_bh"
    )[1]

    df = df.sort_values("p_corrected")

    output_file.parent.mkdir(parents=True, exist_ok=True)
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

    parser.add_argument(
        "--input_folder_tree",
        type=Path,
        required=True,
        help="Tree input folder",
    )
    parser.add_argument(
        "--input_folder_allel",
        type=Path,
        required=True,
        help="Allele input folder",
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

    tree_csv_files = get_csv_files_mode(
        input_folder_tree,
        grouped_by,
        seq_len_group,
        sample_size_group
    )

    if not tree_csv_files:
        raise ValueError(
            f"No tree files found in {input_folder_tree}"
        )

    allel_csv_files = get_csv_files_mode(
        input_folder_allel,
        grouped_by,
        seq_len_group,
        sample_size_group
    )

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