from pathlib import Path
import json
import yaml
import os
from glob import glob
import itertools
from itertools import combinations
import pandas as pd
import math
import re
import re

from project_paths import *
from simulators import *

'''
ROOT_DIR = Path(".")
BIN_DIR = ROOT_DIR / "bin"
CONFIG_DIR = ROOT_DIR / "config"

# Directories for parameters and demographic models
PARAMETERS_DIR = CONFIG_DIR / "parameter_setting"
DEM_MODELS_DIR = PARAMETERS_DIR / "demographic_models"
VARIABLE_FILE_PATH = PARAMETERS_DIR / "simulation_parameters.yaml"

# Directories for data (simulation output, output after standardization and feature extraction)
DATA_DIR = ROOT_DIR / "data" #Path("/hits/fast/cme/kuchve/without_recombination/data")
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "cleaned"
FEATURES_DIR = DATA_DIR / "features"
TREE_FEATURES_DIR = FEATURES_DIR / "tree"
ALLEL_FEATURES_DIR = FEATURES_DIR / "allele"


# Directories for results after evaluation (classification and statistical tests)
RESULTS_DIR = ROOT_DIR / "results" #Path("/hits/fast/cme/kuchve/without_recombination/results_final")
TOPFEATURES_RESULTS_DIR = RESULTS_DIR/ "top_features"
STATISTICS_RESULTS_DIR = RESULTS_DIR / "statistics"

ALLEL_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "allele"
ALLELFEATURES_PLOTS_DIR = PLOTS_RESULTS_DIR/ "allele"
TREE_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "tree"
TREEFEATURES_PLOTS_DIR = PLOTS_RESULTS_DIR/ "tree"
TOGETHER_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "together"


# Simulatiors paths
DEFAULT_MS_PATH = BIN_DIR / "ms"
DEFAULT_MSMS_PATH = ROOT_DIR/ "lib"/ "msms.jar"
DEFAULT_MS_MOD_PATH = BIN_DIR / "ms-mod.exe"
DEFAULT_SLIM_PATH = BIN_DIR / "slim"




# Was zur Hölle ist das?
TIPTOP_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "top_feature_run"

# MERGED 
MERGES_COLLECT_DIR = RESULTS_DIR / "merged_stats"
COLLECT_DIR = RESULTS_DIR / "collected"
ALLELFEATURES_COLLECT_DIR = COLLECT_DIR / "allele"
# Wofür brauche ich das?
TREEFEATURES_COLLECT_DIR = COLLECT_DIR / "tree"



COMBO_PARAMETERS_YAML = "combo_parameters.yaml"
'''

# Load workflow_config 
with WORKFLOW_CONFIG_FILE.open(encoding="utf-8") as f:
    workflow_config = yaml.safe_load(f)

# Get target simulators
TARGET_SIMULATORS = workflow_config["target_simulators"]         

EXTERNAL_REPS = workflow_config["replicates"]["external"]
INTERNAL_REPS = workflow_config["replicates"]["internal"]
PRECISION = workflow_config["decimal_precision"]
MAX_OVERALL_SAMPLES = workflow_config["max_overall_samples"]


with (FILE_FORMATS).open() as f:
    file_formats = yaml.safe_load(f)

COMBO_PARAMETERS_YAML = file_formats["combo_parameters"]


all_configs = [os.path.splitext(os.path.basename(f))[0] for f in glob(f"{DEM_MODELS_DIR}/*.yaml")]


# Computation of the GROUPING MODES for the grouping for computation of top features
# TODO: VALID_GROUPING_MODES anpassen
VALID_GROUPING_MODES = [
    "recombination_applied", "recombination_skipped",
    "overall", "single", 
    "sample_size_4", "sample_size_8", "sample_size_16",
    "sample_size_32", "sample_size_64", "sample_size_128",
    "seq_len_1000", "seq_len_10000", "seq_len_100000", "seq_len_1000000"
]

with open(VARIABLE_FILE_PATH, "r", encoding="utf-8") as f:
    variables = yaml.safe_load(f)

GROUPING_MODES_TREE = [
    f"sample_size_{sample_size}"
    for sample_size in variables["sample_sizes"]
]

GROUPING_MODES_ALLEL = [
    f"seq_len_{seq_len}"
    for seq_len in variables["sequence_lengths"]
]



def get_num_sampled_populations(config_path):
    num_pops = 0
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    for pop in config.get("demes"):
        epochs_time = {epoch.get("end_time", 0) for epoch in pop.get("epochs")}
        if any(t == 0 for t in epochs_time):
            num_pops += 1
    return num_pops


def get_combinations(config_path, variable_file):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vars_dict = parse_variable_file(variable_file)

    if get_num_sampled_populations(config_path) < 2:
        vars_dict.pop("population_sizes", None)

    keys = list(vars_dict.keys())
    combos = itertools.product(*(vars_dict[key] for key in keys))

    return keys, combos


def parse_variable_file(filepath):
    """Read simulation parameters from a YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        variables = yaml.safe_load(f)

    if not isinstance(variables, dict):
        raise ValueError(
            f"Expected a dictionary in parameter file: {filepath}"
        )

    return variables


def format_combination_as_folder_name(combo):
    return "_".join(f"{k}-{v}" for k, v in combo.items())

combinations_by_config = {}
tree_modes_by_config = {}
combo_folder_to_json = {}
all_combo_folders = set()

for config_name in all_configs:
    config_file = Path(DEM_MODELS_DIR) / f"{config_name}.yaml"
    keys, all_combos_by_config = get_combinations(config_file, VARIABLE_FILE_PATH)
    num_populations = get_num_sampled_populations(config_file)

    tree_modes_by_config[config_name] = []
    
    combos = []
    for values in all_combos_by_config:
        combo = dict(zip(keys, values))

        # --- Filter: remove combos with usePrecision=True AND recombination_rate != 0 ---
        if not bool(combo.get("usePrecision")) and float(combo.get("recombination_rates", 0)) > 0.0:
            continue
        
        if num_populations * combo.get("sample_sizes") > MAX_OVERALL_SAMPLES:
            continue

        possible_mode = f"sample_size_{int(combo.get('sample_sizes'))}"
        if possible_mode in GROUPING_MODES_TREE:
            tree_modes_by_config[config_name].append(possible_mode)

        possible_mode = f"sample_size_{int(combo.get('sample_sizes'))}"
        if possible_mode in GROUPING_MODES_TREE:
            tree_modes_by_config[config_name].append(possible_mode)

        combo_folder = format_combination_as_folder_name(combo)
        all_combo_folders.add(combo_folder)

        combos.append(combo_folder)
        combo_folder_to_json[combo_folder] = json.dumps(combo)
    
    combinations_by_config[config_name] = combos

    for mode in GROUPING_MODES_TREE:
        if not "sample_size" in mode:
            tree_modes_by_config[config_name].append(mode)


# Generate all feature file names for each simulator group
# and create a mapping from feature file to the corresponding simulator group

def format_feature_ranking_file(group):
    return file_formats["feature_ranking_with_summary"].format(
        group="_".join(group)
    )

def get_evaluation_input(config, mode, feature_type, group=None, combo=None):
    result = []
    if type(group) is str:
        group = group.split("_")
    elif group is None:
        group = TARGET_SIMULATORS

    grouped_by = mode.split(",")
    normalized_grouped_by = []
    sample_size_group = None
    seq_len_group = None

    for m in grouped_by:
        if m.startswith("sample_size"):
            para = re.search(r"\d+", m)
            if not para:
                raise ValueError("sample_size mode needs a number")
            sample_size_group = int(para.group())
            normalized_grouped_by.append("sample_size")

        elif m.startswith("seq_len"):
            para = re.search(r"\d+", m)
            if not para:
                raise ValueError("seq_len mode needs a number")
            seq_len_group = int(para.group())
            normalized_grouped_by.append("seq_len")

        else:
            normalized_grouped_by.append(m)

    grouped_by = set(normalized_grouped_by)
    
    if feature_type == "tree":
        root_folder = TREE_FEATURES_DIR
        format_file = "tree_features"
    elif feature_type == "allel":
        root_folder = ALLEL_FEATURES_DIR
        format_file = "allel_features"
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")

    if "single" in grouped_by:
        if combo is None:
            raise ValueError("Mode 'single' is not supported for top features input generation without combo.")

        return [
            f"{root_folder}/{config}/{combo}/rep{i+1}/{file_formats[format_file].format(SIM=sim)}"
            for sim in group
            for i in range(EXTERNAL_REPS)
        ]

    for combo_folder in combinations_by_config[config]:
        combo_string = combo_folder_to_json[combo_folder]
        combo = json.loads(combo_string)
        rate = float(combo.get("recombination_rates"))
        sample_size = int(combo.get("sample_sizes")) 
        sample_size = int(combo.get("sample_sizes")) 
        recombination_rate = int(combo.get("recombination_rates", 0)) 
        seq_len = int(combo.get("sequence_lengths"))
        usePrecision = bool(combo.get("usePrecision", True))

        if (
            (("recombination_applied" not in grouped_by) or (recombination_rate > 0)) and
            (("recombination_skipped" not in grouped_by) or (recombination_rate == 0)) and
            (("sample_size" not in grouped_by) or (sample_size == sample_size_group)) and
            (("seq_len" not in grouped_by) or (seq_len == seq_len_group))
        ):
            result.append(combo_folder)
    
    return [
        f"{root_folder}/{config}/{combo}/rep{i+1}/{file_formats[format_file].format(SIM=sim)}"
        for combo in result
        for sim in group
        for i in range(EXTERNAL_REPS)
    ] + [
        f"{root_folder}/{config}/{combo}/{COMBO_PARAMETERS_YAML}"
        for combo in result
    ]


def get_topfeatures_by_config(config, sim_group, feature_type):
    if feature_type == "tree":
        root_folder = TREE_TOPFEATURES_DIR
        modes = tree_modes_by_config[config]
    elif feature_type == "allel":
        root_folder = ALLEL_TOPFEATURES_DIR
        modes = GROUPING_MODES_ALLEL
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")

    if type(sim_group) is str:
        sim_group = sim_group.split("_")


    return [
        f"{root_folder}/{config}/{mode}/{format_feature_ranking_file(sim_group)}"
        for mode in modes
    ]

combinations_mode = {}
for config in all_configs:
    combinations_mode[config] = [
        f"{tree_mode},{grouping_mode}"
        for tree_mode in tree_modes_by_config[config]
        for grouping_mode in GROUPING_MODES_ALLEL
    ]

def get_topfeatures_by_config(config, sim_group, feature_type):
    if feature_type == "tree":
        root_folder = TREE_TOPFEATURES_DIR
        modes = tree_modes_by_config[config]
    elif feature_type == "allel":
        root_folder = ALLEL_TOPFEATURES_DIR
        modes = GROUPING_MODES_ALLEL
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")

    if type(sim_group) is str:
        sim_group = sim_group.split("_")


    return [
        f"{root_folder}/{config}/{mode}/{format_feature_ranking_file(sim_group)}"
        for mode in modes
    ]

combinations_mode = {}
for config in all_configs:
    combinations_mode[config] = [
        f"{tree_mode},{grouping_mode}"
        for tree_mode in tree_modes_by_config[config]
        for grouping_mode in GROUPING_MODES_ALLEL
    ]

#----------------------------------
# Zwischenstadien für snakemake
#----------------------------------

RAW_INPUTS = [
    f"{RAW_DATA_DIR}/{config}/{combo}/rep{i+1}/{file_formats['raw'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in TARGET_SIMULATORS
    for i in range(EXTERNAL_REPS)
]

ALL_COMBO_YAML = [
    f"{RAW_DATA_DIR}/{config}/{combo}/{COMBO_PARAMETERS_YAML}"
    for config in all_configs
    for combo in combinations_by_config[config]
]

CLEAN_INPUTS = [
    f"{CLEAN_DIR}/{config}/{combo}/rep{i+1}/{file_formats['cleaned'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in TARGET_SIMULATORS
    for i in range(EXTERNAL_REPS)
]

ALL_FEATURES = [
    f"{TREE_FEATURES_DIR}/{config}/{combo}/rep{i+1}/{file_formats['tree_features'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in TARGET_SIMULATORS
    for i in range(EXTERNAL_REPS)
] + [
    f"{ALLEL_FEATURES_DIR}/{config}/{combo}/rep{i+1}/{file_formats['allel_features'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in TARGET_SIMULATORS
    for i in range(EXTERNAL_REPS)
]

ALL_SLIM_FEATURES = [
    f"{TREE_FEATURES_DIR}/{config}/{combo}/rep{i+1}/{file_formats['tree_features'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in ["slim"]
    for i in range(EXTERNAL_REPS)
] + [
    f"{ALLEL_FEATURES_DIR}/{config}/{combo}/rep{i+1}/{file_formats['allel_features'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in ["slim"]
    for i in range(EXTERNAL_REPS)
]

CREATE_COMBO_PARAMETERS_YAML = [
    f"{TREE_FEATURES_DIR}/{config}/{combo}/{COMBO_PARAMETERS_YAML}"
    for config in all_configs
    for combo in combinations_by_config[config]
] + [
    f"{ALLEL_FEATURES_DIR}/{config}/{combo}/{COMBO_PARAMETERS_YAML}"
    for config in all_configs
    for combo in combinations_by_config[config]
]

#-----------------------------
#Final results for snakemake
#-----------------------------

simulator_pairs = list(combinations(TARGET_SIMULATORS, 2))
simulator_combination_all = tuple(TARGET_SIMULATORS)
simulator_combinations = simulator_pairs + [simulator_combination_all]


ALL_TREE_TOPFEATURES = [
    f"{TREE_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_file(simulator_combination_all)}"
    for config in all_configs
    for mode in combinations_mode[config]
] 

ALL_ALLEL_TOPFEATURES = [
    f"{ALLEL_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_file(simulator_combination_all)}"
    for config in all_configs
    for mode in combinations_mode[config]
] 

ALL_TOGETHER_TOPFEATURES = [
    f"{TOGETHER_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_file(sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for sim_group in simulator_pairs
]

STATS_RESULTS = [
    f"{STATISTICS_RESULTS_DIR}/{config}/{mode}/{group_string}_feature_comparison.csv"
    for config in all_configs
    for mode in combinations_mode[config]
    for group_string in [
        "_".join(group) for group in simulator_pairs
    ]
]

COMPLETE_OUTPUTS = ALL_TREE_TOPFEATURES + ALL_ALLEL_TOPFEATURES + ALL_TOGETHER_TOPFEATURES + STATS_RESULTS


#-------------------------
# Optuna Search Space
#-------------------------

def format_optuna_report_file(group):
    return file_formats["optuna_report_file"].format(
        group="_".join(group)
    )


ALL_TREE_OPTUNA = [
    f"{TREE_TOPFEATURES_DIR}/{config}/{mode}/{format_optuna_report_file(sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for sim_group in simulator_combinations
]

ALL_ALLEL_OPTUNA = [
    f"{ALLEL_TOPFEATURES_DIR}/{config}/{mode}/{format_optuna_report_file(sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for sim_group in simulator_combinations
]

#-----------------------------------
#COmpute stats for merged positions in coalescent simulators
#-----------------------------------
#TODO: check if also for slim

simulators_merged_stats = [
    simulator
    for simulator in TARGET_SIMULATORS
    if simulator in [MS, MSMS, MSPMS]
]

def get_merged_stats_overall_input(config, simulator):
    return [
    f"{MERGES_COLLECT_DIR}/{config}/{combo}/rep{i + 1}/{simulator}_merge_stats.csv"
    for combo in combinations_by_config[config]
    for i in range(EXTERNAL_REPS)
]

MERGED_STATS = [
    f"{MERGES_COLLECT_DIR}/{config}/{simulator}_merged_positions_overall.csv"
    for config in all_configs
    for simulator in simulators_merged_stats
] + [
    f"{MERGES_COLLECT_DIR}/{config}/{simulator}_merged_positions_sample_size.csv"
    for config in all_configs
    for simulator in simulators_merged_stats
]



#---------------------------------------------
# Estimation of cache
#---------------------------------------------


def estimate_mem_mb(config, combo, func, simulator=None):
    """
    Estimate memory (in MB) needed for different simulation/stat steps.
    """
    if isinstance(combo, dict):
        combo_json = combo
    else:
        combo_json = json.loads(combo_folder_to_json[combo])

    if isinstance(config, Path):
        config_file = config
    else:
        config = Path(config)

        # Case 1: full path already given
        if config.exists():
            config_file = config

        # Case 2: only basename (with or without .yaml)
        else:
            if config.suffix != ".yaml":
                config = config.with_suffix(".yaml")
            config_file = Path(DEM_MODELS_DIR) / config

    # optional: safety check
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    seq_len = int(combo_json.get("sequence_lengths"))
    samps = int(combo_json.get("sample_sizes"))
    n_haps = samps * get_num_sampled_populations(config_file)

    # ==============================
    #  calc_allelstatistics (very heavy)
    # ==============================
    if func == "calc_allelstatistics":
        # ~1 byte/base/hap × overhead
        base_mb = n_haps * seq_len / (1024**2)
        mem_mb = int(base_mb * 2.0)          # conservative
        return max(500, min(mem_mb, 70_000)) # at least 500 MB
    elif func == "simulate":

        sim = simulator.lower()

        # ==============================
        #  ms / classical coalescent
        # ==============================
        if sim == MS:
            # ms is text based; memory ~ output size
            # assume 0.2 byte per site×hap
            return min(int(n_haps * seq_len * 0.2 / 1e6), 10_000)

        # ==============================
        #  msprime (tree sequences)
        # ==============================
        if sim == MSPRIME:
            # msprime memory is modest even for long seqs
            # if no recombination, very cheap
            base_mb = n_haps * seq_len * 0.1 / 1e6
            # slight extra for recombination if present
            rho = float(combo_json.get("recombination_rates", 0))
            if rho > 0:
                base_mb *= (1.0 + (n_haps * rho)**0.5)
            return max(50, min(int(base_mb), 5000))

        # ==============================
        #  msms (Java)
        # ==============================
        if sim == MSMS:
            recomb_rate = float(combo_json.get("recombination_rates", 0))
            mu = float(combo_json.get("mutation_rates", 1e-8))

            Ne = combo_json.get("population_sizes")
            if Ne is None:
                with open(config_file, "r") as f:
                    config_yaml = yaml.safe_load(f)
                Ne = config_yaml.get("population_size")
            if Ne is None:
                raise ValueError("Ne required for msms")
            Ne = int(Ne)

            theta = 4 * Ne * mu * seq_len
            Hn = math.log(n_haps) + 0.577
            S = theta * Hn

            # Mutation memory
            site_mb = (S * (120 + 8 * n_haps)) / (1024**2)

            # Recombination memory
            if recomb_rate > 0:
                rho = 4 * Ne * recomb_rate * seq_len
                arg_mb = 0.006 * n_haps * rho * math.log(rho + 10)
            else:
                arg_mb = 0.0

            mem_mb = site_mb + arg_mb

            reps = int(combo_json.get("nreps", INTERNAL_REPS))

            # ====== Ultra-conservative peak multipliers ======
            mem_mb *= 10.0            # Mutation/output peak
            mem_mb *= 2.5             # Java GC + fragmentation
            mem_mb *= (1 + 0.5*reps)  # Multiple replicates

            # Absolute floor and ceiling
            return max(8192, min(int(mem_mb), 1_000_000))



    return 100


# global cache
size_all_csv = {}


# global cache
size_all_csv = {}


def estimate_mem_mb_classification(
    input_folder,
    config,
    group,
    mode,
    dtype_bytes=4,          # float32
    pandas_factor=2.0,
    split_factor=1.5,
    model_factor=2.0,
    safety_factor=1.3,
    min_mem_mb=512
):
    """
    Schnelle Approximation des RAM-Bedarfs (MB) für Classification:
    - scannt alle CSVs unter input_folder nur EINMAL (caching in size_all_csv)
    - skaliert dann pro Job über group-Anteil + mode-Anteil + (milde) Größe von sample_size/seq_len

    Erwartet, dass SIMULATORS, tree_modes_by_config, GROUPING_MODES_ALLEL,
    sowie ALLEL_FEATURES_DIR / TREE_FEATURES_DIR global existieren.
    """

    """
    Schnelle Approximation des RAM-Bedarfs (MB) für Classification:
    - scannt alle CSVs unter input_folder nur EINMAL (caching in size_all_csv)
    - skaliert dann pro Job über group-Anteil + mode-Anteil + (milde) Größe von sample_size/seq_len

    Erwartet, dass SIMULATORS, tree_modes_by_config, GROUPING_MODES_ALLEL,
    sowie ALLEL_FEATURES_DIR / TREE_FEATURES_DIR global existieren.
    """

    input_folder = Path(input_folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"Input directory {input_folder} does not exist.")

    # --- 1) Basisschätzung für "alles unter input_folder" (einmalig, gecacht) ---
    if input_folder not in size_all_csv:
        csv_files = list(input_folder.rglob("*.csv"))
        if not csv_files:
            size_all_csv[input_folder] = min_mem_mb
        else:
            total_rows = 0
            n_cols = None

            for csv_file in csv_files:
                # Zeilen zählen (ohne Header)
                try:
                    with csv_file.open("r") as f:
                        rows = sum(1 for _ in f) - 1
                except OSError:
                    continue

                rows = max(rows, 0)
                total_rows += rows

                # Spalten nur einmal aus einem nicht-leeren File bestimmen
                if n_cols is None and rows > 0:
                    try:
                        df_head = pd.read_csv(csv_file, nrows=1)
                        n_cols = len(df_head.columns)  # dep var NICHT abziehen (wie gewünscht)
                    except Exception:
                        n_cols = 0

            if total_rows <= 0 or not n_cols or n_cols <= 0:
                size_all_csv[input_folder] = min_mem_mb
            else:
                base_bytes = total_rows * n_cols * dtype_bytes
                total_bytes = (
                    base_bytes
                    * pandas_factor
                    * split_factor
                    * model_factor
                    * safety_factor
                )
                mem_mb = math.ceil(total_bytes / 1024**2)
                size_all_csv[input_folder] = max(mem_mb, min_mem_mb)

    # --- 1) Basisschätzung für "alles unter input_folder" (einmalig, gecacht) ---
    if input_folder not in size_all_csv:
        csv_files = list(input_folder.rglob("*.csv"))
        if not csv_files:
            size_all_csv[input_folder] = min_mem_mb
        else:
            total_rows = 0
            n_cols = None

            for csv_file in csv_files:
                # Zeilen zählen (ohne Header)
                try:
                    with csv_file.open("r") as f:
                        rows = sum(1 for _ in f) - 1
                except OSError:
                    continue

                rows = max(rows, 0)
                total_rows += rows

                # Spalten nur einmal aus einem nicht-leeren File bestimmen
                if n_cols is None and rows > 0:
                    try:
                        df_head = pd.read_csv(csv_file, nrows=1)
                        n_cols = len(df_head.columns)  # dep var NICHT abziehen (wie gewünscht)
                    except Exception:
                        n_cols = 0

            if total_rows <= 0 or not n_cols or n_cols <= 0:
                size_all_csv[input_folder] = min_mem_mb
            else:
                base_bytes = total_rows * n_cols * dtype_bytes
                total_bytes = (
                    base_bytes
                    * pandas_factor
                    * split_factor
                    * model_factor
                    * safety_factor
                )
                mem_mb = math.ceil(total_bytes / 1024**2)
                size_all_csv[input_folder] = max(mem_mb, min_mem_mb)

    base_mem_mb = size_all_csv[input_folder]

    # --- 2) group_percentage: Anteil Simulatoren im aktuellen group-string ---
    if isinstance(group, str):
        group_list = [g for g in group.split("_") if g]
    else:
        group_list = list(group)

    group_percentage = len(group_list) / max(len(SIMULATORS), 1)
    group_percentage = max(group_percentage, 1.0 / max(len(SIMULATORS), 1))  # niemals 0

    # --- 3) mode_percentage: grob wie viele Modes partitionieren den Datensatz ---
    mode_percentage = 1.0
    mode_split = [t.strip() for t in (mode or "").split(",") if t.strip()]

    for token in mode_split:
        if "sample_size" in token:
            denom = max(len(tree_modes_by_config.get(config, [])), 1)
            mode_percentage *= 1.0 / denom
        elif "seq_len" in token:
            denom = max(len(GROUPING_MODES_ALLEL), 1)
            mode_percentage *= 1.0 / denom

    # --- 4) extra Skalierung: größere sample size / seq len => etwas mehr RAM ---
    # Regex lokal (wie gewünscht)
    re_sample_size = re.compile(r"sample_size_(\d+)")
    re_seq_len = re.compile(r"seq_len_(\d+)")

    scale = 1.0

    m = re_sample_size.search(mode or "")
    if m:
        n_val = int(m.group(1))
        # baseline bei 4; wächst mild mit log2(n/4)
        scale *= (1.0 + 0.10 * max(math.log2(n_val / 4), 0.0))

    m = re_seq_len.search(mode or "")
    if m:
        L_val = int(m.group(1))
        # baseline bei 1000; wächst mild mit log10(L/1000)
        scale *= (1.0 + 0.08 * max(math.log10(L_val / 1000), 0.0))

    # --- 5) Final ---
    est_mb = base_mem_mb * group_percentage * mode_percentage * scale
    return max(min_mem_mb, int(math.ceil(est_mb)))

    base_mem_mb = size_all_csv[input_folder]

    # --- 2) group_percentage: Anteil Simulatoren im aktuellen group-string ---
    if isinstance(group, str):
        group_list = [g for g in group.split("_") if g]
    else:
        group_list = list(group)

    group_percentage = len(group_list) / max(len(SIMULATORS), 1)
    group_percentage = max(group_percentage, 1.0 / max(len(SIMULATORS), 1))  # niemals 0

    # --- 3) mode_percentage: grob wie viele Modes partitionieren den Datensatz ---
    mode_percentage = 1.0
    mode_split = [t.strip() for t in (mode or "").split(",") if t.strip()]

    for token in mode_split:
        if "sample_size" in token:
            denom = max(len(tree_modes_by_config.get(config, [])), 1)
            mode_percentage *= 1.0 / denom
        elif "seq_len" in token:
            denom = max(len(GROUPING_MODES_ALLEL), 1)
            mode_percentage *= 1.0 / denom

    # --- 4) extra Skalierung: größere sample size / seq len => etwas mehr RAM ---
    # Regex lokal (wie gewünscht)
    re_sample_size = re.compile(r"sample_size_(\d+)")
    re_seq_len = re.compile(r"seq_len_(\d+)")

    scale = 1.0

    m = re_sample_size.search(mode or "")
    if m:
        n_val = int(m.group(1))
        # baseline bei 4; wächst mild mit log2(n/4)
        scale *= (1.0 + 0.10 * max(math.log2(n_val / 4), 0.0))

    m = re_seq_len.search(mode or "")
    if m:
        L_val = int(m.group(1))
        # baseline bei 1000; wächst mild mit log10(L/1000)
        scale *= (1.0 + 0.08 * max(math.log10(L_val / 1000), 0.0))

    # --- 5) Final ---
    est_mb = base_mem_mb * group_percentage * mode_percentage * scale
    return max(min_mem_mb, int(math.ceil(est_mb)))
