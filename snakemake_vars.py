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
from scenario_types import *

# Load workflow_config 
with WORKFLOW_CONFIG_FILE.open(encoding="utf-8") as f:
    workflow_config = yaml.safe_load(f)

with VARIABLE_FILE_PATH.open(encoding="utf-8") as f:
    parameters_config = yaml.safe_load(f)

# Get target simulators
TARGET_SIMULATORS = workflow_config["target_simulators"]
RECOMBINATION_RATES = [
    float(rate) if rate is not None else 0.0
    for rate in parameters_config["recombination_rates"]
]

if not RECOMBINATION_RATES:
    raise ValueError("At least one recombination rate must be specified.")

if any(rate < 0 for rate in RECOMBINATION_RATES):
    raise ValueError("Recombination rates must not be negative.")

# Validate recombination rates
if len(RECOMBINATION_RATES) > 2:
    raise ValueError(
        "More than two recombination rates are specified. "
        "The current directory structure supports at most two rates "
        "(one without recombination and one with recombination). "
        "If you want to use more rates, the data and results directory "
        "structure must be adapted to accommodate additional comparison scenarios."
    )

if len(RECOMBINATION_RATES) == 2:
    has_zero = 0 in RECOMBINATION_RATES
    has_nonzero = any(rate != 0 for rate in RECOMBINATION_RATES)

    if not (has_zero and has_nonzero):
        raise ValueError(
            "If exactly two recombination rates are specified, "
            "one must be 0 and the other must be non-zero."
        )


SELECTED_SCENARIOS = workflow_config["selected_scenarios"]
AVAILABLE_SCENARIOS = []

if 0 in RECOMBINATION_RATES:
    AVAILABLE_SCENARIOS.append(COALESCENT_WITHOUT_RECOMBINATION)

    if SLIM in TARGET_SIMULATORS:
        AVAILABLE_SCENARIOS.append(SLIM_COMPARISON)

if any(rate > 0 for rate in RECOMBINATION_RATES):
    AVAILABLE_SCENARIOS.append(COALESCENT_WITH_RECOMBINATION)

unavailable_scenarios = (
    set(SELECTED_SCENARIOS) - set(AVAILABLE_SCENARIOS)
)

if unavailable_scenarios:
    raise ValueError(
        "The following selected scenarios cannot be executed "
        "with the current simulator and recombination settings: "
        f"{sorted(unavailable_scenarios)}"
    )

SIMULATORS_BY_SCENARIO = {
    scenario: [
        simulator for simulator in TARGET_SIMULATORS
        if scenario == SLIM_COMPARISON or simulator != SLIM
    ]
    for scenario in SELECTED_SCENARIOS
}

for scenario, simulators in SIMULATORS_BY_SCENARIO.items():
    if len(simulators) < 2:
        raise ValueError(
            f"Comparison scenario '{scenario}' requires at least two simulators. "
            f"Currently selected simulators: {simulators}"
        )

EXTERNAL_REPS = workflow_config["replicates"]["external"]
INTERNAL_REPS = workflow_config["replicates"]["internal"]
PRECISION = workflow_config["decimal_precision"]
MAX_OVERALL_SAMPLES = workflow_config["max_overall_samples"]


with (FILE_FORMATS).open() as f:
    file_formats = yaml.safe_load(f)

COMBO_PARAMETERS_YAML = file_formats["combo_parameters"]


all_configs = [os.path.splitext(os.path.basename(f))[0] for f in glob(f"{DEM_MODELS_DIR}/*.yaml")]
config_by_scenario = {
    COALESCENT_WITHOUT_RECOMBINATION: all_configs.copy(),

    COALESCENT_WITH_RECOMBINATION: [
        config for config in all_configs
        if config not in {"lavretsky_ducks", "li_african3epoch"}
    ],

    SLIM_COMPARISON: [
        config for config in all_configs
        if config != "freedman_wolf"
    ],
}


# Computation of the GROUPING MODES for the grouping for computation of top features
def is_valid_grouping_mode(mode):
    return (
        mode in {
            "overall",
            "single",
        }
        or re.fullmatch(r"sample_size_\d+", mode) is not None
        or re.fullmatch(r"seq_len_\d+", mode) is not None
    )

GROUPING_MODES_SEQ_LEN = [
    f"seq_len_{seq_len}"
    for seq_len in parameters_config["sequence_lengths"]
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

combinations_by_config_and_scenario = {}
modes_sample_size_by_config = {}
combo_folder_to_json = {}
all_combo_folders = set()

for config_name in all_configs:
    config_file = Path(DEM_MODELS_DIR) / f"{config_name}.yaml"
    keys, all_combos_by_config = get_combinations(
        config_file, VARIABLE_FILE_PATH
    )
    num_populations = get_num_sampled_populations(config_file)

    modes_sample_size_by_config[config_name] = []

    combos_by_scenario = {
        scenario: []
        for scenario in SELECTED_SCENARIOS
    }

    for values in all_combos_by_config:
        combo = dict(zip(keys, values))

        if num_populations * combo.get("sample_sizes") > MAX_OVERALL_SAMPLES:
            continue

        possible_mode = f"sample_size_{int(combo.get('sample_sizes'))}"
        if is_valid_grouping_mode(possible_mode):
            modes_sample_size_by_config[config_name].append(possible_mode)

        combo_folder = format_combination_as_folder_name(combo)
        all_combo_folders.add(combo_folder)
        combo_folder_to_json[combo_folder] = json.dumps(combo)

        recombination_rate = float(combo.get("recombination_rates") or 0)

        if recombination_rate == 0:
            for scenario in [COALESCENT_WITHOUT_RECOMBINATION, SLIM_COMPARISON]:
                if (
                    scenario in SELECTED_SCENARIOS
                    and config_name in config_by_scenario[scenario]
                ):
                    combos_by_scenario[scenario].append(combo_folder)

        elif (
            COALESCENT_WITH_RECOMBINATION in SELECTED_SCENARIOS
            and config_name not in {"lavretsky_ducks", "li_african3epoch"}
        ):
            combos_by_scenario[COALESCENT_WITH_RECOMBINATION].append(combo_folder)

    for scenario in SELECTED_SCENARIOS:
        combinations_by_config_and_scenario.setdefault(scenario, {})
        combinations_by_config_and_scenario[scenario][config_name] = (
            combos_by_scenario[scenario]
        )

simulators_by_scenario = {}
simulator_groups_by_scenario = {}

for scenario in SELECTED_SCENARIOS:
    if scenario == SLIM_COMPARISON:
        simulators = TARGET_SIMULATORS.copy()
    else:
        simulators = [
            simulator for simulator in TARGET_SIMULATORS
            if simulator != SLIM
        ]

    if len(simulators) < 2:
        raise ValueError(
            f"At least two simulators are required for '{scenario}'. "
            f"Selected simulators: {simulators}"
        )

    simulators_by_scenario[scenario] = simulators

    if scenario == SLIM_COMPARISON:
        groups = [
            pair for pair in itertools.combinations(simulators, 2)
            if SLIM in pair
        ]
    else:
        groups = []
        if len(simulators) > 2:
            groups = [
                pair for pair in itertools.combinations(simulators, 2)
            ]
        groups.append(tuple(simulators))


    simulator_groups_by_scenario[scenario] = groups

# Generate all feature file names for each simulator group
# and create a mapping from feature file to the corresponding simulator group

def format_feature_ranking_file(group):
    return file_formats["feature_ranking_with_summary"].format(
        group="_".join(group)
    )

def get_evaluation_input(config, mode, feature_type, scenario, group=None, combo=None):
    result = []
    if type(group) is str:
        group = group.split("_")
    elif group is None:
        group = simulators_by_scenario[scenario]

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
        root_folder = TREE_FEATURES_DIR[scenario]
        format_file = "tree_features"
    elif feature_type == "allel":
        root_folder = ALLEL_FEATURES_DIR[scenario]
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

    for combo_folder in combinations_by_config_and_scenario[scenario][config]:
        combo_string = combo_folder_to_json[combo_folder]
        combo = json.loads(combo_string)
        rate = float(combo.get("recombination_rates"))
        sample_size = int(combo.get("sample_sizes")) 
        sample_size = int(combo.get("sample_sizes")) 
        recombination_rate = float(combo.get("recombination_rates", 0))
        seq_len = int(combo.get("sequence_lengths"))
        usePrecision = bool(combo.get("usePrecision", True))

        if (
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
        f"{folder}/{config}/{combo}/{COMBO_PARAMETERS_YAML}"
        for folder in [root_folder, CLEAN_DIR[scenario]]
        for combo in result
    ]


#----------------------------------
# Zwischenstadien für snakemake
#----------------------------------

RAW_INPUTS = [
    f"{RAW_DATA_DIR[scenario]}/{config}/{combo}/rep{i+1}/{file_formats['raw'].format(SIM=sim)}"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for combo in combinations_by_config_and_scenario[scenario][config]
    for sim in simulators_by_scenario[scenario]
    for i in range(EXTERNAL_REPS)
]

ALL_COMBO_YAML = [
    f"{RAW_DATA_DIR[scenario]}/{config}/{combo}/{COMBO_PARAMETERS_YAML}"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for combo in combinations_by_config_and_scenario[scenario][config]
]

CLEAN_INPUTS = [
    f"{CLEAN_DIR[scenario]}/{config}/{combo}/rep{i+1}/{file_formats['cleaned'].format(SIM=sim)}"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for combo in combinations_by_config_and_scenario[scenario][config]
    for sim in simulators_by_scenario[scenario]
    for i in range(EXTERNAL_REPS)
]

ALL_FEATURES = [
    f"{TREE_FEATURES_DIR[scenario]}/{config}/{combo}/rep{i+1}/{file_formats['tree_features'].format(SIM=sim)}"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for combo in combinations_by_config_and_scenario[scenario][config]
    for sim in simulators_by_scenario[scenario]
    for i in range(EXTERNAL_REPS)
] + [
    f"{ALLEL_FEATURES_DIR[scenario]}/{config}/{combo}/rep{i+1}/{file_formats['allel_features'].format(SIM=sim)}"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for combo in combinations_by_config_and_scenario[scenario][config]
    for sim in simulators_by_scenario[scenario]
    for i in range(EXTERNAL_REPS)
]

CREATE_COMBO_PARAMETERS_YAML = [
    f"{TREE_FEATURES_DIR[scenario]}/{config}/{combo}/{COMBO_PARAMETERS_YAML}"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for combo in combinations_by_config_and_scenario[scenario][config]
] + [
    f"{ALLEL_FEATURES_DIR[scenario]}/{config}/{combo}/{COMBO_PARAMETERS_YAML}"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for combo in combinations_by_config_and_scenario[scenario][config]
]

#-----------------------------
#Final results for snakemake
#-----------------------------

combinations_mode = {}
for config in all_configs:
    combinations_mode[config] = [
        f"{sample_size_mode},{seq_len_mode}"
        for sample_size_mode in modes_sample_size_by_config[config]
        for seq_len_mode in GROUPING_MODES_SEQ_LEN
    ]



ALL_TREE_TOPFEATURES = [
    f"{TREE_TOPFEATURES_DIR[scenario]}/{config}/{mode}/{format_feature_ranking_file(sim_group)}"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for mode in combinations_mode[config]
    for sim_group in simulator_groups_by_scenario[scenario]
] 

ALL_ALLEL_TOPFEATURES = [
    f"{ALLEL_TOPFEATURES_DIR[scenario]}/{config}/{mode}/{format_feature_ranking_file(sim_group)}"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for mode in combinations_mode[config]
    for sim_group in simulator_groups_by_scenario[scenario]
] 

ALL_TOGETHER_TOPFEATURES = [
    f"{TOGETHER_TOPFEATURES_DIR[scenario]}/{config}/{mode}/{format_feature_ranking_file(sim_group)}"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for mode in combinations_mode[config]
    for sim_group in simulator_groups_by_scenario[scenario]
]

STATS_RESULTS = [
    f"{STATISTICS_RESULTS_DIR[scenario]}/{config}/{mode}/{'_'.join(group)}_feature_comparison.csv"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for mode in combinations_mode[config]
    for group in simulator_groups_by_scenario[scenario]
    if len(group) == 2
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
    f"{TREE_TOPFEATURES_DIR[scenario]}/{config}/{mode}/{format_optuna_report_file(sim_group)}"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for mode in combinations_mode[config]
    for sim_group in simulator_groups_by_scenario[scenario]
] 

ALL_ALLEL_OPTUNA = [
    f"{ALLEL_TOPFEATURES_DIR[scenario]}/{config}/{mode}/{format_optuna_report_file(sim_group)}"
    for scenario in SELECTED_SCENARIOS
    for config in config_by_scenario[scenario]
    for mode in combinations_mode[config]
    for sim_group in simulator_groups_by_scenario[scenario]
] 

#-----------------------------------
#Compute stats for merged positions in coalescent simulators
#-----------------------------------

MERGE_STATS_SCENARIOS = [scenario for scenario in SELECTED_SCENARIOS if scenario != SLIM_COMPARISON]

def get_merged_stats_overall_input(config, simulator):
    return [
    f"{MERGES_COLLECT_DIR[scenario]}/{config}/{combo}/rep{i + 1}/{simulator}_merge_stats.csv"
    for scenario in MERGE_STATS_SCENARIOS
    for config in config_by_scenario[scenario]
    for combo in combinations_by_config_and_scenario[scenario][config]
    for sim in simulators_by_scenario[scenario]
    for i in range(EXTERNAL_REPS)
]

MERGED_STATS = [
    f"{MERGES_COLLECT_DIR[scenario]}/{config}/{simulator}_merged_positions_overall.csv"
    for scenario in MERGE_STATS_SCENARIOS
    for config in config_by_scenario[scenario]
    for simulator in simulators_by_scenario[scenario]
] + [
    f"{MERGES_COLLECT_DIR[scenario]}/{config}/{simulator}_merged_positions_sample_size.csv"
    for scenario in MERGE_STATS_SCENARIOS
    for config in config_by_scenario[scenario]
    for simulator in simulators_by_scenario[scenario]
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

            # TODO: das ist hier iwie falsch
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

    Erwartet, dass SIMULATORS, modes_sample_size_by_config, GROUPING_MODES_ALLEL,
    sowie ALLEL_FEATURES_DIR / TREE_FEATURES_DIR global existieren.
    """

    """
    Schnelle Approximation des RAM-Bedarfs (MB) für Classification:
    - scannt alle CSVs unter input_folder nur EINMAL (caching in size_all_csv)
    - skaliert dann pro Job über group-Anteil + mode-Anteil + (milde) Größe von sample_size/seq_len

    Erwartet, dass SIMULATORS, modes_sample_size_by_config, GROUPING_MODES_ALLEL,
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
            denom = max(len(modes_sample_size_by_config.get(config, [])), 1)
            mode_percentage *= 1.0 / denom
        elif "seq_len" in token:
            denom = max(len(GROUPING_MODES_SEQ_LEN), 1)
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
            denom = max(len(modes_sample_size_by_config.get(config, [])), 1)
            mode_percentage *= 1.0 / denom
        elif "seq_len" in token:
            denom = max(len(GROUPING_MODES_SEQ_LEN), 1)
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
