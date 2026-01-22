from pathlib import Path
import json
import yaml
import os
from glob import glob
import itertools
from itertools import combinations
import pandas as pd
import math

ROOT_DIR = Path(".")
BIN_DIR = ROOT_DIR / "bin"
CONFIG_DIR = ROOT_DIR / "stdpopsim-config"
# CONFIG_DIR = ROOT_DIR / "configs"
CMD_DIR = ROOT_DIR / "commands"
LOG_DIR = ROOT_DIR / "logs"

DATA_DIR = Path("/hits/fast/cme/kuchve/run2/data")
RESULTS_DIR = Path("/hits/fast/cme/kuchve/run2/results")

DEFAULT_MS_PATH = BIN_DIR / "ms"
DEFAULT_MSMS_PATH = ROOT_DIR/ "lib"/ "msms.jar"
DEFAULT_MS_MOD_PATH = BIN_DIR / "ms-mod"
DEFAULT_SLIM_PATH = BIN_DIR / "slim"

VARIABLE_FILE_PATH = CMD_DIR / "variables.txt"

RAW_DATA_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
TREE_FEATURES_DIR = FEATURES_DIR / "tree"
ALLEL_FEATURES_DIR = FEATURES_DIR / "allele"

PIGOT_RHO_DIR = FEATURES_DIR / "pigot_rho"

CLEAN_DIR = DATA_DIR / "cleaned"

PLOTS_RESULTS_DIR = RESULTS_DIR / "plots"
TOPFEATURES_RESULTS_DIR = RESULTS_DIR/ "top_features"

ALLEL_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "allele"
ALLELFEATURES_PLOTS_DIR = PLOTS_RESULTS_DIR/ "allele"
TREE_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "tree"
TREEFEATURES_PLOTS_DIR = PLOTS_RESULTS_DIR/ "tree"



SANITY_RESULTS_DIR = RESULTS_DIR / "sanity_checks"
TREEPLOTS_RESULTS_DIR = RESULTS_DIR/ "treeplots"

CMD_SUBDIR = "commands"

COMBO_PARAMETERS_YAML = "combo_parameters.yaml"

SIMULATORS = ["msms", "msprime", "ms", "slim"] #msms_ms behaves the same as just msms
REPS = 50
NUM_REPLICATES = 2
PRECISION = 17
MAX_OVERALL_SAMPLES = 200

NUM_SANITY_CLASSES = 5
SANITY_REPS = 5

with open("file_formats.yaml") as f:
    FILE_FORMATS = yaml.safe_load(f)

XGB_STR = "xgboost"
RF_STR = "random_forest"
LR_STR = "logistic_regression"
CLASSIFIERS = [XGB_STR, RF_STR, LR_STR]


all_configs = [os.path.splitext(os.path.basename(f))[0] for f in glob(f"{CONFIG_DIR}/*.yaml")]
all_configs.remove("freedman_wolf")  # exclude for now due to long runtime

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
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vars_dict = parse_variable_file(variable_file)
    if get_num_sampled_populations(config_path) < 2:
        vars_dict.pop("population_sizes", None)
    keys = list(vars_dict.keys())
    combos = itertools.product(*vars_dict.values())

    return keys, combos


def parse_variable_file(filepath):
    """Reads the variable file and returns a dict of parameter lists."""
    variables = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, values = line.split(":")
            key = key.strip()
            vals = [v.strip() for v in values.split(",")]
            def convert(v):
                try:
                    return int(v) if "." not in v else float(v)
                except ValueError:
                    return v
            variables[key] = [convert(v) for v in vals]
    return variables


def format_combination_as_folder_name(combo):
    return "_".join(f"{k}-{v}" for k, v in combo.items())

combinations_by_config = {}
combo_folder_to_json = {}
all_combo_folders = set()

for config_name in all_configs:
    config_file = Path(CONFIG_DIR) / f"{config_name}.yaml"
    keys, all_combos_by_config = get_combinations(config_file, VARIABLE_FILE_PATH)
    num_populations = get_num_sampled_populations(config_file)
    
    combos = []
    for values in all_combos_by_config:
        combo = dict(zip(keys, values))

        # --- Filter: remove combos with usePrecision=True AND recombination_rate != 0 ---
        if not bool(combo.get("usePrecision")) and float(combo.get("recombination_rates", 0)) > 0.0:
            continue

        if num_populations * combo.get("sample_sizes") > MAX_OVERALL_SAMPLES:
            continue

        combo_folder = format_combination_as_folder_name(combo)
        all_combo_folders.add(combo_folder)

        combos.append(combo_folder)
        combo_folder_to_json[combo_folder] = json.dumps(combo)
    
    combinations_by_config[config_name] = combos

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
            config_file = Path(CONFIG_DIR) / config

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
        if sim == "ms":
            # ms is text based; memory ~ output size
            # assume 0.2 byte per site×hap
            return min(int(n_haps * seq_len * 0.2 / 1e6), 10_000)

        # ==============================
        #  msprime (tree sequences)
        # ==============================
        if sim == "msprime":
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
        if sim == "msms":
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

            reps = int(combo_json.get("nreps", NUM_REPLICATES))

            # ====== Ultra-conservative peak multipliers ======
            mem_mb *= 10.0            # Mutation/output peak
            mem_mb *= 2.5             # Java GC + fragmentation
            mem_mb *= (1 + 0.5*reps)  # Multiple replicates

            # Absolute floor and ceiling
            return max(8192, min(int(mem_mb), 1_000_000))



    return 100


def estimate_mem_mb_classification(
    input_folder,
    dtype_bytes=4,          # float32
    pandas_factor=2.0,
    split_factor=1.5,
    model_factor=2.0,
    safety_factor=1.3,
    min_mem_mb=512
):
    input_folder = Path(input_folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"Input directory {input_folder} does not exist.")
    csv_files = list(input_folder.rglob("*.csv"))

    total_rows = 0
    n_features = None

    for csv in csv_files:
        csv = Path(csv)

        # --- number of rows (cheap) ---
        with csv.open("r") as f:
            rows = sum(1 for _ in f) - 1
        total_rows += max(rows, 0)

        # --- number of columns (read only header) ---
        if n_features is None:
            df_head = pd.read_csv(csv, nrows=1)
            n_features = len(df_head.columns) - 1  # minus 'simulator'

    if total_rows <= 0 or n_features <= 0:
        return min_mem_mb

    # --- base data size ---
    base_bytes = total_rows * n_features * dtype_bytes

    # --- apply overhead factors ---
    total_bytes = (
        base_bytes
        * pandas_factor
        * split_factor
        * model_factor
        * safety_factor
    )

    mem_mb = math.ceil(total_bytes / 1024**2)

    return max(mem_mb, min_mem_mb)


def estimate_mem_mb_plot_features(
    input_folder,
    dtype_bytes=4,           # float32
    pandas_factor=2.0,       # Overhead für DataFrame
    plot_factor=1.2,         # Boxplots/Histogramme temporär im Speicher
    safety_factor=1.3,       # Sicherheitspuffer
    min_mem_mb=512
):
    """
    Schätzt den Speicherbedarf (MB) für das Plot-Features-Skript.
    Lädt keine CSVs, sondern nutzt Zeilen- und Spaltenzählung.
    """
    input_folder = Path(input_folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"Input directory {input_folder} does not exist.")
    
    csv_files = list(input_folder.rglob("*.csv"))
    total_rows = 0
    n_features = None

    for csv_file in csv_files:
        # --- Anzahl Zeilen (ohne Header) ---
        with csv_file.open("r") as f:
            rows = sum(1 for _ in f) - 1
            total_rows += max(rows, 0)

        # --- Anzahl Spalten (nur Header lesen) ---
        if n_features is None and rows > 0:
            df_head = pd.read_csv(csv_file, nrows=1)
            n_features = len(df_head.columns)

    if total_rows <= 0 or n_features <= 0:
        return min_mem_mb

    # --- Basisspeicher in Bytes ---
    base_bytes = total_rows * n_features * dtype_bytes

    # --- Multiplikatoren für pandas, Plot-Overhead und Sicherheit ---
    total_bytes = base_bytes * pandas_factor * plot_factor * safety_factor

    mem_mb = math.ceil(total_bytes / 1024**2)
    return max(mem_mb, min_mem_mb)


# Generate all feature file names for each simulator group
# and create a mapping from feature file to the corresponding simulator group
VALID_GROUPING_MODES = [
    "precision_applied", "precision_skipped",
    "recombination_applied", "recombination_skipped",
    "overall", "single"
]
GROUPING_MODES = ["overall"]

def format_feature_ranking_file(group):
    return f"feature_ranking_{'_'.join(group)}_with_summary.csv"

def format_feature_ranking_barplot(group):
    return f"feature_ranking_{'_'.join(group)}_barplot.png"

def format_classifier_importance_plot(classifier, group):
    return f"feature_importance_{'_'.join(group)}_{classifier}.png"
            
simulator_groups = list(combinations(SIMULATORS, 2))
simulator_groups.append(tuple(SIMULATORS))
simulator_groups = list(dict.fromkeys(simulator_groups))

def get_topfeatures_input(config, group, mode, feature_type, combo=None):
    result = []
    if type(group) is str:
        group = group.split("_")

    if mode not in VALID_GROUPING_MODES:
        raise ValueError(f"Invalid mode: {mode}")
    
    if feature_type == "tree":
        root_folder = TREE_FEATURES_DIR
        format_file = "tree_features"
    elif feature_type == "allel":
        root_folder = ALLEL_FEATURES_DIR
        format_file = "allel_features"
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")

    if mode == "single":
        if combo is None:
            raise ValueError("Mode 'single' is not supported for top features input generation without combo.")

        return [
            f"{root_folder}/{config}/{combo}/rep{i+1}/{FILE_FORMATS[format_file].format(SIM=sim)}"
            for sim in group
            for i in range(REPS)
        ]

    for combo_folder in combinations_by_config[config]:
        combo_string = combo_folder_to_json[combo_folder]
        combo = json.loads(combo_string)
        rate = float(combo.get("recombination_rates"))
        has_recombination = rate is not None and rate > 0

        if mode == "precision_applied" and bool(combo["usePrecision"]):
            result.append(combo_folder)
        elif mode == "precision_skipped" and not bool(combo["usePrecision"]):
            result.append(combo_folder)
        elif mode == "recombination_applied" and has_recombination:
            result.append(combo_folder)
        elif mode == "recombination_skipped" and not has_recombination:
            result.append(combo_folder)
        elif mode == "overall":
            result.append(combo_folder)
    
    return [
        f"{root_folder}/{config}/{combo}/rep{i+1}/{FILE_FORMATS[format_file].format(SIM=sim)}"
        for combo in result
        for sim in group
        for i in range(REPS)
    ] + [
        f"{root_folder}/{config}/{combo}/{COMBO_PARAMETERS_YAML}"
        for combo in result
    ]


def get_pigotrho_input(simulators):
    if isinstance(simulators, str):
        simulators = [simulators]

    return [
        f"{PIGOT_RHO_DIR}/{config}/{combo}/rep{i+1}/{sim}_pigotrho_features.csv"
        for config in all_configs
        for combo in combinations_by_config[config]
        for sim in simulators
        for i in range(REPS)
    ]


RAW_INPUTS = [
    f"{RAW_DATA_DIR}/{config}/{combo}/rep{i+1}/{FILE_FORMATS['raw'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in SIMULATORS
    for i in range(REPS)
]

ALL_COMBO_YAML = [
    f"{RAW_DATA_DIR}/{config}/{combo}/{COMBO_PARAMETERS_YAML}"
    for config in all_configs
    for combo in combinations_by_config[config]
]

CLEAN_INPUTS = [
    f"{CLEAN_DIR}/{config}/{combo}/rep{i+1}/{FILE_FORMATS['cleaned'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in SIMULATORS
    for i in range(REPS)
]

ALL_FEATURES = [
    f"{TREE_FEATURES_DIR}/{config}/{combo}/rep{i+1}/{FILE_FORMATS['tree_features'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in SIMULATORS
    for i in range(REPS)
] + [
    f"{ALLEL_FEATURES_DIR}/{config}/{combo}/rep{i+1}/{FILE_FORMATS['allel_features'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in SIMULATORS
    for i in range(REPS)
]

ALL_SLIM_FEATURES = [
    f"{TREE_FEATURES_DIR}/{config}/{combo}/rep{i+1}/{FILE_FORMATS['tree_features'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in ["slim"]
    for i in range(REPS)
] + [
    f"{ALLEL_FEATURES_DIR}/{config}/{combo}/rep{i+1}/{FILE_FORMATS['allel_features'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in ["slim"]
    for i in range(REPS)
]


ALL_TREE_TOPFEATURES = [
    f"{TREE_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_file(sim_group)}"
    for config in all_configs
    for mode in GROUPING_MODES
    for sim_group in simulator_groups
] + [
    f"{TREE_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_barplot(sim_group)}"
    for config in all_configs
    for mode in GROUPING_MODES
    for sim_group in simulator_groups
] + [
    f"{TREE_TOPFEATURES_DIR}/{config}/{mode}/{format_classifier_importance_plot(classifier, sim_group)}"
    for config in all_configs
    for mode in GROUPING_MODES
    for sim_group in simulator_groups
    for classifier in CLASSIFIERS
]


ALL_ALLEL_TOPFEATURES = [
    f"{ALLEL_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_file(sim_group)}"
    for config in all_configs
    for mode in GROUPING_MODES
    for sim_group in simulator_groups
] + [
    f"{ALLEL_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_barplot(sim_group)}"
    for config in all_configs
    for mode in GROUPING_MODES
    for sim_group in simulator_groups
] + [
    f"{ALLEL_TOPFEATURES_DIR}/{config}/{mode}/{format_classifier_importance_plot(classifier, sim_group)}"
    for config in all_configs
    for mode in GROUPING_MODES
    for sim_group in simulator_groups
    for classifier in CLASSIFIERS
]


ALL_PLOTS = [
    f"{TREEFEATURES_PLOTS_DIR}/{config}/{mode}/.done"
    for config in all_configs
    for mode in GROUPING_MODES
] + [
    f"{ALLELFEATURES_PLOTS_DIR}/{config}/{mode}/.done"
    for config in all_configs
    for mode in GROUPING_MODES
]

COMPLETE_OUTPUTS = ALL_TREE_TOPFEATURES + ALL_ALLEL_TOPFEATURES + ALL_PLOTS






RAW_SINGLE = [
    f"{RAW_DATA_DIR}/zigzag/{combo}/rep{i+1}/{FILE_FORMATS['raw'].format(SIM=sim)}"
    for combo in combinations_by_config["zigzag"]
    for sim in SIMULATORS
    for i in range(REPS)
]

