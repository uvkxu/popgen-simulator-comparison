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

ROOT_DIR = Path(".")
BIN_DIR = ROOT_DIR / "bin"
CONFIG_DIR = ROOT_DIR / "stdpopsim-config"
# CONFIG_DIR = ROOT_DIR / "configs"
CMD_DIR = ROOT_DIR / "commands"
LOG_DIR = ROOT_DIR / "logs"

DATA_DIR = Path("/hits/fast/cme/kuchve/without_recombination/data")
RESULTS_DIR = Path("/hits/fast/cme/kuchve/without_recombination/results_final")

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
STATISTICS_RESULTS_DIR = RESULTS_DIR / "statistics"

ALLEL_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "allele"
ALLELFEATURES_PLOTS_DIR = PLOTS_RESULTS_DIR/ "allele"
TREE_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "tree"
TREEFEATURES_PLOTS_DIR = PLOTS_RESULTS_DIR/ "tree"
TOGETHER_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "together"
TIPTOP_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "top_feature_run"

MERGES_COLLECT_DIR = RESULTS_DIR / "merged_stats"
COLLECT_DIR = RESULTS_DIR / "collected"
ALLELFEATURES_COLLECT_DIR = COLLECT_DIR / "allele"
TREEFEATURES_COLLECT_DIR = COLLECT_DIR / "tree"

#-------COPY-FILE---------------
COPY_DIR = Path("/hits/fast/cme/kuchve/run2/data")
COPY_RAW_DATA_DIR = COPY_DIR/ "raw"
COPY_CLEAN_DIR = COPY_DIR/ "cleaned"
COPY_FEATURES_DIR = COPY_DIR / "features"
COPY_TREE_FEATURES_DIR = COPY_FEATURES_DIR / "tree"
COPY_ALLEL_FEATURES_DIR = COPY_FEATURES_DIR / "allele"
#-------------------------------

SANITY_RESULTS_DIR = RESULTS_DIR / "sanity_checks"
TREEPLOTS_RESULTS_DIR = RESULTS_DIR/ "treeplots"

CMD_SUBDIR = "commands"

COMBO_PARAMETERS_YAML = "combo_parameters.yaml"

SIMULATORS = ["ms", "msms", "msprime"] #msms_ms behaves the same as just msms , , "msprime"
REPS = 200
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
CLASSIFIERS = [XGB_STR]


all_configs = [os.path.splitext(os.path.basename(f))[0] for f in glob(f"{CONFIG_DIR}/*.yaml")]

VALID_GROUPING_MODES = [
    "precision_applied", "precision_skipped",
    "recombination_applied", "recombination_skipped",
    "overall", "single", 
    "sample_size_4", "sample_size_8", "sample_size_16",
    "sample_size_32", "sample_size_64", "sample_size_128",
    "seq_len_1000", "seq_len_10000", "seq_len_100000", "seq_len_1000000"
]
GROUPING_MODES_TREE = ["sample_size_4", "sample_size_8", "sample_size_16",
    "sample_size_32", "sample_size_64", "sample_size_128"]

GROUPING_MODES_ALLEL = ["seq_len_1000", "seq_len_10000", "seq_len_100000", "seq_len_1000000"]

def get_num_sampled_populations(config_path):
    num_pops = 0
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if CONFIG_DIR.name != "own_configs":
        for pop in config.get("demes"):
            epochs_time = {epoch.get("end_time", 0) for epoch in pop.get("epochs")}
            if any(t == 0 for t in epochs_time):
                num_pops += 1
    else:
        if "populations" not in config:
            return 1
        for pop in config.get("populations", []):
            if pop.get("sample_size", 0) > 0:
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
tree_modes_by_config = {}
combo_folder_to_json = {}
all_combo_folders = set()

for config_name in all_configs:
    config_file = Path(CONFIG_DIR) / f"{config_name}.yaml"
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



# Generate all feature file names for each simulator group
# and create a mapping from feature file to the corresponding simulator group

def format_feature_ranking_file(group):
    return f"feature_ranking_{'_'.join(group)}_with_summary.csv"

def format_feature_ranking_barplot(group):
    return f"feature_ranking_{'_'.join(group)}_barplot.png"

def format_classifier_importance_plot(classifier, group):
    return f"feature_importance_{'_'.join(group)}_{classifier}.png"
            
# simulator_groups = list(combinations(SIMULATORS, 2))
simulator_groups = []
# simulator_groups.append(tuple(SIMULATORS))
simulator_groups = list(dict.fromkeys(simulator_groups))
# simulator_groups.append(("msms", "msprime", "ms"))

def get_topfeatures_input(config, mode, feature_type, group=None, combo=None):
    result = []
    if type(group) is str:
        group = group.split("_")
    elif group is None:
        group = SIMULATORS

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
            f"{root_folder}/{config}/{combo}/rep{i+1}/{FILE_FORMATS[format_file].format(SIM=sim)}"
            for sim in group
            for i in range(REPS)
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
            (("precision_applied" not in grouped_by) or usePrecision) and
            (("precision_skipped" not in grouped_by) or (not usePrecision)) and
            (("recombination_applied" not in grouped_by) or (recombination_rate > 0)) and
            (("recombination_skipped" not in grouped_by) or (recombination_rate == 0)) and
            (("sample_size" not in grouped_by) or (sample_size == sample_size_group and usePrecision)) and
            (("seq_len" not in grouped_by) or (seq_len == seq_len_group and usePrecision))
        ):
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
    ] + [
        f"{root_folder}/{config}/{mode}/{format_feature_ranking_barplot(sim_group)}"
        for mode in modes
    ] + [
        f"{root_folder}/{config}/{mode}/{format_classifier_importance_plot(classifier, sim_group)}"
        for mode in modes
        for classifier in CLASSIFIERS
    ]

combinations_mode = {}
for config in all_configs:
    combinations_mode[config] = [
        f"{tree_mode},{grouping_mode}"
        for tree_mode in tree_modes_by_config[config]
        for grouping_mode in GROUPING_MODES_ALLEL
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
    ] + [
        f"{root_folder}/{config}/{mode}/{format_feature_ranking_barplot(sim_group)}"
        for mode in modes
    ] + [
        f"{root_folder}/{config}/{mode}/{format_classifier_importance_plot(classifier, sim_group)}"
        for mode in modes
        for classifier in CLASSIFIERS
    ]

combinations_mode = {}
for config in all_configs:
    combinations_mode[config] = [
        f"{tree_mode},{grouping_mode}"
        for tree_mode in tree_modes_by_config[config]
        for grouping_mode in GROUPING_MODES_ALLEL
    ]


ALL_TREE_TOPFEATURES = [
    f"{TREE_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_file(sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for mode in combinations_mode[config]
    for sim_group in simulator_groups
] + [
    f"{TREE_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_barplot(sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for mode in combinations_mode[config]
    for sim_group in simulator_groups
] + [
    f"{TREE_TOPFEATURES_DIR}/{config}/{mode}/{format_classifier_importance_plot(classifier, sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for mode in combinations_mode[config]
    for sim_group in simulator_groups
    for classifier in CLASSIFIERS
]


ALL_ALLEL_TOPFEATURES = [
    f"{ALLEL_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_file(sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for sim_group in simulator_groups
] + [
    f"{ALLEL_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_barplot(sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for sim_group in simulator_groups
] + [
    f"{ALLEL_TOPFEATURES_DIR}/{config}/{mode}/{format_classifier_importance_plot(classifier, sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for sim_group in simulator_groups
    for classifier in CLASSIFIERS
]

ALL_TOGETHER_TOPFEATURES = [
    f"{TOGETHER_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_file(sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for sim_group in simulator_groups
] + [
    f"{TOGETHER_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_barplot(sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for sim_group in simulator_groups
] + [
    f"{TOGETHER_TOPFEATURES_DIR}/{config}/{mode}/{format_classifier_importance_plot(classifier, sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for sim_group in simulator_groups
    for classifier in CLASSIFIERS
]


ALL_PLOTS = [
    f"{TREEFEATURES_PLOTS_DIR}/{config}/{mode}/.done"
    for config in all_configs
    for mode in combinations_mode[config]
] + [
    f"{ALLELFEATURES_PLOTS_DIR}/{config}/{mode}/.done"
    for config in all_configs
    for mode in combinations_mode[config]
]


COMPLETE_OUTPUTS = ALL_TREE_TOPFEATURES + ALL_ALLEL_TOPFEATURES + ALL_PLOTS

STATS_RESULTS = [
    f"{STATISTICS_RESULTS_DIR}/{config}/{mode}/{group_string}_feature_comparison.csv"
    for config in all_configs
    for mode in combinations_mode[config]
    for group_string in [
        "_".join(group) for group in simulator_groups
    ]
]

print("=== STATS_RESULTS (first 5) ===")
print(STATS_RESULTS[:5])
print("=================================")


MEAN_SAMPLE_SIZE_OUTPUT_TREE = [
    path
    for config in all_configs
    for group in simulator_groups
    for path in [
        f"{TREE_TOPFEATURES_DIR}/{config}/aggregated/mean_aggregated_{'_'.join(group)}_with_summary.csv",
        f"{TREE_TOPFEATURES_DIR}/{config}/aggregated/all_features_{'_'.join(group)}_by_sample_size.csv",
        f"{TREE_TOPFEATURES_DIR}/{config}/aggregated/all_features_{'_'.join(group)}_vs_sample_size.png"
    ]
]

MEAN_SAMPLE_SIZE_OUTPUT_ALLEL = [
    path
    for config in all_configs
    for group in simulator_groups
    for path in [
        f"{ALLEL_TOPFEATURES_DIR}/{config}/aggregated/mean_aggregated_{'_'.join(group)}_with_summary.csv",
        f"{ALLEL_TOPFEATURES_DIR}/{config}/aggregated/all_features_{'_'.join(group)}_by_seq_len.csv",
        f"{ALLEL_TOPFEATURES_DIR}/{config}/aggregated/all_features_{'_'.join(group)}_vs_seq_len.png"
    ]
]

RAW_SINGLE = [
    f"{RAW_DATA_DIR}/zigzag/{combo}/rep{i+1}/{FILE_FORMATS['raw'].format(SIM=sim)}"
    for combo in combinations_by_config["zigzag"]
    for sim in SIMULATORS
    for i in range(REPS)
]

EXCLUDE_CONFIGS = {"zigzag", "one_population"}

EFF_CHECK_OUTPUT = [
    f"{RESULTS_DIR}/{sim}/{typ}/{config}/{mode}/{fname}"
    for sim in SIMULATORS
    for typ in ("tree", "allele")
    for config in all_configs
    if config not in EXCLUDE_CONFIGS
    for mode in combinations_mode[config]
    for fname in (
        f"check_eff_pop_size_{sim}_summary.csv",
        f"check_eff_pop_size_{sim}_plot.png",
    )
]



TOPFEATURES_AGG_GLOBAL_DIR = TOPFEATURES_RESULTS_DIR / "aggregated_global"
TOPFEATURES_AGG_GLOBAL_PLOTS_DIR = TOPFEATURES_AGG_GLOBAL_DIR / "plots"

AGG_OVERALL = TOPFEATURES_AGG_GLOBAL_DIR / "agg_overall.csv"
AGG_FEAT    = TOPFEATURES_AGG_GLOBAL_DIR / "agg_feature_accuracy.csv"
AGG_PASSED  = TOPFEATURES_AGG_GLOBAL_DIR / "agg_passed_frequency.csv"
AGG_INDEX   = TOPFEATURES_AGG_GLOBAL_DIR / "agg_files_index.csv"

# These are "key plots" that your plot script writes (always)
PLOT_BAR_OVERALL = TOPFEATURES_AGG_GLOBAL_PLOTS_DIR / "bar_overall_by_config_tree_vs_allele.png"
PLOT_LINE_SEQLEN = TOPFEATURES_AGG_GLOBAL_PLOTS_DIR / "line_accuracy_vs_sequence_length.png"

# These are only produced if you pass --demes-root
PLOT_LINE_RATIO_MEAN = TOPFEATURES_AGG_GLOBAL_PLOTS_DIR / "line_accuracy_vs_sample_over_meanNe.png"
PLOT_LINE_RATIO_MIN  = TOPFEATURES_AGG_GLOBAL_PLOTS_DIR / "line_accuracy_vs_sample_over_minNe.png"
PLOT_BAR_MIG         = TOPFEATURES_AGG_GLOBAL_PLOTS_DIR / "bar_accuracy_by_continuous_migration_tree_vs_allele.png"


def list_all_topfeature_summary_csv():
    """
    Collect all feature_ranking_*_with_summary.csv under:
      results/top_features/tree/{config}/{mode}/...
      results/top_features/allele/{config}/{mode}/...
    """
    return [str(p) for p in TOPFEATURES_RESULTS_DIR.rglob("*with_summary.csv")]


def format_optuna_report_file(group):
    return f"xgb_params_esgrid_{'_'.join(group)}.yaml"


ALL_TREE_OPTUNA = [
    f"{TREE_TOPFEATURES_DIR}/{config}/{mode}/{format_optuna_report_file(sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for sim_group in simulator_groups
]

ALL_ALLEL_OPTUNA = [
    f"{ALLEL_TOPFEATURES_DIR}/{config}/{mode}/{format_optuna_report_file(sim_group)}"
    for config in all_configs
    for mode in combinations_mode[config]
    for sim_group in simulator_groups
]


ALL_OWN_FEATURES = [
    f"{TREE_FEATURES_DIR}/{config}/{combo}/rep{i+1}/{sim}_own_tree_features.csv"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in SIMULATORS
    for i in range(REPS)
] + [
    f"{ALLEL_FEATURES_DIR}/{config}/{combo}/rep{i+1}/{sim}_own_allel_features.csv"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in SIMULATORS
    for i in range(REPS)
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

TOPFEATURES_ACCURACY = [
    f"{TIPTOP_TOPFEATURES_DIR}/{config}/{mode}/{format_feature_ranking_file(sim_group)}"
        for config in all_configs
        for mode in combinations_mode[config]
        for sim_group in simulator_groups
]

COLLECT_FEATURES = [
    f"{COLLECT_DIR}/{type_}/{config}/simulator_feature_values.csv"
        for type_ in ("tree", "allele")
        for config in all_configs
]

OWN_COMPARISON_OUTPUT = [
    f"{ALLEL_TOPFEATURES_DIR}/{config}/{combo}/feature_ranking_with_summary.csv"
    for config in all_configs
    for combo in combinations_by_config[config]
] + [
    f"{TREE_TOPFEATURES_DIR}/{config}/{combo}/feature_ranking_with_summary.csv"
    for config in all_configs
    for combo in combinations_by_config[config]
]

MERGED_STATS = [
    f"{MERGES_COLLECT_DIR}/{config}/{combo}/rep{i + 1}/{simulator}_merge_stats.csv"
    for config in all_configs
    for combo in combinations_by_config[config]
    for i in range(REPS)
    for simulator in ["ms", "msms"]
]

MERGED_STATS_OVERALL = [
    f"{ALLELFEATURES_COLLECT_DIR}/{config}/{simulator}_merged_positions_overall.csv"
    for config in all_configs
    for simulator in ["ms", "msms"]
] + [
    f"{ALLELFEATURES_COLLECT_DIR}/{config}/{simulator}_merged_positions_sample_size.csv"
    for config in all_configs
    for simulator in ["ms", "msms"]
]

