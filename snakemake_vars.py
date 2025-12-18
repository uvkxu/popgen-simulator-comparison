from pathlib import Path
import json
import yaml
import os
from glob import glob
import itertools
from itertools import combinations

ROOT_DIR = Path(".")
BIN_DIR = ROOT_DIR / "bin"
CONFIG_DIR = ROOT_DIR / "stdpopsim-config"
# CONFIG_DIR = ROOT_DIR / "configs"
CMD_DIR = ROOT_DIR / "commands"
LOG_DIR = ROOT_DIR / "logs"

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_MS_PATH = BIN_DIR / "ms.exe"
DEFAULT_MSMS_PATH = ROOT_DIR/ "lib"/ "msms.jar"
DEFAULT_MS_MOD_PATH = BIN_DIR / "ms-mod.exe"
DEFAULT_SLIM_PATH = BIN_DIR / "slim"

VARIABLE_FILE_PATH = CMD_DIR / "variables.txt"

RAW_DATA_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
TREE_FEATURES_DIR = FEATURES_DIR / "tree"
ALLEL_FEATURES_DIR = FEATURES_DIR / "allele"

CLEAN_DIR = DATA_DIR / "cleaned"

PLOTS_RESULTS_DIR = RESULTS_DIR / "plots"
TOPFEATURES_RESULTS_DIR = RESULTS_DIR/ "top_features"

ALLEL_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "allele"
ALLELFEATURES_PLOTS_DIR = PLOTS_RESULTS_DIR/ "allele"
TREE_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "tree"
TREEFEATURES_PLOTS_DIR = PLOTS_RESULTS_DIR/ "tree"



SANITY_RESULTS_DIR = RESULTS_DIR / "sanity_checks"

CMD_SUBDIR = "commands"

COMBO_PARAMETERS_YAML = "combo_parameters.yaml"

SIMULATORS = ["msms", "msprime", "ms"] #msms_ms behaves the same as just msms
REPS = 10
NUM_REPLICATES = 10
PRECISION = 17

NUM_SANITY_CLASSES = 5
SANITY_REPS = 5

with open("file_formats.yaml") as f:
    FILE_FORMATS = yaml.safe_load(f)

XGB_STR = "xgboost"
RF_STR = "random_forest"
LR_STR = "logistic_regression"
CLASSIFIERS = [XGB_STR, RF_STR, LR_STR]


all_configs = [os.path.splitext(os.path.basename(f))[0] for f in glob(f"{CONFIG_DIR}/*.yaml")]
# all_own_configs = [os.path.splitext(os.path.basename(f))[0] for f in glob(f"{OWN_CONFIG_DIR}/*.yaml")]


def get_combinations(config_path, variable_file):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    def has_only_one_pop():
        demes = config.get("demes", [])
        return demes is None or len(config.get("demes", [])) == 1

    vars_dict = parse_variable_file(variable_file)
    if has_only_one_pop():
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
    
    combos = []
    for values in all_combos_by_config:
        combo = dict(zip(keys, values))

        # --- Filter: remove combos with usePrecision=True AND recombination_rate != 0 ---
        if not bool(combo.get("usePrecision")) and float(combo.get("recombination_rates", 0)) > 0.0:
            continue

        combo_folder = format_combination_as_folder_name(combo)
        all_combo_folders.add(combo_folder)

        combos.append(combo_folder)
        combo_folder_to_json[combo_folder] = json.dumps(combo)
    
    combinations_by_config[config_name] = combos


# Generate all feature file names for each simulator group
# and create a mapping from feature file to the corresponding simulator group
VALID_GROUPING_MODES = [
    "precision_applied", "precision_skipped",
    "recombination_applied", "recombination_skipped",
    "overall", "single"
]
GROUPING_MODES = [m for m in VALID_GROUPING_MODES if m != "single"]

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

ALL_TREE_FEATURES = [
    f"{TREE_FEATURES_DIR}/{config}/{combo}/rep{i+1}/{FILE_FORMATS['tree_features'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in SIMULATORS
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







RAW_BROWNING = [
    f"{RAW_DATA_DIR}/browning_america/{combo}/rep{i+1}/{FILE_FORMATS['raw'].format(SIM=sim)}"
    for combo in combinations_by_config["browning_america"]
    for sim in SIMULATORS
    for i in range(REPS)
]


RAW_ZIGZAG_MSMS = [
    f"{RAW_DATA_DIR}/zigzag/{combo}/rep{i+1}/{FILE_FORMATS['raw'].format(SIM=sim)}"
    for combo in combinations_by_config["zigzag"]
    for sim in ["msms"]
    for i in range(REPS)
]


ALLEL_FEATURES_ZIGZAG = [
    f"{ALLEL_FEATURES_DIR}/zigzag/{combo}/rep{i+1}/{FILE_FORMATS['allel_features'].format(SIM=sim)}"
    for combo in combinations_by_config["zigzag"]
    for sim in SIMULATORS
    for i in range(REPS)
]

TREE_FEATURES_ZIGZAG = [
    f"{TREE_FEATURES_DIR}/zigzag/{combo}/rep{i+1}/{FILE_FORMATS['tree_features'].format(SIM=sim)}"
    for combo in combinations_by_config["zigzag"]
    for sim in SIMULATORS
    for i in range(REPS)
]

TREE_FEATURES_FREEDMAN = [
    f"{TREE_FEATURES_DIR}/freedman_wolf/{combo}/rep{i+1}/{FILE_FORMATS['tree_features'].format(SIM=sim)}"
    for combo in combinations_by_config["freedman_wolf"]
    for sim in SIMULATORS
    for i in range(REPS)
]

ALLEL_FEATURES_BROWNING = [
    f"{ALLEL_FEATURES_DIR}/browning_america/{combo}/rep{i+1}/{FILE_FORMATS['allel_features'].format(SIM=sim)}"
    for combo in combinations_by_config["browning_america"]
    for sim in SIMULATORS
    for i in range(REPS)
]

TREE_FEATURES_BROWNING = [
    f"{TREE_FEATURES_DIR}/browning_america/{combo}/rep{i+1}/{FILE_FORMATS['tree_features'].format(SIM=sim)}"
    for combo in combinations_by_config["browning_america"]
    for sim in SIMULATORS
    for i in range(REPS)
]



TOPFEATURES_TREE_BROWNING = [
    f"{TREE_TOPFEATURES_DIR}/browning_america/{mode}/{format_feature_ranking_file(sim_group)}"
    for mode in ["precision_applied", "precision_skipped", "recombination_applied", "recombination_skipped", "overall"]
    for sim_group in simulator_groups
] + [
    f"{TREE_TOPFEATURES_DIR}/browning_america/{mode}/{format_feature_ranking_barplot(sim_group)}"
    for mode in ["precision_applied", "precision_skipped", "recombination_applied", "recombination_skipped", "overall"]
    for sim_group in simulator_groups
] + [
    f"{TREE_TOPFEATURES_DIR}/browning_america/{mode}/{format_classifier_importance_plot(classifier, sim_group)}"
    for mode in ["precision_applied", "precision_skipped", "recombination_applied", "recombination_skipped", "overall"]
    for sim_group in simulator_groups
    for classifier in CLASSIFIERS
]


TOPFEATURES_ALLEL_BROWNING = [
    f"{ALLEL_TOPFEATURES_DIR}/browning_america/{mode}/{format_feature_ranking_file(sim_group)}"
    for mode in ["precision_applied", "precision_skipped"]
    for sim_group in simulator_groups
] + [
    f"{ALLEL_TOPFEATURES_DIR}/browning_america/{mode}/{format_feature_ranking_barplot(sim_group)}"
    for mode in ["precision_applied", "precision_skipped"]
    for sim_group in simulator_groups
] + [
    f"{ALLEL_TOPFEATURES_DIR}/browning_america/{mode}/{format_classifier_importance_plot(classifier, sim_group)}"
    for mode in ["precision_applied", "precision_skipped"]
    for sim_group in simulator_groups
    for classifier in CLASSIFIERS
]