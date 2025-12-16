from pathlib import Path
import json
import yaml
import os
from glob import glob
import itertools

ROOT_DIR = Path(".")
BIN_DIR = ROOT_DIR / "bin"
CONFIG_DIR = ROOT_DIR / "configs"
OWN_CONFIG_DIR = ROOT_DIR / "own_configs"
CMD_DIR = ROOT_DIR / "commands"

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

BOXPLOTS_RESULTS_DIR = RESULTS_DIR / "boxplots"
TOPFEATURES_RESULTS_DIR = RESULTS_DIR/ "top_features"
ALLEL_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "allele"
ALLELFEATURES_BOXPLOTS_DIR = BOXPLOTS_RESULTS_DIR/ "allele"

TREE_ZEROB_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "tree"/ "without_zero_branches"
TREE_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "tree"/ "with_zero_branches"
TREEFEATURES_ZEROB_BOXPLOTS_DIR = BOXPLOTS_RESULTS_DIR/ "tree"/ "without_zero_branches"
TREEFEATURES_BOXPLOTS_DIR = BOXPLOTS_RESULTS_DIR/ "tree"/ "with_zero_branches"

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
    combinations = itertools.product(*vars_dict.values())

    return keys, combinations


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

for config_name in all_configs:
    config_file = Path(CONFIG_DIR) / f"{config_name}.yaml"
    keys, combinations = get_combinations(config_file, VARIABLE_FILE_PATH)
    
    combos = []
    for values in combinations:
        combo = dict(zip(keys, values))
        combo_folder = format_combination_as_folder_name(combo)

        combos.append(combo_folder)
        combo_folder_to_json[combo_folder] = json.dumps(combo)
    
    combinations_by_config[config_name] = combos


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

ALLEL_FEATURES_ZIGZAG = [
    f"{ALLEL_FEATURES_DIR}/zigzag/{combo}/rep{i+1}/{FILE_FORMATS['allel_features'].format(SIM=sim)}"
    for combo in combinations_by_config["zigzag"]
    for sim in SIMULATORS
    for i in range(REPS)
]