
from pathlib import Path
import yaml
import platform
from scenario_types import COMPARISON_SCENARIOS


SYSTEM = platform.system().lower()

if SYSTEM == "windows":
    PLATFORM = "windows"
elif SYSTEM == "linux":
    PLATFORM = "linux"
else:
    raise RuntimeError(f"Unsupported operating system: {SYSTEM}")

# This file is located directly in the project root
PROJECT_ROOT = Path(".")

with (PROJECT_ROOT / "config" / "paths.yaml").open(
    encoding="utf-8"
) as f:
    cfg = yaml.safe_load(f)


def resolve(path):
    path = Path(path).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


# Project paths
BIN_DIR = resolve(cfg["directories"]["bin"])
CONFIG_DIR = resolve(cfg["directories"]["config"])
LIB_DIR = resolve(cfg["directories"]["lib"])

# Independent storage roots
DATA_DIR = resolve(cfg["directories"]["data"])
RESULTS_DIR = resolve(cfg["directories"]["results"])

# Scientific configuration stays inside the project
PARAMETERS_DIR = CONFIG_DIR / cfg["parameters"]["directory"]
DEM_MODELS_DIR = PARAMETERS_DIR / cfg["parameters"]["demographic_models"]
VARIABLE_FILE_PATH = PARAMETERS_DIR / cfg["parameters"]["simulation_parameters"]

# Data
FEATURE_TYPES = cfg["feature_types"]


# Data directories
RAW_DATA_DIR = {
    scenario: DATA_DIR / scenario / cfg["data_layout"]["raw"]
    for scenario in COMPARISON_SCENARIOS
}

CLEAN_DIR = {
    scenario: DATA_DIR / scenario / cfg["data_layout"]["cleaned"]
    for scenario in COMPARISON_SCENARIOS
}

FEATURES_DATA_DIR = {
    scenario: DATA_DIR / scenario / cfg["data_layout"]["features"]
    for scenario in COMPARISON_SCENARIOS
}

FEATURE_DIRS = {
    scenario: {
        feature_type: FEATURES_DATA_DIR[scenario] / feature_type
        for feature_type in FEATURE_TYPES
    }
    for scenario in COMPARISON_SCENARIOS
}

TREE_FEATURES_DIR = {
    scenario: dirs["tree"]
    for scenario, dirs in FEATURE_DIRS.items()
}

ALLEL_FEATURES_DIR = {
    scenario: dirs["allele"]
    for scenario, dirs in FEATURE_DIRS.items()
}

RESULTS_PATHS = {
    scenario: {
        name: RESULTS_DIR / scenario / subdir
        for name, subdir in cfg["results_layout"].items()
    }
    for scenario in COMPARISON_SCENARIOS
}

# Results
# Results directories
TOPFEATURES_RESULTS_DIR = {
    scenario: RESULTS_PATHS[scenario]["top_features"]
    for scenario in COMPARISON_SCENARIOS
}

STATISTICS_RESULTS_DIR = {
    scenario: RESULTS_PATHS[scenario]["statistics"]
    for scenario in COMPARISON_SCENARIOS
}

PLOTS_RESULTS_DIR = {
    scenario: RESULTS_PATHS[scenario]["plots"]
    for scenario in COMPARISON_SCENARIOS
}

MERGES_COLLECT_DIR = {
    scenario: RESULTS_PATHS[scenario]["merged_stats"]
    for scenario in COMPARISON_SCENARIOS
}

COLLECT_DIR = {
    scenario: RESULTS_PATHS[scenario]["collected"]
    for scenario in COMPARISON_SCENARIOS
}

AGGREGATED_GLOBAL_DIR = {
    scenario: RESULTS_PATHS[scenario]["aggregated_global"]
    for scenario in COMPARISON_SCENARIOS
}

TOPFEATURES_TYPES = cfg["topfeatures_types"]

TOPFEATURES_DIRS = {
    scenario: {
        name: TOPFEATURES_RESULTS_DIR[scenario] / name
        for name in TOPFEATURES_TYPES
    }
    for scenario in COMPARISON_SCENARIOS
}

TREE_TOPFEATURES_DIR = {
    scenario: dirs["tree"]
    for scenario, dirs in TOPFEATURES_DIRS.items()
}

ALLEL_TOPFEATURES_DIR = {
    scenario: dirs["allele"]
    for scenario, dirs in TOPFEATURES_DIRS.items()
}

TOGETHER_TOPFEATURES_DIR = {
    scenario: dirs["together"]
    for scenario, dirs in TOPFEATURES_DIRS.items()
}

# Executables
def get_executable(name):
    filename = cfg["executables"][name][PLATFORM]

    if name == "msms":
        return LIB_DIR / filename

    return BIN_DIR / filename


DEFAULT_MS_PATH = get_executable("ms")
DEFAULT_MS_MOD_PATH = get_executable("ms_mod")
DEFAULT_MSMS_PATH = get_executable("msms")
DEFAULT_SLIM_PATH = get_executable("slim")

FILES = {
    name: resolve(path)
    for name, path in cfg["files"].items()
}

WORKFLOW_CONFIG_FILE = FILES["workflow_config"]
SIMULATORS_CONFIG_FILE = FILES["simulators"]
FILE_FORMATS = FILES["file_formats"]
