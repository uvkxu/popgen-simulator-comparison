
from pathlib import Path
import yaml
import platform


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
RAW_DATA_DIR = DATA_DIR / cfg["data_layout"]["raw"]
CLEAN_DIR = DATA_DIR / cfg["data_layout"]["cleaned"]
FEATURES_DIR = DATA_DIR / cfg["data_layout"]["features"]

FEATURE_DIRS = {
    name: FEATURES_DIR / name
    for name in cfg["feature_types"]
}

TREE_FEATURES_DIR = FEATURE_DIRS["tree"]
ALLEL_FEATURES_DIR = FEATURE_DIRS["allele"]

# Results
TOPFEATURES_RESULTS_DIR = RESULTS_DIR / cfg["results_layout"]["top_features"]
STATISTICS_RESULTS_DIR = RESULTS_DIR / cfg["results_layout"]["statistics"]
PLOTS_RESULTS_DIR = RESULTS_DIR / cfg["results_layout"]["plots"]
MERGES_COLLECT_DIR = RESULTS_DIR / cfg["results_layout"]["merged_stats"]
COLLECT_DIR = RESULTS_DIR / cfg["results_layout"]["collected"]

TOPFEATURES_DIRS = {
    name: TOPFEATURES_RESULTS_DIR / name
    for name in cfg["topfeatures_types"]
}

TREE_TOPFEATURES_DIR = TOPFEATURES_DIRS["tree"]
ALLEL_TOPFEATURES_DIR = TOPFEATURES_DIRS["allele"]
TOGETHER_TOPFEATURES_DIR = TOPFEATURES_DIRS["together"]

PLOTS_DIRS = {
    name: PLOTS_RESULTS_DIR / name
    for name in cfg["feature_types"]
}

TREEFEATURES_PLOTS_DIR = PLOTS_DIRS["tree"]
ALLELFEATURES_PLOTS_DIR = PLOTS_DIRS["allele"]

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
