from pathlib import Path

ROOT_DIR = Path(".")
BIN_DIR = ROOT_DIR / "bin"
CONFIG_DIR = ROOT_DIR / "configs"
OWN_CONFIG_DIR = ROOT_DIR / "own_configs"
CMD_DIR = ROOT_DIR / "commands"

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_MS_PATH = BIN_DIR / "ms"
DEFAULT_MSMS_PATH = BIN_DIR / "msms"
DEFAULT_MS_MOD_PATH = BIN_DIR / "ms-mod"

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

SIMULATORS = ["ms", "msprime", "msms", "mspms"] #msms_ms behaves the same as just msms
REPS = 20
REC_REPS = 100
NUM_SANITY_CLASSES = 5
SANITY_REPS = 5

FILE_FORMATS = {
    "raw": "{SIM}.txt",
    "FEATURES": "{SIM}_FEATURES.csv",
    "reps": "/rep{REP}",
    "sanity_check_file": "sanity_check_FEATURES.txt"
}

import os
from glob import glob

all_configs = [os.path.splitext(os.path.basename(f))[0] for f in glob(f"{CONFIG_DIR}/*.yaml")]
all_own_configs = [os.path.splitext(os.path.basename(f))[0] for f in glob(f"{OWN_CONFIG_DIR}/*.yaml")]
