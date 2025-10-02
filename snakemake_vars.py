from pathlib import Path

CONFIG_DIR = Path("configs")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

RAW_DATA_DIR = DATA_DIR / "raw"
TREESTATS_DATA_DIR = DATA_DIR / "treestats"
BOXPLOTS_RESULTS_DIR = RESULTS_DIR / "boxplots"
GBT_RESULTS_DIR = RESULTS_DIR/ "gbt"
SANITY_RESULTS_DIR = RESULTS_DIR / "sanity_checks"

SIMULATORS = ["ms", "msprime", "msms"]
REPS = 5
NUM_SANITY_CLASSES = 5
SANITY_REPS = 5

FILE_FORMATS = {
    "raw": "{SIM}.txt",
    "treestats": "{SIM}_stats.csv",
    "reps": "/rep{REP}",
    "sanity_check_file": "sanity_check_stats.txt"
}

import os
from glob import glob

all_configs = [os.path.splitext(os.path.basename(f))[0] for f in glob(f"{CONFIG_DIR}/*.yaml")]
