from pathlib import Path

CONFIG_DIR = Path("configs")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TREESTATS_DATA_DIR = PROCESSED_DATA_DIR / "treestats"

BOXPLOTS_RESULTS_DIR = RESULTS_DIR / "boxplots"
TOPFEATURES_RESULTS_DIR = RESULTS_DIR/ "top_features"
SANITY_RESULTS_DIR = RESULTS_DIR / "sanity_checks"
TREEPLOTS_RESULTS_DIR = RESULTS_DIR/ "treeplots"

SIMULATORS = ["ms", "msprime", "msms", "msms_ms"]
REPS = 5
REC_REPS = 200
NUM_SANITY_CLASSES = 5
SANITY_REPS = 5

FILE_FORMATS = {
    "raw": "{SIM}.txt",
    "stats": "{SIM}_stats.csv",
    "reps": "/rep{REP}",
    "sanity_check_file": "sanity_check_stats.txt"
}

import os
from glob import glob

all_configs = [os.path.splitext(os.path.basename(f))[0] for f in glob(f"{CONFIG_DIR}/*.yaml")]
