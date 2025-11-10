from pathlib import Path

CONFIG_DIR = Path("configs")
CMD_DIR = Path("commands")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TREESTATS_DATA_DIR = PROCESSED_DATA_DIR / "treestats"
ALLELSTATS_DATA_DIR = PROCESSED_DATA_DIR / "allelstats"

BOXPLOTS_RESULTS_DIR = RESULTS_DIR / "boxplots"
TOPFEATURES_RESULTS_DIR = RESULTS_DIR/ "top_features"
TREESTATS_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "treestats"
ALLELSTATS_TOPFEATURES_DIR = TOPFEATURES_RESULTS_DIR/ "allelstats"
TREESTATS_BOXPLOTS_DIR = BOXPLOTS_RESULTS_DIR/ "treestats"
ALLELSTATS_BOXPLOTS_DIR = BOXPLOTS_RESULTS_DIR/ "allelstats"

SANITY_RESULTS_DIR = RESULTS_DIR / "sanity_checks"

CMD_SUBDIR = "commands"

SIMULATORS = ["ms", "msprime", "msms"] #msms_ms behaves the same as just msms
REPS = 10
REC_REPS = 50
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
