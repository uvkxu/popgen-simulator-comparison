CONFIG_DIR = "configs"
RAW_DATA_DIR = "data/raw"
TREESTATS_DATA_DIR = "data/treestats"

import os
from glob import glob

all_configs = [os.path.splitext(os.path.basename(f))[0] for f in glob(f"{CONFIG_DIR}/*.yaml")]


