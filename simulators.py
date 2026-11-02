from pathlib import Path

import yaml
from project_paths import SIMULATORS_CONFIG_FILE

with SIMULATORS_CONFIG_FILE.open(encoding="utf-8") as f:
    config = yaml.safe_load(f)
  

# Simulator-IDs als Konstanten
MS = config["simulators"]["MS"]
MSMS = config["simulators"]["MSMS"]
MSPMS = config["simulators"]["MSPMS"]
MSPRIME = config["simulators"]["MSPRIME"]
SLIM = config["simulators"]["SLIM"]

SIMULATORS = [MS, MSMS, MSPMS, MSPRIME, SLIM]