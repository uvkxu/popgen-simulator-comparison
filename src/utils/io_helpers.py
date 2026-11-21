from pathlib import Path
import pandas as pd
import math
import itertools
import yaml
import os 
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from simulators import *

CONFIG_ORDER = [
    "one_population",
    "zigzag",
    "li_african3epoch",
    "lavretsky_ducks",
    "locke_orangutan",
    "gutenkunst_ooa",
    "browning_america",
    "pawar_gorilla",
    "freedman_wolf",
    "kamm_westerneurasia",
    "jacobs_papuans",
]


def log_step(master_log_file, step_name, combo, simulator, success_inc, failed_inc, info: str = ""):
    """
    Log a step in the master log file, counting successes and failures.
    
    Columns: step_name | combo | simulator | success | failed | info
    """
    master_log_file.parent.mkdir(parents=True, exist_ok=True)
    lines = {}

    # Load existing log
    if master_log_file.exists():
        with open(master_log_file, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split("|")
                    # remove surrounding whitespace
                    parts = [p.strip() for p in parts]
                    if len(parts) < 5:
                        continue  # malformed line
                    key = (parts[0], parts[1], parts[2])  # (step_name, combo, simulator)
                    succ = int(parts[3])
                    fail = int(parts[4])
                    info_old = parts[5] if len(parts) > 5 else ""
                    lines[key] = [succ, fail, info_old]

    # Update or add
    key = (step_name, combo, simulator)
    if key in lines:
        lines[key][0] += success_inc
        lines[key][1] += failed_inc
        if info:
            if lines[key][2]:
                lines[key][2] += ", " + info
            else:
                lines[key][2] = info
    else:
        lines[key] = [success_inc, failed_inc, info]

    # Write back
    with open(master_log_file, "w") as f:
        for (step, cmb, sim), (succ, fail, info_text) in lines.items():
            f.write(f"{step} | {cmb} | {sim} | {succ} | {fail} | {info_text}\n")


def detect_simulator_from_yaml(input_file, metadata=None):
    """
    Detects the simulator used for a given input file based on a YAML metadata dict.

    Args:
        metadata (dict): Loaded YAML configuration as a Python dictionary.
        input_file (str): Name or path of the input file to check.

    Returns:
        str: Name of the simulator, or None if not detected.
    """
    input_file_name = Path(input_file).name  # just the filename
    
    # 1. Check all mappings in the YAML that have dict values
    if metadata:
        for key, mapping in metadata.items():
            if isinstance(mapping, dict):
                for simulator, file_name in mapping.items():
                    if Path(file_name).name == input_file_name:
                        return simulator

    # Be cautious - it is not as stable as the mapping
    for sim in sorted(SIMULATORS, key=len, reverse=True):
        if f"{sim}_" in input_file_name:
            return sim
    
    # 3. No match found
    return None


def collect_csv_files(input_dir):
    """
    Recursively collect CSV files from input_dir.
    
    Returns a flat list of Path objects.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    # collect all CSVs recursively
    csv_files = list(input_dir.rglob("*.csv"))

    if not csv_files:
        return None

    return csv_files


def assign_simulator_labels(csv_files, selected_simulators):
    """
    Load CSVs and add a 'simulator' column based on filename.
    Expects a list of Path objects.
    """
    if csv_files is None:
        return []

    all_dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        sim = detect_simulator_from_yaml(
                input_file=file
            )

        if sim in selected_simulators:
            df['simulator'] = sim
            all_dfs.append(df)

    return all_dfs