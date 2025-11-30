from pathlib import Path
import pandas as pd
import math


def detect_simulator_from_yaml(metadata, input_file, simulators=None):
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
    for key, mapping in metadata.items():
        if isinstance(mapping, dict):
            for simulator, file_name in mapping.items():
                if Path(file_name).name == input_file_name:
                    return simulator
    
    # 2. Check if any simulator name appears in the input file name
    if not simulators:
        simulators = metadata.get('simulators', [])

    # Be cautious - it is not as stable as the mapping
    for sim in sorted(simulators, key=len, reverse=True):
        if sim.lower() in input_file_name:
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


def assign_simulator_labels(metadata, csv_files, selected_simulators):
    """
    Load CSVs and add a 'simulator' column based on filename.
    Expects a list of Path objects.
    """
    if csv_files is None:
        return []

    all_dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        sim = detect_simulator_from_yaml(metadata, file, selected_simulators)

        if selected_simulators is None or sim in selected_simulators:
            df['simulator'] = sim
            all_dfs.append(df)

    return all_dfs


def assign_rep_labels(csv_files, num_reps=2):
    """
    Add a 'rep' column to CSV files based on the parent 'repX' folder.
    Only include files where rep number <= num_reps.
    
    Parameters:
        csv_files (list of Path): list of CSV file paths
        num_reps (int): maximum number of repetitions to include
    
    Returns:
        list of pd.DataFrame: each DataFrame has an additional 'rep' column
    """
    if not csv_files:
        return []

    all_dfs = []

    for file in csv_files:
        # Parent folder (assume structure .../repX/filename.csv)
        parent_folder = file.parent.name.lower()

        if parent_folder.startswith("rep"):
            try:
                rep_num = int(parent_folder.replace("rep", ""))
            except ValueError:
                continue  # skip if rep folder doesn't have a number
            if rep_num > num_reps:
                continue  # skip if rep > allowed
        else:
            continue  # skip files not in a rep folder

        df = pd.read_csv(file)
        df['rep'] = rep_num
        all_dfs.append(df)
    
    return all_dfs
