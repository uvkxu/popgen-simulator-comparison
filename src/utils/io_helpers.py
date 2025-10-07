from pathlib import Path
import pandas as pd

def collect_csv_files(input_dir, csv_types=None):
    """
    Recursively collect CSV files from input_dir.
    
    Returns a flat list of Path objects.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    if csv_types is None:
        # collect all CSVs recursively
        csv_files = list(input_dir.rglob("*.csv"))
    else:
        # collect only specified CSV types
        csv_files = []
        for csv_name in csv_types:
            csv_files.extend(input_dir.rglob(csv_name))

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
        fname = file.name.lower()

        for sim in selected_simulators:
            if sim in fname:
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
