from pathlib import Path
import pandas as pd
import math


def clean_segsites(sim_output, sequence_length=None, decimals=None):
    """
    Clean and normalize ms output.

    Args:
        sim_output (str): Path to the raw output file.
        sequence_length (int, optional): If set, floating-point positions are scaled 
            to integer positions (0..sequence_length). 
        decimals (int, optional): If set, positions are rounded to this many decimal places.
    Returns:
        list[str]: Cleaned ms output lines.
    """
    lines = sim_output.splitlines()

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Handle segsites + positions + haplotypes
        if line.startswith("segsites:"):
            segsites = int(line.split()[1])

            if segsites == 0:
                new_lines.append(line)
                i += 1
                continue

            # --- positions line ---
            pos_line = lines[i + 1].strip()
            assert pos_line.startswith("positions:")
            positions = list(map(float, pos_line.split()[1:]))

            # --- haplotype matrix ---
            haplotypes = []
            j = i + 2
            while j < len(lines):
                hapline = lines[j].strip()
                if not hapline or not set(hapline).issubset({"0", "1"}):
                    break
                haplotypes.append(list(map(int, hapline)))
                j += 1
            haplotypes = list(map(list, haplotypes))  # make mutable

            # print(f"Len of haploytypes: {len(haplotypes)}")

            # --- normalize positions ---
            norm_positions = []
            for p in positions:
                if sequence_length is not None:
                    # scale to discrete int
                    p = math.floor(p * sequence_length)
                    if p == sequence_length:
                        p = sequence_length - 1
                if decimals is not None and sequence_length is None:
                    p = round(p, decimals)
                norm_positions.append(p)

            # --- merge duplicates---
            merged_positions = [norm_positions[0]]
            merged_haplotypes = [[h[0]] for h in haplotypes]  # start with first column

            for idx in range(1, len(norm_positions)):
                p = norm_positions[idx]
                if p == merged_positions[-1]:
                    # same as previous -> OR the bits into last column
                    for k in range(len(haplotypes)):
                        merged_haplotypes[k][-1] |= haplotypes[k][idx]
                else:
                    # new unique position -> add new column
                    merged_positions.append(p)
                    for k in range(len(haplotypes)):
                        merged_haplotypes[k].append(haplotypes[k][idx])

            # write new segsites + positions + haplotypes
            new_lines.append(f"segsites: {len(merged_positions)}")
            pos_str = " ".join(map(str, merged_positions))
            new_lines.append("positions: " + pos_str)
            for row in merged_haplotypes:
                new_lines.append("".join(map(str, row)))

            i = j
            continue
        else:
            new_lines.append(line)

        i += 1

    return "\n".join(new_lines) + "\n"


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
