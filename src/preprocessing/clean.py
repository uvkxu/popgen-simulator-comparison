import re
import yaml
import os
from pathlib import Path
import argparse
import math
import numpy as np
import sys
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from snakemake_vars import FILE_FORMATS, PRECISION, SIMULATORS


def clean_sim_output(input_file, output_file, sequence_length,
                            branch_decimals=None, position_decimals=None,
                            factor=1.0, branches_already_scaled=False):

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        lines_buffer = []
        while True:
            line = fin.readline()
            if not line:
                break
            line = line.strip()

            # --- Handle segsites + positions + haplotypes ---
            if line.startswith("segsites:"):
                segsites = int(line.split()[1])
                if segsites == 0:
                    fout.write(line + "\n")
                    continue

                # Positionen
                pos_line = fin.readline().strip()
                assert pos_line.startswith("positions:")
                positions = list(map(float, pos_line.split()[1:]))

                # Haplotypen
                haplotypes = []
                while True:
                    hapline = fin.readline()
                    if not hapline:  # EOF
                        break
                    hapline = hapline.strip()
                    if not hapline or not hapline.isdigit():
                        break  # Ende des Blocks
                    haplotypes.append(list(map(int, hapline)))

                haplotypes = np.array(haplotypes, dtype=int)

                # Binarize haplotypes to ensure comparability across simulators
                # (SLiM allows stacked mutations, ms/msprime do not)
                haplotypes[haplotypes > 1] = 1

                # Normalisierung der Positionen
                all_integer_like = np.all(np.abs(positions - np.round(positions)) < 1e-9)
                if all_integer_like:
                    norm_positions = [int(round(p)) for p in positions]
                else:
                    norm_positions = []
                    for p in positions:
                        if position_decimals is not None:
                            p = round(p, position_decimals)
                        p = int(round(p * sequence_length))
                        if p == sequence_length:
                            p = sequence_length - 1
                        norm_positions.append(p)

                # Merge duplicates
                merged_positions = [norm_positions[0]]
                merged_haplotypes = [[h[0]] for h in haplotypes]
                for idx in range(1, len(norm_positions)):
                    p = norm_positions[idx]
                    if p == merged_positions[-1]:
                        for k in range(len(haplotypes)):
                            merged_haplotypes[k][-1] |= haplotypes[k][idx]
                    else:
                        merged_positions.append(p)
                        for k in range(len(haplotypes)):
                            merged_haplotypes[k].append(haplotypes[k][idx])

                # Schreiben
                fout.write(f"segsites: {len(merged_positions)}\n")
                fout.write("positions: " + " ".join(map(str, merged_positions)) + "\n")
                for row in merged_haplotypes:
                    fout.write("".join(map(str, row)) + "\n")

            # --- Handle tree lines with branch lengths ---
            elif "(" in line and ":" in line:
                def scale_branch(m):
                    val = float(m.group(1))
                    if branches_already_scaled:
                        if branch_decimals is not None:
                            val = (val / factor)
                            val = round(val, branch_decimals)
                            val *= factor
                        return f":{val:.{branch_decimals if branch_decimals else 6}f}"
                    else:
                        if branch_decimals is not None:
                            val = round(val, branch_decimals)
                        val *= factor
                        return f":{val:.{branch_decimals if branch_decimals else 6}f}"
                scaled_line = re.sub(r":([0-9]*\.?[0-9]+)", scale_branch, line) if factor != 1 else line
                fout.write(scaled_line + "\n")

            else:
                fout.write(line + "\n")



def main():
    parser = argparse.ArgumentParser(
        description="Clean ms-style simulator outputs in bulk."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing raw simulation outputs"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where cleaned files will be written"
    )

    parser.add_argument(
        "--combo_yaml",
        type=Path,
        required=True,
        help="Yaml file, which contains all combo parameters"
    )

    parser.add_argument(
        "--sim",
        type=str,
        required=True,
        choices=SIMULATORS,
        help="Simulator used to generate the input files"
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=PRECISION,
        help=f"Decimal precision for numerical outputs (default: {PRECISION})"
    )

    parser.add_argument(
        "--copy_file",
        type=Path,
        default=None,
        help="If raw file is just wished to be copied, put it here"
    )

    args = parser.parse_args()
    output_file = args.output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    copy_file = args.copy_file

    if copy_file is not None and copy_file.exists():
        shutil.copy(copy_file, output_file)
        print(f"Copied existing file {copy_file} -> {output_file}")
        sys.exit(0) 

    sim = args.sim

    input_file = args.input
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    combo_yaml = args.combo_yaml
    if not combo_yaml.exists():
        raise FileNotFoundError(f"Combo parameter file not found: {combo_yaml}")

    with combo_yaml.open("r") as f:
        combo_params = yaml.safe_load(f)

    precision = args.precision if bool(combo_params["usePrecision"]) else None
    
    factor = combo_params.get("population_size", 0.25)
    factor = 4 * factor if factor else 1

    clean_sim_output(
        input_file=input_file,
        output_file=output_file,
        sequence_length=combo_params["sequence_length"],
        branch_decimals=precision,
        position_decimals=precision,
        factor=factor,
        branches_already_scaled=(sim == "msprime" or sim == "slim" or "slim" in sim)
    )

if __name__ == "__main__":
    main()