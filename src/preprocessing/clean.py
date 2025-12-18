import re
import yaml
import os
from pathlib import Path
import argparse
import math
import numpy as np
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from snakemake_vars import FILE_FORMATS, PRECISION, SIMULATORS


def clean_sim_output(
    sim_output,
    sequence_length,
    branch_decimals=None,
    position_decimals=None,
    factor=1.0,
    branches_already_scaled=False
):
    """
    Clean and normalize ms-style simulator output.

    Args:
        sim_output (str): Raw ms output as a string.
        sequence_length (int): Sequence length for scaling positions.
        branch_decimals (int, optional): Number of decimal places to round branch lengths to before scaling.
        position_decimals (int, optional): Number of decimal places to round variant positions to before scaling.
        factor (float): Branch length scaling factor (default = 1.0).
        branches_already_scaled (bool): Whether branch lengths are already scaled by factor.

    Returns:
        str: Cleaned and normalized ms output.
    """
    lines = sim_output.splitlines()
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # --- Handle segsites + positions + haplotypes ---
        if line.startswith("segsites:"):
            segsites = int(line.split()[1])

            if segsites == 0:
                new_lines.append(line)
                i += 1
                continue

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

            haplotypes = np.array(list(map(list, haplotypes)), dtype=int)  # make mutable

            all_integer_like = np.all(np.abs(positions - np.round(positions)) < 1e-9)

            if all_integer_like:
                # alle Positionen schon ganzzahlig → direkt int casten
                norm_positions = [int(round(p)) for p in positions]
            else:
                # normale Skalierung
                norm_positions = []
                for p in positions:
                    if position_decimals is not None:
                        p = round(p, position_decimals)
                    p = int(round(p * sequence_length))
                    if p == sequence_length:
                        p = sequence_length - 1
                    norm_positions.append(p)

            # --- merge duplicates ---
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

            # --- write results ---
            new_lines.append(f"segsites: {len(merged_positions)}")
            pos_str = " ".join(map(str, merged_positions))
            new_lines.append("positions: " + pos_str)
            for row in merged_haplotypes:
                new_lines.append("".join(map(str, row)))

            i = j
            continue

        # --- Handle tree lines with branch lengths ---
        elif "(" in line and ":" in line:

            def scale_branch(m):
                val = float(m.group(1))

                # Wenn Branches schon skaliert, dann ggf. ent-/wiederskalieren
                if branches_already_scaled:
                    if branch_decimals is not None:
                        val = (val / factor)
                        val = round(val, branch_decimals)
                        val *= factor
                    return f":{val:.{branch_decimals if branch_decimals else 6}f}"

                # Normales Verhalten (Branches müssen skaliert werden)
                else:
                    if branch_decimals is not None:
                        val = round(val, branch_decimals)
                    val *= factor
                    return f":{val:.{branch_decimals if branch_decimals else 6}f}"

            scaled_line = re.sub(r":([0-9]*\.?[0-9]+)", scale_branch, line)
            new_lines.append(scaled_line)

        else:
            new_lines.append(line)

        i += 1

    return "\n".join(new_lines) + "\n"


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

    args = parser.parse_args()

    sim = args.sim

    input_file = args.input
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r") as infile:
        sim_output = infile.read()
    
    combo_yaml = args.combo_yaml
    if not combo_yaml.exists():
        raise FileNotFoundError(f"Combo parameter file not found: {combo_yaml}")

    with combo_yaml.open("r") as f:
        combo_params = yaml.safe_load(f)

    precision = args.precision if bool(combo_params["usePrecision"]) else None

    cleaned_output = clean_sim_output(
        sim_output=sim_output,
        sequence_length=combo_params["sequence_length"],
        branch_decimals=precision,
        position_decimals=precision,
        factor= 4 * combo_params["population_size"],
        branches_already_scaled=(sim == "msprime" or sim == "slim")
    )

    output_file = args.output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as outfile:
        outfile.write(cleaned_output)

if __name__ == "__main__":
    main()