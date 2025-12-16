import re
import yaml
import os
from pathlib import Path
import argparse
import math
import numpy as np
import sys

UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)
from utils.io_helpers import detect_simulator_from_yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from utils.io_helpers import get_combinations, format_combination_as_folder_name
from snakemake_vars import FILE_FORMATS


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
                        merged_haplotypes[k][-1] += haplotypes[k][idx]
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


def analyze_precision(sim_output_path):
    """
    Analyze how many decimal places positions and branch lengths have in an ms-style output file.

    Args:
        sim_output_path (str): Path to the ms output file.

    Returns:
        dict: {
            "positions_decimals": int or None,
            "branchlength_decimals": int or None,
        }
    """
    with open(sim_output_path, "r") as f:
        content = f.read().splitlines()

    position_decimals = []
    branch_decimals = []

    for line in content:
        line = line.strip()

        # --- Positions line ---
        if line.startswith("positions:"):
            floats = line.split()[1:]
            for val in floats:
                if "." in val:
                    position_decimals.append(len(val.split(".")[1]))
                else:
                    position_decimals.append(0)

        # --- Branch lengths in Newick-like tree lines ---
        elif "(" in line and ":" in line:
            matches = re.findall(r":([0-9]*\.?[0-9]+)", line)
            for val in matches:
                if "." in val:
                    branch_decimals.append(len(val.split(".")[1]))
                else:
                    branch_decimals.append(0)

    # Compute max precision (most digits after decimal)
    pos_dec = max(position_decimals) if position_decimals else None
    if not pos_dec:
        pos_dec = math.inf
    br_dec = max(branch_decimals) if branch_decimals else None
    if not br_dec:
        br_dec = math.inf

    return {
        "positions_decimals": pos_dec,
        "branchlength_decimals": br_dec
    }


def process_folder(input_folder, output_folder, usePrecision=True):
    """
    Process all simulation output files in input_folder, preserving folder structure.
    Uses metadata.yaml to determine simulator types and sequence length.
    msprime outputs are copied unchanged but still written to output_dir.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # --- Alle Ordner durchsuchen ---
    for root, dirs, files in os.walk(input_folder):
        root_path = Path(root)

        # --- metadata.yaml laden ---
        metadata_files = [f for f in files if f.endswith(".yaml") or f.endswith(".yml")]
        if not metadata_files:
            continue
        
        metadata_path = root_path / metadata_files[0]
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        # --- Pflichtfelder prüfen ---
        if "sequence_length" not in metadata:
            raise ValueError(f"sequence_length missing in metadata file {metadata_path}")
        if "population_size" not in metadata:
            raise ValueError(f"population_size missing in metadata file {metadata_path}")
        if "simulator_outputs" not in metadata:
            raise ValueError(f"simulator_outputs missing in metadata file {metadata_path}")

        sequence_length = metadata["sequence_length"]
        population_size = metadata["population_size"]
        simulator_outputs = metadata["simulator_outputs"]
        factor = 4 * population_size
        usePrecision = bool(metadata.get("usePrecision", usePrecision))

        # --- Unterordner (z.B. Replikate) durchgehen ---
        for rep_root, rep_dirs, rep_files in os.walk(root_path):
            rep_root_path = Path(rep_root)
            if not rep_files:
                continue

            # --- Precision optional berechnen ---
            pos_prec, branch_prec = None, None
            if usePrecision:
                pos_precisions, branch_precisions = [], []
                for f in rep_files:
                    if not f.endswith(".txt"):
                        continue
                    file_path = rep_root_path / f
                    prec = analyze_precision(file_path)
                    if prec["positions_decimals"] is not None:
                        pos_precisions.append(prec["positions_decimals"])
                    if prec["branchlength_decimals"] is not None:
                        branch_precisions.append(prec["branchlength_decimals"])
                pos_prec = min((p for p in pos_precisions if math.isfinite(p)), default=None)
                branch_prec = min((b for b in branch_precisions if math.isfinite(b)), default=None)
                print(f"Processing folder {rep_root_path.relative_to(input_folder)} "
                      f"(positions_decimals={pos_prec}, branch_decimals={branch_prec})")

            # --- Alle simulatorspezifischen Outputs durchgehen ---
            for sim, filename in simulator_outputs.items():
                file_path = Path(filename)
                input_file = file_path if file_path.is_absolute() else rep_root_path / file_path

                if not input_file.exists():
                    continue

                # --- Ausgabe-Datei ---
                output_file = output_folder / rep_root_path.relative_to(input_folder) / FILE_FORMATS["cleaned"].format(SIM=sim)
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # --- Dateiinhalt lesen ---
                with open(input_file, "r") as infile:
                    content = infile.read()

                cleaned = clean_sim_output(
                    content,
                    sequence_length=sequence_length,
                    position_decimals=pos_prec,
                    branch_decimals=branch_prec,
                    factor=factor,
                    branches_already_scaled=(sim == "msprime" or sim == "slim"),
                )
                # print(f"✅ Cleaned {sim} output: {input_file}")

                # --- Datei ins Output-Verzeichnis schreiben ---
                with open(output_file, "w") as outfile:
                    outfile.write(cleaned)

                # --- Pfad in metadata ergänzen ---
                metadata.setdefault("cleaned_outputs", {})[sim] = str(output_file.name)

        # --- metadata.yaml aktualisiert abspeichern ---
        out_metadata_path = output_folder / root_path.relative_to(input_folder) / metadata_files[0]
        out_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_metadata_path, "w") as f_out:
            yaml.dump(metadata, f_out, sort_keys=False)

    print(f"✅ Finished processing all folders. Cleaned outputs written to {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Clean ms-style simulator outputs in bulk.")
    parser.add_argument("--input", type=Path, required=True, help="Input folder with raw simulation outputs")
    parser.add_argument("--output", type=Path, required=True, help="Output folder for cleaned files")
    parser.add_argument("--skip_precision", action="store_true", help="Skip analyze_precision and use default decimals")
    args = parser.parse_args()

    process_folder(
        input_folder=args.input,
        output_folder=args.output,
        usePrecision=not args.skip_precision
    )

if __name__ == "__main__":
    main()