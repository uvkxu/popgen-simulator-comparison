import argparse
import yaml
import numpy as np
from pathlib import Path
import csv


def count_merges(input_file, sequence_length, position_decimals=None, overall_samples=0):

    individuals_total = 0
    individuals_with_merge = 0
    positions_total = 0
    positions_merged = 0

    with open(input_file, "r") as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = line.strip()

            if line.startswith("segsites:"):
                segsites = int(line.split()[1])

                # --- no mutations ---
                if segsites == 0:
                    individuals_total += overall_samples
                    continue

                positions_total += segsites

                # positions
                pos_line = fin.readline().strip()
                assert pos_line.startswith("positions:")
                positions = list(map(float, pos_line.split()[1:]))

                # haplotypes
                haplotypes = []
                while True:
                    hapline = fin.readline()
                    if not hapline:
                        break
                    hapline = hapline.strip()
                    if not hapline or not hapline.isdigit():
                        break
                    haplotypes.append(list(map(int, hapline)))

                haplotypes = np.array(haplotypes, dtype=int)

                n_ind = haplotypes.shape[0]
                individuals_total += n_ind

                # --- normalize positions ---
                positions = np.array(positions)

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

                # --- count merges ---
                merges = 0
                prev = norm_positions[0]

                for p in norm_positions[1:]:
                    if p == prev:
                        merges += 1
                    else:
                        prev = p

                positions_merged += merges

                if merges > 0:
                    individuals_with_merge += n_ind

    return individuals_total, individuals_with_merge, positions_total, positions_merged


def main():

    parser = argparse.ArgumentParser(
        description="Count how often normalized positions merge."
    )

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combo_yaml", type=Path, required=True)
    parser.add_argument("--precision", type=int, default=None)

    args = parser.parse_args()

    with args.combo_yaml.open("r") as f:
        combo_params = yaml.safe_load(f)

    sequence_length = combo_params["sequence_length"]
    overall_samples = combo_params["overall_samples"]

    individuals_total, individuals_with_merge, positions_total, positions_merged = count_merges(
        input_file=args.input,
        sequence_length=sequence_length,
        position_decimals=args.precision,
        overall_samples=overall_samples
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", newline="") as fout:
        writer = csv.writer(fout)

        writer.writerow([
            "overall_samples",
            "individuals_total",
            "individuals_with_merge",
            "positions_total",
            "positions_merged"
        ])

        writer.writerow([
            overall_samples,
            individuals_total,
            individuals_with_merge,
            positions_total,
            positions_merged
        ])


if __name__ == "__main__":
    main()