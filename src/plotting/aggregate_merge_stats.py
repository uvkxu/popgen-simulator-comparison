import argparse
from pathlib import Path
import csv
from collections import defaultdict
import os
import sys

UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from utils.io_helpers import detect_simulator_from_yaml


def main():

    parser = argparse.ArgumentParser(
        description="Aggregate merge statistics."
    )

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_overall", type=Path, required=True)
    parser.add_argument("--output_by_sample_size", type=Path, required=True)
    parser.add_argument("--simulator", type=str, required=True)

    args = parser.parse_args()

    input_dir = args.input

    overall = {
        "individuals_total": 0,
        "individuals_with_merge": 0,
        "positions_total": 0,
        "positions_merged": 0
    }

    grouped = defaultdict(lambda: {
        "individuals_total": 0,
        "individuals_with_merge": 0,
        "positions_total": 0,
        "positions_merged": 0
    })

    csv_files = list(input_dir.rglob("*.csv"))

    if not csv_files:
        raise RuntimeError(f"No CSV files found in {input_dir}")

    for file in csv_files:
        sim = detect_simulator_from_yaml(file)
        if sim != args.simulator:
            continue

        with open(file, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:

                sample_size = int(row["overall_samples"])

                it = int(row["individuals_total"])
                im = int(row["individuals_with_merge"])
                pt = int(row["positions_total"])
                pm = int(row["positions_merged"])

                # overall aggregation
                overall["individuals_total"] += it
                overall["individuals_with_merge"] += im
                overall["positions_total"] += pt
                overall["positions_merged"] += pm

                # grouped aggregation
                grouped[sample_size]["individuals_total"] += it
                grouped[sample_size]["individuals_with_merge"] += im
                grouped[sample_size]["positions_total"] += pt
                grouped[sample_size]["positions_merged"] += pm

    # ---- write overall ----

    args.output_overall.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_overall, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "individuals_total",
            "individuals_with_merge",
            "positions_total",
            "positions_merged"
        ])

        writer.writerow([
            overall["individuals_total"],
            overall["individuals_with_merge"],
            overall["positions_total"],
            overall["positions_merged"]
        ])

    # ---- write grouped ----

    args.output_by_sample_size.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_by_sample_size, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "overall_samples",
            "individuals_total",
            "individuals_with_merge",
            "positions_total",
            "positions_merged"
        ])

        for sample_size in sorted(grouped):
            g = grouped[sample_size]

            writer.writerow([
                sample_size,
                g["individuals_total"],
                g["individuals_with_merge"],
                g["positions_total"],
                g["positions_merged"]
            ])


if __name__ == "__main__":
    main()