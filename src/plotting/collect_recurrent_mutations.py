#!/usr/bin/env python3
"""
Parallel collection of recurrent mutation statistics from SLiM outputs.

Unit of analysis: combo directory (contains combo_parameters.yaml).

Usage:
  python collect_recurrent_mutations_parallel.py <input_dir> <output_csv> [--workers N]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml


SLIM_FILE = "slim_output.txt"


def read_overall_samples(yaml_path: Path) -> int:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "overall_samples" not in data:
        raise ValueError(f"overall_samples not found in {yaml_path}")
    return int(data["overall_samples"])


def parse_slim_output(path: Path, overall_samples: int) -> Tuple[int, int, int, int]:
    """
    Returns:
      (num_individuals,
       num_with_recurrent_mutation,
       total_segsites,
       positions_with_gt1)
    """
    total_individuals = 0
    total_recurrent = 0
    total_segsites = 0
    positions_with_gt1 = 0

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("segsites:"):
            segsites = int(line.split(":")[1].strip())
            total_segsites += segsites
            total_individuals += overall_samples

            i += 1

            # skip positions line
            if i < len(lines) and lines[i].startswith("positions:"):
                i += 1

            if segsites == 0:
                continue

            rows = []
            counted = 0

            while i < len(lines):
                row = lines[i].strip()

                if not row or row.startswith("//") or row.startswith("segsites:"):
                    break

                # check if this individual has a recurrent mutation
                if any(ch not in "01" for ch in row):
                    total_recurrent += 1

                rows.append(row)
                counted += 1
                i += 1

            if counted != overall_samples:
                print(
                    f"[warn] {path}: expected {overall_samples} individuals, "
                    f"found {counted}"
                )

            # check positions for values >1
            for col in range(segsites):
                column_values = [r[col] for r in rows if len(r) > col]

                if any(v not in "01" for v in column_values):
                    positions_with_gt1 += 1

            continue

        i += 1

    return total_individuals, total_recurrent, total_segsites, positions_with_gt1


def process_combo(combo_dir: Path) -> Dict[str, object]:
    """
    Worker function: process a single combo directory.
    """
    yaml_path = combo_dir / "combo_parameters.yaml"
    overall_samples = read_overall_samples(yaml_path)

    total_individuals = 0
    total_recurrent = 0
    total_segsites = 0
    total_positions_gt1 = 0

    for rep_dir in combo_dir.glob("rep*"):
        slim_path = rep_dir / SLIM_FILE
        if not slim_path.exists():
            continue

        n_ind, n_rec, segsites, pos_gt1 = parse_slim_output(
            slim_path, overall_samples
        )

        total_individuals += n_ind
        total_recurrent += n_rec
        total_segsites += segsites
        total_positions_gt1 += pos_gt1

    frac = (
        total_recurrent / total_individuals
        if total_individuals > 0
        else ""
    )

    return {
        "config_dir": combo_dir.parent.name,
        "combo_dir": combo_dir.name,
        "overall_samples": overall_samples,
        "num_individuals_total": total_individuals,
        "num_individuals_with_recurrent_mutation": total_recurrent,
        "frac_with_recurrent_mutation": frac,
        "total_segsites": total_segsites,
        "positions_with_gt1": total_positions_gt1,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path, help="Root directory (e.g. data/raw)")
    ap.add_argument("output_csv", type=Path, help="Output CSV file")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.input_dir}")

    combo_dirs = sorted(
        p.parent
        for p in args.input_dir.rglob("combo_parameters.yaml")
    )

    if not combo_dirs:
        raise SystemExit("No combo_parameters.yaml found.")

    rows = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_combo, combo): combo
            for combo in combo_dirs
        }

        for fut in as_completed(futures):
            combo = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"[error] {combo}: {e}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config_dir",
                "combo_dir",
                "overall_samples",
                "num_individuals_total",
                "num_individuals_with_recurrent_mutation",
                "frac_with_recurrent_mutation",
                "total_segsites",
                "positions_with_gt1",
            ],
        )

        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda r: (r["config_dir"], r["combo_dir"]))
        )

    print(f"✅ Wrote {args.output_csv}")
    print(f"   Combos processed: {len(rows)}")


if __name__ == "__main__":
    main()