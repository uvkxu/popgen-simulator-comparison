#!/usr/bin/env python3
"""
collect_recurrent_mutations.py

Scan config directories for SLiM outputs and count:
  - total number of individuals
  - number of individuals with recurrent mutations (any allele > 1)

Directory structure (assumed):
data/raw/<config-dir>/
  combo_parameters.yaml   (contains overall_samples)
  rep*/slim_output.txt

Usage:
  python collect_recurrent_mutations.py <input_dir> <output_csv>
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Tuple
import yaml


SLIM_FILE = "slim_output.txt"


def read_overall_samples(yaml_path: Path) -> int | None:
    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("overall_samples")
    except Exception:
        return None


def parse_slim_output(path: Path, overall_samples: int) -> tuple[int, int]:
    """
    Returns:
      (num_individuals, num_with_recurrent_mutation)
    """
    total_individuals = 0
    total_recurrent = 0

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("segsites:"):
            segsites = int(line.split(":")[1].strip())

            # Every block has overall_samples individuals
            total_individuals += overall_samples

            i += 1  # move past segsites

            # skip positions line if present
            if i < len(lines) and lines[i].startswith("positions:"):
                i += 1

            if segsites == 0:
                # no mutations at all → no recurrent possible
                continue

            counted = 0
            while i < len(lines):
                row = lines[i].strip()
                if not row or row.startswith("//") or row.startswith("segsites:"):
                    break

                if any(ch not in "01" for ch in row):
                    total_recurrent += 1

                counted += 1
                i += 1

            # sanity check (optional)
            if counted != overall_samples:
                print(
                    f"[warn] {path}: expected {overall_samples} individuals, "
                    f"found {counted}"
                )
            continue

        i += 1

    return total_individuals, total_recurrent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path, help="Root directory containing config dirs")
    ap.add_argument("output_csv", type=Path, help="Output CSV file")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.input_dir}")

    rows = []

    for config_dir in sorted(p for p in args.input_dir.iterdir() if p.is_dir()):
        yaml_path = config_dir / "combo_parameters.yaml"
        overall_samples = read_overall_samples(yaml_path)

        total_individuals = 0
        total_recurrent = 0

        for rep_dir in config_dir.glob("rep*"):
            slim_path = rep_dir / SLIM_FILE
            if not slim_path.exists():
                continue

            n_ind, n_rec = parse_slim_output(slim_path)
            total_individuals += n_ind
            total_recurrent += n_rec

        if total_individuals == 0:
            frac = ""
        else:
            frac = total_recurrent / total_individuals

        rows.append(
            {
                "config_dir": config_dir.name,
                "overall_samples": overall_samples if overall_samples is not None else "",
                "num_individuals_total": total_individuals,
                "num_individuals_with_recurrent_mutation": total_recurrent,
                "frac_with_recurrent_mutation": frac,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config_dir",
                "overall_samples",
                "num_individuals_total",
                "num_individuals_with_recurrent_mutation",
                "frac_with_recurrent_mutation",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Wrote {args.output_csv}")
    print(f"   Configs processed: {len(rows)}")


if __name__ == "__main__":
    main()