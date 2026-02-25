#!/usr/bin/env python3
"""
collect_non_binary_features.py

Scan an input folder recursively for CSV files with a given name/pattern
(default: slim_tree_features.csv), extract columns:
  - non_binary_nodes
  - non_binary_nodes_frac
and write a merged CSV for later plotting (boxplots etc.).

Usage:
  python collect_non_binary_features.py <input_dir> <output_file> [pattern]

Examples:
  python collect_non_binary_features.py results/ merged_non_binary.csv
  python collect_non_binary_features.py results/ merged_non_binary.csv slim_tree_features.csv
  python collect_non_binary_features.py results/ merged_non_binary.csv "*tree_features*.csv"
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Optional


REQUIRED_COLS = ("non_binary_nodes", "non_binary_nodes_frac")


def find_csv_files(input_dir: Path, pattern: str) -> List[Path]:
    # Allow either exact filename (default) or glob pattern
    # We search recursively.
    if any(ch in pattern for ch in "*?[]"):
        # Glob pattern
        paths = glob.glob(str(input_dir / "**" / pattern), recursive=True)
    else:
        # Exact filename
        paths = glob.glob(str(input_dir / "**" / pattern), recursive=True)
    return [Path(p) for p in paths if Path(p).is_file()]


def safe_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    if x == "" or x.upper() == "NA" or x.upper() == "NAN":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def read_required_columns(csv_path: Path) -> Tuple[List[Dict[str, object]], Optional[str]]:
    """
    Returns (rows, error). If error != None, rows will be empty.
    Each row contains required cols + source_file.
    """
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return [], "no header"

            missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
            if missing:
                return [], f"missing columns: {', '.join(missing)}"

            out: List[Dict[str, object]] = []
            for row in reader:
                out.append(
                    {
                        "non_binary_nodes": safe_float(row.get("non_binary_nodes", "")),
                        "non_binary_nodes_frac": safe_float(row.get("non_binary_nodes_frac", "")),
                        "source_file": str(csv_path),
                    }
                )
            return out, None
    except Exception as e:
        return [], f"read error: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path, help="Input directory to scan recursively")
    ap.add_argument("output_file", type=Path, help="Output CSV path to write merged rows")
    ap.add_argument(
        "pattern",
        nargs="?",
        default="slim_tree_features.csv",
        help="Filename or glob pattern to match (default: slim_tree_features.csv)",
    )
    args = ap.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {args.input_dir}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    files = find_csv_files(args.input_dir, args.pattern)
    if not files:
        raise SystemExit(f"No files found matching '{args.pattern}' under {args.input_dir}")

    merged_rows: List[Dict[str, object]] = []
    kept = 0
    skipped = 0

    for p in sorted(files):
        rows, err = read_required_columns(p)
        if err is not None:
            skipped += 1
            print(f"[skip] {p} ({err})")
            continue
        kept += 1
        merged_rows.extend(rows)

    if not merged_rows:
        raise SystemExit("No usable rows collected (all files skipped or empty).")

    # Write merged CSV
    with args.output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(REQUIRED_COLS) + ["source_file"])
        writer.writeheader()
        for r in merged_rows:
            # Keep None as empty in CSV
            writer.writerow(
                {
                    "non_binary_nodes": "" if r["non_binary_nodes"] is None else r["non_binary_nodes"],
                    "non_binary_nodes_frac": "" if r["non_binary_nodes_frac"] is None else r["non_binary_nodes_frac"],
                    "source_file": r["source_file"],
                }
            )

    # Quick summary (ignoring Nones)
    nodes = [r["non_binary_nodes"] for r in merged_rows if r["non_binary_nodes"] is not None]
    frac = [r["non_binary_nodes_frac"] for r in merged_rows if r["non_binary_nodes_frac"] is not None]

    def qstats(vals: List[float]) -> str:
        if not vals:
            return "no numeric values"
        vals2 = sorted(vals)
        n = len(vals2)
        med = vals2[n // 2] if n % 2 == 1 else 0.5 * (vals2[n // 2 - 1] + vals2[n // 2])
        return f"min={vals2[0]} median={med} max={vals2[-1]} (n={n})"

    print(f"\n✅ Wrote: {args.output_file}")
    print(f"   Files scanned: {len(files)} | kept: {kept} | skipped: {skipped}")
    print(f"   Rows collected: {len(merged_rows)}")
    print("\nQuick summary:")
    print(f"  non_binary_nodes:      {qstats(nodes)}")
    print(f"  non_binary_nodes_frac: {qstats(frac)}")


if __name__ == "__main__":
    main()