#!/usr/bin/env python3

import argparse
import re
import yaml
from pathlib import Path
import csv

# ------------------------------------------------------------
# Regex: matcht NUR das Zeitformat, egal was davor steht
# ------------------------------------------------------------
TIME_REGEX = re.compile(
    r"(\d+)h\s*(\d+)m\s*([\d.]+)s",
    re.IGNORECASE
)

# ------------------------------------------------------------
# Zeit aus Datei extrahieren (robust)
# ------------------------------------------------------------
def extract_time_from_file(path: Path, max_lines: int = 5) -> float:
    """
    Extract simulation time in seconds from the first few lines of a file.
    """
    with path.open() as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            m = TIME_REGEX.search(line)
            if m:
                h, m_, s = m.groups()
                return int(h) * 3600 + int(m_) * 60 + float(s)

    raise ValueError(f"No simulation time found in {path}")

# ------------------------------------------------------------
# YAML Parameter laden
# ------------------------------------------------------------
def load_params(yaml_path: Path) -> dict:
    with yaml_path.open() as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------
# Hauptlogik
# ------------------------------------------------------------
def collect_runtimes(input_dir: Path):
    rows = []

    for config_dir in input_dir.iterdir():
        if not config_dir.is_dir():
            continue

        config_name = config_dir.name

        for param_dir in config_dir.iterdir():
            if not param_dir.is_dir():
                continue

            # YAML im Parameter-Ordner finden
            yaml_files = list(param_dir.glob("*.yaml"))
            if not yaml_files:
                continue

            params = load_params(yaml_files[0])

            rep_dir = param_dir / "rep1"
            if not rep_dir.exists():
                continue

            for txt_file in rep_dir.glob("*_output.txt"):
                simulator = txt_file.name.replace("_output.txt", "")

                try:
                    time_sec = extract_time_from_file(txt_file)
                except ValueError:
                    continue

                rows.append({
                    "config": config_name,
                    "simulator": simulator,
                    "sequence_length": params.get("sequence_length"),
                    "recombination_rate": params.get("recombination_rate"),
                    "overall_samples": params.get("overall_samples"),
                    "time_seconds": round(time_sec, 3),
                })

    return rows

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Collect simulation runtimes from output files"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Base data directory"
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Output CSV file"
    )

    args = parser.parse_args()

    rows = collect_runtimes(args.input_dir)

    if not rows:
        print("No simulation times found.")
        return

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} entries to {args.output_file}")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
