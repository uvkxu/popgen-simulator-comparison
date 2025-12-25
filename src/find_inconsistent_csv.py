#!/usr/bin/env python3
import argparse
from pathlib import Path
import csv


def check_csv_consistency(csv_path, delimiter=","):
    """
    Checks whether all rows in a CSV have the same number of fields.
    Returns:
        expected_fields (int)
        bad_rows: list of (line_number, field_count)
    """
    bad_rows = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return 0, []

        expected_fields = len(header)

        for line_no, row in enumerate(reader, start=2):
            if len(row) != expected_fields:
                bad_rows.append((line_no, len(row)))

    return expected_fields, bad_rows


def main(input_folder, delimiter):
    csv_files = sorted(Path(input_folder).rglob("*.csv"))

    if not csv_files:
        print("⚠️  No CSV files found.")
        return

    inconsistent = False

    for csv_file in csv_files:
        expected, bad_rows = check_csv_consistency(csv_file, delimiter)

        if bad_rows:
            inconsistent = True
            print("\n❌ Inconsistent CSV:")
            print(f"   File: {csv_file}")
            print(f"   Expected fields: {expected}")
            print(f"   Problematic rows:")

            for line_no, field_count in bad_rows[:10]:
                print(f"     - line {line_no}: {field_count} fields")

            if len(bad_rows) > 10:
                print(f"     ... and {len(bad_rows) - 10} more")

    if not inconsistent:
        print("✅ All CSV files are consistent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find CSV files with inconsistent number of columns."
    )
    parser.add_argument(
        "--input_folder",
        type=Path,
        help="Folder containing CSV files (searched recursively)"
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: ,)"
    )

    args = parser.parse_args()
    main(args.input_folder, args.delimiter)
