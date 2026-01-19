#!/usr/bin/env python3
import argparse
from pathlib import Path
import csv
import sys


def log(msg, log_path):
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def check_csv_consistency(csv_path, delimiter=",", max_rows_logged=20):
    """
    Checks whether all rows in a CSV have the same number of fields.
    Returns:
        status (str)
        expected_fields (int or None)
        bad_rows: list of (line_number, field_count)
    """
    bad_rows = []

    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)

            try:
                header = next(reader)
            except StopIteration:
                return "EMPTY_FILE", None, []

            expected_fields = len(header)

            if expected_fields == 0:
                return "EMPTY_HEADER", expected_fields, []

            for line_no, row in enumerate(reader, start=2):
                if len(row) != expected_fields:
                    bad_rows.append((line_no, len(row)))
                    if len(bad_rows) >= max_rows_logged:
                        break

        if bad_rows:
            return "ROW_LENGTH_MISMATCH", expected_fields, bad_rows

        return "OK", expected_fields, []

    except Exception as e:
        return "EXCEPTION", None, [(type(e).__name__, str(e))]


def main(input_csv, log_path, delimiter):
    status, expected, bad_rows = check_csv_consistency(input_csv, delimiter)

    if status == "OK":
        log("OK", log_path)
        return

    if status == "EMPTY_FILE":
        log("ERROR\tEMPTY_FILE", log_path)
        return

    if status == "EMPTY_HEADER":
        log("ERROR\tEMPTY_HEADER", log_path)
        return

    if status == "ROW_LENGTH_MISMATCH":
        log(f"ERROR\tROW_LENGTH_MISMATCH\texpected_fields={expected}", log_path)
        for line_no, field_count in bad_rows:
            log(f"DETAIL\tline={line_no}\tfields={field_count}", log_path)
        return

    if status == "EXCEPTION":
        err_type, msg = bad_rows[0]
        log(f"ERROR\tEXCEPTION\ttype={err_type}\tmsg={msg}", log_path)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check a single CSV file for column consistency."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="CSV file to check"
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Log file to write results to"
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: ,)"
    )

    args = parser.parse_args()
    main(args.input_csv, args.log, args.delimiter)
