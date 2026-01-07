#!/usr/bin/env python3
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
from tqdm import tqdm

# -----------------------------
# CSV Consistency Checker
# -----------------------------
def check_csv_consistency(csv_path, delimiter=",", max_rows_logged=20):
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


# -----------------------------
# Worker function for Pool
# -----------------------------
def worker(csv_path_str):
    csv_path = Path(csv_path_str)
    status, expected, bad_rows = check_csv_consistency(csv_path)

    if status == "OK":
        return None  # OK wird nicht geschrieben

    details_list = []

    if status == "ROW_LENGTH_MISMATCH":
        details_list.append(f"expected_fields={expected}")
        details_list.extend(f"line={ln},fields={fc}" for ln, fc in bad_rows)

    elif status == "EMPTY_FILE":
        details_list.append("empty_file")

    elif status == "EMPTY_HEADER":
        details_list.append(f"empty_header, expected_fields={expected}")

    elif status == "EXCEPTION":
        err_type, msg = bad_rows[0]
        details_list.append(f"{err_type}: {msg}")

    # Alles als ein TSV-String
    details_str = "; ".join(details_list)
    return f"{csv_path}\t{status}\t{details_str}"


# -----------------------------
# Main
# -----------------------------
def main(csv_dir, output_file, n_processes=None):
    csv_dir = Path(csv_dir)
    csv_files = list(csv_dir.rglob("*.csv"))

    output_file = Path(output_file)
    # Ordner automatisch erstellen, falls er nicht existiert
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if n_processes is None:
        n_processes = cpu_count()  # nutzt automatisch alle verfügbaren CPU-Kerne

    print(f"Found {len(csv_files)} CSV files. Using {n_processes} processes...")

    with Pool(n_processes) as pool, open(output_file, "w", encoding="utf-8") as out:
        out.write("csv_path\tstatus\tdetails\n")
        # Fortschrittsbalken mit tqdm
        for result in tqdm(pool.imap_unordered(worker, [str(f) for f in csv_files], chunksize=100),
                   total=len(csv_files),
                   desc="Checking CSVs"):
            if result is not None:  # nur Fehler schreiben
                out.write(result + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel CSV consistency checker")
    parser.add_argument("csv_dir", help="Directory containing CSV files")
    parser.add_argument("output_file", help="TSV file to write results")
    parser.add_argument("--processes", type=int, default=None, help="Number of parallel processes (default: all cores)")
    args = parser.parse_args()
    main(args.csv_dir, args.output_file, args.processes)