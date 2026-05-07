from pathlib import Path
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse


def check_tree_format(file_path: Path):
    problems = 0
    trees = 0

    try:
        with open(file_path, "r") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()

                if (
                    not line
                    or line.startswith("//")
                    or line.startswith("segsites")
                    or line.startswith("positions")
                ):
                    continue

                if "(" in line:
                    trees += 1

                    norm = re.sub(r";+$", ";", line)

                    if norm.count(";") != 1:
                        problems += 1

                    if not norm.endswith(";"):
                        problems += 1

                    if norm.count("(") != norm.count(")"):
                        problems += 1

        return (str(file_path), trees, problems)

    except Exception as e:
        return (str(file_path), 0, f"ERROR: {e}")


def process_all_files(root_dir: Path, pattern="*.txt", n_workers=1):
    files = list(root_dir.rglob(pattern))
    print(f"Found {len(files)} files")

    total_trees = 0
    total_problems = 0

    with Pool(n_workers) as pool:
        # 🔥 tqdm + imap → progress bar funktioniert sauber
        for file, trees, problems in tqdm(
            pool.imap(check_tree_format, files),
            total=len(files),
            desc="Checking files"
        ):
            if isinstance(problems, str):
                print(f"{file}: {problems}")
                continue

            total_trees += trees
            total_problems += problems

            if problems > 0:
                print(f"[BAD] {file}: {problems} problems")

    print("\n--- SUMMARY ---")
    print(f"Total trees: {total_trees}")
    print(f"Total problems: {total_problems}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count() // 2)"
    )
    parser.add_argument("--input_folder", type=Path, required=True, help="Input folder containing .txt files")

    return parser.parse_args()

def main():
    args = parse_args()
    n_workers = args.n_workers or max(1, cpu_count() // 2)
    print(f"Using {n_workers} workers")

    process_all_files(args.input_folder, n_workers=n_workers)