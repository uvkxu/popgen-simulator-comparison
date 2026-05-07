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


def process_all_files(root_dir: Path, pattern="*.txt", n_workers=1, use_threads=True):
    # 🔥 Generator statt Liste → kein RAM-Bottleneck
    files = root_dir.rglob(pattern)

    print(f"Scanning files in: {root_dir}")
    print(f"Pattern: {pattern}")

    total_trees = 0
    total_problems = 0
    processed_files = 0

    # 👉 ThreadPool ist für I/O oft besser
    if use_threads:
        from multiprocessing.dummy import Pool  # ThreadPool
        print("Using THREAD pool (I/O optimized)")
    else:
        from multiprocessing import Pool
        print("Using PROCESS pool")

    with Pool(n_workers) as pool:
        for file, trees, problems in tqdm(
            pool.imap_unordered(check_tree_format, files),
            desc="Checking files",
            smoothing=0.1
        ):
            processed_files += 1

            if isinstance(problems, str):
                print(f"{file}: {problems}")
                continue

            total_trees += trees
            total_problems += problems

            if problems > 0:
                print(f"[BAD] {file}: {problems} problems")

            # 🔥 optional: heartbeat für lange Runs
            if processed_files % 10000 == 0:
                print(f"Processed {processed_files} files...")

    print("\n--- SUMMARY ---")
    print(f"Processed files: {processed_files}")
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

if __name__ == "__main__":
    main()