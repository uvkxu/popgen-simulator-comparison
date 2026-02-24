#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from pandas.errors import ParserError
from tqdm import tqdm


TOKENIZE_PATTERN = re.compile(r"Expected\s+\d+\s+fields\s+in\s+line\s+\d+,\s+saw\s+\d+")


def is_tokenize_mismatch(err_msg: str) -> bool:
    return bool(TOKENIZE_PATTERN.search(err_msg))


def iter_files(root: Path, exts: list[str]) -> list[Path]:
    files: list[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(set(files))


def check_file(args_tuple: tuple[str, str | None, str | None, int | None]) -> tuple[str, bool, str]:
    """
    Worker-friendly function (must be top-level for multiprocessing).
    Returns: (filepath, is_bad, error_msg)
    """
    path_str, sep, encoding, nrows = args_tuple
    path = Path(path_str)

    try:
        read_kwargs = {}
        if sep is not None:
            read_kwargs["sep"] = sep
        if encoding is not None:
            read_kwargs["encoding"] = encoding
        if nrows is not None:
            read_kwargs["nrows"] = nrows

        # Keep default C engine to reproduce the exact tokenizer error.
        pd.read_csv(path, **read_kwargs)
        return (path_str, False, "")
    except ParserError as e:
        msg = str(e)
        if is_tokenize_mismatch(msg):
            return (path_str, True, msg)
        return (path_str, False, "")
    except Exception:
        # Ignore other errors by default (only flag the specific mismatch error)
        return (path_str, False, "")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parallel scan for pandas tokenizing ParserError (Expected fields ... saw ...), with a progress bar."
    )
    ap.add_argument("--input_dir", required=True, help="Folder to scan (recursively).")
    ap.add_argument("--output_file", required=True, help="Path to write results to.")
    ap.add_argument(
        "--ext",
        default=".csv",
        help="Comma-separated file extensions (default: .csv). Example: .csv,.tsv",
    )
    ap.add_argument("--sep", default=None, help="Optional delimiter for pandas (e.g. ',' or ';').")
    ap.add_argument("--encoding", default=None, help="Optional encoding (e.g. 'utf-8', 'latin1').")
    ap.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional: only read first N rows (speed). WARNING: might miss errors after row N.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: os.cpu_count()).",
    )
    ap.add_argument(
        "--write_error_message",
        action="store_true",
        help="Also write the pandas error message after each file path.",
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    exts = [e.strip() for e in args.ext.split(",") if e.strip()]

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input dir does not exist or is not a directory: {input_dir}")

    files = iter_files(input_dir, exts)
    tasks = [(str(p), args.sep, args.encoding, args.nrows) for p in files]

    bad_files: list[tuple[str, str]] = []

    # Use processes (not threads) for robustness with heavy parsing
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(check_file, t) for t in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning files", unit="file"):
            path_str, is_bad, msg = fut.result()
            if is_bad:
                bad_files.append((path_str, msg))

    bad_files.sort(key=lambda x: x[0])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out:
        for path_str, msg in bad_files:
            if args.write_error_message:
                out.write(f"{path_str}\t{msg}\n")
            else:
                out.write(f"{path_str}\n")

    print(f"Scanned: {len(files)} files")
    print(f"Tokenizing mismatches found: {len(bad_files)}")
    print(f"Wrote results to: {output_file}")


if __name__ == "__main__":
    main()