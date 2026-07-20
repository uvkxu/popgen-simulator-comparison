#!/usr/bin/env python3

import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def collect_files(root: Path):
    """
    Liefert alle CSV-Dateien relativ zum Root-Verzeichnis.
    """
    return {
        f.relative_to(root)
        for f in root.rglob("*.csv")
    }


def count_lines(file_path: Path):
    """
    Zählt die Zeilen einer Datei.
    """
    try:
        with open(file_path, "rb") as f:
            return sum(1 for _ in f)
    except Exception as e:
        return e


def compare_file(root1: Path, root2: Path, rel_path: Path):
    f1 = root1 / rel_path
    f2 = root2 / rel_path

    c1 = count_lines(f1)
    c2 = count_lines(f2)

    if isinstance(c1, Exception):
        return ("ERROR", rel_path, str(c1))

    if isinstance(c2, Exception):
        return ("ERROR", rel_path, str(c2))

    if c1 == c2:
        return ("OK", rel_path, c1)
    else:
        return ("DIFF", rel_path, c1, c2)


def main():
    parser = argparse.ArgumentParser(
        description="Vergleicht zwei Ordnerstrukturen und CSV-Zeilenzahlen."
    )
    parser.add_argument("folder1")
    parser.add_argument("folder2")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() * 2,
        help="Anzahl paralleler Worker",
    )

    args = parser.parse_args()

    root1 = Path(args.folder1)
    root2 = Path(args.folder2)

    if not root1.is_dir():
        print(f"Ordner existiert nicht: {root1}")
        return

    if not root2.is_dir():
        print(f"Ordner existiert nicht: {root2}")
        return

    files1 = collect_files(root1)
    files2 = collect_files(root2)

    only1 = sorted(files1 - files2)
    only2 = sorted(files2 - files1)
    common = sorted(files1 & files2)

    print("=" * 60)
    print("Ordnervergleich")
    print("=" * 60)

    if only1:
        print("\nNur in Ordner 1:")
        for f in only1:
            print("  ", f)

    if only2:
        print("\nNur in Ordner 2:")
        for f in only2:
            print("  ", f)

    print(f"\nGemeinsame CSV-Dateien: {len(common)}")

    print("\nVergleiche Zeilenzahlen...")

    ok = 0
    diff = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(compare_file, root1, root2, rel): rel
            for rel in common
        }

        for future in as_completed(futures):
            result = future.result()

            if result[0] == "OK":
                ok += 1

            elif result[0] == "DIFF":
                diff += 1
                _, rel, c1, c2 = result
                print(f"[DIFF] {rel}")
                print(f"       {c1} Zeilen vs {c2} Zeilen")

            else:
                errors += 1
                _, rel, err = result
                print(f"[ERROR] {rel}: {err}")

    print("\n" + "=" * 60)
    print("Zusammenfassung")
    print("=" * 60)
    print(f"CSV-Dateien nur in Ordner1 : {len(only1)}")
    print(f"CSV-Dateien nur in Ordner2 : {len(only2)}")
    print(f"Verglichen                 : {len(common)}")
    print(f"Gleich                     : {ok}")
    print(f"Unterschiedliche Zeilen    : {diff}")
    print(f"Fehler                     : {errors}")


if __name__ == "__main__":
    main()