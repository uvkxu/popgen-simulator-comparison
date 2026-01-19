import os
import sys
import pandas as pd
from collections import defaultdict
import numpy as np


def find_csv_files(root_dir):
    """Recursively collects all CSV files under the given root directory."""
    csv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, file))
    return csv_files


def compare_weighted_unweighted(df):
    """Compares weighted and unweighted features within a single DataFrame."""
    differences = []
    identical = []

    for col in df.columns:
        if col.endswith("_unweighted"):
            base = col.replace("_unweighted", "")
            if base in df.columns:
                weighted = df[base]
                unweighted = df[col]

                # Compare values with NaN tolerance
                if not np.allclose(weighted, unweighted, equal_nan=True):
                    differences.append(base)
                else:
                    identical.append(base)
    return differences, identical


def check_global_invariance(all_dfs):
    """Checks which features are constant across all CSV files."""
    global_invariant = []
    all_cols = set().union(*[df.columns for df in all_dfs])

    for col in all_cols:
        values = []
        for df in all_dfs:
            if col in df.columns:
                vals = df[col].dropna().unique()
                values.extend(vals)

        values = [v for v in values if not pd.isna(v)]
        if len(set(values)) <= 1:
            global_invariant.append(col)
    return global_invariant


def main(root_dir):
    csv_files = find_csv_files(root_dir)
    if not csv_files:
        print(f"❌ No CSV files found in '{root_dir}'.")
        return

    print(f"🔍 Found CSV files: {len(csv_files)}\n")

    all_dfs = []
    all_differences = defaultdict(list)
    all_identical = defaultdict(list)

    # Process each CSV file
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

            diffs, same = compare_weighted_unweighted(df)
            if diffs:
                all_differences[csv_path] = diffs
            if same:
                all_identical[csv_path] = same

        except Exception as e:
            print(f"⚠️ Error reading {csv_path}: {e}")

    # Print differences
    print("📄 Differences between weighted and unweighted features:")
    if not all_differences:
        print("✅ No differences found!")
    else:
        for f, cols in all_differences.items():
            print(f"  • {f}: {', '.join(cols)}")

    # Print identical features
    print("\n📄 Features that are identical (weighted == unweighted):")
    if not all_identical:
        print("❌ No identical features found.")
    else:
        for f, cols in all_identical.items():
            print(f"  • {f}: {', '.join(cols)}")

    # Check globally invariant features
    global_invariant = check_global_invariance(all_dfs)
    print("\n🌍 Features that are constant across all CSV files:")
    if not global_invariant:
        print("❌ No globally invariant features found.")
    else:
        for col in sorted(global_invariant):
            print(f"  • {col}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❗ Usage: python compare_weighted_unweighted_features.py <input_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"❌ Invalid directory: {input_dir}")
        sys.exit(1)

    main(input_dir)
