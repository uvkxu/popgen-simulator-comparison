#!/usr/bin/env python3
"""
Aggregate ALL_FEATURES accuracy across sample sizes where the mode is encoded
in the folder name (sample_size_<n>), and create a plot.
"""

from pathlib import Path
import argparse
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_n_from_mode_folder(folder_name: str) -> int:
    m = re.search(r"sample_size_(\d+)", folder_name)
    if not m:
        raise ValueError(f"Not a sample_size folder: {folder_name}")
    return int(m.group(1))


def read_row(df: pd.DataFrame, key: str) -> pd.Series:
    row = df.loc[df["feature"] == key]
    if row.empty:
        raise ValueError(f"Missing row '{key}' in CSV")
    return row.iloc[0]


def plot_accuracy_vs_sample_size(df: pd.DataFrame, output_path: Path, title: str):
    df = df.sort_values("sample_size")

    plt.figure(figsize=(8, 5))

    plt.plot(df["sample_size"], df["xgboost"], marker="o", label="XGBoost")
    plt.plot(df["sample_size"], df["random_forest"], marker="o", label="Random Forest")
    plt.plot(df["sample_size"], df["logistic_regression"], marker="o", label="Logistic Regression")

    if "chance_baseline" in df.columns and df["chance_baseline"].notna().any():
        chance_mean = float(df["chance_baseline"].mean())
        plt.hlines(
            y=chance_mean,
            xmin=df["sample_size"].min(),
            xmax=df["sample_size"].max(),
            linestyles="dashed",
            colors="black",
            label=f"Chance baseline (mean={chance_mean:.2f})",
        )

    plt.xlabel("Sample size")
    plt.ylabel("Accuracy (ALL_FEATURES)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir",
        type=Path,
        required=True,
        help="Directory containing sample_size_<n>/ subfolders",
    )
    ap.add_argument(
        "--group_string",
        type=str,
        required=True,
        help="E.g. ms_msprime_scrm",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where aggregated CSVs and plots are written",
    )
    ap.add_argument(
        "--plot_title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    args = ap.parse_args()

    base_dir: Path = args.base_dir
    group_string: str = args.group_string
    output_dir: Path = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    mode_dirs = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("sample_size_")]
    )
    if not mode_dirs:
        raise RuntimeError(f"No sample_size_* folders found in {base_dir}")

    per_n_rows = []

    for mode_dir in mode_dirs:
        n = extract_n_from_mode_folder(mode_dir.name)
        csv_path = mode_dir / f"feature_ranking_{group_string}_with_summary.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path).dropna(how="all")

        all_row = read_row(df, "ALL_FEATURES")
        entry = {
            "sample_size": n,
            "mode": mode_dir.name,
            "xgboost": float(all_row["xgboost"]),
            "random_forest": float(all_row["random_forest"]),
            "logistic_regression": float(all_row["logistic_regression"]),
        }

        if "CHANCE_BASELINE" in set(df["feature"]):
            ch_row = read_row(df, "CHANCE_BASELINE")
            entry["chance_baseline"] = float(ch_row["max_accuracy"])

        per_n_rows.append(entry)

    if not per_n_rows:
        raise RuntimeError("No matching CSVs found for the given group_string.")

    per_n = pd.DataFrame(per_n_rows).sort_values("sample_size").reset_index(drop=True)

    # --- Macro mean across sample sizes ---
    mean_summary = {
        "group_string": group_string,
        "n_sample_sizes": int(per_n["sample_size"].nunique()),
        "mean_xgboost": float(per_n["xgboost"].mean()),
        "mean_random_forest": float(per_n["random_forest"].mean()),
        "mean_logistic_regression": float(per_n["logistic_regression"].mean()),
    }

    if "chance_baseline" in per_n.columns:
        mean_summary["mean_chance_baseline"] = float(per_n["chance_baseline"].mean())

    # --- Output paths ---
    by_n_csv = output_dir / f"mean_all_features_{group_string}_by_sample_size.csv"
    mean_csv = output_dir / f"mean_all_features_{group_string}.csv"
    plot_png = output_dir / f"mean_all_features_{group_string}_vs_sample_size.png"

    per_n.to_csv(by_n_csv, index=False)
    pd.DataFrame([mean_summary]).to_csv(mean_csv, index=False)

    title = args.plot_title or f"ALL_FEATURES accuracy vs. sample size ({group_string})"
    plot_accuracy_vs_sample_size(per_n, plot_png, title)

    print("Wrote:", by_n_csv)
    print("Wrote:", mean_csv)
    print("Wrote:", plot_png)


if __name__ == "__main__":
    main()
