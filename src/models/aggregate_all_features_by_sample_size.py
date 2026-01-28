#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_n_from_mode(mode_dir_name: str) -> int:
    m = re.search(r"sample_size_(\d+)", mode_dir_name)
    if not m:
        raise ValueError(f"Not a sample_size mode dir: {mode_dir_name}")
    return int(m.group(1))


def find_sample_size_csvs(base_dir: Path, group_string: str) -> list[tuple[int, Path]]:
    """
    base_dir contains mode folders, e.g. <...>/<config>/
      sample_size_64/feature_ranking_<group>_with_summary.csv
    Returns list of (sample_size, csv_path).
    """
    out = []
    for mode_dir in sorted(base_dir.iterdir()):
        if not mode_dir.is_dir():
            continue
        if not mode_dir.name.startswith("sample_size_"):
            continue
        csv_path = mode_dir / f"feature_ranking_{group_string}_with_summary.csv"
        if csv_path.exists():
            out.append((extract_n_from_mode(mode_dir.name), csv_path))
    return out


def aggregate_mean_same_format(csvs: list[Path]) -> pd.DataFrame:
    """
    Reads multiple feature_ranking_*_with_summary.csv files and returns a single
    DataFrame with per-feature mean across files, keeping the same columns.

    Strategy:
    - drop fully empty rows
    - keep rows where 'feature' is present
    - numeric columns are averaged per feature
    - output keeps special rows first if present: CHANCE_BASELINE, ALL_FEATURES, PASSED_FEATURES
    - rest of features sorted by mean 'max_accuracy' descending (stable and readable)
    - OVERALL_MEAN is dropped (you said it should be out)
    """
    dfs = []
    for p in csvs:
        df = pd.read_csv(p).dropna(how="all")
        if "feature" not in df.columns:
            raise ValueError(f"Missing 'feature' column in {p}")
        # keep only rows with a feature name
        df = df[df["feature"].notna()].copy()
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Remove rows you don't want in the final aggregation
    all_df = all_df[~all_df["feature"].isin(["OVERALL_MEAN"])].copy()

    # Identify numeric columns to average (everything except 'feature')
    numeric_cols = [c for c in all_df.columns if c != "feature"]

    # Ensure numeric conversion where possible
    for c in numeric_cols:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    # Mean per feature
    mean_df = (
        all_df.groupby("feature", as_index=False)[numeric_cols]
        .mean(numeric_only=True)
    )

    # Reorder: special rows first (if present)
    special = ["CHANCE_BASELINE", "ALL_FEATURES", "PASSED_FEATURES"]
    df_special = mean_df[mean_df["feature"].isin(special)].copy()
    df_rest = mean_df[~mean_df["feature"].isin(special)].copy()

    # Sort rest by mean max_accuracy (if exists), else alphabetical
    if "max_accuracy" in df_rest.columns:
        df_rest = df_rest.sort_values(["max_accuracy", "feature"], ascending=[False, True])
    else:
        df_rest = df_rest.sort_values("feature")

    out = pd.concat([df_special, df_rest], ignore_index=True)

    # Optional: add a blank row after PASSED_FEATURES to mimic your formatting
    # (purely cosmetic; safe for parsing because feature is NaN)
    # If you want it, uncomment:
    # out = pd.concat([out.iloc[:3], pd.DataFrame([{c: np.nan for c in out.columns}]), out.iloc[3:]], ignore_index=True)

    return out


def plot_all_features_vs_sample_size(per_n: pd.DataFrame, out_png: Path, title: str):
    """
    per_n columns: sample_size, xgboost, random_forest, logistic_regression, (optional) chance_baseline
    """
    per_n = per_n.sort_values("sample_size")

    plt.figure(figsize=(8, 5))
    plt.plot(per_n["sample_size"], per_n["xgboost"], marker="o", label="XGBoost")
    plt.plot(per_n["sample_size"], per_n["random_forest"], marker="o", label="Random Forest")
    plt.plot(per_n["sample_size"], per_n["logistic_regression"], marker="o", label="Logistic Regression")

    if "chance_baseline" in per_n.columns and per_n["chance_baseline"].notna().any():
        chance_mean = float(per_n["chance_baseline"].mean())
        plt.hlines(
            y=chance_mean,
            xmin=float(per_n["sample_size"].min()),
            xmax=float(per_n["sample_size"].max()),
            linestyles="dashed",
            colors="black",
            label=f"Chance baseline (mean={chance_mean:.2f})",
        )

    plt.xlabel("Sample size")
    plt.ylabel("Accuracy (ALL_FEATURES)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=Path, required=True,
                    help="Directory containing sample_size_<n>/ subfolders (i.e., .../<config>/)")
    ap.add_argument("--group_string", type=str, required=True,
                    help="E.g. ms_msprime_scrm")
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="Where to write aggregated CSV and plots")
    ap.add_argument("--plot_title", type=str, default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ss_csvs = find_sample_size_csvs(args.base_dir, args.group_string)
    if not ss_csvs:
        raise RuntimeError(f"No sample_size_* CSVs found under {args.base_dir} for group {args.group_string}")

    sample_sizes = [n for n, _ in ss_csvs]
    csv_paths = [p for _, p in ss_csvs]

    # 1) Aggregate mean per feature (same format)
    mean_df = aggregate_mean_same_format(csv_paths)

    out_csv = args.output_dir / f"mean_aggregated_{args.group_string}_with_summary.csv"
    mean_df.to_csv(out_csv, index=False)

    # 2) Build per-sample-size table for ALL_FEATURES and plot it
    per_n_rows = []
    for n, p in ss_csvs:
        df = pd.read_csv(p).dropna(how="all")
        df = df[df["feature"].notna()].copy()
        df_cols = df.columns.tolist()

        all_row = df[df["feature"] == "ALL_FEATURES"]
        if all_row.empty:
            continue
        r = all_row.iloc[0]

        row = {
            "sample_size": n,
            "xgboost": float(pd.to_numeric(r.get("xgboost"), errors="coerce")),
            "random_forest": float(pd.to_numeric(r.get("random_forest"), errors="coerce")),
            "logistic_regression": float(pd.to_numeric(r.get("logistic_regression"), errors="coerce")),
        }

        ch = df[df["feature"] == "CHANCE_BASELINE"]
        if not ch.empty:
            row["chance_baseline"] = float(pd.to_numeric(ch.iloc[0].get("max_accuracy"), errors="coerce"))

        per_n_rows.append(row)

    per_n = pd.DataFrame(per_n_rows).sort_values("sample_size").reset_index(drop=True)
    out_by_n = args.output_dir / f"all_features_{args.group_string}_by_sample_size.csv"
    per_n.to_csv(out_by_n, index=False)

    out_png = args.output_dir / f"all_features_{args.group_string}_vs_sample_size.png"
    title = args.plot_title or f"ALL_FEATURES accuracy vs. sample size ({args.group_string})"
    plot_all_features_vs_sample_size(per_n, out_png, title)

    print("Wrote:", out_csv)
    print("Wrote:", out_by_n)
    print("Wrote:", out_png)
    print("Aggregated over sample sizes:", sorted(sample_sizes))


if __name__ == "__main__":
    main()