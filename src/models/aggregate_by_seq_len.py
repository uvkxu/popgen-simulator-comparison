#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEQ_LEN_DIR_RE = re.compile(r"^seq_len_(\d+)$")


def extract_L_from_mode(mode_dir_name: str) -> int:
    """
    Expects directory name exactly like: seq_len_10000
    """
    m = SEQ_LEN_DIR_RE.match(mode_dir_name)
    if not m:
        raise ValueError(f"Not a seq_len mode dir: {mode_dir_name}")
    return int(m.group(1))


def find_seq_len_csvs(base_dir: Path, group_string: str) -> list[tuple[int, Path]]:
    """
    base_dir contains mode folders, e.g. <...>/<config>/
      seq_len_10000/feature_ranking_<group>_with_summary.csv

    Returns list of (seq_len, csv_path).
    """
    out: list[tuple[int, Path]] = []
    for mode_dir in sorted(base_dir.iterdir()):
        if not mode_dir.is_dir():
            continue

        m = SEQ_LEN_DIR_RE.match(mode_dir.name)
        if not m:
            continue

        csv_path = mode_dir / f"feature_ranking_{group_string}_with_summary.csv"
        if csv_path.exists():
            out.append((int(m.group(1)), csv_path))

    return out


def aggregate_mean_same_format(csvs: list[Path]) -> pd.DataFrame:
    """
    Reads multiple feature_ranking_*_with_summary.csv files and returns a single
    DataFrame with per-feature mean across files, keeping the same columns.

    - drop fully empty rows
    - keep rows where 'feature' is present
    - numeric columns are averaged per feature
    - output keeps special rows first if present: CHANCE_BASELINE, ALL_FEATURES, PASSED_FEATURES
    - rest sorted by mean 'max_accuracy' desc (then feature name)
    - OVERALL_MEAN dropped
    """
    dfs = []
    for p in csvs:
        df = pd.read_csv(p).dropna(how="all")
        if "feature" not in df.columns:
            raise ValueError(f"Missing 'feature' column in {p}")
        df = df[df["feature"].notna()].copy()
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Drop rows you do not want
    all_df = all_df[~all_df["feature"].isin(["OVERALL_MEAN"])].copy()

    numeric_cols = [c for c in all_df.columns if c != "feature"]
    for c in numeric_cols:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    mean_df = (
        all_df.groupby("feature", as_index=False)[numeric_cols]
        .mean(numeric_only=True)
    )

    special = ["CHANCE_BASELINE", "ALL_FEATURES", "PASSED_FEATURES"]
    df_special = mean_df[mean_df["feature"].isin(special)].copy()
    df_rest = mean_df[~mean_df["feature"].isin(special)].copy()

    if "max_accuracy" in df_rest.columns:
        df_rest = df_rest.sort_values(["max_accuracy", "feature"], ascending=[False, True])
    else:
        df_rest = df_rest.sort_values("feature")

    return pd.concat([df_special, df_rest], ignore_index=True)


def plot_all_features_vs_seq_len(per_L: pd.DataFrame, out_png: Path, title: str):
    """
    per_L columns: seq_len, xgboost, random_forest, logistic_regression, (optional) chance_baseline
    """
    per_L = per_L.sort_values("seq_len")

    plt.figure(figsize=(8, 5))
    plt.plot(per_L["seq_len"], per_L["xgboost"], marker="o", label="XGBoost")
    plt.plot(per_L["seq_len"], per_L["random_forest"], marker="o", label="Random Forest")
    plt.plot(per_L["seq_len"], per_L["logistic_regression"], marker="o", label="Logistic Regression")

    if "chance_baseline" in per_L.columns and per_L["chance_baseline"].notna().any():
        chance_mean = float(per_L["chance_baseline"].mean())
        plt.hlines(
            y=chance_mean,
            xmin=float(per_L["seq_len"].min()),
            xmax=float(per_L["seq_len"].max()),
            linestyles="dashed",
            colors="black",
            label=f"Chance baseline (mean={chance_mean:.2f})",
        )

    plt.xlabel("Sequence length (seq_len)")
    plt.ylabel("Accuracy (ALL_FEATURES)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir", type=Path, required=True,
        help="Directory containing seq_len_<L>/ subfolders (i.e., .../<config>/)"
    )
    ap.add_argument(
        "--group", type=str, required=True,
        help="Group string exactly as used in file names, e.g. ms_msprime_scrm (already '_'.join(group))"
    )
    ap.add_argument(
        "--output_dir", type=Path, required=True,
        help="Output directory, e.g. .../<config>/aggregated/"
    )
    ap.add_argument("--plot_title", type=str, default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    L_csvs = find_seq_len_csvs(args.base_dir, args.group)
    if not L_csvs:
        raise RuntimeError(
            f"No seq_len_* CSVs found under {args.base_dir} for group {args.group}"
        )

    seq_lens = [L for L, _ in L_csvs]
    csv_paths = [p for _, p in L_csvs]

    # 1) Aggregate mean per feature (same format)
    mean_df = aggregate_mean_same_format(csv_paths)
    out_csv = args.output_dir / f"mean_aggregated_{args.group}_with_summary.csv"
    mean_df.to_csv(out_csv, index=False)

    # 2) Build per-seq-len table for ALL_FEATURES and plot it
    per_L_rows = []
    for L, p in L_csvs:
        df = pd.read_csv(p).dropna(how="all")
        df = df[df["feature"].notna()].copy()

        all_row = df[df["feature"] == "ALL_FEATURES"]
        if all_row.empty:
            continue
        r = all_row.iloc[0]

        row = {
            "seq_len": L,
            "xgboost": float(pd.to_numeric(r.get("xgboost"), errors="coerce")),
            "random_forest": float(pd.to_numeric(r.get("random_forest"), errors="coerce")),
            "logistic_regression": float(pd.to_numeric(r.get("logistic_regression"), errors="coerce")),
        }

        ch = df[df["feature"] == "CHANCE_BASELINE"]
        if not ch.empty:
            row["chance_baseline"] = float(pd.to_numeric(ch.iloc[0].get("max_accuracy"), errors="coerce"))

        per_L_rows.append(row)

    per_L = pd.DataFrame(per_L_rows).sort_values("seq_len").reset_index(drop=True)

    out_by_L = args.output_dir / f"all_features_{args.group}_by_seq_len.csv"
    per_L.to_csv(out_by_L, index=False)

    out_png = args.output_dir / f"all_features_{args.group}_vs_seq_len.png"
    title = args.plot_title or f"ALL_FEATURES accuracy vs. seq_len ({args.group})"
    plot_all_features_vs_seq_len(per_L, out_png, title)

    print("Wrote:", out_csv)
    print("Wrote:", out_by_L)
    print("Wrote:", out_png)
    print("Aggregated over seq_len:", sorted(seq_lens))


if __name__ == "__main__":
    main()
