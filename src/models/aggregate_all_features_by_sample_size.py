#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np


def parse_sample_size_from_filename(stem: str) -> int:
    """
    Accepts filenames containing either:
      - 'sample_size_128'
      - '_n128'
    """
    m = re.search(r"(?:sample_size_|_n)(\d+)", stem)
    if not m:
        raise ValueError(f"Could not parse sample size from filename: {stem}")
    return int(m.group(1))


def normalize_group_string(group_items: list[str]) -> str:
    """
    Create a stable identifier for a simulator group.
    Example: ['msprime','ms','scrm'] -> 'ms-msprime-scrm' (sorted)
    """
    return "-".join(sorted([g.strip() for g in group_items if g.strip()]))


def find_group_files(input_folder: Path, group_id: str) -> list[Path]:
    """
    Finds evaluation CSVs for the given group id.
    We assume your files contain the group id somewhere in the filename.
    Example filenames:
      feature_ranking_ms-msprime-scrm_sample_size_128.csv
      feature_ranking_ms-msprime-scrm_n128.csv
    """
    candidates = []
    for p in input_folder.glob("*.csv"):
        name = p.name
        if group_id in name and ("sample_size_" in name or re.search(r"_n\d+", name)):
            candidates.append(p)
    return sorted(candidates)


def read_all_features_row(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path).dropna(how="all")
    if "feature" not in df.columns:
        raise ValueError(f"Missing 'feature' column in {csv_path}")

    row = df.loc[df["feature"] == "ALL_FEATURES"]
    if row.empty:
        raise ValueError(f"No ALL_FEATURES row found in {csv_path}")

    r = row.iloc[0].to_dict()
    # expected columns in your output
    needed = ["xgboost", "random_forest", "logistic_regression"]
    for col in needed:
        if col not in r:
            raise ValueError(f"Missing column '{col}' in {csv_path}")

    return {
        "xgboost": float(r["xgboost"]),
        "random_forest": float(r["random_forest"]),
        "logistic_regression": float(r["logistic_regression"]),
    }


def load_chance_baselines(input_folder: Path, group_id: str) -> pd.DataFrame | None:
    """
    Optional: If you saved chance baselines per sample size into chance_baselines.csv
    with at least columns: group, sample_size, chance_baseline
    optionally: n_rows (for weighted mean)
    """
    p = input_folder / "chance_baselines.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    req = {"group", "sample_size", "chance_baseline"}
    if not req.issubset(set(df.columns)):
        return None
    df = df[df["group"] == group_id].copy()
    return df


def main():
    ap = argparse.ArgumentParser(description="Aggregate ALL_FEATURES accuracy across sample sizes.")
    ap.add_argument("--input_folder", type=Path, required=True, help="Folder containing per-sample-size evaluation CSVs")
    ap.add_argument("--output_file", type=Path, required=True, help="Output CSV with mean accuracies")
    ap.add_argument(
        "--group",
        type=str,
        required=True,
        help="Comma-separated simulator group, e.g. 'msprime,ms,scrm' (must match filenames)",
    )
    ap.add_argument(
        "--pattern_hint",
        type=str,
        default="feature_ranking",
        help="Optional hint: only consider CSVs that contain this substring (default: feature_ranking)",
    )
    args = ap.parse_args()

    input_folder: Path = args.input_folder
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    group_items = [s.strip() for s in args.group.split(",")]
    group_id = normalize_group_string(group_items)

    # collect files
    files = []
    for p in input_folder.glob("*.csv"):
        if args.pattern_hint and args.pattern_hint not in p.name:
            continue
        if group_id in p.name and ("sample_size_" in p.name or re.search(r"_n\d+", p.stem)):
            files.append(p)
    files = sorted(files)

    if not files:
        raise ValueError(
            f"No matching CSVs found in {input_folder} for group '{group_id}'. "
            f"Expected filenames containing '{group_id}' and 'sample_size_<n>' or '_n<n>'."
        )

    # build per-n table
    rows = []
    for p in files:
        n = parse_sample_size_from_filename(p.stem)
        acc = read_all_features_row(p)
        rows.append({
            "group": group_id,
            "sample_size": n,
            "file": p.name,
            "xgboost_acc": acc["xgboost"],
            "random_forest_acc": acc["random_forest"],
            "logreg_acc": acc["logistic_regression"],
        })

    per_n = pd.DataFrame(rows).sort_values("sample_size").reset_index(drop=True)

    # optionally join chance baselines
    chance_df = load_chance_baselines(input_folder, group_id)
    if chance_df is not None:
        per_n = per_n.merge(
            chance_df[["sample_size", "chance_baseline"] + (["n_rows"] if "n_rows" in chance_df.columns else [])],
            on="sample_size",
            how="left",
        )

    # compute macro mean (each n equally)
    acc_cols = ["xgboost_acc", "random_forest_acc", "logreg_acc"]
    macro_means = per_n[acc_cols].mean(numeric_only=True).to_dict()

    summary = {
        "group": group_id,
        "n_sample_sizes": int(per_n["sample_size"].nunique()),
        **{f"macro_mean_{k}": float(v) for k, v in macro_means.items()},
    }

    # weighted mean if n_rows exists
    if "n_rows" in per_n.columns and per_n["n_rows"].notna().all():
        w = per_n["n_rows"].astype(float)
        for c in acc_cols:
            summary[f"weighted_mean_{c}"] = float((per_n[c] * w).sum() / w.sum())

    # also macro mean chance if available
    if "chance_baseline" in per_n.columns and per_n["chance_baseline"].notna().any():
        summary["macro_mean_chance_baseline"] = float(per_n["chance_baseline"].mean())

    # write outputs
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1) per sample size table (sidecar)
    per_n_out = args.output_file.with_suffix("").as_posix() + "_by_sample_size.csv"
    per_n.to_csv(per_n_out, index=False)

    # 2) summary table
    pd.DataFrame([summary]).to_csv(args.output_file, index=False)

    print(f"Wrote per-sample-size table: {per_n_out}")
    print(f"Wrote summary means: {args.output_file}")


if __name__ == "__main__":
    main()
