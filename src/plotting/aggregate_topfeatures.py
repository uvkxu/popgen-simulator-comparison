#!/usr/bin/env python3
# aggregate_topfeatures.py
#
# Aggregates your per-config feature-ranking CSVs into:
#   - index.csv                 (file inventory + parsed metadata)
#   - overall_agg.csv           (mean overall accuracies per config)
#   - feature_agg.csv           (mean feature "max_accuracy" per config+feature)
#   - passed_agg.csv            (how often a feature appeared in passed section)
#
# Important fix vs older version:
#   If PASSED_FEATURES is empty, the old parser could get "stuck" in passed_section
#   and mis-classify all subsequent features as passed. We now ALWAYS switch to
#   all_section at the first separator after the PASSED_FEATURES row.

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

CONFIG_RE = re.compile(r"sample_size_(?P<sample>\d+),seq_len_(?P<seqlen>\d+)")


@dataclass
class ParsedSummary:
    chance_baseline: float
    all_features_overall: Optional[float]
    passed_features_overall: Optional[float]
    overall_by_model: Dict[str, Optional[float]]  # xgboost, random_forest, logistic_regression

    # feature -> max_accuracy (in your current pipeline: this is permuted accuracy = baseline - mean_drop)
    all_feature_acc: Dict[str, float]
    # list of features that passed threshold (as appearing in the 'passed' section)
    passed_features: List[str]


def infer_feature_type(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    if "tree" in parts:
        return "tree"
    if "allele" in parts:
        return "allele"
    return None


def infer_config_dir(path: Path) -> Optional[Path]:
    """
    Walk up from file path to find the directory whose name matches:
    sample_size_{n},seq_len_{L}
    """
    for parent in [path.parent] + list(path.parents):
        if CONFIG_RE.search(parent.name):
            return parent
    return None


def extract_config_params(config_dir: Path) -> Tuple[int, int]:
    m = CONFIG_RE.search(config_dir.name)
    if not m:
        raise ValueError(f"Could not parse config_dir name: {config_dir}")
    return int(m.group("sample")), int(m.group("seqlen"))


def _normalize_header(header: List[str]) -> List[str]:
    # normalize potential BOM / whitespace
    return [h.strip().lstrip("\ufeff") for h in header]


def _looks_like_summary_csv(header: List[str]) -> bool:
    header = _normalize_header(header)
    return ("feature" in header) and ("max_accuracy" in header)


def read_summary_csv_with_sections(csv_path: Path, default_chance: float = 1.0 / 3.0) -> ParsedSummary:
    """
    Parses your special CSV that may contain:
      - special rows: CHANCE_BASELINE, ALL_FEATURES, PASSED_FEATURES, OVERALL_MEAN
      - separators: empty lines or rows whose first column is empty
      - "passed features section": starts AFTER PASSED_FEATURES row, continues until first separator
      - "all features section": all other feature rows afterwards

    NOTE (important):
      If no passed-features exist, PASSED_FEATURES is followed immediately by a separator.
      We MUST still transition into all_section to avoid mis-classifying everything as passed.
    """
    chance = None
    all_overall = None
    passed_overall = None
    model_cols = {"xgboost": None, "random_forest": None, "logistic_regression": None}

    all_feature_acc: Dict[str, float] = {}
    passed_features: List[str] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {csv_path}")

        header = _normalize_header(header)
        col_idx = {name: i for i, name in enumerate(header)}
        if "feature" not in col_idx or "max_accuracy" not in col_idx:
            raise ValueError(f"Unexpected header in {csv_path}: {header}")

        state = "pre"  # pre -> passed_section -> all_section

        def get_cell(row: List[str], col: str) -> str:
            if col not in col_idx:
                return ""
            i = col_idx[col]
            return row[i].strip() if i < len(row) else ""

        def get_float(row: List[str], col: str) -> Optional[float]:
            s = get_cell(row, col)
            if s == "":
                return None
            try:
                return float(s)
            except ValueError:
                return None

        for row in reader:
            # Treat fully-empty rows as separators
            if (not row) or all((c.strip() == "" for c in row)):
                if state == "passed_section":
                    # IMPORTANT FIX: always switch after first separator following PASSED_FEATURES
                    state = "all_section"
                continue

            # Treat rows with empty first column ("feature") as separators too
            if get_cell(row, "feature") == "":
                if state == "passed_section":
                    state = "all_section"
                continue

            feat = get_cell(row, "feature")

            if feat == "CHANCE_BASELINE":
                v = get_float(row, "max_accuracy")
                if v is not None:
                    chance = v
                continue

            if feat == "ALL_FEATURES":
                all_overall = get_float(row, "max_accuracy")
                model_cols["xgboost"] = get_float(row, "xgboost")
                model_cols["random_forest"] = get_float(row, "random_forest")
                model_cols["logistic_regression"] = get_float(row, "logistic_regression")
                continue

            if feat == "PASSED_FEATURES":
                passed_overall = get_float(row, "max_accuracy")
                state = "passed_section"
                continue

            if feat == "OVERALL_MEAN":
                continue

            # regular feature row
            max_acc = get_float(row, "max_accuracy")
            if max_acc is None:
                continue

            if state == "passed_section":
                passed_features.append(feat)
            else:
                all_feature_acc[feat] = max_acc

    if chance is None:
        chance = default_chance

    return ParsedSummary(
        chance_baseline=chance,
        all_features_overall=all_overall,
        passed_features_overall=passed_overall,
        overall_by_model=model_cols,
        all_feature_acc=all_feature_acc,
        passed_features=passed_features,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate feature-ranking CSV outputs across configs (tree/allele)."
    )
    ap.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory to search recursively for summary CSV files.",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for aggregated CSVs.",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*with_summary.csv",
        help="Glob pattern used under input-root (default: '*with_summary.csv').",
    )
    ap.add_argument(
        "--default-chance",
        type=float,
        default=1.0 / 3.0,
        help="Default chance baseline if CHANCE_BASELINE is missing.",
    )
    args = ap.parse_args()

    input_root: Path = args.input_root
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(input_root.rglob(args.pattern))
    if not candidate_files:
        raise SystemExit(f"No files matching pattern '{args.pattern}' found under {input_root}")

    # Optionally filter out non-matching CSVs by header
    csv_files: List[Path] = []
    for fp in candidate_files:
        try:
            with fp.open("r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and _looks_like_summary_csv(header):
                    csv_files.append(fp)
        except Exception:
            # skip unreadable
            continue

    if not csv_files:
        raise SystemExit(
            f"Found {len(candidate_files)} candidates, but none looked like summary CSVs "
            f"(missing header columns feature/max_accuracy)."
        )

    index_rows = []
    overall_rows = []
    feat_rows = []
    passed_rows = []

    for fp in csv_files:
        ftype = infer_feature_type(fp)
        if ftype is None:
            continue

        config_dir = infer_config_dir(fp)
        if config_dir is None:
            continue

        sample_size, seq_len = extract_config_params(config_dir)
        model_name = config_dir.parent.name  # demographic model folder

        parsed = read_summary_csv_with_sections(fp, default_chance=args.default_chance)

        index_rows.append(
            {
                "file": str(fp),
                "feature_type": ftype,
                "model": model_name,
                "config": config_dir.name,
                "sample_size": sample_size,
                "seq_len": seq_len,
            }
        )

        overall_rows.append(
            {
                "feature_type": ftype,
                "model": model_name,
                "config": config_dir.name,
                "sample_size": sample_size,
                "seq_len": seq_len,
                "chance_baseline": parsed.chance_baseline,
                "overall_all_features": parsed.all_features_overall,
                "overall_passed_features": parsed.passed_features_overall,
                "overall_xgboost": parsed.overall_by_model.get("xgboost"),
                "overall_random_forest": parsed.overall_by_model.get("random_forest"),
                "overall_logistic_regression": parsed.overall_by_model.get("logistic_regression"),
            }
        )

        for feat, acc in parsed.all_feature_acc.items():
            feat_rows.append(
                {
                    "feature_type": ftype,
                    "model": model_name,
                    "config": config_dir.name,
                    "sample_size": sample_size,
                    "seq_len": seq_len,
                    "feature": feat,
                    "max_accuracy": acc,
                }
            )

        for feat in parsed.passed_features:
            passed_rows.append(
                {
                    "feature_type": ftype,
                    "model": model_name,
                    "config": config_dir.name,
                    "sample_size": sample_size,
                    "seq_len": seq_len,
                    "feature": feat,
                    "passed": 1,
                }
            )

    df_index = pd.DataFrame(index_rows)
    df_overall = pd.DataFrame(overall_rows)
    df_feat = pd.DataFrame(feat_rows)
    df_passed = pd.DataFrame(passed_rows)

    # Write raw inventories
    df_index.to_csv(outdir / "index.csv", index=False)
    df_overall.to_csv(outdir / "overall_raw.csv", index=False)
    df_feat.to_csv(outdir / "feature_raw.csv", index=False)
    df_passed.to_csv(outdir / "passed_raw.csv", index=False)

    # Aggregate
    group_keys = ["feature_type", "model", "config", "sample_size", "seq_len"]

    df_overall_agg = (
        df_overall.groupby(group_keys, dropna=False)
        .agg(
            chance_baseline=("chance_baseline", "mean"),
            overall_all_features=("overall_all_features", "mean"),
            overall_passed_features=("overall_passed_features", "mean"),
            overall_xgboost=("overall_xgboost", "mean"),
            overall_random_forest=("overall_random_forest", "mean"),
            overall_logistic_regression=("overall_logistic_regression", "mean"),
            n_files=("overall_all_features", "count"),
        )
        .reset_index()
    )

    # Feature aggregation: mean max_accuracy per (config, feature)
    # NOTE: in your pipeline "max_accuracy" is permuted accuracy (baseline - mean_drop).
    # Lower values => larger mean_drop => more important (if you interpret it that way).
    df_feat_agg = (
        df_feat.groupby(group_keys + ["feature"], dropna=False)
        .agg(
            mean_max_accuracy=("max_accuracy", "mean"),
            std_max_accuracy=("max_accuracy", "std"),
            n=("max_accuracy", "count"),
        )
        .reset_index()
    )

    # Passed aggregation: how often a feature was in passed section for that config
    if not df_passed.empty:
        df_passed_agg = (
            df_passed.groupby(group_keys + ["feature"], dropna=False)
            .agg(
                passed_count=("passed", "sum"),
                n=("passed", "count"),
            )
            .reset_index()
        )
    else:
        df_passed_agg = pd.DataFrame(columns=group_keys + ["feature", "passed_count", "n"])

    # For convenience: compute "importance-like" proxy = (overall_all_features - mean_max_accuracy)
    # This is an estimate of mean_drop if overall_all_features corresponds to baseline_acc.
    # Only valid if your per-feature max_accuracy is perm_acc computed on SAME validation set.
    if not df_feat_agg.empty and not df_overall_agg.empty:
        df_feat_agg = df_feat_agg.merge(
            df_overall_agg[group_keys + ["overall_all_features"]],
            on=group_keys,
            how="left",
        )
        df_feat_agg["mean_drop_proxy"] = df_feat_agg["overall_all_features"] - df_feat_agg["mean_max_accuracy"]

    # Save aggregates
    df_overall_agg.to_csv(outdir / "overall_agg.csv", index=False)
    df_feat_agg.to_csv(outdir / "feature_agg.csv", index=False)
    df_passed_agg.to_csv(outdir / "passed_agg.csv", index=False)

    print(f"[OK] Wrote:")
    print(f"  - {outdir / 'index.csv'}")
    print(f"  - {outdir / 'overall_raw.csv'}")
    print(f"  - {outdir / 'feature_raw.csv'}")
    print(f"  - {outdir / 'passed_raw.csv'}")
    print(f"  - {outdir / 'overall_agg.csv'}")
    print(f"  - {outdir / 'feature_agg.csv'}")
    print(f"  - {outdir / 'passed_agg.csv'}")


if __name__ == "__main__":
    main()
