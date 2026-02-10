#!/usr/bin/env python3
# aggregate_topfeatures.py

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

    # feature -> max_accuracy
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


def read_summary_csv_with_sections(csv_path: Path, default_chance: float = 1.0 / 3.0) -> ParsedSummary:
    """
    Parses your special CSV that may contain:
      - special rows: CHANCE_BASELINE, ALL_FEATURES, PASSED_FEATURES, OVERALL_MEAN
      - separators: empty lines or rows whose first column is empty
      - "passed features section": starts AFTER PASSED_FEATURES row, continues until first separator after at least 1 feature
      - "all features section": all other feature rows afterwards
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

        # Map indices
        # expected: feature,max_accuracy,xgboost,random_forest,logistic_regression
        col_idx = {name: i for i, name in enumerate(header)}
        if "feature" not in col_idx or "max_accuracy" not in col_idx:
            raise ValueError(f"Unexpected header in {csv_path}: {header}")

        state = "pre"  # pre -> passed_section -> all_section
        saw_any_passed = False

        for row in reader:
            # normalize row length
            if not row or all((c.strip() == "" for c in row)):
                # blank line = separator
                if state == "passed_section" and saw_any_passed:
                    state = "all_section"
                continue

            # if first column empty -> separator line
            if row[col_idx["feature"]].strip() == "":
                if state == "passed_section" and saw_any_passed:
                    state = "all_section"
                continue

            feat = row[col_idx["feature"]].strip()

            def get_float(col: str) -> Optional[float]:
                if col not in col_idx:
                    return None
                s = row[col_idx[col]].strip() if col_idx[col] < len(row) else ""
                if s == "":
                    return None
                try:
                    return float(s)
                except ValueError:
                    return None

            if feat == "CHANCE_BASELINE":
                v = get_float("max_accuracy")
                if v is not None:
                    chance = v
                continue

            if feat == "ALL_FEATURES":
                all_overall = get_float("max_accuracy")
                model_cols["xgboost"] = get_float("xgboost")
                model_cols["random_forest"] = get_float("random_forest")
                model_cols["logistic_regression"] = get_float("logistic_regression")
                continue

            if feat == "PASSED_FEATURES":
                passed_overall = get_float("max_accuracy")
                # after this row, next feature rows until separator = passed feature list
                state = "passed_section"
                saw_any_passed = False
                continue

            if feat == "OVERALL_MEAN":
                # ignore (redundant)
                continue

            # regular feature row
            max_acc = get_float("max_accuracy")
            if max_acc is None:
                # can't use it
                continue

            if state == "passed_section":
                passed_features.append(feat)
                saw_any_passed = True
            else:
                # all-section or pre
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
    ap = argparse.ArgumentParser(description="Aggregate top_features CSV outputs across configs (tree/allele).")
    ap.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory to search recursively for '*with_summary.csv' files (e.g., results/plots).",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for aggregated CSVs.",
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

    csv_files = sorted(input_root.rglob("*with_summary.csv"))
    if not csv_files:
        raise SystemExit(f"No '*with_summary.csv' found under {input_root}")

    index_rows = []
    overall_rows = []
    feat_rows = []
    passed_rows = []

    for fp in csv_files:
        ftype = infer_feature_type(fp)
        if ftype is None:
            # skip unknown
            continue

        config_dir = infer_config_dir(fp)
        if config_dir is None:
            # skip if we can't find config dir
            continue

        sample_size, seq_len = extract_config_params(config_dir)

        # model key = parent directory of config_dir (often the demographic model folder)
        model_name = config_dir.parent.name

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

    # Aggregate:
    # overall: mean across files that map to the same (feature_type, model, config, sample_size, seq_len)
    group_keys = ["feature_type", "model", "config", "sample_size", "seq_len"]

    df_overall_agg = (
        df_overall.groupby(group_keys, dropna=False)
        .agg(
            {
                "chance_baseline": "mean",
                "overall_all_features": "mean",
                "overall_passed_features": "mean",
                "overall_xgboost": "mean",
                "overall_random_forest": "mean",
                "overall_logistic_regression": "mean",
            }
        )
        .reset_index()
    )

    # feature accuracie
