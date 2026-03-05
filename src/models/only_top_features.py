from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import sys
import os
import yaml
import argparse
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import (
    VALID_GROUPING_MODES,
    format_feature_ranking_file,
    format_feature_ranking_barplot,
    XGB_STR,
)
from utils.io_helpers import collect_csv_files, detect_simulator_from_yaml


# --------------------------------------------------
# Fixed feature sets
# --------------------------------------------------
FEATURE_SETS = {
    "SMALLEST": [
        "tree_height_uw_mean",
        "tree_height_uw_max",
        "num_trees",
    ],
    "WITH_MORE": [
        "tree_height_uw_mean",
        "tree_height_uw_max",
        "crown_age_std",
        "num_branches_mean",
        "num_trees",
        "mean_branch_length_mean",
        "bollback_multinomial",
        "tree_height_uw_std",
        "mntd_std",
    ]
}


# --------------------------------------------------
# CSV merge helpers
# --------------------------------------------------
def _relative_after_dir(p: Path, anchor: str) -> Path:
    parts = p.parts
    if anchor not in parts:
        raise ValueError(f"'{anchor}' not found in path: {p}")
    idx = parts.index(anchor)
    return Path(*parts[idx + 1:])


def collect_and_merge_tree_allele_features(
    tree_csv_files,
    allele_csv_files,
    simulators,
):
    allele_lookup = {}
    for a in allele_csv_files:
        allele_lookup[_relative_after_dir(a, "allele")] = a

    merged = []

    for tree_csv in tree_csv_files:
        key = _relative_after_dir(tree_csv, "tree")
        allele_key = key.with_name(
            tree_csv.name.replace("_tree_features.csv", "_allel_features.csv")
        )

        if allele_key not in allele_lookup:
            raise FileNotFoundError(tree_csv)

        df_tree = pd.read_csv(tree_csv)
        df_allel = pd.read_csv(allele_lookup[allele_key])

        if len(df_tree) != len(df_allel):
            raise ValueError("Row mismatch")

        sim = detect_simulator_from_yaml(tree_csv)
        if sim not in simulators:
            continue

        merged.append(
            pd.concat(
                [
                    pd.Series([sim] * len(df_tree), name="simulator"),
                    df_tree.drop(columns=["simulator"], errors="ignore"),
                    df_allel.drop(columns=["simulator"], errors="ignore"),
                ],
                axis=1,
            )
        )

    if not merged:
        raise ValueError("No CSVs merged")

    return merged


# --------------------------------------------------
# Main evaluation logic (NO Optuna, NO permutation)
# --------------------------------------------------
def run_feature_evaluation(
    simulator_group,
    tree_csv_files,
    allel_csv_files,
    output_dir,
    es_rounds=30,
):
    all_dfs = collect_and_merge_tree_allele_features(
        tree_csv_files, allel_csv_files, simulator_group
    )
    data = pd.concat(all_dfs, ignore_index=True)

    if "difficulty" in data.columns:
        data["difficulty"] = data["difficulty"].replace(-1, np.nan)

    y = LabelEncoder().fit_transform(data["simulator"])
    num_class = len(simulator_group)

    csv_out = output_dir / format_feature_ranking_file(simulator_group)

    xgb_params = {
        "n_estimators": 5000,
        "learning_rate": 0.08,
        "max_depth": 5,
        "min_child_weight": 8.68,
        "gamma": 2.21,
        "subsample": 0.79,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.12,
        "reg_alpha": 0.22,
        "objective": "multi:softprob",
        "num_class": num_class,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": 1,
        "random_state": 0,
    }

    records = []

    for label, features in FEATURE_SETS.items():
        features = [f for f in features if f in data.columns]
        X = data[features].copy()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=0
        )

        if "difficulty" in X_train.columns:
            med = X_train["difficulty"].median(skipna=True)
            X_train["difficulty"] = X_train["difficulty"].fillna(med)
            X_val["difficulty"] = X_val["difficulty"].fillna(med)

        clf = xgb.XGBClassifier(**xgb_params)
        clf.set_params(early_stopping_rounds=es_rounds)

        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        ir = (0, clf.best_iteration + 1)
        proba = clf.predict_proba(X_val, iteration_range=ir)
        preds = np.argmax(proba, axis=1)

        acc = accuracy_score(y_val, preds)

        records.append(
            {
                "feature": label,
                "max_accuracy": acc,
                XGB_STR: acc,
            }
        )

    chance = np.bincount(y).max() / len(y)

    records.insert(
        0,
        {
            "feature": "CHANCE_BASELINE",
            "max_accuracy": chance,
            XGB_STR: np.nan,
        },
    )

    df = pd.DataFrame(records)
    df.to_csv(csv_out, index=False)


# --------------------------------------------------
# Grouped-by CSV selection (UNCHANGED)
# --------------------------------------------------
def csv_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def get_csv_files_mode(input_folder, grouped_by, seq_len_group=None, sample_size_group=None):
    csvs = []

    for root, _, files in os.walk(input_folder):
        root = Path(root)
        yamls = [f for f in files if f.endswith(".yaml") or f.endswith(".yml")]
        if not yamls:
            continue

        with open(root / yamls[0]) as f:
            params = yaml.safe_load(f)

        usePrecision = bool(params.get("usePrecision"))
        recomb = float(params.get("recombination_rate") or 0)
        sample_size = int(params["sample_structure"][0])
        seq_len = int(params["sequence_length"])

        if (
            (("precision_applied" not in grouped_by) or usePrecision)
            and (("precision_skipped" not in grouped_by) or not usePrecision)
            and (("recombination_applied" not in grouped_by) or recomb > 0)
            and (("recombination_skipped" not in grouped_by) or recomb == 0)
            and (("sample_size" not in grouped_by) or sample_size == sample_size_group)
            and (("seq_len" not in grouped_by) or seq_len == seq_len_group)
        ):
            csvs += collect_csv_files(root)

    return csvs


# --------------------------------------------------
# CLI + NORMALIZED GROUPING (1:1 wie bei dir)
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_folder_tree", type=Path, required=True)
    p.add_argument("--input_folder_allel", type=Path, required=True)
    p.add_argument("--output_folder", type=Path, required=True)
    p.add_argument("--simulators", type=lambda s: s.split(","), required=True)
    p.add_argument("--grouped_by", type=csv_list, default=["single"])
    p.add_argument("--es_rounds", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()

    grouped_by = args.grouped_by

    normalized_grouped_by = []
    sample_size_group = None
    seq_len_group = None

    for mode in grouped_by:
        if mode.startswith("sample_size"):
            m = re.search(r"\d+", mode)
            if not m:
                raise ValueError("sample_size mode needs a number")
            sample_size_group = int(m.group())
            normalized_grouped_by.append("sample_size")

        elif mode.startswith("seq_len"):
            m = re.search(r"\d+", mode)
            if not m:
                raise ValueError("seq_len mode needs a number")
            seq_len_group = int(m.group())
            normalized_grouped_by.append("seq_len")

        else:
            normalized_grouped_by.append(mode)

    grouped_by = set(normalized_grouped_by)

    args.output_folder.mkdir(parents=True, exist_ok=True)

    tree_csvs = get_csv_files_mode(
        args.input_folder_tree, grouped_by, seq_len_group, sample_size_group
    )
    allel_csvs = get_csv_files_mode(
        args.input_folder_allel, grouped_by, seq_len_group, sample_size_group
    )

    if not tree_csvs or not allel_csvs:
        raise ValueError("No CSVs found")

    run_feature_evaluation(
        simulator_group=args.simulators,
        tree_csv_files=tree_csvs,
        allel_csv_files=allel_csvs,
        output_dir=args.output_folder,
        es_rounds=args.es_rounds,
    )


if __name__ == "__main__":
    main()