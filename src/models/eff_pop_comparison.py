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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import sys
import os
import yaml
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import VALID_GROUPING_MODES, XGB_STR, RF_STR, LR_STR, CLASSIFIERS
from utils.io_helpers import collect_csv_files, detect_simulator_from_yaml


def evaluate_feature(X, y, feature, classifiers, repeats=2, test_size=0.2):
    """Train each classifier on a single feature multiple times and return max accuracy per classifier."""
    acc_dict = {clf_name: [] for clf_name in classifiers}
    X_feat = X[[feature]].values

    for clf_name, clf in classifiers.items():
        for _ in range(repeats):
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_feat, y, test_size=test_size, stratify=y
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(X_valid)
            acc_dict[clf_name].append(accuracy_score(y_valid, preds))

    return {clf_name: float(np.max(acc_list)) for clf_name, acc_list in acc_dict.items()}


def evaluate_classifiers(X, y, classifiers, repeats=2, test_size=0.2):
    """Evaluate classifiers on all features together; returns mean accuracy per classifier."""
    acc_dict = {clf_name: [] for clf_name in classifiers}

    for clf_name, clf in classifiers.items():
        for _ in range(repeats):
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=test_size, stratify=y
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(X_valid)
            acc_dict[clf_name].append(accuracy_score(y_valid, preds))

    return {clf_name: float(np.mean(acc_list)) for clf_name, acc_list in acc_dict.items()}


def plot_feature_performance(csv_path, simulator_name, chance_baseline=None):
    """
    Reads the evaluation CSV and produces:
    - one accuracy bar plot
    """
    df = pd.read_csv(csv_path)

    # Remove empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Convert numeric columns to float
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only actual feature rows
    df_features = df[~df["feature"].isin(["ALL_FEATURES", "PASSED_FEATURES", "OVERALL_MEAN"])]

    features_acc = df_features["feature"]
    metrics = ["max_accuracy"] + list(CLASSIFIERS)
    x = np.arange(len(features_acc))
    width = 0.2

    plt.figure(figsize=(12, 8))

    if chance_baseline is not None:
        plt.axhline(
            y=chance_baseline,
            linestyle="--",
            linewidth=1.2,
            label=f"Chance baseline: {chance_baseline:.2f}",
        )

    for i, metric in enumerate(metrics):
        if metric in df_features.columns:
            plt.bar(x + i * width, df_features[metric], width=width, label=metric)

    plt.xticks(x + width * 1.5, features_acc, rotation=90, fontsize=6)
    plt.ylabel("Accuracy")
    plt.xlabel("Feature")
    plt.title("Model Performance per Feature")
    plt.legend()
    plt.tight_layout()

    output_ranking_plot = f"check_eff_pop_size_{simulator_name}_plot.png"
    out_path = os.path.join(os.path.dirname(csv_path), output_ranking_plot)
    plt.savefig(out_path, dpi=300)
    plt.close()


def clean_nan_data(
    data: pd.DataFrame,
    output_csv: Path,
    dep_var: str,
    max_row_loss_frac: float = 0.05,
):
    log_path = output_csv.parent / f"nan_cleanup_{output_csv.stem}.log"
    log_lines = []

    # Replace inf with NaN
    data = data.replace([np.inf, -np.inf], np.nan)

    n_rows_initial = len(data)
    if n_rows_initial == 0:
        return data  # empty DF

    for col in list(data.columns):
        if col == dep_var:
            continue

        if data[col].isna().any():
            n_nan = int(data[col].isna().sum())
            frac_nan = n_nan / n_rows_initial

            if frac_nan < max_row_loss_frac:
                data = data[~data[col].isna()]
                log_lines.append(
                    f"Dropped {n_nan} rows ({frac_nan:.2%}) due to NaNs in column '{col}'."
                )
            else:
                data = data.drop(columns=[col])
                log_lines.append(
                    f"Dropped column '{col}' ({frac_nan:.2%} NaNs >= {max_row_loss_frac:.2%})."
                )

    if log_lines:
        with open(log_path, "w") as f:
            for line in log_lines:
                f.write(line + "\n")

    return data


def run_feature_evaluation(simulator, csv_files_by_ne, output_dir, acc_factor=1.0):
    """
    csv_files_by_ne: dict[int, list[Path]] mapping effective_pop_size -> list of csv files
    """
    dep_var = "effective_pop_size"

    # Build one big DF and label each row with Ne
    dfs = []
    for eff_pop_size, files in csv_files_by_ne.items():
        for f in files:
            df = pd.read_csv(f)
            df[dep_var] = int(eff_pop_size)
            dfs.append(df)

    if not dfs:
        raise ValueError("No data loaded: dfs is empty. Check your input folder and filtering.")

    data = pd.concat(dfs, ignore_index=True)
    del dfs

    csv_file = output_dir / f"check_eff_pop_size_{simulator}_summary.csv"

    # Clean inf/NaN
    data = clean_nan_data(data, output_csv=csv_file, dep_var=dep_var)

    # Cast float64 -> float32
    num_cols = data.select_dtypes(include=["float64"]).columns
    data[num_cols] = data[num_cols].astype(np.float32)

    # Keep only classes that are present in dict (robustness)
    classes = sorted(int(k) for k in csv_files_by_ne.keys())
    data = data[data[dep_var].isin(classes)].copy()

    # Prepare X/y
    le = LabelEncoder()
    y = le.fit_transform(data[dep_var].astype(int))
    X = data.drop(columns=[dep_var])
    features = X.columns.tolist()
    del data

    # Classifiers
    classifiers = {
        XGB_STR: xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective="multi:softmax",
            num_class=len(le.classes_),
            eval_metric="mlogloss",
        ),
        RF_STR: RandomForestClassifier(n_estimators=100, max_depth=5),
        LR_STR: LogisticRegression(max_iter=500),
    }

    # Chance baseline (majority class)
    class_counts = np.bincount(y)
    majority_prop = float(class_counts.max() / class_counts.sum())
    min_accuracy = majority_prop * acc_factor

    feature_records = []
    passed_features = []

    for feature in features:
        try:
            acc_per_clf = evaluate_feature(X, y, feature, classifiers)
            max_acc = float(np.nanmax(list(acc_per_clf.values())))
        except ValueError as e:
            if "only one class" in str(e):
                acc_per_clf = {clf: np.nan for clf in classifiers}
                max_acc = np.nan
            else:
                raise

        if not np.isnan(max_acc) and max_acc >= min_accuracy:
            passed_features.append(feature)

        record = {"feature": feature, "max_accuracy": max_acc}
        record.update(acc_per_clf)
        feature_records.append(record)

    feature_records_sorted = sorted(feature_records, key=lambda x: x["max_accuracy"], reverse=True)

    # Evaluate all features
    all_features_acc = evaluate_classifiers(X, y, classifiers)

    # Evaluate passed features
    if passed_features:
        X_passed = X[passed_features]
        passed_acc = evaluate_classifiers(X_passed, y, classifiers)
    else:
        passed_acc = {clf_name: None for clf_name in classifiers}

    overall_mean_acc = float(np.mean(list(all_features_acc.values())))

    # Summary
    summary_records = []

    summary_records.append(
        {
            "feature": "ALL_FEATURES",
            "max_accuracy": float(np.max(list(all_features_acc.values()))),
            **all_features_acc,
        }
    )

    valid_values = [v for v in passed_acc.values() if v is not None]
    max_acc_passed = float(np.max(valid_values)) if valid_values else None

    summary_records.append(
        {
            "feature": "PASSED_FEATURES",
            "max_accuracy": max_acc_passed,
            **passed_acc,
        }
    )

    if feature_records_sorted:
        blank_row = {col: np.nan for col in feature_records_sorted[0].keys()}
        summary_records.append(blank_row)

        passed_records = [r for r in feature_records_sorted if r["feature"] in passed_features]
        failed_records = [r for r in feature_records_sorted if r["feature"] not in passed_features]

        summary_records.extend(passed_records)
        summary_records.append(blank_row)
        summary_records.append(blank_row)
        summary_records.extend(failed_records)

    # OVERALL_MEAN row
    mean_row = {"feature": "OVERALL_MEAN", "max_accuracy": np.nan, "mean_accuracy": overall_mean_acc}
    mean_row.update(all_features_acc)
    summary_records.append(mean_row)

    df_summary = pd.DataFrame(summary_records)

    cols = ["feature", "max_accuracy"] + list(classifiers.keys())
    # keep existing cols only
    cols = [c for c in cols if c in df_summary.columns]
    df_summary = df_summary[cols]

    df_summary.to_csv(csv_file, index=False)

    plot_feature_performance(csv_file, simulator_name=simulator, chance_baseline=majority_prop)

def csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def parse_args():
    parser = argparse.ArgumentParser(description="Extract top features from simulation results.")

    parser.add_argument("--input_folder", type=Path, required=True, help="Input folder containing result files")
    parser.add_argument("--output_folder", type=Path, required=True, help="Output folder path")
    parser.add_argument("--acc_factor", type=float, default=1.2, help="Accuracy factor (default: 1.2)")
    parser.add_argument("--simulator", type=str, default=None, help="Simulator name to filter on (required)")
    parser.add_argument(
        "--grouped_by",
        type=csv_list,
        default=["single"],
        help=(
            "Comma-separated list of grouping modes, e.g. "
            "'precision_applied,recombination_skipped,sample_size500'. "
            f"Base choices: {', '.join(VALID_GROUPING_MODES)} "
            "(sample_sizeN / seq_lenN allowed)."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    acc_factor = args.acc_factor
    grouped_by = args.grouped_by
    simulator = args.simulator
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

    if simulator is None:
        raise ValueError("--simulator must be provided to filter CSV files by simulator.")

    output_folder.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: dict must live OUTSIDE os.walk loop
    assign_eff_pop_size = {}

    for root, _, files in os.walk(input_folder):
        root_path = Path(root)

        combo_yamls = [f for f in files if f.endswith(".yaml") or f.endswith(".yml")]
        if not combo_yamls:
            continue

        combo_yaml_path = root_path / combo_yamls[0]
        with open(combo_yaml_path, "r") as f:
            combo_params = yaml.safe_load(f) or {}

        usePrecision = bool(combo_params.get("usePrecision"))
        recombination_rate = float(combo_params.get("recombination_rate") or 0.0)
        effective_pop_size = int(combo_params.get("population_size"))

        csv_files = collect_csv_files(root_path)

        # grouping filter
        if (
            (("precision_applied" not in grouped_by) or usePrecision) and
            (("precision_skipped" not in grouped_by) or (not usePrecision)) and
            (("recombination_applied" not in grouped_by) or (recombination_rate > 0)) and
            (("recombination_skipped" not in grouped_by) or (recombination_rate == 0)) and
            (("sample_size" not in grouped_by) or (sample_size == sample_size_group and usePrecision)) and
            (("seq_len" not in grouped_by) or (seq_len == seq_len_group and usePrecision))
        ):
            # simulator filter
            csv_files = [f for f in csv_files if detect_simulator_from_yaml(f) == simulator]

            if csv_files:
                assign_eff_pop_size.setdefault(effective_pop_size, []).extend(csv_files)

    if not assign_eff_pop_size:
        raise ValueError("No CSV files matched your filters. Check --simulator, --grouped_by, and input_folder.")

    run_feature_evaluation(
        simulator=simulator,
        csv_files_by_ne=assign_eff_pop_size,
        output_dir=output_folder,
        acc_factor=acc_factor,
    )


if __name__ == "__main__":
    main()
