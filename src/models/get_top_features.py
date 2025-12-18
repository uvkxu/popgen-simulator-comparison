from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import sys
import os
import yaml
from itertools import combinations
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import VALID_GROUPING_MODES, format_feature_ranking_file, \
                        format_feature_ranking_barplot, format_classifier_importance_plot, \
                        XGB_STR, RF_STR, LR_STR, CLASSIFIERS
from utils.io_helpers import collect_csv_files, assign_simulator_labels


def evaluate_feature(X, y, feature, classifiers, repeats=2, test_size=0.2):
    """Train each classifier on a single feature multiple times and return max accuracy per classifier."""
    acc_dict = {clf_name: [] for clf_name in classifiers}
    X_feat = X[[feature]].values

    for clf_name, clf in classifiers.items():
        for r in range(repeats):
            X_train, X_valid, y_train, y_valid = train_test_split(X_feat, y, test_size=test_size)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_valid)
            acc_dict[clf_name].append(accuracy_score(y_valid, preds))

    return {clf_name: np.max(acc_list) for clf_name, acc_list in acc_dict.items()}


def evaluate_classifiers(X, y, classifiers, repeats=2, test_size=0.2):
    """Evaluate classifiers on all features together."""
    acc_dict = {clf_name: [] for clf_name in classifiers}
    importance_dict = {}

    feature_names = X.columns.tolist()

    for clf_name, clf in classifiers.items():
        for _ in range(repeats):
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_valid)
            acc_dict[clf_name].append(accuracy_score(y_valid, preds))
        
        importance_dict[clf_name] = get_feature_importance(clf, feature_names)
    
    acc_dict_mean = {clf_name: np.mean(acc_list) for clf_name, acc_list in acc_dict.items()}

    return acc_dict_mean, importance_dict


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):  # RF, XGB
        importance = model.feature_importances_
    elif isinstance(model, LogisticRegression):
        if model.coef_.ndim == 1 or model.coef_.shape[0] == 1:
            importance = np.abs(model.coef_).flatten()
        else:
            importance = np.mean(np.abs(model.coef_), axis=0)
    else:
        raise ValueError(f"Unsupported model type {type(model)}")

    return dict(zip(feature_names, importance))


def plot_feature_performance(feature_importance_dict, csv_path, group, chance_baseline=None):
    """
    Reads the evaluation CSV and produces:
    - one accuracy bar plot
    - one combined feature importance plot
    - one feature importance plot per classifier (new!)
    """

    # --- Accuracy plot ---
    df = pd.read_csv(csv_path)

    # Remove empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Convert numeric columns to float
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only actual feature rows
    df_features = df[~df["feature"].isin(["ALL_FEATURES", "PASSED_FEATURES", "OVERALL_MEAN"])]

    features_acc = df_features["feature"]
    metrics = ["max_accuracy"] + CLASSIFIERS
    x = np.arange(len(features_acc))
    width = 0.2

    plt.figure(figsize=(12, 8))

    if chance_baseline is not None:
        plt.axhline(
            y=chance_baseline,
            linestyle="--",
            linewidth=1.2,
            label=f"Chance baseline: {chance_baseline:.2f}"
        )

    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, df_features[metric], width=width, label=metric)

    plt.xticks(x + width*1.5, features_acc, rotation=90, fontsize=6)
    plt.ylabel("Accuracy")
    plt.xlabel("Feature")
    plt.title("Model Performance per Feature")
    plt.legend()
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_ranking_plot = format_feature_ranking_barplot(group)
    out_path = os.path.join(os.path.dirname(csv_path), output_ranking_plot)
    plt.savefig(out_path, dpi=300)
    plt.close()
    # print(f"Accuracy plot saved at: {out_path}")

    # --- One plot per classifier (NEW) ---
    for clf_name, importance_dict_single in feature_importance_dict.items():
        features = list(importance_dict_single.keys())
        values = [importance_dict_single[f] for f in features]

        plt.figure(figsize=(10, 6))
        plt.bar(features, values)
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Feature Importance")
        plt.xlabel("Feature")
        plt.title(f"Feature Importance – {clf_name}")
        plt.tight_layout()

        out_path_single = os.path.join(
            os.path.dirname(csv_path),
            format_classifier_importance_plot(clf_name, group)
        )

        plt.savefig(out_path_single, dpi=300)
        plt.close()

        # print(f"Feature importance plot for {clf_name} saved at: {out_path_single}")


def run_feature_evaluation(
        simulator_group,
        csv_files,
        output_dir,
        acc_factor=1.0
    ):
    # print(f"\n=== Evaluating simulator group: {simulator_group} ===")

    # --- 2. Load and label data ---
    all_dfs = assign_simulator_labels(csv_files, simulator_group)
    data = pd.concat(all_dfs, ignore_index=True)

    # --- Handle inf / NaN ---
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(axis=1, inplace=True)

    # --- Prepare features ---
    dep_var = 'simulator'
    features = [c for c in data.columns if c != dep_var]
    data = data[data[dep_var].isin(simulator_group)]
    y = LabelEncoder().fit_transform(data[dep_var])
    X = data[features]

    # --- Print simulator distribution ---
    print("\nSimulator distribution:")
    counts = data["simulator"].value_counts()
    for sim in simulator_group:
        print(f"  {sim}: {counts.get(sim, 0)} samples")


    # --- Classifiers ---
    classifiers = {
        XGB_STR: xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=len(simulator_group),
            eval_metric='mlogloss'
        ),
        RF_STR: RandomForestClassifier(n_estimators=100, max_depth=5),
        LR_STR: LogisticRegression(max_iter=500)
    }

    # --- Feature evaluation ---
    class_counts = np.bincount(y)
    majority_prop = class_counts.max() / class_counts.sum()

    min_accuracy = majority_prop * acc_factor
    feature_records = []
    passed_features = []

    for feature in features:
        try:
            acc_per_clf = evaluate_feature(X, y, feature, classifiers)
            max_acc = np.nanmax(list(acc_per_clf.values()))
        except ValueError as e:
            if "only one class" in str(e):
                # print(f"⚠️ Skipping feature '{feature}' – only one class.")
                acc_per_clf = {clf: np.nan for clf in classifiers}
                max_acc = np.nan
            else:
                raise

        if not np.isnan(max_acc) and max_acc >= min_accuracy:
            passed_features.append(feature)

        record = {"feature": feature, "max_accuracy": max_acc}
        record.update(acc_per_clf)
        feature_records.append(record)

    feature_records_sorted = sorted(
        feature_records,
        key=lambda x: x["max_accuracy"],
        reverse=True
    )

    # --- Evaluate classifiers with all features ---
    all_features_acc, importance_dict = evaluate_classifiers(X, y, classifiers)

    # --- Evaluate with passed features ---
    if passed_features:
        X_passed = X[passed_features]
        passed_acc, _ = evaluate_classifiers(X_passed, y, classifiers)
    else:
        passed_acc = {clf_name: None for clf_name in classifiers}

    # --- Compute mean accuracy ---
    overall_mean_acc = np.mean(list(all_features_acc.values()))

    # --- Summary output ---
    summary_records = []

    # All feature row
    summary_records.append({
        "feature": "ALL_FEATURES",
        "max_accuracy": np.max(list(all_features_acc.values())),
        **all_features_acc
    })

    # Passed features
    valid_values = [v for v in passed_acc.values() if v is not None]
    max_acc_passed = np.max(valid_values) if valid_values else None

    summary_records.append({
        "feature": "PASSED_FEATURES",
        "max_accuracy": max_acc_passed,
        **passed_acc
    })

    # Blank row
    summary_records.append({col: np.nan for col in feature_records_sorted[0].keys()})

    passed_records = [r for r in feature_records_sorted if r["feature"] in passed_features]
    failed_records = [r for r in feature_records_sorted if r["feature"] not in passed_features]

    summary_records.extend(passed_records)
    summary_records.append({col: np.nan for col in feature_records_sorted[0].keys()})
    summary_records.append({col: np.nan for col in feature_records_sorted[0].keys()})
    summary_records.extend(failed_records)

    summary_records.append({
        "feature": "OVERALL_MEAN",
        "mean_accuracy": overall_mean_acc,
        **all_features_acc
    })

    df_summary = pd.DataFrame(summary_records)

    cols = ["feature", "max_accuracy"] + [c for c in classifiers.keys()]
    df_summary = df_summary[cols]
    csv_file = output_dir / format_feature_ranking_file(simulator_group)

    df_summary.to_csv(csv_file, index=False)

    plot_feature_performance(importance_dict, csv_file, simulator_group, chance_baseline=majority_prop)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract top features from simulation results."
    )

    parser.add_argument(
        "--input_folder",
        type=Path,
        required=True,
        help="Input folder containing result files",
    )

    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Output folder path",
    )

    parser.add_argument(
        "--acc_factor",
        type=float,
        default=1.2,
        help="Accuracy factor (default: 1.2)",
    )

    parser.add_argument(
        "--simulators",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated list of simulators (e.g. sim1,sim2)",
    )

    parser.add_argument(
        "--grouped_by",
        type=str,
        choices=VALID_GROUPING_MODES,
        default="single",
        help=f"Grouping mode. Choices: {', '.join(VALID_GROUPING_MODES)} (default: single)",
    )

    return parser.parse_args()


def assert_non_empty(csv_files, mode):
    if not csv_files:
        raise ValueError(
            f"No CSV files found for mode '{mode}'. "
            "This indicates an inconsistent input directory or configuration."
        )


def main():
    args = parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    acc_factor = args.acc_factor
    simulator_group = args.simulators
    grouped_by = args.grouped_by

    output_folder.mkdir(parents=True, exist_ok=True)

    csv_with_precision = []
    csv_without_precision = []

    csv_with_recombination = []
    csv_without_recombination = []

    if "single" == grouped_by or  "overall" == grouped_by:
        csv_files = collect_csv_files(input_folder)
        run_feature_evaluation(
            simulator_group=simulator_group,
            csv_files=csv_files,
            output_dir=output_folder,
            acc_factor=acc_factor
        )
        return

    for root, _, files in os.walk(input_folder):
        root_path = Path(root)

        combo_yamls = [f for f in files if f.endswith(".yaml") or f.endswith(".yml")]
        if not combo_yamls:
            continue
        
        combo_yaml_path = root_path / combo_yamls[0]
        with open(combo_yaml_path, "r") as f:
            combo_params = yaml.safe_load(f)

        usePrecision = bool(combo_params.get("usePrecision"))
        recombination_rate = float(combo_params.get("recombination_rate") or 0)

        csv_files = collect_csv_files(root_path)

        if usePrecision:
            csv_with_precision += csv_files
            if recombination_rate:
                csv_with_recombination += csv_files
            else:
                csv_without_recombination += csv_files
        else:
            csv_without_precision += csv_files

    if "precision_applied" == grouped_by:
        assert_non_empty(csv_files=csv_with_precision, mode=grouped_by)
        run_feature_evaluation(
            simulator_group=simulator_group,
            csv_files=csv_with_precision,
            output_dir=output_folder,
            acc_factor=acc_factor,
        )
            
    elif "precision_skipped" == grouped_by:
        assert_non_empty(csv_files=csv_without_precision, mode=grouped_by)
        run_feature_evaluation(
            simulator_group=simulator_group,
            csv_files=csv_without_precision,
            output_dir=output_folder,
            acc_factor=acc_factor,
        )
    
    elif "recombination_applied" == grouped_by:
        assert_non_empty(csv_files=csv_with_recombination, mode=grouped_by)
        run_feature_evaluation(
            simulator_group=simulator_group,
            csv_files=csv_with_recombination,
            output_dir=output_folder,
            acc_factor=acc_factor,
        )

    elif "recombination_skipped" == grouped_by:
        assert_non_empty(csv_files=csv_without_precision, mode=grouped_by)
        run_feature_evaluation(
            simulator_group=simulator_group,
            csv_files=csv_without_recombination,
            output_dir=output_folder,
            acc_factor=acc_factor,
        )


if __name__ == "__main__":
    main()





