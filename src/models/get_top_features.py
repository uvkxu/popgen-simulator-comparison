from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import sys
import os
from itertools import combinations

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import SIMULATORS, FILE_FORMATS
from utils.io_helpers import collect_csv_files, assign_simulator_labels

def evaluate_feature(X, y, feature, classifiers, repeats=2, test_size=0.2):
    """Train each classifier on a single feature multiple times and return max accuracy per classifier."""
    acc_dict = {clf_name: [] for clf_name in classifiers}
    X_feat = X[[feature]].values

    for clf_name, clf in classifiers.items():
        for _ in range(repeats):
            X_train, X_valid, y_train, y_valid = train_test_split(X_feat, y, test_size=test_size)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_valid)
            acc_dict[clf_name].append(accuracy_score(y_valid, preds))

    return {clf_name: np.max(acc_list) for clf_name, acc_list in acc_dict.items()}

def evaluate_classifiers(X, y, classifiers, repeats=2, test_size=0.2):
    """Evaluate classifiers on all features together."""
    acc_dict = {clf_name: [] for clf_name in classifiers}

    for clf_name, clf in classifiers.items():
        for _ in range(repeats):
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_valid)
            acc_dict[clf_name].append(accuracy_score(y_valid, preds))

    return {clf_name: np.mean(acc_list) for clf_name, acc_list in acc_dict.items()}

def main():
    if len(sys.argv) < 3:
        print("Usage: python get_top_features.py <input_folder> <output_folder> [--acc_factor=FLOAT] [--simulators=sim1,sim2,...]")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    acc_factor = 1.2
    selected_simulators = SIMULATORS

    for arg in sys.argv[3:]:
        if arg.startswith("--acc_factor"):
            acc_factor = float(arg.split("=")[1])
        elif arg.startswith("--simulators"):
            selected_simulators = arg.split("=")[1].split(",")

    # --- 1. Generate simulator groups ---
    simulator_groups = list(set(combinations(selected_simulators, 2)))
    simulator_groups.append(tuple(selected_simulators))
    if set(selected_simulators) == set(SIMULATORS):
        without_msms_ms = tuple(s for s in selected_simulators if s != "msms_ms")
        simulator_groups.append(without_msms_ms)
    simulator_groups = list(dict.fromkeys(simulator_groups))

    for group in simulator_groups:
        print(f"\n=== Evaluating simulator group: {group} ===")

        CSV_TYPES = [FILE_FORMATS["stats"].format(SIM=sim) for sim in group]
        csv_files_dict = collect_csv_files(input_folder, CSV_TYPES)
        all_dfs = assign_simulator_labels(csv_files_dict, group)
        data = pd.concat(all_dfs, ignore_index=True)

        # --- 2. Handle Inf and missing values ---
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(axis=1, inplace=True)

        # --- 3. Prepare features and target ---
        dep_var = 'simulator'
        features = [c for c in data.columns if c != dep_var]
        data = data[data[dep_var].isin(group)]
        y = LabelEncoder().fit_transform(data[dep_var])
        X = data[features]

        # --- 4. Define classifiers ---
        classifiers = {
            "xgboost": xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                objective='multi:softmax',
                num_class=len(group),
                eval_metric='mlogloss'
            ),
            "random_forest": RandomForestClassifier(n_estimators=100, max_depth=5),
            "logistic_regression": LogisticRegression(max_iter=500)
        }

        # --- 5. Evaluate each feature ---
        min_accuracy = 1 / len(group) * acc_factor
        feature_records = []
        passed_features = []

        for feature in features:
            acc_per_clf = evaluate_feature(X, y, feature, classifiers)
            max_acc = np.max(list(acc_per_clf.values()))
            if max_acc >= min_accuracy:
                passed_features.append(feature)
            record = {"feature": feature, "max_accuracy": max_acc}
            record.update(acc_per_clf)
            feature_records.append(record)

        feature_records_sorted = sorted(feature_records, key=lambda x: x["max_accuracy"], reverse=True)

        # --- 6. Evaluate classifiers with all features ---
        all_features_acc = evaluate_classifiers(X, y, classifiers)
        print(f"All features accuracy per classifier: {all_features_acc}")

        # --- 7. Evaluate classifiers with only passed features ---
        if passed_features:
            X_passed = X[passed_features]
            passed_acc = evaluate_classifiers(X_passed, y, classifiers)
        else:
            passed_acc = {clf_name: None for clf_name in classifiers}
        print(f"Passed features accuracy per classifier: {passed_acc}")

        # --- 8. Overall mean accuracy ---
        overall_mean_acc = np.mean(list(all_features_acc.values()))
        print(f"Overall mean accuracy (all features, all classifiers): {overall_mean_acc:.4f}")

        # --- 9. Save results (passed features first, spacing) ---
        output_folder.mkdir(exist_ok=True)
        sim_part = "_".join(group)
        summary_records = []

        # ALL_FEATURES row
        summary_records.append({"feature": "ALL_FEATURES", "max_accuracy": np.max(list(all_features_acc.values())), **all_features_acc})

        # PASSED_FEATURES row
        valid_values = [v for v in passed_acc.values() if v is not None]
        if valid_values:
            max_acc = np.max(valid_values)
        else:
            max_acc = None  # or float('nan'), depending on your downstream usage

        summary_records.append({"feature": "PASSED_FEATURES", "max_accuracy": max_acc, **passed_acc})

        # Blank row
        summary_records.append({col: np.nan for col in feature_records_sorted[0].keys()})

        # Separate passed and failed features
        passed_feature_records = [r for r in feature_records_sorted if r["feature"] in passed_features]
        failed_feature_records = [r for r in feature_records_sorted if r["feature"] not in passed_features]

        # Add passed features
        summary_records.extend(passed_feature_records)

        # Two blank rows for spacing
        summary_records.append({col: np.nan for col in feature_records_sorted[0].keys()})
        summary_records.append({col: np.nan for col in feature_records_sorted[0].keys()})

        # Add failed features
        summary_records.extend(failed_feature_records)

        # OVERALL_MEAN row
        summary_records.append({"feature": "OVERALL_MEAN", "mean_accuracy": overall_mean_acc, **all_features_acc})

        df_summary = pd.DataFrame(summary_records)

        # Column order
        cols = ["feature", "max_accuracy"] + [c for c in classifiers.keys()]
        df_summary = df_summary[cols]

        output_file = output_folder / f"feature_ranking_{sim_part}_with_summary.csv"
        df_summary.to_csv(output_file, index=False)
        print(f"Feature ranking and classifier accuracies saved to {output_file}")

if __name__ == "__main__":
    main()




