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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import SIMULATORS, FILE_FORMATS
from utils.io_helpers import collect_csv_files, assign_simulator_labels

def evaluate_feature(X, y, feature, classifiers, repeats=3, test_size=0.2):
    """Train each classifier on a single feature multiple times and return mean accuracy."""
    acc_list = []
    X_feat = X[[feature]].values  # nur die eine Feature-Spalte

    for clf_name, clf in classifiers.items():
        for _ in range(repeats):
            X_train, X_valid, y_train, y_valid = train_test_split(X_feat, y, test_size=test_size)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_valid)
            acc_list.append(accuracy_score(y_valid, preds))
    return np.mean(acc_list)

def main():
    if len(sys.argv) < 3:
        print("Usage: python get_top_features.py <input_folder> <output_folder> [--top_n=N] [--simulators=sim1,sim2,...]")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    top_n = 5  # default
    selected_simulators = SIMULATORS  # default: alle

    # Parse optional arguments
    for arg in sys.argv[3:]:
        if arg.startswith("--top_n"):
            top_n = int(arg.split("=")[1])
        elif arg.startswith("--simulators"):
            selected_simulators = arg.split("=")[1].split(",")

    # ----------------------------
    # 1. Read CSV files and label them by simulator
    # ----------------------------
    CSV_TYPES = [FILE_FORMATS["stats"].format(SIM=sim) for sim in selected_simulators]
    csv_files_dict = collect_csv_files(input_folder, CSV_TYPES)
    all_dfs = assign_simulator_labels(csv_files_dict)
    data = pd.concat(all_dfs, ignore_index=True)

    # ----------------------------
    # 2. Handle Inf and missing values
    # ----------------------------
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(axis=1, inplace=True)

    # ----------------------------
    # 3. Prepare features and target
    # ----------------------------
    dep_var = 'simulator'
    features = [c for c in data.columns if c != dep_var]

    # Filter rows for selected simulators only
    data = data[data[dep_var].isin(selected_simulators)]

    y = LabelEncoder().fit_transform(data[dep_var])
    X = data[features]

    # ----------------------------
    # 4. Define classifiers
    # ----------------------------
    classifiers = {
        "xgboost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=len(selected_simulators),
            eval_metric='mlogloss'
        ),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=5),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=5),
        "logistic_regression": LogisticRegression(max_iter=500)
    }

    # ----------------------------
    # 5. Evaluate each feature
    # ----------------------------
    min_accuracy = 1/len(selected_simulators) * 1.1
    feature_scores = []
    low_scores = []
    for feature in features:
        mean_acc = evaluate_feature(X, y, feature, classifiers)
        print(f"{feature}: {mean_acc:.4f}")
        if mean_acc >= min_accuracy:
            feature_scores.append((feature, mean_acc, True))
        else:
            low_scores.append((feature, mean_acc, False))

    all_features = feature_scores + low_scores
    all_features.sort(key=lambda x: x[1], reverse=True)

    # ----------------------------
    # 6. Save top_n features with dynamic filename
    # ----------------------------
    output_folder.mkdir(exist_ok=True)
    if set(selected_simulators) == set(SIMULATORS):
        sim_part = "all"
    else:
        sim_part = "_".join(selected_simulators)

    df_top = pd.DataFrame(all_features, columns=["feature", "mean_accuracy", "passed_threshold"])
    output_file = output_folder / f"feature_ranking_{sim_part}_with_threshold.csv"
    df_top.to_csv(output_file, index=False)
    print(f"Feature ranking saved to {output_file}")

if __name__ == "__main__":
    main()

