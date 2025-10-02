from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import SIMULATORS, FILE_FORMATS, NUM_SANITY_CLASSES, SANITY_REPS
from src.utils.io_helpers import collect_csv_files, assign_rep_labels

# ----------------------------
# Define classifiers here
# ----------------------------
CLASSIFIERS = {
    "gradient_boosted_tree": lambda: xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=NUM_SANITY_CLASSES,
        eval_metric='mlogloss'
    )
    # Hier später weitere Classifier hinzufügen
}

def run_classifier_multiple_times(clf_factory, X, y, reps=5):
    """Train a classifier 'reps' times and return the mean validation accuracy."""
    accs = []
    for _ in range(reps):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        clf = clf_factory()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_valid)
        accs.append(accuracy_score(y_valid, preds))
    return np.mean(accs), len(accs)


def main():
    if len(sys.argv) < 3:
        print("Usage: python sanity_check.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_base_folder = Path(sys.argv[2])

    config_folder = input_folder.name
    output_folder = output_base_folder / config_folder
    output_folder.mkdir(exist_ok=True)

    output_file = output_folder / FILE_FORMATS["sanity_check_file"]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    report_lines = []

    baseline = 1.0 / NUM_SANITY_CLASSES

    CSV_TYPES = [FILE_FORMATS["treestats"].format(SIM=sim) for sim in SIMULATORS]

    for typ in CSV_TYPES:
        csv_files_dict = collect_csv_files(input_folder, [typ])
        if not csv_files_dict:
            report_lines.append(f"No CSV files found for type: {typ}\n")
            continue

        all_dfs = assign_rep_labels(csv_files_dict, NUM_SANITY_CLASSES)
        data = pd.concat(all_dfs, ignore_index=True)

        # ----------------------------
        # Handle Inf and missing values
        # ----------------------------
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        cols_with_nan = [col for col in data.columns if data[col].isna().any()]

        report_lines.append(f"Columns to be dropped ({len(cols_with_nan)}):")
        for col in cols_with_nan:
            report_lines.append(f"- {col}")
        data.drop(columns=cols_with_nan, inplace=True)
        report_lines.append(f"Number of columns remaining: {data.shape[1]}\n")

        # ----------------------------
        # Features & Target
        # ----------------------------
        dep_var = 'rep'
        cont_names = list(data.columns)
        cont_names.remove(dep_var)
        y = LabelEncoder().fit_transform(data[dep_var])
        X = data[cont_names]

        # ----------------------------
        # Train and evaluate each classifier SANITY_REPS times
        # ----------------------------
        for clf_name, clf_factory in CLASSIFIERS.items():
            mean_acc, runs = run_classifier_multiple_times(clf_factory, X, y, reps=SANITY_REPS)
            report_lines.append(
                f"Classifier '{clf_name}' | config '{config_folder}' | simulator '{typ}' | "
                f"mean accuracy over {runs} runs: {mean_acc:.4f}"
            )

            if mean_acc > baseline * 1.3:
                warn_msg = (
                    f"⚠️ WARNING: Classifier '{clf_name}' | config '{config_folder}' | simulator '{typ}' "
                    f"mean accuracy ({mean_acc:.4f}) is much higher than random baseline ({baseline:.4f})"
                )
                print(warn_msg)
                report_lines.append(warn_msg)

        report_lines.append("")

    # ----------------------------
    # Write report
    # ----------------------------
    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Sanity check report written to {output_file}")


if __name__ == "__main__":
    main()
