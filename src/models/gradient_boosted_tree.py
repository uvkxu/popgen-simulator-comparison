from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import SIMULATORS, FILE_FORMATS
from utils.io_helpers import collect_csv_files, assign_simulator_labels


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_boxplots.py <input_folder> <output_folder> [--simulators sim1,sim2] [--stats stat1,stat2]")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])

    # Parse optional arguments
    selected_simulators = SIMULATORS  # default: all
    selected_stats = None  # default: all features

    for arg in sys.argv[3:]:
        if arg.startswith("--simulators"):
            selected_simulators = arg.split("=")[1].split(",")
        elif arg.startswith("--stats"):
            selected_stats = arg.split("=")[1].split(",")

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
    cols_with_nan = [col for col in data.columns if data[col].isna().any()]

    print(f"Columns to be dropped ({len(cols_with_nan)}):")
    for col in cols_with_nan:
        print(f"- {col}")
    data.drop(columns=cols_with_nan, inplace=True)
    print(f"Number of columns remaining: {data.shape[1]}")

    # ----------------------------
    # 3. Prepare features and target
    # ----------------------------
    dep_var = 'simulator'
    cont_names = list(data.columns)
    cont_names.remove(dep_var)

    # Apply stat selection if provided
    if selected_stats is not None:
        cont_names = [stat for stat in cont_names if stat in selected_stats]

    y = LabelEncoder().fit_transform(data[dep_var])
    X = data[cont_names]

    # ----------------------------
    # 4. Split into train and validation sets
    # ----------------------------
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    # ----------------------------
    # 5. Train Gradient Boosted Tree (XGBoost)
    # ----------------------------
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=len(selected_simulators),
        eval_metric='mlogloss'
    )

    model.fit(X_train, y_train)

    # ----------------------------
    # 6. Evaluate model
    # ----------------------------
    preds = model.predict(X_valid)
    print("Validation Accuracy:", accuracy_score(y_valid, preds))

    # ----------------------------
    # 7. Feature importance plots
    # ----------------------------
    output_folder.mkdir(exist_ok=True)

    fig1 = xgb.plot_importance(model, max_num_features=20, importance_type='weight')
    fig1.figure.tight_layout()
    fig1.figure.savefig(output_folder / "feature_importance_weight.png")
    plt.close(fig1.figure)

    fig2 = xgb.plot_importance(model, max_num_features=20, importance_type='gain')
    fig2.figure.tight_layout()
    fig2.figure.savefig(output_folder / "feature_importance_gain.png")
    plt.close(fig2.figure)


if __name__ == "__main__":
    main()
