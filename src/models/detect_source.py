from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def collect_ms_csvs(folder: Path):
    """
    Collect all CSV files containing 'ms_' in the filename.
    """
    return [p for p in folder.rglob("*.csv") if p.name.startswith("ms_")]


def load_dataset(dir1: Path, dir2: Path):

    files1 = collect_ms_csvs(dir1)
    files2 = collect_ms_csvs(dir2)

    if not files1:
        raise ValueError(f"No ms_ CSV files found in {dir1}")

    if not files2:
        raise ValueError(f"No ms_ CSV files found in {dir2}")

    dfs = []

    for f in files1:
        df = pd.read_csv(f)
        df["label"] = 0
        dfs.append(df)

    for f in files2:
        df = pd.read_csv(f)
        df["label"] = 1
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # difficulty handling (same logic as your other script)
    if "difficulty" in data.columns:
        data["difficulty_missing"] = (data["difficulty"] == -1).astype(np.int8)
        data.loc[data["difficulty"] == -1, "difficulty"] = np.nan

    return data


def clean_data(df: pd.DataFrame):

    df = df.replace([np.inf, -np.inf], np.nan)

    # drop rows containing NaNs
    df = df.dropna()

    return df


def compute_dataset_stats(data):

    counts = data["label"].value_counts().to_dict()

    n_dir1 = counts.get(0, 0)
    n_dir2 = counts.get(1, 0)

    total = n_dir1 + n_dir2

    chance_baseline = max(n_dir1, n_dir2) / total

    balanced = n_dir1 == n_dir2

    return n_dir1, n_dir2, chance_baseline, balanced


def train_xgb(X, y, output_path):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=0,
    )

    if "tree" in str(output_path):

        xgb_params = {
            "n_estimators": 300,
            "learning_rate": 0.08,
            "max_depth": 5,
            "min_child_weight": 8.68,
            "gamma": 2.21,
            "subsample": 0.79,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.12,
            "reg_alpha": 0.22,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 0,
        }

    elif "allele" in str(output_path):

        xgb_params = {
            "n_estimators": 300,
            "learning_rate": 0.10,
            "max_depth": 5,
            "min_child_weight": 8.92,
            "gamma": 1.62,
            "subsample": 0.71,
            "colsample_bytree": 0.75,
            "reg_lambda": 3.87,
            "reg_alpha": 0.08,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 0,
        }

    else:
        raise ValueError("Output file path must contain 'tree' or 'allele'.")

    # difficulty median imputation
    if "difficulty" in X_train.columns:
        median_diff = X_train["difficulty"].median(skipna=True)
        X_train["difficulty"] = X_train["difficulty"].fillna(median_diff)
        X_test["difficulty"] = X_test["difficulty"].fillna(median_diff)

    model = xgb.XGBClassifier(**xgb_params)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    return acc


def save_results(output_file: Path, n1, n2, chance, balanced, accuracy):

    rows = [
        {"metric": "N_DIR1", "value": n1},
        {"metric": "N_DIR2", "value": n2},
        {"metric": "BALANCED_DATASET", "value": balanced},
        {"metric": "CHANCE_BASELINE", "value": chance},
        {"metric": "MODEL_ACCURACY", "value": accuracy},
    ]

    df = pd.DataFrame(rows)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir1", type=Path, required=True)
    parser.add_argument("--input_dir2", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)

    args = parser.parse_args()

    data = load_dataset(args.input_dir1, args.input_dir2)

    data = clean_data(data)

    n1, n2, chance, balanced = compute_dataset_stats(data)

    y = data["label"]
    X = data.drop(columns=["label"])

    accuracy = train_xgb(X, y, args.output_file)

    save_results(
        args.output_file,
        n1,
        n2,
        chance,
        balanced,
        accuracy
    )


if __name__ == "__main__":
    main()