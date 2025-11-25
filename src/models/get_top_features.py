from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import sys
import os
import yaml
from itertools import combinations

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import SIMULATORS, FILE_FORMATS
from utils.io_helpers import collect_csv_files, assign_simulator_labels

def evaluate_feature(X, y, feature, classifiers, repeats=2, test_size=0.2, balance_train=None):
    """Train each classifier on a single feature multiple times and return max accuracy per classifier."""
    acc_dict = {clf_name: [] for clf_name in classifiers}
    X_feat = X[[feature]].values

    for clf_name, clf in classifiers.items():
        for r in range(repeats):
            X_train, X_valid, y_train, y_valid = train_test_split(X_feat, y, test_size=test_size)

            if balance_train == "over":
                X_train, y_train = balanced_train_oversample(X_train, y_train, random_state=42 + r)
            elif balance_train == "under":
                X_train, y_train = balanced_train_undersample(X_train, y_train, random_state=42 + r)
            
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


def balance_train_oversample(X_train, y_train, random_state=None):
    """
    Oversample minority classes in the training data so all classes are equally represented.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training target labels.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_resampled, y_resampled : balanced training data
    """
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    df_train = X_train.copy()
    df_train["target"] = y_train.values

    max_size = df_train["target"].value_counts().max()
    df_list = []

    for label, df_class in df_train.groupby("target"):
        df_resampled = resample(
            df_class,
            replace=True,
            n_samples=max_size,
            random_state=random_state,
        )
        df_list.append(df_resampled)

    df_balanced = pd.concat(df_list, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    y_balanced = df_balanced["target"]
    X_balanced = df_balanced.drop(columns=["target"])

    return X_balanced, y_balanced


def balanced_train_undersample(X_train, y_train, random_state=None):
    """Return an undersampled, class-balanced version of (X_train, y_train)."""
    df = pd.DataFrame(X_train).copy()
    df['__target__'] = y_train
    counts = df['__target__'].value_counts()
    min_count = counts.min()

    balanced = []
    for cls in counts.index:
        cls_df = df[df['__target__'] == cls]
        sampled = cls_df.sample(n=min_count, random_state=random_state)
        balanced.append(sampled)

    df_bal = pd.concat(balanced).sample(frac=1, random_state=random_state)  # shuffle
    y_bal = df_bal['__target__'].values
    X_bal = df_bal.drop(columns='__target__').values
    return X_bal, y_bal


def plot_feature_performance(csv_path):
    """
    Reads the evaluation CSV and produces a plot stored in the same folder.
    """

    # Read CSV
    df = pd.read_csv(csv_path)

    # Remove completely empty rows (your file includes empty blocks)
    df = df.dropna(how="all")

    # Remove empty columns (due to trailing commas)
    df = df.dropna(axis=1, how="all")

    # Convert numeric columns to float
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only keep rows that contain actual feature names (exclude headers like ALL_FEATURES)
    df_features = df[~df["feature"].isin(["ALL_FEATURES", "PASSED_FEATURES", "OVERALL_MEAN"])]

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(df_features["feature"], df_features["max_accuracy"], label="max_accuracy")
    plt.plot(df_features["feature"], df_features["xgboost"], label="xgboost")
    plt.plot(df_features["feature"], df_features["random_forest"], label="random_forest")
    plt.plot(df_features["feature"], df_features["logistic_regression"], label="logistic_regression")

    plt.xticks(rotation=90, fontsize=6)
    plt.ylabel("Accuracy")
    plt.xlabel("Feature")
    plt.title("Model Performance per Feature")
    plt.legend()
    plt.tight_layout()

    # Save file next to the CSV
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(os.path.dirname(csv_path), f"{base}.png")

    # save plot
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Plot gespeichert unter: {out_path}")
    

def main():
    if len(sys.argv) < 3:
        print("Usage: python get_top_features.py <input_folder> <output_folder> "
              "[--acc_factor=FLOAT] [--simulators=sim1,sim2,...] "
              "[--filter_zero_branches] [--resample=none|under|over]")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    acc_factor = 1.2
    selected_simulators = None
    filter_zero_branches = False
    resample_mode = "none"

    for arg in sys.argv[3:]:
        if arg.startswith("--acc_factor"):
            acc_factor = float(arg.split("=")[1])
        elif arg.startswith("--simulators"):
            selected_simulators = arg.split("=")[1].split(",")
        elif arg == "--filter_zero_branches":
            filter_zero_branches = True  # 🟢 activate filtering
        elif arg.startswith("--resample="):
            resample_mode = arg.split("=")[1].strip().lower()
            if resample_mode not in ["none", "under", "over"]:
                print("Invalid --resample value. Use one of: none, under, over.")
                sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    for root, _, files in os.walk(input_folder):
        root_path = Path(root)

        metadata_files = [f for f in files if f.endswith(".yaml") or f.endswith(".yml")]
        if not metadata_files:
            continue
        
        metadata_path = root_path / metadata_files[0]
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
        
        if not selected_simulators and "simulators" in metadata:
            selected_simulators = metadata.get("simulators")

        # --- 1. Generate simulator groups ---
        simulator_groups = list(set(combinations(selected_simulators, 2)))
        simulator_groups.append(tuple(selected_simulators))
        simulator_groups = list(dict.fromkeys(simulator_groups))

        for group in simulator_groups:
            print(f"\n=== Evaluating simulator group: {group} ===")

            csv_files_dict = collect_csv_files(input_folder)

            all_dfs = assign_simulator_labels(metadata, csv_files_dict, group)
            data = pd.concat(all_dfs, ignore_index=True)
            meta_rows = []

            relative = root_path.relative_to(input_folder)
            target_folder = output_folder / relative
            target_folder.mkdir(parents=True, exist_ok=True)
    
            sim_part = "_".join(group)
            output_file = target_folder/ f"feature_ranking_{sim_part}_with_summary.csv"

            # --- 🧹 Optional Filtering ---
            if filter_zero_branches:
                before = len(data)
                class_counts_before = data["simulator"].value_counts().to_dict()
                cond = (data["zero_branch_count"] == 0) & (data["tree_type"] != 0) & (data["cut_edge_weight"] != -1)
                data = data[cond]
                after = len(data)
                print(f"Filtered out {before - after} entries (zero_branch_count != 0 or tree_type == 0 or cut_edge_weight == -1) "
                    f"({after}/{before} remaining).")

                class_counts_after = data["simulator"].value_counts().to_dict()

                meta_rows = [
                    f"Trees before filtering: {before}",
                    f"Trees after filtering: {after}",
                    f"Class distribution after filtering: {class_counts_after}"
                ]
                
                # 🛑 Check if enough data left
                if any(sim not in class_counts_after or class_counts_after[sim]/class_counts_before[sim] < 0.25 for sim in group):
                    print("⚠️ Too many trees removed due to zero branches — not enough data left for training.")
                    
                    df_empty = pd.DataFrame({
                        "message": [
                            "Too many trees removed due to zero_branch_count > 0.",
                            "Not enough data left to train classifiers.",
                            f"Remaining samples: {after} (originally {before}).",
                            "No analysis performed."
                        ]
                    })
                    df_empty.to_csv(output_file, index=False)
                    print(f"⚠️ Created summary file with warning message: {output_file}")
                    continue  # skip to next group

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
                try:
                    acc_per_clf = evaluate_feature(X, y, feature, classifiers, balance_train=resample_mode)
                    max_acc = np.nanmax(list(acc_per_clf.values()))
                except ValueError as e:
                    if "only one class" in str(e):
                        print(f"⚠️ Skipping feature '{feature}' – only one class present in training data.")
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

            df_summary.to_csv(output_file, index=False)
            if meta_rows:
                with open(output_file, "a") as f:
                    f.write("\n".join(meta_rows) + "\n")
            plot_feature_performance(output_file)
            print(f"Feature ranking and classifier accuracies saved to {output_file}")
    
    yaml_files = [f for f in input_folder.iterdir() if f.is_file() and f.suffix in {'.yaml', '.yml'}]

    if not yaml_files:
        yaml_paths = list(Path(input_folder).rglob("*.yaml")) \
                + list(Path(input_folder).rglob("*.yml"))

        yaml_files = []
        for path in yaml_paths:
            with open(path, "r") as f:
                yaml_files.append(yaml.safe_load(f))

        if not selected_simulators:
            # Alle simulatoren einsammeln und vereinigen
            all_sims = set()
            for f in yaml_files:
                sims = f.get("simulators", [])
                all_sims.update(sims)

            selected_simulators = list(all_sims)
        
        # --- 1. Generate simulator groups ---
        simulator_groups = list(set(combinations(selected_simulators, 2)))
        simulator_groups.append(tuple(selected_simulators))
        simulator_groups = list(dict.fromkeys(simulator_groups))
        print(f"---------- All simulator groups: {simulator_groups} ---------- ")

        for group in simulator_groups:
            print(f"\n=== Evaluating simulator group: {group} ===")

            csv_files_dict = collect_csv_files(input_folder)
            all_dfs = assign_simulator_labels(metadata, csv_files_dict, list(group))
            data = pd.concat(all_dfs, ignore_index=True)
            meta_rows = []
            
            sim_part = "_".join(group)
            all_folder = output_folder / "all"
            all_folder.mkdir(parents=True, exist_ok=True)

            output_file = all_folder/ f"feature_ranking_{sim_part}_with_summary.csv"

            # --- 🧹 Optional Filtering ---
            if filter_zero_branches:
                before = len(data)
                class_counts_before = data["simulator"].value_counts().to_dict()
                cond = (data["zero_branch_count"] == 0) & (data["tree_type"] != 0) & (data["cut_edge_weight"] != -1)
                data = data[cond]
                after = len(data)
                print(f"Filtered out {before - after} entries (zero_branch_count != 0 or tree_type == 0 or cut_edge_weight == -1) "
                    f"({after}/{before} remaining).")

                class_counts_after = data["simulator"].value_counts().to_dict()

                meta_rows = [
                    f"Trees before filtering: {before}",
                    f"Trees after filtering: {after}",
                    f"Class distribution after filtering: {class_counts_after}"
                ]

                # 🛑 Check if enough data left
                if any(sim not in class_counts_after or class_counts_after[sim]/class_counts_before[sim] < 0.25 for sim in group):
                    print("⚠️ Too many trees removed due to zero branches — not enough data left for training.")
                    
                    # Save informative CSV
                   
                    
                    df_empty = pd.DataFrame({
                        "message": [
                            "Too many trees removed due to zero_branch_count > 0.",
                            "Not enough data left to train classifiers.",
                            f"Remaining samples: {after} (originally {before}).",
                            "No analysis performed."
                        ]
                    })
                    df_empty.to_csv(output_file, index=False)
                    print(f"⚠️ Created summary file with warning message: {output_file}")
                    continue  # skip to next group

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
                try:
                    acc_per_clf = evaluate_feature(X, y, feature, classifiers, balance_train=resample_mode)
                    max_acc = np.nanmax(list(acc_per_clf.values()))
                except ValueError as e:
                    if "only one class" in str(e):
                        print(f"⚠️ Skipping feature '{feature}' – only one class present in training data.")
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

            df_summary.to_csv(output_file, index=False)
            if meta_rows:
                with open(output_file, "a") as f:
                    f.write("\n".join(meta_rows) + "\n")
            plot_feature_performance(output_file)
            print(f"Feature ranking and classifier accuracies saved to {output_file}")


if __name__ == "__main__":
    main()




