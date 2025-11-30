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
    Reads the evaluation CSV and produces a bar plot stored in the same folder.
    """

    # Read CSV
    df = pd.read_csv(csv_path)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove empty columns
    df = df.dropna(axis=1, how="all")

    # Convert numeric columns to float
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only keep rows that contain actual feature names
    df_features = df[~df["feature"].isin(["ALL_FEATURES", "PASSED_FEATURES", "OVERALL_MEAN"])]

    # Plot as grouped bar chart
    features = df_features["feature"]
    metrics = ["max_accuracy", "xgboost", "random_forest", "logistic_regression"]
    x = np.arange(len(features))  # the label locations
    width = 0.2  # width of the bars

    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, df_features[metric], width=width, label=metric)

    plt.xticks(x + width*1.5, features, rotation=90, fontsize=6)
    plt.ylabel("Accuracy")
    plt.xlabel("Feature")
    plt.title("Model Performance per Feature")
    plt.legend()
    plt.tight_layout()

    # Save file next to the CSV
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(os.path.dirname(csv_path), f"{base}_barplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Plot gespeichert unter: {out_path}")


def run_feature_evaluation(
        selected_simulators,
        csv_files,
        output_folder,
        metadata,
        filter_zero_branches=False,
        acc_factor=1.0,
        resample_mode=None
    ):
    """
    Runs feature evaluation for combinations of simulators.
    
    Parameters
    ----------
    selected_simulators : list[str]
        Simulators to evaluate.
    csv_files : list[Path]
        Path of every csv file.
    output_folder : Path
        Base output folder for results.
    metadata : dict or DataFrame
        Metadata used in assign_simulator_labels().
    usePrecision : bool, default=False
        If True, updates csv_with_precision; otherwise csv_without_precision.
    filter_zero_branches : bool, default=False
        Whether to filter based on zero_branch_count etc.
    acc_factor : float, default=1.0
        Multiplier for minimum accuracy threshold.
    resample_mode : str or None, default=None
        Whether training should be balanced.
    return_precision_sets : bool, default=False
        If True, return dictionaries for precision/non-precision CSVs.

    Returns
    -------
    dict or None
        If return_precision_sets=True:
            {
                "csv_with_precision": {...},
                "csv_without_precision": {...}
            }
        Otherwise None.
    """

    # --- 1. Build simulator groups ---
    simulator_groups = list(set(combinations(selected_simulators, 2)))
    simulator_groups.append(tuple(selected_simulators))
    simulator_groups = list(dict.fromkeys(simulator_groups))

    for group in simulator_groups:
        print(f"\n=== Evaluating simulator group: {group} ===")

        # --- 2. Load and label data ---
        all_dfs = assign_simulator_labels(metadata, csv_files, group)
        data = pd.concat(all_dfs, ignore_index=True)

        meta_rows = []

        sim_part = "_".join(group)
        output_file = output_folder / f"feature_ranking_{sim_part}_with_summary.csv"

        if output_file.exists():
            continue

        # --- Optional filtering ---
        if filter_zero_branches:
            before = len(data)
            class_counts_before = data["simulator"].value_counts().to_dict()
            cond = (data["zero_branch_count"] == 0) & (data["tree_type"] != 0) & (data["cut_edge_weight"] != -1)
            data = data[cond]
            after = len(data)

            print(f"Filtered out {before - after} entries "
                  f"({after}/{before} remaining).")

            class_counts_after = data["simulator"].value_counts().to_dict()

            meta_rows = [
                f"Trees before filtering: {before}",
                f"Trees after filtering: {after}",
                f"Class distribution after filtering: {class_counts_after}"
            ]

            # Not enough samples left?
            if any(sim not in class_counts_after or 
                   class_counts_after[sim] / class_counts_before[sim] < 0.25 
                   for sim in group):

                print("⚠️ Too many trees removed — not enough data left.")

                df_empty = pd.DataFrame({
                    "message": [
                        "Too many trees removed due to zero_branch_count > 0.",
                        "Not enough data left to train classifiers.",
                        f"Remaining samples: {after} (originally {before}).",
                        "No analysis performed."
                    ]
                })
                df_empty.to_csv(output_file, index=False)
                continue

        # --- Handle inf / NaN ---
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(axis=1, inplace=True)

        # --- Prepare features ---
        dep_var = 'simulator'
        features = [c for c in data.columns if c != dep_var]
        data = data[data[dep_var].isin(group)]
        y = LabelEncoder().fit_transform(data[dep_var])
        X = data[features]

        # --- Classifiers ---
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

        # --- Feature evaluation ---
        min_accuracy = 1 / len(group) * acc_factor
        feature_records = []
        passed_features = []

        for feature in features:
            try:
                acc_per_clf = evaluate_feature(X, y, feature, classifiers, balance_train=resample_mode)
                max_acc = np.nanmax(list(acc_per_clf.values()))
            except ValueError as e:
                if "only one class" in str(e):
                    print(f"⚠️ Skipping feature '{feature}' – only one class.")
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
        all_features_acc = evaluate_classifiers(X, y, classifiers)

        # --- Evaluate with passed features ---
        if passed_features:
            X_passed = X[passed_features]
            passed_acc = evaluate_classifiers(X_passed, y, classifiers)
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

        df_summary.to_csv(output_file, index=False)

        if meta_rows:
            with open(output_file, "a") as f:
                f.write("\n".join(meta_rows) + "\n")

        plot_feature_performance(output_file)


#TODO: mapping wird nur einmal akzeptiert... aber naja
def merge_csv_mappings(yaml_paths):
    """
    Merge multiple YAML files, collecting only mappings that point to CSV files.

    Parameters
    ----------
    yaml_paths : list[Path]
        List of paths to YAML files to merge.

    Returns
    -------
    dict
        Merged dictionary containing only key-value pairs where the value ends with '.csv'.
        Duplicate keys are avoided. In case of a conflict (same key, different value), 
        the first occurrence is kept and a warning is printed.
    """
    
    merged = {}

    for path in yaml_paths:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not data:
            continue

        # Recursively traverse all key-value pairs
        def extract_csv_mappings(d):
            result = {}
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, dict):
                        result.update(extract_csv_mappings(v))
                    elif isinstance(v, str) and v.endswith(".csv"):
                        result[k] = v
            return result

        csv_mappings = extract_csv_mappings(data)

        for k, v in csv_mappings.items():
            # Only add if key doesn't already exist
            if k not in merged:
                merged[k] = v
            else:
                if merged[k] != v:
                    print(f"⚠️ Conflict for key '{k}': keeping '{merged[k]}', discarding '{v}'")

    return merged


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

    csv_with_precision = []
    csv_without_precision = []

    csv_with_recombination = []
    csv_without_recombination = []

    for root, _, files in os.walk(input_folder):
        root_path = Path(root)

        metadata_files = [f for f in files if f.endswith(".yaml") or f.endswith(".yml")]
        if not metadata_files:
            continue
        
        metadata_path = root_path / metadata_files[0]
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        relative = root_path.relative_to(input_folder)
        target_folder = output_folder / relative
        target_folder.mkdir(parents=True, exist_ok=True)

        usePrecision = bool(metadata.get("usePrecision", True))
        recombination_rate = float(metadata.get("recombination_rate") or 0)
        
        if not selected_simulators and "simulators" in metadata:
            selected_simulators = metadata.get("simulators")

        csv_files = collect_csv_files(root_path)

        if usePrecision:
            csv_with_precision += csv_files
            if recombination_rate:
                csv_with_recombination += csv_files
            else:
                csv_without_recombination += csv_files
        else:
            csv_without_precision += csv_files


        run_feature_evaluation(
            selected_simulators=selected_simulators,
            csv_files=csv_files,
            output_folder=target_folder,
            metadata=metadata,
            filter_zero_branches=filter_zero_branches,
            acc_factor=acc_factor,
            resample_mode=resample_mode
        )

    
    yaml_files = [f for f in input_folder.iterdir() if f.is_file() and f.suffix in {'.yaml', '.yml'}]

    if not yaml_files:
        yaml_paths = list(Path(input_folder).rglob("*.yaml")) \
                + list(Path(input_folder).rglob("*.yml"))

        with open(yaml_paths[0], "r") as f:
            metadata = yaml.safe_load(f)

        if not selected_simulators and "simulators" in metadata:
            selected_simulators = metadata.get("simulators")

        '''
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
        '''

        all_folder = output_folder / "all"
        all_folder.mkdir(parents=True, exist_ok=True)

        # metadata = merge_csv_mappings(yaml_paths)

        csv_files = collect_csv_files(input_folder)

        run_feature_evaluation(
            selected_simulators=selected_simulators,
            csv_files=csv_files,
            output_folder=all_folder,
            metadata=metadata,
            filter_zero_branches=filter_zero_branches,
            acc_factor=acc_factor,
            resample_mode=resample_mode
        )

        if csv_with_precision and csv_without_precision:
            precision_applied_folder = output_folder / "precision_applied"
            precision_applied_folder.mkdir(parents=True, exist_ok=True)

            run_feature_evaluation(
                selected_simulators=selected_simulators,
                csv_files=csv_with_precision,
                output_folder=precision_applied_folder,
                metadata=metadata,
                filter_zero_branches=filter_zero_branches,
                acc_factor=acc_factor,
                resample_mode=resample_mode
            )
            
            precision_skipped_folder = output_folder / "precision_skipped"
            precision_skipped_folder.mkdir(parents=True, exist_ok=True)

            run_feature_evaluation(
                selected_simulators=selected_simulators,
                csv_files=csv_without_precision,
                output_folder=precision_skipped_folder,
                metadata=metadata,
                filter_zero_branches=filter_zero_branches,
                acc_factor=acc_factor,
                resample_mode=resample_mode
            )
        
        if csv_with_recombination and csv_without_recombination:
            recombination_applied_folder = output_folder / "recombination_applied"
            recombination_applied_folder.mkdir(parents=True, exist_ok=True)

            run_feature_evaluation(
                selected_simulators=selected_simulators,
                csv_files=csv_with_recombination,
                output_folder=recombination_applied_folder,
                metadata=metadata,
                filter_zero_branches=filter_zero_branches,
                acc_factor=acc_factor,
                resample_mode=resample_mode
            )

            recombination_skipped_folder = output_folder / "recombination_skipped"
            recombination_skipped_folder.mkdir(parents=True, exist_ok=True)

            run_feature_evaluation(
                selected_simulators=selected_simulators,
                csv_files=csv_with_recombination,
                output_folder=recombination_skipped_folder,
                metadata=metadata,
                filter_zero_branches=filter_zero_branches,
                acc_factor=acc_factor,
                resample_mode=resample_mode
            )


if __name__ == "__main__":
    main()




