from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

import optuna
import optuna.logging

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
    format_classifier_importance_plot,
    XGB_STR,
)
from utils.io_helpers import collect_csv_files, assign_simulator_labels


TOP_TREE_FEATURES = [
    "tree_height_uw",
    "num_branches",
]

TOP_ALLEL_FEATURES = [
    "difficulty",
    "bollback_multinomial",
]

EXCLUDE_FEATURES = {
    "non_binary_nodes", "non_binary_nodes_frac",
    # Laplacian spectrum outputs
    "laplace_spectrum_a",
    "laplace_spectrum_p",
    "laplace_spectrum_e",
    "laplace_spectrum_g",

    # min/max of laplacian/adjacency derived from (binary-resolved) matrices
    "min_laplace",
    "max_laplace",
    "min_adj",
    "max_adj",

    # stats_laplace(...) outputs (highly sensitive to added nodes/edges)
    "lap_max",
    "lap_fiedler",
    "lap_var",
    "lap_skew",
    "lap_mean",
    "lap_kurt",
    "lap_second_min",
    "lap_entropy",
    "lap_sum",
    "cut_weight",
    "conductance",
    "normalized_cut",
    "bottleneck_strength",
    "density_pos",
    "density_neg",
    "modularity",
    "fiedler_sd",

    # graph centrality / distance measures
    "max_betweenness",
    "max_closeness",
    "max_closenessW",
    "eigen_centrality",
    "eigen_centralityW",
    "diameter",
}


def _relative_after_dir(p: Path, anchor: str) -> Path:
    """
    Return path relative to the directory named `anchor`
    e.g. .../tree/configX/rep1/file.csv -> configX/rep1/file.csv
    """
    parts = p.parts
    if anchor not in parts:
        raise ValueError(f"'{anchor}' not found in path: {p}")

    idx = parts.index(anchor)
    return Path(*parts[idx + 1:])

def collect_and_merge_tree_allele_features(
    tree_csv_files: list[Path],
    allele_csv_files: list[Path],
    simulators: list[str],
    suffix_tree: str = "_tree_features.csv",
    suffix_allele: str = "_allel_features.csv",
) -> list[pd.DataFrame]:
    """
    Merge tree and allele feature CSVs.
    Paths are assumed identical below `tree/` vs `allele/`.
    Feature names are NOT prefixed.
    """

    # --- build allele lookup: relative path below allele/ ---
    allele_lookup: dict[Path, Path] = {}

    for a in allele_csv_files:
        key = _relative_after_dir(a, "allele")
        allele_lookup[key] = a

    merged_dfs: list[pd.DataFrame] = []

    for tree_csv in tree_csv_files:
        tree_key = _relative_after_dir(tree_csv, "tree")

        allele_key = tree_key.with_name(
            tree_csv.name.replace(suffix_tree, suffix_allele)
        )

        if allele_key not in allele_lookup:
            raise FileNotFoundError(
                f"No matching allele CSV for:\n{tree_csv}"
            )

        allele_csv = allele_lookup[allele_key]

        df_tree = pd.read_csv(tree_csv)
        df_allele = pd.read_csv(allele_csv)

        if len(df_tree) != len(df_allele):
            raise ValueError(
                f"Row mismatch:\n{tree_csv}\n{allele_csv}"
            )

        # --- infer simulator from filename ---
        simulator_name = None
        for sim in simulators:
            if f"{sim}_" in tree_csv.name:
                simulator_name = sim
                break

        if simulator_name is None:
            continue

        simulator_col = pd.Series(
            [simulator_name] * len(df_tree),
            name="simulator"
        )

        tree_feats = df_tree.drop(columns=["simulator"], errors="ignore")
        allele_feats = df_allele.drop(columns=["simulator"], errors="ignore")

        df_merged = pd.concat(
            [simulator_col, tree_feats, allele_feats],
            axis=1
        )

        merged_dfs.append(df_merged)

    if not merged_dfs:
        raise ValueError("No CSV pairs were merged.")

    return merged_dfs

# -----------------------------
# Helpers for XGBoost Early Stopping (works with xgboost >= 2.1)
# -----------------------------
def _fit_xgb_with_early_stopping(
    clf: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    early_stopping_rounds: int,
    verbose: bool = False,
) -> int:
    """
    Fits an XGBClassifier with early stopping.
    For xgboost >= 2.1.0, early_stopping_rounds is configured on the estimator (constructor/set_params),
    NOT passed to fit().

    Returns best iteration index (best_iteration).
    """
    # set early stopping on the estimator (NOT in fit)
    clf.set_params(early_stopping_rounds=int(early_stopping_rounds))

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose,
    )

    best_iter = getattr(clf, "best_iteration", None)
    if best_iter is None:
        best_iter = clf.get_booster().best_iteration
    return int(best_iter)


def _iter_range(best_iter: int) -> tuple[int, int]:
    # xgboost expects (start, end) with end exclusive
    return (0, int(best_iter) + 1)


# -----------------------------
# Optuna tuning for XGBoost
# -----------------------------
def tune_xgb_with_optuna(
    X: pd.DataFrame,
    y: np.ndarray,
    num_class: int,
    n_trials: int = 50,
    test_size: float = 0.2,
    seed: int = 0,
    early_stopping_rounds: int = 30,
) -> tuple[dict, float]:
    """
    Tune XGBClassifier hyperparameters using Optuna.
    Uses a single stratified train/val split and minimizes validation mlogloss.
    Returns (best_params, best_val_mlogloss).
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=seed
    )
    if "difficulty" in X_train.columns:
        median_diff = X_train["difficulty"].median(skipna=True)
        X_train["difficulty"] = X_train["difficulty"].fillna(median_diff)
        X_val["difficulty"] = X_val["difficulty"].fillna(median_diff)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "objective": "multi:softprob",
            "num_class": num_class,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 0,
        }

        clf = xgb.XGBClassifier(**params)

        best_iter = _fit_xgb_with_early_stopping(
            clf=clf,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

        # Use best iteration for evaluation
        y_proba = clf.predict_proba(X_val, iteration_range=_iter_range(best_iter))
        return log_loss(y_val, y_proba, labels=list(range(num_class)))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best.update(
        {
            "n_estimators": 5000,
            "objective": "multi:softprob",
            "num_class": num_class,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 0,
        }
    )
    return best, float(study.best_value)


# -----------------------------
# Permutation-importance helpers
# -----------------------------
def permutation_importance_accuracy(
    clf: xgb.XGBClassifier,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 1,
    seed: int = 0,
    iteration_range: tuple[int, int] | None = None,
) -> tuple[dict[str, float], dict[str, float], float]:
    """
    Computes permutation importance w.r.t. accuracy on a *fixed* trained model.

    Returns:
      mean_drop: dict feature -> mean(baseline_acc - permuted_acc)
      std_drop:  dict feature -> std(baseline_acc - permuted_acc)
      baseline_acc: accuracy on unpermuted X_valid
    """
    rng = np.random.default_rng(seed)

    baseline_preds = clf.predict(X_valid, iteration_range=iteration_range)
    baseline_acc = accuracy_score(y_valid, baseline_preds)

    mean_drop: dict[str, float] = {}
    std_drop: dict[str, float] = {}

    Xv = X_valid.copy()

    for f in feature_names:
        drops = []
        original = Xv[f].to_numpy(copy=True)

        for _ in range(n_repeats):
            shuffled = original.copy()
            rng.shuffle(shuffled)
            Xv[f] = shuffled

            preds_perm = clf.predict(Xv, iteration_range=iteration_range)
            acc_perm = accuracy_score(y_valid, preds_perm)
            drops.append(baseline_acc - acc_perm)

        # restore
        Xv[f] = original

        mean_drop[f] = float(np.mean(drops))
        std_drop[f] = float(np.std(drops, ddof=0))

    return mean_drop, std_drop, float(baseline_acc)


def get_feature_importance_xgb(model: xgb.XGBClassifier, feature_names: list[str]) -> dict:
    importance = model.feature_importances_
    return dict(zip(feature_names, importance))


def plot_feature_performance(
    feature_importance_dict: dict,
    csv_path: Path,
    group,
    chance_baseline: float | None = None,
):
    """
    Reads the evaluation CSV and produces:
    - one accuracy bar plot
    - one feature importance plot for XGB
    """
    df = pd.read_csv(csv_path)

    df = df.dropna(how="all").dropna(axis=1, how="all")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    special = ["CHANCE_BASELINE", "ALL_FEATURES", "PASSED_FEATURES", "TOP_FEATURES"]
    df_special = df[df["feature"].isin(special)]
    df_regular = df[~df["feature"].isin(special + ["OVERALL_MEAN"])]
    df_features = pd.concat([df_special, df_regular], ignore_index=True)

    features_acc = df_features["feature"].astype(str)
    metrics = ["max_accuracy", XGB_STR]
    x = np.arange(len(features_acc))
    width = 0.35

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

    plt.xticks(x + width * 0.5, features_acc, rotation=90, fontsize=6)
    plt.ylabel("Accuracy / Accuracy drop proxy")
    plt.xlabel("Feature")
    plt.title("Model Performance per Feature")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(
        os.path.dirname(str(csv_path)),
        format_feature_ranking_barplot(group),
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    if XGB_STR in feature_importance_dict:
        imp = feature_importance_dict[XGB_STR]
        features = list(imp.keys())
        values = [imp[f] for f in features]

        plt.figure(figsize=(10, 6))
        plt.bar(features, values)
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Feature Importance")
        plt.xlabel("Feature")
        plt.title(f"Feature Importance – {XGB_STR}")
        plt.tight_layout()

        out_path_single = os.path.join(
            os.path.dirname(str(csv_path)),
            format_classifier_importance_plot(XGB_STR, group),
        )
        plt.savefig(out_path_single, dpi=300)
        plt.close()


def clean_nan_data(
    data: pd.DataFrame,
    output_csv: Path,
    max_row_loss_frac: float = 0.05,
):
    log_path = output_csv.parent / f"nan_cleanup_{output_csv.stem}.log"
    log_lines: list[str] = []

    data = data.replace([np.inf, -np.inf], np.nan)

    n_rows_initial = len(data)
    if n_rows_initial == 0:
        return data

    for col in data.columns:
        if col == "simulator" or col == "difficulty":
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


def run_feature_evaluation(
    simulator_group,
    tree_csv_files,
    allel_csv_files,
    output_dir,
    use_optuna: bool = False,
    optuna_trials: int = 50,
    es_rounds: int = 30,
    es_grid: list[int] | None = None,
    perm_repeats: int = 1,
    drop_threshold: float = 0.0,
):
    # --- Load and label data ---
    all_dfs =  collect_and_merge_tree_allele_features(
        tree_csv_files=tree_csv_files,
        allele_csv_files=allel_csv_files,
        simulators=simulator_group
    )
    data = pd.concat(all_dfs, ignore_index=True)
    del all_dfs

    # Drop excluded columns if present (optional columns)
    drop_cols = [c for c in EXCLUDE_FEATURES if c in data.columns]
    if drop_cols:
        data = data.drop(columns=drop_cols)

    if "difficulty" in data.columns:
        data["difficulty_missing"] = (data["difficulty"] == -1).astype(np.int8)
        data.loc[data["difficulty"] == -1, "difficulty"] = np.nan

    csv_file = output_dir / format_feature_ranking_file(simulator_group)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean NaNs / infs (same as before)
    data = clean_nan_data(data, output_csv=csv_file)

    # numeric downcast
    num_cols = data.select_dtypes(include=["float64"]).columns
    data[num_cols] = data[num_cols].astype(np.float32)

    dep_var = "simulator"
    data = data[data[dep_var].isin(simulator_group)].copy()

    # Features: everything except label, and also not in EXCLUDE_FEATURES (belt & suspenders)
    feature_names = [c for c in data.columns if c != dep_var and c not in EXCLUDE_FEATURES]

    y = LabelEncoder().fit_transform(data[dep_var])
    X_all = data.drop(columns=[dep_var])
    del data

    num_class = len(simulator_group)

    xgb_params = {
        "n_estimators": 5000,  # high, early stopping will cut
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

    # One split reused for all single-feature fits (fair comparison)
    X_train_all, X_valid_all, y_train, y_valid = train_test_split(
        X_all, y, stratify=y, test_size=0.2, random_state=0
    )

    # Fill difficulty NaNs using train median (same as your code)
    if "difficulty" in X_train_all.columns:
        median_diff = X_train_all["difficulty"].median(skipna=True)
        X_train_all["difficulty"] = X_train_all["difficulty"].fillna(median_diff)
        X_valid_all["difficulty"] = X_valid_all["difficulty"].fillna(median_diff)

    # Baselines
    class_counts = np.bincount(y)
    majority_prop = class_counts.max() / class_counts.sum()

    # ALL_FEATURES baseline model
    clf_all = xgb.XGBClassifier(**xgb_params)
    best_iter_all = _fit_xgb_with_early_stopping(
        clf=clf_all,
        X_train=X_train_all,
        y_train=y_train,
        X_val=X_valid_all,
        y_val=y_valid,
        early_stopping_rounds=es_rounds,
        verbose=False,
    )
    ir_all = _iter_range(best_iter_all)
    proba_all = clf_all.predict_proba(X_valid_all, iteration_range=ir_all)
    preds_all = _pred_labels_from_proba(proba_all)
    all_features_acc = accuracy_score(y_valid, preds_all)

    # --- Single-feature evaluation ---
    records = []

    # special rows
    records.append({"feature": "CHANCE_BASELINE", "max_accuracy": float(majority_prop), XGB_STR: np.nan})
    records.append({"feature": "ALL_FEATURES", "max_accuracy": float(all_features_acc), XGB_STR: float(all_features_acc)})
    records.append({"feature": "PASSED_FEATURES", "max_accuracy": np.nan, XGB_STR: np.nan})  # not used here

    # blank separator
    records.append({"feature": np.nan, "max_accuracy": np.nan, XGB_STR: np.nan})

    single_accs = []

    for f in feature_names:
        Xtr = X_train_all[[f]].copy()
        Xva = X_valid_all[[f]].copy()

        # train model on the single column
        clf = xgb.XGBClassifier(**xgb_params)
        best_iter = _fit_xgb_with_early_stopping(
            clf=clf,
            X_train=Xtr,
            y_train=y_train,
            X_val=Xva,
            y_val=y_valid,
            early_stopping_rounds=es_rounds,
            verbose=False,
        )
        ir = _iter_range(best_iter)

        proba = clf.predict_proba(Xva, iteration_range=ir)
        preds = _pred_labels_from_proba(proba)
        acc = float(accuracy_score(y_valid, preds))

        single_accs.append(acc)
        records.append({"feature": f, "max_accuracy": acc, XGB_STR: acc})

    # Optional summary row: mean single-feature accuracy
    records.append({"feature": "SINGLE_FEATURE_MEAN", "max_accuracy": float(np.mean(single_accs)), XGB_STR: float(np.mean(single_accs))})

    df_out = pd.DataFrame(records)[["feature", "max_accuracy", XGB_STR]]
    df_out.to_csv(csv_file, index=False)

    # Optional: barplot reusing your plotting function
    plot_feature_performance({XGB_STR: {}}, csv_file, simulator_group, chance_baseline=float(majority_prop))


def csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract top features from simulation results.")

    parser.add_argument("--input_folder_tree", type=Path, required=True, help="Input folder containing result files")
    parser.add_argument("--input_folder_allel", type=Path, required=True, help="Input folder containing result files")
    parser.add_argument("--output_folder", type=Path, required=True, help="Output folder path")

    parser.add_argument(
        "--simulators",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated list of simulators (e.g. sim1,sim2)",
    )
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

    parser.add_argument(
        "--use_optuna",
        default=False,
        action="store_true",
        help="If set: tune XGB hyperparameters with Optuna and write a parameter report.",
    )
    parser.add_argument("--optuna_trials", type=int, default=50, help="Number of Optuna trials (default: 50).")

    parser.add_argument("--es_rounds", type=int, default=30, help="Early stopping rounds for XGB fits (default: 30).")

    parser.add_argument(
        "--es_grid",
        type=csv_list,
        default=None,
        help="Comma-separated list of early stopping rounds for Optuna grid, e.g. '10,30,50'. "
             "If set with --use_optuna: runs Optuna for each ES and stores all results in one YAML.",
    )

    parser.add_argument(
        "--perm_repeats",
        type=int,
        default=50,
        help="Permutation repeats per feature (default: 20). Increase to reduce noise.",
    )

    parser.add_argument(
        "--drop_threshold",
        type=float,
        default=0.01,
        help="Permutation importance pass threshold applied to (mean_drop - std_drop) > threshold. (default: 0.0)",
    )

    return parser.parse_args()


def assert_non_empty(csv_files, mode):
    if not csv_files:
        raise ValueError(
            f"No CSV files found for mode '{mode}'. "
            "This indicates an inconsistent input directory or configuration."
        )


def get_csv_files_mode(input_folder, grouped_by, seq_len_group=None, sample_size_group=None):
    csv_my_mode = []

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
            sample_size = int(combo_params["sample_structure"][0])
            seq_len = int(combo_params["sequence_length"])

            if (
                (("precision_applied" not in grouped_by) or usePrecision)
                and (("precision_skipped" not in grouped_by) or (not usePrecision))
                and (("recombination_applied" not in grouped_by) or (recombination_rate > 0))
                and (("recombination_skipped" not in grouped_by) or (recombination_rate == 0))
                and (("sample_size" not in grouped_by) or (sample_size == sample_size_group and usePrecision))
                and (("seq_len" not in grouped_by) or (seq_len == seq_len_group and usePrecision))
            ):
                csv_my_mode += collect_csv_files(root_path)

    return csv_my_mode

def main():
    args = parse_args()

    input_folder_tree = args.input_folder_tree
    input_folder_allel = args.input_folder_allel
    output_folder = args.output_folder
    simulator_group = args.simulators
    grouped_by = args.grouped_by
    use_optuna = args.use_optuna
    optuna_trials = args.optuna_trials
    es_rounds = args.es_rounds
    es_grid = args.es_grid
    perm_repeats = args.perm_repeats
    drop_threshold = args.drop_threshold

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

    output_folder.mkdir(parents=True, exist_ok=True)
    tree_csv_files = get_csv_files_mode(input_folder_tree, grouped_by, seq_len_group, sample_size_group)
    allel_csv_files = get_csv_files_mode(input_folder_allel, grouped_by, seq_len_group, sample_size_group)

    assert len(tree_csv_files) == len(allel_csv_files)

    run_feature_evaluation(
        simulator_group=simulator_group,
        tree_csv_files=tree_csv_files,
        allel_csv_files=allel_csv_files,
        output_dir=output_folder,
        use_optuna=use_optuna,
        optuna_trials=optuna_trials,
        es_rounds=es_rounds,
        es_grid=es_grid,
        perm_repeats=perm_repeats,
        drop_threshold=drop_threshold,
    )


if __name__ == "__main__":
    main()