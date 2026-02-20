from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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
    XGB_STR,
)
from utils.io_helpers import collect_csv_files

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
    clf.set_params(early_stopping_rounds=int(early_stopping_rounds))
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)

    best_iter = getattr(clf, "best_iteration", None)
    if best_iter is None:
        best_iter = clf.get_booster().best_iteration
    return int(best_iter)


def _iter_range(best_iter: int) -> tuple[int, int]:
    return (0, int(best_iter) + 1)


# -----------------------------
# Labeling: binary own vs demes (based on filename substring)
# -----------------------------
def label_from_filename(path: Path) -> str:
    # If "own" occurs anywhere in the filename -> own else demes
    return "own" if "own" in path.name else "demes"


def load_and_label_csvs(csv_files: list[Path]) -> pd.DataFrame:
    dfs = []
    for p in csv_files:
        df = pd.read_csv(p)
        df["simulator"] = label_from_filename(Path(p))
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# -----------------------------
# NaN cleanup (unchanged)
# -----------------------------
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
        if col == "simulator":
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


# -----------------------------
# Optuna tuning for *binary* XGBoost
# -----------------------------
def tune_xgb_with_optuna_binary(
    X: pd.DataFrame,
    y: np.ndarray,
    n_trials: int = 50,
    test_size: float = 0.2,
    seed: int = 0,
    early_stopping_rounds: int = 30,
) -> tuple[dict, float]:
    """
    Tune XGBClassifier hyperparameters using Optuna for binary classification.
    Minimizes validation logloss on a single stratified train/val split.
    Returns (best_params, best_val_logloss).
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=seed
    )

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
            "objective": "binary:logistic",
            "eval_metric": "logloss",
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

        y_proba = clf.predict_proba(X_val, iteration_range=_iter_range(best_iter))[:, 1]
        return log_loss(y_val, y_proba, labels=[0, 1])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best.update(
        {
            "n_estimators": 5000,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 0,
        }
    )
    return best, float(study.best_value)


# -----------------------------
# Overall evaluation only (no per-feature evaluation, no permutation importance)
# -----------------------------
def run_overall_accuracy(
    csv_files: list[Path],
    output_dir: Path,
    use_optuna: bool = False,
    optuna_trials: int = 50,
    es_rounds: int = 30,
    es_grid: list[int] | None = None,
):
    # --- Load and binary-label data ---
    data = load_and_label_csvs(csv_files)

    out_csv = output_dir / format_feature_ranking_file("binary_own_vs_demes")

    # --- Handle inf / NaN ---
    data = clean_nan_data(data, output_csv=out_csv)

    num_cols = data.select_dtypes(include=["float64"]).columns
    data[num_cols] = data[num_cols].astype(np.float32)

    dep_var = "simulator"
    data = data[data[dep_var].isin(["own", "demes"])].copy()

    feature_names = [c for c in data.columns if c != dep_var]
    X = data[feature_names]
    y = (data[dep_var].values == "own").astype(int)  # own=1, demes=0
    del data

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=0
    )

    # --- Optuna: (optional) ES-grid report, but still binary ---
    if use_optuna:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        es_values = [10, 30, 50] if es_grid is None else [int(v) for v in es_grid]

        results: list[dict] = []
        best_overall: dict | None = None

        for es in es_values:
            best_params, best_val_logloss = tune_xgb_with_optuna_binary(
                X=X,
                y=y,
                n_trials=optuna_trials,
                test_size=0.2,
                seed=0,
                early_stopping_rounds=es,
            )
            entry = {
                "early_stopping_rounds": int(es),
                "optuna_trials": int(optuna_trials),
                "best_val_logloss": float(best_val_logloss),
                "best_params": best_params,
            }
            results.append(entry)

            if best_overall is None or entry["best_val_logloss"] < best_overall["best_val_logloss"]:
                best_overall = entry

        params_out = output_dir / "xgb_params_esgrid_binary_own_vs_demes.yaml"
        report = {
            "use_optuna": True,
            "selection_metric": "val_logloss",
            "es_grid": [int(e) for e in es_values],
            "results": results,
            "best_overall": best_overall,
        }
        with open(params_out, "w") as f:
            yaml.safe_dump(report, f, sort_keys=False)

        print(f"[INFO] Optuna ES-grid complete. Wrote parameter report to: {params_out}")
        return

    # --- Fixed params (binary objective) ---
    # keep your tree/allele branching if you want; here: simplest stable default
    xgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.10,
        "max_depth": 5,
        "min_child_weight": 8.0,
        "gamma": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": 1,
        "random_state": 0,
    }

    clf = xgb.XGBClassifier(**xgb_params)
    best_iter = _fit_xgb_with_early_stopping(
        clf=clf,
        X_train=X_train,
        y_train=y_train,
        X_val=X_valid,
        y_val=y_valid,
        early_stopping_rounds=es_rounds,
        verbose=False,
    )

    ir = _iter_range(best_iter)
    preds = clf.predict(X_valid, iteration_range=ir)
    acc = float(accuracy_score(y_valid, preds))

    # chance baseline: majority class proportion
    class_counts = np.bincount(y)
    majority_prop = float(class_counts.max() / class_counts.sum())

    # --- Write a small CSV report (overall only) ---
    df = pd.DataFrame(
        [
            {"metric": "CHANCE_BASELINE", "value": majority_prop},
            {"metric": "ACCURACY", "value": acc},
        ]
    )
    df.to_csv(out_csv, index=False)

    # --- Simple plot (overall only) ---
    plt.figure(figsize=(5, 4))
    plt.bar(["chance", "accuracy"], [majority_prop, acc])
    plt.ylabel("Value")
    plt.title("Overall accuracy (own vs demes)")
    plt.tight_layout()
    plt.savefig(output_dir / format_feature_ranking_barplot("binary_own_vs_demes"), dpi=300)
    plt.close()


def csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Binary XGB classification: own vs demes (overall accuracy only).")

    parser.add_argument("--input_folder", type=Path, required=True, help="Input folder containing result files")
    parser.add_argument("--output_folder", type=Path, required=True, help="Output folder path")

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
    use_optuna = args.use_optuna
    optuna_trials = args.optuna_trials
    es_rounds = args.es_rounds
    es_grid = args.es_grid
    grouped_by = args.grouped_by

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
    csv_my_mode: list[Path] = []

    if "single" in grouped_by or "overall" in grouped_by:
        csv_files = collect_csv_files(input_folder)
        assert_non_empty(csv_files=csv_files, mode=grouped_by)
        run_overall_accuracy(
            csv_files=csv_files,
            output_dir=output_folder,
            use_optuna=use_optuna,
            optuna_trials=optuna_trials,
            es_rounds=es_rounds,
            es_grid=es_grid,
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

    assert_non_empty(csv_files=csv_my_mode, mode=grouped_by)

    run_overall_accuracy(
        csv_files=csv_my_mode,
        output_dir=output_folder,
        use_optuna=use_optuna,
        optuna_trials=optuna_trials,
        es_rounds=es_rounds,
        es_grid=es_grid,
    )


if __name__ == "__main__":
    main()