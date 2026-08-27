#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# --- allow imports from utils/ ---
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from utils.io_helpers import collect_csv_files, detect_simulator_from_yaml  # noqa: E402


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV robustly; if broken lines exist, skip them."""
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def simulator_group(sim: str) -> str:
    """Map simulator name -> forward/backward."""
    s = (sim or "").strip().lower()
    # be forgiving: slim, SLiM4, slim-v4, etc.
    return "forward" if s.startswith("slim") else "backward"


def numeric_feature_means(df: pd.DataFrame) -> pd.Series:
    """Return mean per numeric feature column, coercing numeric strings."""
    drop_cols = {"simulator", "label", "group", "config", "combo"}
    cols_to_drop = [c for c in df.columns if c in drop_cols]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df2 = df.copy()
    for c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    df2 = df2.select_dtypes(include=[np.number])
    if df2.shape[1] == 0:
        return pd.Series(dtype=float)

    return df2.mean(axis=0, skipna=True)


def compute_means_for_files(csv_files: List[Path], row_level: bool, allowed_sim) -> Tuple[pd.Series, pd.Series, Dict[str, int]]:
    """
    Compute feature means for forward (SLiM) and backward (all others)
    within ONE subgroup.

    Returns (mean_slim, mean_backward, counts)
    """
    counts = {"slim_files": 0, "backward_files": 0, "skipped_files": 0}

    # -----------------
    # File-level: each CSV counts once
    # -----------------
    if not row_level:
        slim_means: List[pd.Series] = []
        back_means: List[pd.Series] = []

        for f in csv_files:
            try:
                sim = detect_simulator_from_yaml(input_file=str(f))

                if sim not in allowed_sim and sim != "slim":
                    continue
            except Exception:
                counts["skipped_files"] += 1
                continue

            grp = simulator_group(str(sim))

            try:
                df = read_csv_robust(f)
            except Exception:
                counts["skipped_files"] += 1
                continue

            if df.empty:
                counts["skipped_files"] += 1
                continue

            m = numeric_feature_means(df)
            if m.empty:
                counts["skipped_files"] += 1
                continue

            if grp == "forward":
                slim_means.append(m)
                counts["slim_files"] += 1
            else:
                back_means.append(m)
                counts["backward_files"] += 1

        if len(slim_means) == 0 or len(back_means) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float), counts

        mean_slim = pd.DataFrame(slim_means).mean(axis=0, skipna=True)
        mean_back = pd.DataFrame(back_means).mean(axis=0, skipna=True)
        return mean_slim, mean_back, counts

    # -----------------
    # Row-level: each ROW counts; files weighted by their row count
    # -----------------
    slim_sum: Optional[pd.Series] = None
    slim_cnt: Optional[pd.Series] = None
    back_sum: Optional[pd.Series] = None
    back_cnt: Optional[pd.Series] = None

    for f in csv_files:
        try:
            sim = detect_simulator_from_yaml(input_file=str(f))

            if sim not in allowed_sim and sim != "slim":
                    continue
        except Exception:
            counts["skipped_files"] += 1
            continue

        grp = simulator_group(str(sim))

        try:
            df = read_csv_robust(f)
        except Exception:
            counts["skipped_files"] += 1
            continue

        if df.empty:
            counts["skipped_files"] += 1
            continue

        # drop obvious non-feature columns if present
        drop_cols = {"simulator", "label", "group", "config", "combo"}
        cols_to_drop = [c for c in df.columns if c in drop_cols]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.select_dtypes(include=[np.number])

        if df.shape[1] == 0:
            counts["skipped_files"] += 1
            continue

        sums = df.sum(axis=0, skipna=True)
        cnts = df.notna().sum(axis=0)

        if grp == "forward":
            counts["slim_files"] += 1
            slim_sum = sums if slim_sum is None else slim_sum.add(sums, fill_value=0.0)
            slim_cnt = cnts if slim_cnt is None else slim_cnt.add(cnts, fill_value=0.0)
        else:
            counts["backward_files"] += 1
            back_sum = sums if back_sum is None else back_sum.add(sums, fill_value=0.0)
            back_cnt = cnts if back_cnt is None else back_cnt.add(cnts, fill_value=0.0)

    if slim_sum is None or back_sum is None or slim_cnt is None or back_cnt is None:
        return pd.Series(dtype=float), pd.Series(dtype=float), counts

    mean_slim = slim_sum / slim_cnt.replace(0, np.nan)
    mean_back = back_sum / back_cnt.replace(0, np.nan)
    return mean_slim, mean_back, counts


def list_subgroups(config_dir: Path) -> List[Path]:
    """Immediate subdirectories under config_dir are treated as subgroups."""
    subs = [p for p in config_dir.iterdir() if p.is_dir()]
    subs.sort(key=lambda p: p.name)
    return subs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute per-subgroup feature mean deltas: (SLiM - backward) for a single config dir."
    )
    ap.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Config directory (e.g. data/features/tree/<config>). Subgroups are immediate subfolders.",
    )
    ap.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Output CSV to write (one config dir).",
    )
    ap.add_argument(
        "--row_level",
        action="store_true",
        help="Aggregate per row (weights files by #rows). Default is file-level (each CSV counts once).",
    )
    ap.add_argument(
        "--ext",
        default=".csv",
        help="Comma-separated extensions to collect (default: .csv).",
    )
    ap.add_argument(
        "--coalescent-simulator",
        nargs="+",
        choices=["ms", "msms", "mspms"],
        default=["ms", "msms", "mspms"],
        help="Allowed coalescent simulators.",
    )

    args = ap.parse_args()
    cfg_path = args.input_dir.expanduser().resolve()
    if not cfg_path.exists() or not cfg_path.is_dir():
        raise SystemExit(f"input_dir does not exist or is not a directory: {cfg_path}")

    config_name = cfg_path.name
    subgroups = list_subgroups(cfg_path)
    if not subgroups:
        raise SystemExit(f"No subgroup directories found under: {cfg_path}")

    exts = [e.strip() for e in args.ext.split(",") if e.strip()]
    records: List[dict] = []

    for sub in subgroups:
        # collect all csv under subgroup (includes rep*/... recursion)
        all_files = [Path(p) for p in collect_csv_files(str(sub))]
        csvs = [p for p in all_files if any(str(p).endswith(ext) for ext in exts)]
        csvs = sorted(set(csvs))

        if not csvs:
            continue

        mean_slim, mean_back, counts = compute_means_for_files(
            csvs,
            row_level=args.row_level,
            allowed_sim=args.coalescent_simulator,
        )

        if mean_slim.empty or mean_back.empty:
            # Still record that subgroup could not be computed (useful for debugging)
            records.append(
                dict(
                    config=config_name,
                    subgroup=sub.name,
                    feature="__NO_COMPUTABLE_FEATURES__",
                    mean_slim=np.nan,
                    mean_backward=np.nan,
                    delta_slim_minus_backward=np.nan,
                    abs_delta=np.nan,
                    n_files_slim=counts["slim_files"],
                    n_files_backward=counts["backward_files"],
                    n_files_skipped=counts["skipped_files"],
                )
            )
            continue

        all_features = sorted(set(mean_slim.index).union(set(mean_back.index)))
        ms = mean_slim.reindex(all_features)
        mb = mean_back.reindex(all_features)

        delta = ms - mb  # SLiM - backward (positive => SLiM higher)
        abs_delta = delta.abs()

        for feat in all_features:
            records.append(
                dict(
                    config=config_name,
                    subgroup=sub.name,
                    feature=feat,
                    mean_slim=float(ms.loc[feat]) if pd.notna(ms.loc[feat]) else np.nan,
                    mean_backward=float(mb.loc[feat]) if pd.notna(mb.loc[feat]) else np.nan,
                    delta_slim_minus_backward=float(delta.loc[feat]) if pd.notna(delta.loc[feat]) else np.nan,
                    abs_delta=float(abs_delta.loc[feat]) if pd.notna(abs_delta.loc[feat]) else np.nan,
                    n_files_slim=counts["slim_files"],
                    n_files_backward=counts["backward_files"],
                    n_files_skipped=counts["skipped_files"],
                )
            )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise SystemExit(f"No results produced for: {cfg_path} (no CSVs found under any subgroup?)")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_file, index=False)

    print("✅ Wrote:", args.output_file)
    print("Delta definition: delta_slim_minus_backward = mean_slim - mean_backward")
    print(f"Aggregation: {'row-level' if args.row_level else 'file-level'}")
    print(f"Config: {config_name}, subgroups scanned: {len(subgroups)}")


if __name__ == "__main__":
    main()