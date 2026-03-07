#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import SIMULATORS
from utils.io_helpers import collect_csv_files, detect_simulator_from_yaml 


# ------------------ feature lists ------------------

TREE_FEATURES = [
    "tree_height_uw",
    "min_laplace",
    "min_adj"
]

TREE_BLOCK_FEATURES = [
    "num_trees"
]

# match e.g.
# sample_sizes-16_sequence_lengths-1000000_...
_SAMPLE_SIZE_RE = re.compile(r"sample_sizes-([0-9.]+)")
_SEQ_LEN_RE = re.compile(r"sequence_lengths-([0-9]+)")


def parse_sample_dir_name(name: str) -> Tuple[Optional[float], Optional[int]]:
    """
    Parses real directory names like:
    sample_sizes-16_sequence_lengths-1000000_population_sizes-10000_...

    Returns:
        sample_size_key (float)
        seq_len (int)
    """
    sample_key = None
    seq_len = None

    m1 = _SAMPLE_SIZE_RE.search(name)
    if m1:
        try:
            sample_key = float(m1.group(1))
        except Exception:
            sample_key = None

    m2 = _SEQ_LEN_RE.search(name)
    if m2:
        try:
            seq_len = int(m2.group(1))
        except Exception:
            seq_len = None

    return sample_key, seq_len


def guess_kind_from_config_dir(config_dir: Path) -> str:
    """
    Expected layout: .../tree/<config-name>/ or .../tree_block/<config-name>/
    """
    parent = config_dir.parent.name.lower()
    if parent in {"tree", "tree_block"}:
        return parent
    # fallback: infer from available feature names later; default tree
    return "tree"


def clean_feature_name(name: str, kind: str) -> str:
    if not isinstance(name, str):
        return str(name)

    s = name.strip()

    # tree_block: remove pooled_ prefix (your issue)

    # rename entropy -> alignment_entropy
    if s == "entropy":
        s = "alignment_entropy"

    return s


def find_yaml_for_simulator(sample_dir: Path) -> Optional[Path]:
    """
    Try to find a yaml/yml inside the subgroup directory.
    If your pipeline uses a specific name, adjust here.
    """
    for pat in ("*.yaml", "*.yml"):
        hits = sorted(sample_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def extract_values_from_df(df: pd.DataFrame, wanted: Iterable[str]) -> Dict[str, float]:
    """
    Robust extraction supporting multiple CSV shapes:
    1) wide: feature names are columns, one row
    2) long: columns like [feature,value] or [stat,value] etc.
    3) key/value: first col key, second col value
    Returns dict feature->value for found wanted features.
    """
    wanted_set = set(wanted)

    # --- Case 1: wide format (columns are features) ---
    cols = {str(c).strip() for c in df.columns}

    if len(cols.intersection(wanted_set)) >= max(1, min(3, len(wanted_set) // 3)):
        rows_out = []

        for _, row in df.iterrows():
            out = {}
            for f in wanted_set.intersection(cols):
                v = safe_float(row.get(f))
                if v is not None:
                    out[f] = v
            rows_out.append(out)

    return rows_out

    # --- Case 2: long format ---
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    key_col = None
    val_col = None

    for cand in ["feature", "stat", "name", "metric", "key"]:
        if cand in lower_cols:
            key_col = lower_cols[cand]
            break
    for cand in ["value", "val", "x", "y"]:
        if cand in lower_cols:
            val_col = lower_cols[cand]
            break

    if key_col is not None and val_col is not None:
        out: Dict[str, float] = {}
        for _, r in df.iterrows():
            k = r.get(key_col)
            if k is None:
                continue
            k = str(k).strip()
            if k in wanted_set:
                v = safe_float(r.get(val_col))
                if v is not None:
                    out[k] = v
        return out

    # --- Case 3: key/value as first two columns ---
    if df.shape[1] >= 2:
        out: Dict[str, float] = {}
        kcol = df.columns[0]
        vcol = df.columns[1]
        for _, r in df.iterrows():
            k = r.get(kcol)
            if k is None:
                continue
            k = str(k).strip()
            if k in wanted_set:
                v = safe_float(r.get(vcol))
                if v is not None:
                    out[k] = v
        return out

    return {}


# ------------------ main collection ------------------
def collect_one_config(config_dir: Path, filename_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Returns tidy dataframe with columns:
      config, kind, simulator, sample_dir, sample_size_key, seq_len,
      feature, value, defined, source_csv
    """
    config_dir = config_dir.resolve()
    config_name = config_dir.name
    kind = guess_kind_from_config_dir(config_dir)

    wanted_raw = TREE_FEATURES if kind == "tree" else TREE_BLOCK_FEATURES
    wanted = [clean_feature_name(x, kind) for x in wanted_raw]

    rows = []

    # iterate over all subgroup dirs
    subgroup_dirs = sorted([p for p in config_dir.iterdir() if p.is_dir()])

    for sd in subgroup_dirs:
        sample_key, seq_len = parse_sample_dir_name(sd.name)

        try:
            csv_paths = collect_csv_files(sd)
            csv_paths = [Path(p) for p in csv_paths]
        except Exception:
            csv_paths = list(sd.rglob("*.csv"))

        if filename_filter:
            csv_paths = [p for p in csv_paths if p.name == filename_filter]

        if not csv_paths:
            continue

        # process EVERY csv independently
        for csv_path in sorted(csv_paths):

            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            # clean column names
            df_cols_clean = {c: clean_feature_name(str(c), kind) for c in df.columns}
            df = df.rename(columns=df_cols_clean)

            for keycol in ["feature", "stat", "name", "metric", "key"]:
                if keycol in df.columns:
                    df[keycol] = df[keycol].apply(
                        lambda x: clean_feature_name(str(x), kind)
                        if isinstance(x, str)
                        else x
                    )

            # detect simulator once per CSV
            sim = detect_simulator_from_yaml(csv_path)

            vals = extract_values_from_df(df, wanted)

            # ---- NEW BEHAVIOUR: store ALL found values immediately ----
            for feat in wanted:

                if feat in vals:
                    v = vals[feat]

                    if feat == "difficulty" and v == -1.0:
                        value = np.nan
                        defined = 0
                    else:
                        value = v
                        defined = 1

                    rows.append({
                        "config": config_name,
                        "kind": kind,
                        "simulator": sim if sim is not None else "",
                        "sample_dir": sd.name,
                        "sample_size_key": sample_key,
                        "seq_len": seq_len,
                        "feature": feat,
                        "value": value,
                        "defined": defined,
                        "source_csv": str(csv_path),
                    })

                else:
                    # feature not present in this CSV
                    rows.append({
                        "config": config_name,
                        "kind": kind,
                        "simulator": sim if sim is not None else "",
                        "sample_dir": sd.name,
                        "sample_size_key": sample_key,
                        "seq_len": seq_len,
                        "feature": feat,
                        "value": np.nan,
                        "defined": 0,
                        "source_csv": str(csv_path),
                    })

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Collect per-subgroup feature/stat values per simulator for one config dir (tree or tree_block)."
    )
    ap.add_argument("--input", required=True, help="Path to a single config directory")
    ap.add_argument("--out", default=None, help="Output CSV path (default: <config_dir>/simulator_feature_values.csv)")
    ap.add_argument("--filename-filter", default=None, help="If set, only read CSVs with exactly this filename")
    args = ap.parse_args()

    config_dir = Path(args.input).expanduser()
    if not config_dir.exists() or not config_dir.is_dir():
        raise SystemExit(f"Not a directory: {config_dir}")

    df = collect_one_config(config_dir, filename_filter=args.filename_filter)

    out = Path(args.out).expanduser().resolve() if args.out else (config_dir / "simulator_feature_values.csv")
    df.to_csv(out, index=False)

    print(f"[OK] wrote {len(df)} rows -> {out}")
    if len(df) == 0:
        print("[WARN] no rows collected. Likely reasons: no CSVs found, YAML not found, or wanted features not present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())