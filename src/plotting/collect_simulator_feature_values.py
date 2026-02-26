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
    "phylogenetic_div",
    "num_branches",
    "min_adj_uw",
    "mean_branch_length",
    "tree_height",
]

ALLELE_STATS = [
    "difficulty",
    "bollback_multinomial",
    "mean_ld",
    "sfs_skewness",
    "alignment_entropy",
]

_SAMPLE_RE = re.compile(r"sample_size_(.*),seq_len_(\d+)$")


# ------------------ helpers ------------------

def parse_sample_dir_name(name: str) -> Tuple[Optional[float], Optional[int]]:
    """
    Parses:
      sample_size_<something>,seq_len_<integer>
    Returns:
      sample_size_key: max numeric token in sample_size part (float)
      seq_len: int
    """
    m = _SAMPLE_RE.match(name)
    if not m:
        return None, None
    sample_part = m.group(1)
    seq_len = int(m.group(2))
    nums = re.findall(r"\d+(?:\.\d+)?", sample_part)
    sample_key = max((float(x) for x in nums), default=None)
    return sample_key, seq_len


def guess_kind_from_config_dir(config_dir: Path) -> str:
    """
    Expected layout: .../tree/<config-name>/ or .../allele/<config-name>/
    """
    parent = config_dir.parent.name.lower()
    if parent in {"tree", "allele"}:
        return parent
    # fallback: infer from available feature names later; default tree
    return "tree"


def clean_feature_name(name: str, kind: str) -> str:
    if not isinstance(name, str):
        return str(name)

    s = name.strip()

    # allele: remove pooled_ prefix (your issue)
    if kind == "allele" and s.startswith("pooled_"):
        s = s[len("pooled_"):]

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
    # if most wanted appear as columns, treat as wide
    if len(cols.intersection(wanted_set)) >= max(1, min(3, len(wanted_set) // 3)):
        row0 = df.iloc[0] if len(df) > 0 else None
        out: Dict[str, float] = {}
        if row0 is not None:
            for f in wanted_set.intersection(cols):
                v = safe_float(row0.get(f))
                if v is not None:
                    out[f] = v
        return out

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


def detect_simulator(sample_dir: Path) -> Optional[str]:
    """
    Uses your detect_simulator_from_yaml utility.
    """
    yml = find_yaml_for_simulator(sample_dir)
    if yml is None:
        return None
    try:
        sim = detect_simulator_from_yaml(yml)
        if sim is None:
            return None
        sim = str(sim).strip()
        # optional sanity check against SIMULATORS list
        if SIMULATORS and sim not in SIMULATORS:
            # still return it, but you’ll notice in output
            return sim
        return sim
    except Exception:
        return None


# ------------------ main collection ------------------

def collect_one_config(config_dir: Path, filename_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Returns tidy dataframe with columns:
      config, kind, simulator, sample_dir, sample_size_key, seq_len, feature, value, source_csv
    """
    config_dir = config_dir.resolve()
    config_name = config_dir.name
    kind = guess_kind_from_config_dir(config_dir)

    wanted_raw = TREE_FEATURES if kind == "tree" else ALLELE_STATS
    # after cleaning (e.g. pooled_ removal, entropy rename), our target set is:
    wanted = [clean_feature_name(x, kind) for x in wanted_raw]

    # subgroup directories are the sample_size...,seq_len... dirs
    subgroup_dirs = sorted([p for p in config_dir.iterdir() if p.is_dir()])

    rows = []
    for sd in subgroup_dirs:
        sample_key, seq_len = parse_sample_dir_name(sd.name)
        sim = detect_simulator(sd)

        # collect all csv files under this subgroup directory
        try:
            csvs = collect_csv_files(sd)  # uses your helper
            csv_paths = [Path(p) for p in csvs]
        except Exception:
            csv_paths = list(sd.rglob("*.csv"))

        # optional filter (if you want only one specific csv type)
        if filename_filter:
            csv_paths = [p for p in csv_paths if p.name == filename_filter]

        if not csv_paths:
            continue

        # merge values from potentially multiple CSVs (first hit wins unless later overwrites)
        found: Dict[str, Tuple[float, Path]] = {}
        for csv_path in sorted(csv_paths):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            # normalize key names in dataframe (clean pooled_ and entropy)
            # apply cleaning to columns if wide
            df_cols_clean = {c: clean_feature_name(str(c), kind) for c in df.columns}
            df = df.rename(columns=df_cols_clean)

            # also clean key column entries in long format if present
            for keycol in ["feature", "stat", "name", "metric", "key"]:
                if keycol in df.columns:
                    df[keycol] = df[keycol].apply(lambda x: clean_feature_name(str(x), kind) if isinstance(x, str) else x)

            vals = extract_values_from_df(df, wanted)
            for feat, v in vals.items():
                # keep first occurrence; change to "overwrite" if you prefer latest
                if feat not in found:
                    found[feat] = (v, csv_path)

        for feat in wanted:
            if feat not in found:
                # explicitly record missing (useful for defined-rate calculations)
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
                    "source_csv": "",
                })
                continue

            v, src = found[feat]

            # Treat undefined difficulty (-1) as missing (NaN)
            if feat == "difficulty" and v == -1.0:
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
                    "source_csv": str(src),
                })
                continue

            rows.append({
                "config": config_name,
                "kind": kind,
                "simulator": sim if sim is not None else "",
                "sample_dir": sd.name,
                "sample_size_key": sample_key,
                "seq_len": seq_len,
                "feature": feat,
                "value": v,
                "defined": 1,
                "source_csv": str(src),
            })

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Collect per-subgroup feature/stat values per simulator for one config dir (tree or allele)."
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