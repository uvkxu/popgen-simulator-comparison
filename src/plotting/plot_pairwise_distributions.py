
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

UTILS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(UTILS_ROOT)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(PROJECT_ROOT)

from utils.io_helpers import CONFIG_ORDER as ORDER
from project_paths import PLOTS_RESULTS_DIR, TOGETHER_TOPFEATURES_DIR


# ============================================================
# SETTINGS
# ============================================================

BASELINE = 0.5

DEFAULT_PAIRS = [
    "ms-msms",
    "msms-mspms",
    "ms-mspms",
]

COLORS = [
    "#8172B3",
    "#CCB974",
    "#64B5CD",
    "#C44E52",
    "#55A868",
    "#4C72B0",
]


# ============================================================
# MATPLOTLIB
# ============================================================

def setup_matplotlib() -> None:

    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


# ============================================================
# CSV READING
# ============================================================

def read_all_features_value(
    path: Path,
    metric: str,
) -> float | None:

    if not path.is_file() or path.stat().st_size == 0:
        print(f"WARNING: Missing or empty file: {path}")
        return None

    try:
        df = pd.read_csv(path)

    except pd.errors.EmptyDataError:
        print(f"WARNING: No columns found in file: {path}")
        return None

    if "feature" not in df.columns or metric not in df.columns:
        print(f"WARNING: Required columns missing in {path}")
        return None

    df["feature"] = df["feature"].astype(str).str.strip()

    sub = df[df["feature"] == "ALL_FEATURES"]

    if sub.empty:
        return None

    v = sub.iloc[0][metric]

    if pd.isna(v):
        return None

    return float(v)


def read_chance_baseline(path: Path) -> float | None:

    if not path.is_file() or path.stat().st_size == 0:
        return None

    try:
        df = pd.read_csv(path)

    except pd.errors.EmptyDataError:
        return None

    if "feature" not in df.columns or "accuracy" not in df.columns:
        return None

    df["feature"] = df["feature"].astype(str).str.strip()

    row = df.loc[df["feature"] == "CHANCE_BASELINE"]

    if row.empty or pd.isna(row.iloc[0]["accuracy"]):
        return None

    return float(row.iloc[0]["accuracy"])


# ============================================================
# COMPARISON PAIRS
# ============================================================

def normalize_pair(pair: str) -> tuple[str, str]:

    parts = pair.split("-")

    if len(parts) != 2:
        raise ValueError(
            f"Invalid comparison pair '{pair}'. "
            "Expected format: A-B"
        )

    return tuple(sorted(parts))


def find_comparison_files(root: Path) -> dict:

    found = {}

    for path in root.rglob("feature_ranking_*_with_summary.csv"):

        name = path.name

        pair_string = (
            name.removeprefix("feature_ranking_")
            .removesuffix("_with_summary.csv")
        )

        parts = pair_string.split("_")

        if len(parts) != 2:
            continue

        pair = tuple(sorted(parts))

        found.setdefault(pair, set()).add(name)

    return found


def build_comparisons(
    root: Path,
    pairs: list[str],
) -> dict:

    available_files = find_comparison_files(root)

    comparisons = {}

    for i, pair_string in enumerate(pairs):

        parts = pair_string.split("-")

        if len(parts) != 2:
            raise ValueError(
                f"Invalid comparison pair '{pair_string}'. "
                "Expected format: A-B"
            )

        pair = tuple(sorted(parts))

        if pair not in available_files:
            raise FileNotFoundError(
                f"No feature ranking file found for pair: {pair_string}"
            )

        comparisons[pair_string] = {
            "filenames": sorted(available_files[pair]),
            "label": f"{parts[0]} vs {parts[1]}",
            "color": COLORS[i % len(COLORS)],
        }

    return comparisons


# ============================================================
# COLLECT VALUES
# ============================================================

def collect_values(
    root: Path,
    filenames: List[str],
    metric: str,
) -> Dict[str, List[float]]:

    out: Dict[str, List[float]] = {}

    order_index = {
        c: i for i, c in enumerate(ORDER)
    }

    cfg_dirs = [
        p for p in root.iterdir()
        if p.is_dir()
    ]

    cfg_dirs.sort(
        key=lambda p: order_index.get(
            p.name,
            float("inf")
        )
    )

    for cfg in cfg_dirs:

        vals: List[float] = []

        for combo in sorted(
            p for p in cfg.iterdir()
            if p.is_dir()
        ):

            # Find matching comparison file
            f = next(
                (
                    combo / filename
                    for filename in filenames
                    if (combo / filename).is_file()
                ),
                None,
            )

            if f is None:
                continue

            v = read_all_features_value(
                f,
                metric,
            )

            if v is not None:
                vals.append(v)

        out[cfg.name] = vals

    return out


def collect_chance_values(
    root: Path,
    comparisons: dict,
) -> List[float]:

    vals: List[float] = []

    filenames = {
        filename
        for comp in comparisons.values()
        for filename in comp["filenames"]
    }

    for scenario in root.iterdir():

        if not scenario.is_dir():
            continue

        for combo in scenario.iterdir():

            if not combo.is_dir():
                continue

            for filename in filenames:

                f = combo / filename

                if not f.is_file():
                    continue

                v = read_chance_baseline(f)

                if v is not None:
                    vals.append(v)

    return vals


# ============================================================
# BOXPLOTS
# ============================================================

def draw_boxes(
    ax,
    data,
    positions,
    color,
    width,
):

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        whis=(0, 100),
        showfliers=False,
        medianprops=dict(
            linewidth=1.5,
            color="black",
        ),
        whiskerprops=dict(
            linewidth=1.1,
        ),
        capprops=dict(
            linewidth=1.1,
        ),
        boxprops=dict(
            linewidth=1.0,
        ),
    )

    for b in bp["boxes"]:
        b.set_facecolor(color)
        b.set_edgecolor(color)
        b.set_alpha(0.35)

    for k in ("whiskers", "caps"):
        for line in bp[k]:
            line.set_color(color)

    return bp


def plot_grouped(
    data: Dict[str, Dict[str, List[float]]],
    comparisons: dict,
    out_path: Path,
    config_gap: float = 1.4,
    box_width: float = 0.26,
) -> None:

    if not data:
        print("WARNING: No data available for plotting.")
        return

    first_comparison = next(iter(data.values()))

    configs = list(first_comparison.keys())

    if not configs:
        print("WARNING: No configurations found.")
        return

    n = len(configs)

    base = np.arange(n) * config_gap

    n_comparisons = len(comparisons)

    offsets = (
        np.arange(n_comparisons)
        - (n_comparisons - 1) / 2
    ) * box_width

    fig_w = max(
        10,
        min(20, 0.75 * n + 6),
    )

    fig, ax = plt.subplots(
        figsize=(fig_w, 6.0),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # Vertical separators
    # --------------------------------------------------------

    for i in range(n - 1):

        ax.axvline(
            (base[i] + base[i + 1]) / 2,
            color="0.2",
            linewidth=0.8,
            alpha=0.12,
            zorder=0,
        )

    # --------------------------------------------------------
    # Chance baseline
    # --------------------------------------------------------

    ax.axhline(
        BASELINE,
        linestyle="--",
        linewidth=1.4,
        color="0.1",
        alpha=0.9,
        zorder=1,
        label="chance baseline",
    )

    # --------------------------------------------------------
    # Draw comparison boxplots
    # --------------------------------------------------------

    for (comp, cfg), offset in zip(
        comparisons.items(),
        offsets,
    ):

        vals = [
            data[comp][c]
            for c in configs
        ]

        pos = base + offset

        valid = [
            (v, p)
            for v, p in zip(vals, pos)
            if len(v) > 0
        ]

        if not valid:
            continue

        valid_vals, valid_positions = zip(*valid)

        draw_boxes(
            ax,
            list(valid_vals),
            np.asarray(valid_positions),
            cfg["color"],
            box_width * 0.85,
        )

    # --------------------------------------------------------
    # Y-axis
    # --------------------------------------------------------

    ax.set_ylabel("Classification accuracy")

    ax.set_xlabel(
        "Demographic models (ordered by number of populations)"
    )

    all_vals = [
        v
        for comp_vals in data.values()
        for cfg_vals in comp_vals.values()
        for v in cfg_vals
    ]

    if all_vals:
        ymax = np.ceil(max(all_vals) * 10) / 10
        ax.set_ylim(0, max(ymax, BASELINE + 0.05))
    else:
        ax.set_ylim(0, 1.0)

    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------

    ax.yaxis.grid(
        True,
        alpha=0.22,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(base)

    ax.set_xticklabels(
        configs,
        rotation=35,
        ha="right",
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_handles = [
        Patch(
            facecolor=cfg["color"],
            edgecolor=cfg["color"],
            alpha=0.35,
            label=cfg["label"],
        )
        for cfg in comparisons.values()
    ]

    legend_handles.append(
        Line2D(
            [],
            [],
            linestyle="--",
            color="0.1",
            linewidth=1.4,
            label="chance baseline",
        )
    )

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_handles),
        frameon=True,
        fontsize=14,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        out_path,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# ACCURACY STATISTICS
# ============================================================

def print_accuracy_statistics(
    collected: Dict[str, Dict[str, List[float]]],
    comparisons: dict,
) -> None:

    print("\n" + "=" * 80)
    print("CLASSIFICATION ACCURACY SUMMARY")
    print("=" * 80)

    # --------------------------------------------------------
    # Per comparison / model statistics
    # --------------------------------------------------------

    for comp, cfg in comparisons.items():

        print(f"\n\n{cfg['label'].upper()}")
        print("-" * 80)

        for model, vals in collected[comp].items():

            arr = np.asarray(vals)

            if len(arr) == 0:
                continue

            print(
                f"{model:25s}"
                f" median={np.median(arr):.3f}"
                f" IQR=({np.percentile(arr, 25):.3f},"
                f" {np.percentile(arr, 75):.3f})"
                f" max={arr.max():.3f}"
            )

    # --------------------------------------------------------
    # Highest observed accuracy
    # --------------------------------------------------------

    best_acc = -np.inf
    best_model = None
    best_comp = None

    for comp, models in collected.items():

        for model, vals in models.items():

            if not vals:
                continue

            m = max(vals)

            if m > best_acc:
                best_acc = m
                best_model = model
                best_comp = comp

    print("\n" + "-" * 80)

    if best_model is not None:

        print(
            f"Highest observed accuracy: "
            f"{best_acc:.3f} "
            f"({best_comp}, {best_model})"
        )

    else:
        print("No valid accuracy values found.")

    # --------------------------------------------------------
    # Global distribution
    # --------------------------------------------------------

    all_vals = []

    for models in collected.values():

        for vals in models.values():
            all_vals.extend(vals)

    arr = np.asarray(all_vals)

    if arr.size == 0:
        print("\nNo valid accuracy values found.")
        return

    print("\nOverall accuracy distribution")

    print(f"Median : {np.median(arr):.3f}")
    print(f"Mean   : {arr.mean():.3f}")
    print(f"Max    : {arr.max():.3f}")

    for threshold in (0.6, 0.7, 0.8):

        n = np.sum(arr > threshold)

        print(
            f"Accuracy > {threshold:.1f}: "
            f"{n}/{len(arr)} "
            f"({100 * n / len(arr):.1f}%)"
        )

    # --------------------------------------------------------
    # Best accuracy per demographic model
    # --------------------------------------------------------

    print("\nHighest observed accuracy by model")
    print("-" * 80)

    for model in ORDER:

        mx = -np.inf

        for comp in collected.values():

            if model not in comp:
                continue

            if comp[model]:
                mx = max(mx, max(comp[model]))

        if mx > -np.inf:
            print(f"{model:25s} {mx:.3f}")


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        type=Path,
        default=TOGETHER_TOPFEATURES_DIR,
    )

    ap.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        metavar="PAIR",
        help="Comparison pairs, e.g. ms-msms msms-mspms ms-mspms",
    )

    ap.add_argument(
        "--outfile",
        type=Path,
        default=PLOTS_RESULTS_DIR / "pairwise_accuracies_overall.png",
    )

    args = ap.parse_args()

    setup_matplotlib()

    root = Path(args.input).expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {root}"
        )

    # --------------------------------------------------------
    # 1. Build comparisons dynamically
    # --------------------------------------------------------

    comparisons = build_comparisons(
        root,
        args.pairs,
    )

    print("Selected comparisons:")

    for pair, cfg in comparisons.items():

        print(
            f"  {cfg['label']} -> {cfg['filenames']}"
        )

    # --------------------------------------------------------
    # 2. Collect accuracy values
    # --------------------------------------------------------

    collected: Dict[str, Dict[str, List[float]]] = {}

    for pair, cfg in comparisons.items():

        collected[pair] = collect_values(
            root=root,
            filenames=cfg["filenames"],
            metric="accuracy",
        )

    # --------------------------------------------------------
    # 3. Print accuracy statistics
    # --------------------------------------------------------

    print_accuracy_statistics(
        collected=collected,
        comparisons=comparisons,
    )

    # --------------------------------------------------------
    # 4. Collect chance baseline values
    # --------------------------------------------------------

    chance_vals = collect_chance_values(
        root=root,
        comparisons=comparisons,
    )

    if chance_vals:

        arr = np.array(chance_vals)

        print("\nChance baseline statistics:")
        print(f"n_runs        = {len(arr)}")
        print(f"min chance    = {np.min(arr):.4f}")
        print(f"max chance    = {np.max(arr):.4f}")
        print()

    # --------------------------------------------------------
    # 5. Create and save plot
    # --------------------------------------------------------

    args.outfile.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_grouped(
        data=collected,
        comparisons=comparisons,
        out_path=args.outfile,
    )

    print(
        f"[OK] Wrote plot to {args.outfile.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
