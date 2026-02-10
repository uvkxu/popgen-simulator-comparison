#!/usr/bin/env python3
# plot_topfeatures.py

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

try:
    import yaml  # PyYAML
except Exception:
    yaml = None


CONFIG_RE = re.compile(r"sample_size_(?P<sample>\d+),seq_len_(?P<seqlen>\d+)")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def paired_barplot(
    df: pd.DataFrame,
    x_col: str,
    y_tree: str,
    y_allele: str,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    sort_by: str = "mean",
    rotate_xticks: bool = True,
) -> None:
    """
    Creates grouped bars: for each x, show two bars (tree, allele) side-by-side.
    Expects df to contain columns: x_col, y_tree, y_allele.
    """
    d = df.copy()

    if sort_by == "mean":
        d["_sort"] = d[[y_tree, y_allele]].mean(axis=1, skipna=True)
    elif sort_by == "tree":
        d["_sort"] = d[y_tree]
    elif sort_by == "allele":
        d["_sort"] = d[y_allele]
    else:
        d["_sort"] = d[[y_tree, y_allele]].mean(axis=1, skipna=True)

    d = d.sort_values("_sort", ascending=True)
    xs = list(range(len(d)))

    width = 0.4
    fig = plt.figure()
    ax = plt.gca()

    ax.bar([x - width / 2 for x in xs], d[y_tree].values, width=width, label="tree")
    ax.bar([x + width / 2 for x in xs], d[y_allele].values, width=width, label="allele")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(d[x_col].astype(str).values, rotation=60 if rotate_xticks else 0, ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def lineplot_two_series(
    df: pd.DataFrame,
    x_col: str,
    y_tree: str,
    y_allele: str,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    x_log: bool = False,
    aggregate_duplicates: bool = True,
) -> None:
    d = df.copy()

    # aggregate duplicates by mean if multiple configs share same x
    if aggregate_duplicates:
        d = d.groupby(x_col, dropna=False)[[y_tree, y_allele]].mean().reset_index()

    d = d.sort_values(x_col)

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(d[x_col].values, d[y_tree].values, marker="o", label="tree")
    ax.plot(d[x_col].values, d[y_allele].values, marker="o", label="allele")

    if x_log:
        ax.set_xscale("log")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def find_demes_yaml(demes_root: Path, model_name: str) -> Optional[Path]:
    """
    Best-effort: find a YAML whose filename contains model_name.
    If you prefer strict mapping, replace this with an explicit mapping dict.
    """
    cand = list(demes_root.rglob(f"*{model_name}*.yml")) + list(demes_root.rglob(f"*{model_name}*.yaml"))
    if cand:
        return cand[0]
    return None


def parse_present_day_sizes_and_migration(yaml_path: Path) -> Tuple[Optional[float], Optional[float], Optional[bool]]:
    """
    Returns (mean_present_day_size, min_present_day_size, has_continuous_migration)
    for present-day demes (epochs with end_time == 0).
    Uses start_size of the last epoch reaching end_time=0 (as in your example).
    """
    if yaml is None:
        return None, None, None

    obj = yaml.safe_load(yaml_path.read_text())

    demes = obj.get("demes", [])
    present_sizes = []
    for d in demes:
        epochs = d.get("epochs", [])
        # find an epoch with end_time == 0, take its start_size
        for ep in epochs:
            if ep.get("end_time", None) == 0:
                sz = ep.get("start_size", None)
                if isinstance(sz, (int, float)) and not math.isnan(float(sz)):
                    present_sizes.append(float(sz))
                break

    if not present_sizes:
        mean_sz = None
        min_sz = None
    else:
        mean_sz = sum(present_sizes) / len(present_sizes)
        min_sz = min(present_sizes)

    migrations = obj.get("migrations", None)
    has_mig = bool(migrations)  # True if list exists and non-empty

    return mean_sz, min_sz, has_mig


def main() -> None:
    ap = argparse.ArgumentParser(description="Create plots from aggregated top_features CSVs.")
    ap.add_argument("--agg-dir", type=Path, required=True, help="Directory containing agg_overall.csv etc.")
    ap.add_argument("--outdir", type=Path, required=True, help="Directory for plots.")
    ap.add_argument("--demes-root", type=Path, default=None, help="Root directory containing demes YAMLs (optional).")
    ap.add_argument("--topk-features", type=int, default=30, help="Top-K features to plot in global feature plots.")
    args = ap.parse_args()

    agg_dir: Path = args.agg_dir
    outdir: Path = args.outdir
    safe_mkdir(outdir)

    df_overall = pd.read_csv(agg_dir / "agg_overall.csv")
    df_feat = pd.read_csv(agg_dir / "agg_feature_accuracy.csv")
    df_passed = pd.read_csv(agg_dir / "agg_passed_frequency.csv")

    # --------------------------
    # Helper: pivot tree/allele together for a config
    # --------------------------
    key_cols = ["model", "config", "sample_size", "seq_len"]

    # Choose overall metric: overall_all_features (your ALL_FEATURES max_accuracy)
    df_overall_piv = (
        df_overall.pivot_table(
            index=key_cols,
            columns="feature_type",
            values="overall_all_features",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"tree": "acc_tree", "allele": "acc_allele"})
    )

    # 1) Barplot per config: paired bars tree vs allele, ranked
    df_overall_piv["config_label"] = df_overall_piv.apply(
        lambda r: f"{r['model']}/{r['config']}", axis=1
    )
    paired_barplot(
        df=df_overall_piv,
        x_col="config_label",
        y_tree="acc_tree",
        y_allele="acc_allele",
        title="Overall accuracy per config (ALL_FEATURES)",
        xlabel="Config",
        ylabel="Accuracy",
        outpath=outdir / "bar_overall_by_config_tree_vs_allele.png",
        sort_by="mean",
        rotate_xticks=True,
    )

    # 4) Lineplot: sequence length vs accuracy (mean over configs per seq_len)
    df_seq = df_overall_piv.groupby("seq_len")[["acc_tree", "acc_allele"]].mean().reset_index()
    lineplot_two_series(
        df=df_seq,
        x_col="seq_len",
        y_tree="acc_tree",
        y_allele="acc_allele",
        title="Accuracy vs sequence length",
        xlabel="Sequence length",
        ylabel="Accuracy",
        outpath=outdir / "line_accuracy_vs_sequence_length.png",
        x_log=True,
        aggregate_duplicates=False,
    )

    # If demes-root provided: compute mean/min present-day Ne and migration-flag
    if args.demes_root is not None:
        demes_root = args.demes_root

        mean_map: Dict[str, Optional[float]] = {}
        min_map: Dict[str, Optional[float]] = {}
        mig_map: Dict[str, Optional[bool]] = {}

        for model in df_overall_piv["model"].unique():
            yp = find_demes_yaml(demes_root, str(model))
            if yp is None:
                mean_map[str(model)] = None
                min_map[str(model)] = None
                mig_map[str(model)] = None
                continue
            mean_sz, min_sz, has_mig = parse_present_day_sizes_and_migration(yp)
            mean_map[str(model)] = mean_sz
            min_map[str(model)] = min_sz
            mig_map[str(model)] = has_mig

        df_overall_piv["mean_present_Ne"] = df_overall_piv["model"].astype(str).map(mean_map)
        df_overall_piv["min_present_Ne"] = df_overall_piv["model"].astype(str).map(min_map)
        df_overall_piv["has_continuous_migration"] = df_overall_piv["model"].astype(str).map(mig_map)

        # 2) Lineplot: (sample_size / mean Ne) vs accuracy
        df_ratio_mean = df_overall_piv.dropna(subset=["mean_present_Ne"]).copy()
        df_ratio_mean["ratio_sample_over_meanNe"] = df_ratio_mean["sample_size"] / df_ratio_mean["mean_present_Ne"]
        lineplot_two_series(
            df=df_ratio_mean[["ratio_sample_over_meanNe", "acc_tree", "acc_allele"]],
            x_col="ratio_sample_over_meanNe",
            y_tree="acc_tree",
            y_allele="acc_allele",
            title="Accuracy vs sampling density (sample_size / mean present-day Ne)",
            xlabel="sample_size / mean(N_e, present demes)",
            ylabel="Accuracy",
            outpath=outdir / "line_accuracy_vs_sample_over_meanNe.png",
            x_log=True,
            aggregate_duplicates=True,
        )

        # 3) Lineplot: (sample_size / min Ne) vs accuracy
        df_ratio_min = df_overall_piv.dropna(subset=["min_present_Ne"]).copy()
        df_ratio_min["ratio_sample_over_minNe"] = df_ratio_min["sample_size"] / df_ratio_min["min_present_Ne"]
        lineplot_two_series(
            df=df_ratio_min[["ratio_sample_over_minNe", "acc_tree", "acc_allele"]],
            x_col="ratio_sample_over_minNe",
            y_tree="acc_tree",
            y_allele="acc_allele",
            title="Accuracy vs sampling density (sample_size / min present-day Ne)",
            xlabel="sample_size / min(N_e, present demes)",
            ylabel="Accuracy",
            outpath=outdir / "line_accuracy_vs_sample_over_minNe.png",
            x_log=True,
            aggregate_duplicates=True,
        )

        # 5) Barplot: continuous migration vs none (paired bars)
        df_mig = df_overall_piv.dropna(subset=["has_continuous_migration"]).copy()
        df_mig["mig_label"] = df_mig["has_continuous_migration"].map(
            {True: "continuous_migration", False: "no_continuous_migration"}
        )
        df_mig_grp = df_mig.groupby("mig_label")[["acc_tree", "acc_allele"]].mean().reset_index()
        paired_barplot(
            df=df_mig_grp.rename(columns={"mig_label": "group"}),
            x_col="group",
            y_tree="acc_tree",
            y_allele="acc_allele",
            title="Accuracy by presence of continuous migration (demes 'migrations')",
            xlabel="Group",
            ylabel="Accuracy",
            outpath=outdir / "bar_accuracy_by_continuous_migration_tree_vs_allele.png",
            sort_by="mean",
            rotate_xticks=False,
        )

    # 7) Global feature accuracies over all configs: top-k per feature_type (separate plots)
    safe_mkdir(outdir / "features_global")
    for ftype in ["tree", "allele"]:
        d = df_feat[df_feat["feature_type"] == ftype].copy()
        if d.empty:
            continue
        d2 = d.groupby("feature")["max_accuracy"].mean().sort_values(ascending=False).head(args.topk_features)
        fig = plt.figure()
        ax = plt.gca()
        ax.barh(d2.index[::-1], d2.values[::-1])
        ax.set_title(f"Top-{args.topk_features} features by mean accuracy ({ftype})")
        ax.set_xlabel("Mean max_accuracy")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        fig.savefig(outdir / "features_global" / f"bar_top{args.topk_features}_feature_accuracy_{ftype}.png", dpi=200)
        plt.close(fig)

    # 9) Global passed frequency over all configs: top-k per feature_type (separate plots)
    safe_mkdir(outdir / "passed_global")
    for ftype in ["tree", "allele"]:
        d = df_passed[df_passed["feature_type"] == ftype].copy()
        if d.empty:
            continue
        d2 = d.groupby("feature")["passed"].sum().sort_values(ascending=False).head(args.topk_features)
        fig = plt.figure()
        ax = plt.gca()
        ax.barh(d2.index[::-1], d2.values[::-1])
        ax.set_title(f"Top-{args.topk_features} features by passed-count ({ftype})")
        ax.set_xlabel("Count (times feature appears in PASSED section)")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        fig.savefig(outdir / "passed_global" / f"bar_top{args.topk_features}_passed_frequency_{ftype}.png", dpi=200)
        plt.close(fig)

    # 6) Per-config feature accuracy plots (separate per feature_type)
    safe_mkdir(outdir / "features_by_config")
    for ftype in ["tree", "allele"]:
        d = df_feat[df_feat["feature_type"] == ftype].copy()
        if d.empty:
            continue
        for (model, config, sample_size, seq_len), g in d.groupby(["model", "config", "sample_size", "seq_len"]):
            g2 = g.groupby("feature")["max_accuracy"].mean().sort_values(ascending=False).head(args.topk_features)
            fig = plt.figure()
            ax = plt.gca()
            ax.barh(g2.index[::-1], g2.values[::-1])
            ax.set_title(f"{ftype}: Top-{args.topk_features} feature accuracies\n{model}/{config}")
            ax.set_xlabel("Mean max_accuracy")
            ax.set_ylabel("Feature")
            fig.tight_layout()
            safe_mkdir(outdir / "features_by_config" / ftype)
            fig.savefig(
                outdir / "features_by_config" / ftype / f"{model}__{config}__top{args.topk_features}.png",
                dpi=200,
            )
            plt.close(fig)

    # 8) Per-config passed frequency plots (separate per feature_type)
    safe_mkdir(outdir / "passed_by_config")
    for ftype in ["tree", "allele"]:
        d = df_passed[df_passed["feature_type"] == ftype].copy()
        if d.empty:
            continue
        for (model, config, sample_size, seq_len), g in d.groupby(["model", "config", "sample_size", "seq_len"]):
            g2 = g.groupby("feature")["passed"].sum().sort_values(ascending=False).head(args.topk_features)
            fig = plt.figure()
            ax = plt.gca()
            ax.barh(g2.index[::-1], g2.values[::-1])
            ax.set_title(f"{ftype}: Top-{args.topk_features} passed-feature counts\n{model}/{config}")
            ax.set_xlabel("Count (appearances in PASSED section)")
            ax.set_ylabel("Feature")
            fig.tight_layout()
            safe_mkdir(outdir / "passed_by_config" / ftype)
            fig.savefig(
                outdir / "passed_by_config" / ftype / f"{model}__{config}__top{args.topk_features}.png",
                dpi=200,
            )
            plt.close(fig)

    print(f"Done. Plots written to: {outdir}")


if __name__ == "__main__":
    main()
