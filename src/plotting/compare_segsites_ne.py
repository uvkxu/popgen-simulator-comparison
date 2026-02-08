#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
COMBO_YAML_NAME = "combo_parameters.yaml"
SIMULATORS = ["ms", "msms", "mspms", "msprime"]

# ALLE segsites-Vorkommen aus Textdateien
SEGSITES_RE = re.compile(r"^\s*segsites:\s*(\d+)\s*$", re.MULTILINE)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def os_cpu_count():
    try:
        import os
        return os.cpu_count() or 2
    except Exception:
        return 2


def safe_load_yaml(p: Path) -> dict:
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def infer_sample_size_from_combo(combo: dict) -> int:
    """
    combo-yaml kann z.B. enthalten:
      - overall_samples: 16
      - sample_structure: [4,4,4,4]
    """
    if combo.get("overall_samples") is not None:
        return int(combo["overall_samples"])

    ss = combo.get("sample_structure")
    if isinstance(ss, list) and ss:
        try:
            return int(sum(int(x) for x in ss))
        except Exception:
            pass

    for k in ("sample_size", "sample_sizes"):
        if combo.get(k) is not None:
            return int(combo[k])

    return -1


def find_sim_output_files(rep_dir: Path, sim: str) -> list[Path]:
    """
    Robust/heuristisch:
    - bevorzugt Dateien, die den Sim-Namen enthalten
    - priorisiert Textdateien; für msprime auch .trees möglich
    """
    sim_lower = sim.lower() + "_"
    candidates = []

    # 1) Name enthält simulator-token
    for p in rep_dir.iterdir():
        if not p.is_file():
            continue
        if sim_lower in p.name.lower():
            candidates.append(p)

    # 2) Fallback: wenn nichts gefunden, nimm alle Dateien im rep_dir
    if not candidates:
        return []

    # Priorisierung
    def score(p: Path):
        suf = p.suffix.lower()
        s = 0
        if suf in (".txt", ".out", ".ms", ".log"):
            s += 10
        if suf == ".trees":
            s += 9 if sim_lower == "msprime_" else -5
        # leichte Präferenz für "nicht winzige" Dateien
        try:
            size = p.stat().st_size
            if size < 50:
                s -= 2
        except OSError:
            pass
        return -s  # sort ascending

    return sorted(candidates, key=score)


def parse_all_segsites_from_text_file(p: Path) -> list[int]:
    """
    Liest ALLE 'segsites:' aus einer Textdatei.
    """
    try:
        text = p.read_text(errors="ignore")
    except Exception:
        return []
    return [int(m.group(1)) for m in SEGSITES_RE.finditer(text)]


def parse_segsites_from_trees(p: Path) -> int | None:
    """
    Optional: falls msprime TreeSequence-Dateien (.trees) vorhanden sind.
    """
    try:
        import tskit
    except Exception:
        return None

    try:
        ts = tskit.load(str(p))
        return int(ts.num_sites)
    except Exception:
        return None


def extract_segsites(rep_dir: Path, sim: str) -> tuple[list[int], Path | None]:
    """
    Gibt (list_of_segsites, file_used) zurück.
    """
    files = find_sim_output_files(rep_dir, sim)
    for p in files:
        # 1) .trees (msprime)
        if p.suffix.lower() == ".trees":
            s = parse_segsites_from_trees(p)
            if s is not None:
                return [int(s)], p

        # 2) Text outputs: ALLE segsites ziehen
        segs = parse_all_segsites_from_text_file(p)
        if segs:
            return segs, p

    return [], None


# ------------------------------------------------------------
# Job generation
# ------------------------------------------------------------
def iter_jobs(raw_root: Path):
    """
    Erwartete Struktur:
      raw_root / <config> / <combo_folder> / combo_parameters.yaml
      raw_root / <config> / <combo_folder> / rep1 / ...
    """
    for config_dir in sorted(raw_root.iterdir()):
        if not config_dir.is_dir():
            continue
        config = config_dir.name

        for combo_dir in sorted(config_dir.iterdir()):
            if not combo_dir.is_dir():
                continue

            combo_yaml = combo_dir / COMBO_YAML_NAME
            if not combo_yaml.exists():
                continue

            combo = safe_load_yaml(combo_yaml)

            # Robust gegenüber alten key-names
            seq_len = int(combo.get("sequence_length", combo.get("sequence_lengths", -1)))
            Ne = int(combo.get("population_size", combo.get("population_sizes", -1)))
            mu = float(combo.get("mutation_rate", combo.get("mutation_rates", float("nan"))))
            sample_size = infer_sample_size_from_combo(combo)

            for rep_dir in sorted(combo_dir.glob("rep*")):
                if not rep_dir.is_dir():
                    continue

                for sim in SIMULATORS:
                    yield {
                        "config": config,
                        "combo_dir": str(combo_dir),
                        "rep": rep_dir.name,
                        "sim": sim,
                        "seq_len": seq_len,
                        "Ne": Ne,
                        "sample_size": sample_size,
                        "mu": mu,
                        "rep_dir": str(rep_dir),
                    }


def worker(job: dict) -> list[dict]:
    """
    Gibt eine LISTE rows zurück: 1 row pro segsites in der Datei.
    """
    rep_dir = Path(job["rep_dir"])
    sim = job["sim"]

    segsites_list, used = extract_segsites(rep_dir, sim)

    rows = []
    if not segsites_list:
        out = dict(job)
        out["segsites"] = None
        out["subrep"] = None
        out["used_file"] = str(used) if used is not None else None
        out["ok"] = False
        rows.append(out)
        return rows

    for i, s in enumerate(segsites_list, start=1):
        out = dict(job)
        out["segsites"] = int(s)
        out["subrep"] = i
        out["used_file"] = str(used) if used is not None else None
        out["ok"] = True
        rows.append(out)

    return rows

def plot_segsites_vs_Ne(df: pd.DataFrame, outdir: Path, prefix: str = "segsites"):
    """
    Für jede (config, seq_len, sample_size) ein Plot:
      x = Ne (numerisch)
      y = mean segsites über Reps
      Linien = Simulatoren
    """
    outdir.mkdir(parents=True, exist_ok=True)
    d = df[df["ok"]].copy()
    if d.empty:
        return

    # Aggregate über Reps (und subreps) innerhalb fixer Bedingungen
    agg = (
        d.groupby(["config", "seq_len", "sample_size", "Ne", "sim"], as_index=False)
         .agg(mean_segsites=("segsites", "mean"),
              std=("segsites", "std"),
              n=("segsites", "size"))
    )

    # Pro (config, L, S) ein Plot
    for (config, L, S), sub in agg.groupby(["config", "seq_len", "sample_size"]):
        sub = sub.sort_values("Ne")

        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()

        for sim in SIMULATORS:
            ss = sub[sub["sim"] == sim]
            if ss.empty:
                continue
            ax.plot(ss["Ne"], ss["mean_segsites"], marker="o", label=sim)

        ax.set_xlabel("Ne")
        ax.set_ylabel("Ø segsites (über Reps)")
        ax.set_title(f"segsites vs Ne | {config} | L={int(L)} | S={int(S)}")
        ax.set_xscale("log")  # meistens hübscher, weil Ne oft in Größenordnungen variiert
        ax.legend()
        plt.tight_layout()

        fname = f"{prefix}_vs_Ne__{config}__L{int(L)}__S{int(S)}.png"
        fig.savefig(outdir / fname, dpi=200)
        plt.close(fig)

def plot_segsites_vs_Ne_per_config_aggregated(df: pd.DataFrame, outdir: Path, prefix: str = "segsites"):
    """
    Pro config EIN Plot:
      x = Ne (log)
      y = segsites (über (seq_len, sample_size) aggregiert, aber korrekt zweistufig)
      Linien = Simulatoren

    Zweistufig:
      1) pro (config, seq_len, sample_size, Ne, sim) Mittel über Reps
      2) pro (config, Ne, sim) Mittel über (seq_len, sample_size)
    """
    outdir.mkdir(parents=True, exist_ok=True)
    d = df[df["ok"]].copy()
    if d.empty:
        return

    # Stufe 1: Rep-Aggregation pro fixe Bedingungen
    lvl1 = (
        d.groupby(["config", "seq_len", "sample_size", "Ne", "sim"], as_index=False)
         .agg(mean_segsites=("segsites", "mean"))
    )

    # Stufe 2: über (seq_len, sample_size) aggregieren
    lvl2 = (
        lvl1.groupby(["config", "Ne", "sim"], as_index=False)
            .agg(mean_segsites=("mean_segsites", "mean"))
    )

    for config, sub in lvl2.groupby("config"):
        sub = sub.sort_values("Ne")

        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()

        for sim in SIMULATORS:
            ss = sub[sub["sim"] == sim]
            if ss.empty:
                continue
            ax.plot(ss["Ne"], ss["mean_segsites"], marker="o", label=sim)

        ax.set_xscale("log")
        ax.set_xlabel("Ne")
        ax.set_ylabel("Ø segsites (über L und Sample Size aggregiert)")
        ax.set_title(f"Aggregiert: segsites vs Ne | {config}")
        ax.legend()
        plt.tight_layout()

        fname = f"{prefix}_vs_Ne__{config}__AGG_over_L_and_S.png"
        fig.savefig(outdir / fname, dpi=200)
        plt.close(fig)


# ------------------------------------------------------------
# Plotting + aggregation
# ------------------------------------------------------------
def make_plots(df: pd.DataFrame, outdir: Path, prefix: str = "segsites"):
    outdir.mkdir(parents=True, exist_ok=True)
    d = df[df["ok"]].copy()
    if d.empty:
        return

    # Plot A: (Ne, L, S)
    d["x_combo"] = d.apply(
        lambda r: f"Ne={int(r['Ne'])}\nL={int(r['seq_len'])}\nS={int(r['sample_size'])}",
        axis=1
    )
    aggA = (
        d.groupby(["x_combo", "sim"], as_index=False)
         .agg(mean_segsites=("segsites", "mean"), std=("segsites", "std"), n=("segsites", "size"))
    )
    x_order = list(dict.fromkeys(aggA["x_combo"].tolist()))

    fig = plt.figure(figsize=(max(12, 0.35 * len(x_order)), 6))
    ax = plt.gca()
    for sim in SIMULATORS:
        sub = aggA[aggA["sim"] == sim].set_index("x_combo").reindex(x_order).reset_index()
        ax.plot(sub["x_combo"], sub["mean_segsites"], marker="o", label=sim)


    ax.set_xlabel("Kombination (Ne, Sequenzlänge, Sample Size)")
    ax.set_ylabel("Ø segsites")
    ax.set_title("Vergleich segsites pro (Ne, L, S) – Mittelwert über Reps")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outdir / f"{prefix}_by_Ne_L_S.png", dpi=200)
    plt.close(fig)

    # Plot B: (Ne, L) – sample_size gemittelt (weniger sauber)
    d["x_Ne_L"] = d.apply(lambda r: f"Ne={int(r['Ne'])}\nL={int(r['seq_len'])}", axis=1)
    aggB = (
        d.groupby(["x_Ne_L", "sim"], as_index=False)
         .agg(mean_segsites=("segsites", "mean"), std=("segsites", "std"), n=("segsites", "size"))
    )
    x_orderB = list(dict.fromkeys(aggB["x_Ne_L"].tolist()))

    fig = plt.figure(figsize=(max(12, 0.35 * len(x_orderB)), 6))
    ax = plt.gca()
    for sim in SIMULATORS:
        sub = aggB[aggB["sim"] == sim].set_index("x_Ne_L").reindex(x_orderB).reset_index()
        ax.plot(sub["x_Ne_L"], sub["mean_segsites"], marker="o", label=sim)

    ax.set_xlabel("Kombination (Ne, Sequenzlänge) – sample_size gemittelt")
    ax.set_ylabel("Ø segsites")
    ax.set_title("Vergleich segsites pro (Ne, L) – über sample_size gemittelt")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outdir / f"{prefix}_by_Ne_L_aggSample.png", dpi=200)
    plt.close(fig)


def make_msprime_diff_table(df: pd.DataFrame, outdir: Path, prefix: str = "segsites"):
    """
    Aggregiert nach (config, seq_len, sample_size, Ne, sim) und berechnet
    absdiff zu msprime-Mittelwerten, wo möglich.
    """
    ok = df[df["ok"]].copy()
    if ok.empty:
        return

    agg = (
        ok.groupby(["config", "seq_len", "sample_size", "Ne", "sim"], as_index=False)
          .agg(mean_segsites=("segsites", "mean"), n=("segsites", "size"))
    )

    piv = agg.pivot_table(
        index=["config", "seq_len", "sample_size", "Ne"],
        columns="sim",
        values="mean_segsites",
        aggfunc="first"
    ).reset_index()

    if "msprime" in piv.columns:
        for sim in ["ms", "msms", "mspms"]:
            if sim in piv.columns:
                piv[f"absdiff_{sim}_vs_msprime"] = (piv[sim] - piv["msprime"]).abs()

    piv.to_csv(outdir / f"{prefix}_agg_pivot_vs_msprime.csv", index=False)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, required=True, help="Pfad zu DATA_DIR/raw")
    ap.add_argument("--outdir", type=str, required=True, help="Output-Verzeichnis (csv + plots)")
    ap.add_argument("--nprocs", type=int, default=max(1, os_cpu_count() - 1), help="Parallel-Prozesse")
    ap.add_argument("--prefix", type=str, default="segsites", help="Datei-Prefix für Outputs")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = list(iter_jobs(raw_root))
    if not jobs:
        raise SystemExit(f"Keine Jobs gefunden. Prüfe Struktur unter: {raw_root}")

    results = []
    with ProcessPoolExecutor(max_workers=args.nprocs) as ex:
        futs = [ex.submit(worker, j) for j in jobs]

        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc="Parsing segsites",
            unit="job"
        ):
            results.extend(fut.result())


    df = pd.DataFrame(results)

    df.to_csv(outdir / f"{args.prefix}_raw_table.csv", index=False)

    missing = df[~df["ok"]].copy()
    if len(missing) > 0:
        missing.to_csv(outdir / f"{args.prefix}_missing.csv", index=False)

    make_plots(df, outdir, prefix=args.prefix)
    plot_segsites_vs_Ne(df, outdir, prefix=args.prefix)
    plot_segsites_vs_Ne_per_config_aggregated(df, outdir, prefix=args.prefix)
    make_msprime_diff_table(df, outdir, prefix=args.prefix)

    print("OK. Outputs:")
    print(f"- {outdir / (args.prefix + '_raw_table.csv')}")
    print(f"- {outdir / (args.prefix + '_missing.csv')} (falls vorhanden)")
    print(f"- {outdir / (args.prefix + '_by_Ne_L_S.png')}")
    print(f"- {outdir / (args.prefix + '_by_Ne_L_aggSample.png')}")
    print(f"- {outdir / (args.prefix + '_agg_pivot_vs_msprime.csv')}")


if __name__ == "__main__":
    main()