#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


COMBO_YAML_NAME = "combo_parameters.yaml"
SIMULATORS = ["ms", "msms", "mspms", "msprime"]
SEGSITES_RE = re.compile(r"^\s*segsites:\s*(\d+)\s*$", re.MULTILINE)


def os_cpu_count():
    try:
        import os
        return os.cpu_count() or 2
    except Exception:
        return 2


def safe_load_yaml(p: Path) -> dict:
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def harmonic_number(n: int) -> float:
    if n <= 1:
        return float("nan")
    return sum(1.0 / i for i in range(1, n))


def infer_sample_size_from_combo(combo: dict) -> int:
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
    Du hast garantiert: Dateinamen enthalten '{sim}_'
    => sicher gegen 'ms' in 'msms/msprime'
    """
    sim_token = sim.lower() + "_"
    candidates = [p for p in rep_dir.iterdir()
                  if p.is_file() and sim_token in p.name.lower()]

    if not candidates:
        return []  # kein Fallback: lieber missing als falscher sim

    def score(p: Path):
        suf = p.suffix.lower()
        s = 0
        if suf in (".txt", ".out", ".ms", ".log"):
            s += 10
        if suf == ".trees":
            s += 9 if sim.lower() == "msprime" else -5
        try:
            if p.stat().st_size < 50:
                s -= 2
        except OSError:
            pass
        return -s

    return sorted(candidates, key=score)


def parse_all_segsites_from_text_file(p: Path) -> list[int]:
    try:
        text = p.read_text(errors="ignore")
    except Exception:
        return []
    return [int(m.group(1)) for m in SEGSITES_RE.finditer(text)]


def parse_segsites_from_trees(p: Path) -> int | None:
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
    files = find_sim_output_files(rep_dir, sim)
    for p in files:
        if p.suffix.lower() == ".trees":
            s = parse_segsites_from_trees(p)
            if s is not None:
                return [int(s)], p
        segs = parse_all_segsites_from_text_file(p)
        if segs:
            return segs, p
    return [], None


def iter_jobs(raw_root: Path):
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
                        "sim": sim,
                        "seq_len": seq_len,
                        "Ne": Ne,
                        "sample_size": sample_size,
                        "mu": mu,
                        "rep_dir": str(rep_dir),
                    }


def worker(job: dict) -> list[dict]:
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


def plot_one_per_config(df: pd.DataFrame, outdir: Path, prefix: str = "segsites"):
    """
    Genau 1 Plot pro config:
      x = Ne (Input)
      y = S_norm = S / (L * a_n)   (normalisiert nach seq_len + sample_size)
      Linien = Simulatoren
    """
    outdir.mkdir(parents=True, exist_ok=True)
    d = df[df["ok"]].copy()
    if d.empty:
        return

    d["a_n"] = d["sample_size"].astype(int).apply(harmonic_number)
    d["S_norm"] = d["segsites"] / (d["seq_len"] * d["a_n"])

    # erst Rep-Aggregation pro (config, L, n, Ne, sim)
    lvl1 = (
        d.groupby(["config", "seq_len", "sample_size", "Ne", "sim"], as_index=False)
         .agg(mean_S_norm=("S_norm", "mean"))
    )

    # dann über (L, n) mitteln => 1 Wert pro (config, Ne, sim)
    lvl2 = (
        lvl1.groupby(["config", "Ne", "sim"], as_index=False)
            .agg(mean_S_norm=("mean_S_norm", "mean"))
    )

    for config, sub in lvl2.groupby("config"):
        sub = sub.sort_values("Ne")

        fig = plt.figure(figsize=(7.5, 5.5))
        ax = plt.gca()

        for sim in SIMULATORS:
            ss = sub[sub["sim"] == sim].sort_values("Ne")
            if ss.empty:
                continue
            ax.plot(ss["Ne"], ss["mean_S_norm"], marker="o", label=sim)

        ax.set_xscale("log")
        ax.set_xlabel("Ne")
        ax.set_ylabel("normalisierte segsites: S / (L * a_n)")
        ax.set_title(f"S_norm vs Ne | {config}")
        ax.legend()
        plt.tight_layout()

        fig.savefig(outdir / f"{prefix}_Snorm_vs_Ne__{config}.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--nprocs", type=int, default=max(1, os_cpu_count() - 1))
    ap.add_argument("--prefix", type=str, default="segsites")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = list(iter_jobs(raw_root))
    if not jobs:
        raise SystemExit(f"Keine Jobs gefunden unter: {raw_root}")

    results = []
    with ProcessPoolExecutor(max_workers=args.nprocs) as ex:
        futs = [ex.submit(worker, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Parsing segsites", unit="job"):
            results.extend(fut.result())

    df = pd.DataFrame(results)
    df.to_csv(outdir / f"{args.prefix}_raw_table.csv", index=False)

    missing = df[~df["ok"]].copy()
    if len(missing) > 0:
        missing.to_csv(outdir / f"{args.prefix}_missing.csv", index=False)

    plot_one_per_config(df, outdir, prefix=args.prefix)

    print("OK:")
    print(f"- {outdir / (args.prefix + '_raw_table.csv')}")
    print(f"- {outdir / (args.prefix + '_missing.csv')} (falls vorhanden)")
    print(f"- pro config: {args.prefix}_Snorm_vs_Ne__<config>.png")


if __name__ == "__main__":
    main()
