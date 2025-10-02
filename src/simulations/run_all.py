import argparse
import sys
import yaml
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import CONFIG_DIR, RAW_DATA_DIR, SIMULATORS, FILE_FORMATS
from pathlib import Path
from ms_runner import run_ms_simulation
from msms_runner import run_msms_simulation
from msprime_runner import run_msprime_simulation, run_mspms_simulation
from utils.io_helpers import make_run_output_dir

SIM_FUNCTIONS = {
    "ms": run_ms_simulation,
    "msprime": run_mspms_simulation,
    "msms": run_msms_simulation
}

def run_all_simulations(config_path, config, output_dir=None, simulators=SIMULATORS, reps=1):
    if output_dir is None:
        base_run_dir = make_run_output_dir(config_path)
    else:
        base_run_dir = Path(output_dir)
        base_run_dir.mkdir(parents=True, exist_ok=True)

    for i in range(reps):
        if reps == 1:
            rep_dir = base_run_dir
        else:
            rep_dir = base_run_dir / f"rep{i+1}"  # rep1, rep2, ...
            rep_dir.mkdir(parents=True, exist_ok=True)

        for sim in simulators:
            if sim not in SIM_FUNCTIONS:
                print(f"⚠️ Simulator {sim} unknown, skipping")
                continue

            output_file = rep_dir / FILE_FORMATS["raw"].format(SIM=sim)
            SIM_FUNCTIONS[sim](config, output_file=output_file)

    print(f"✅ Simulations finished successfully! Results saved in: {base_run_dir}")



parser = argparse.ArgumentParser(
    description="Run simulations with one or more config files."
)
parser.add_argument(
    "--config-dir",
    type=Path,
    default=CONFIG_DIR,
    help="Path to directory containing config files (default: configs/)",
)
parser.add_argument(
    "--configs",
    nargs="+",
    type=str,
    help="One or more config filenames to run (default: all *.yaml in config-dir)",
)
parser.add_argument(
    "--outdir",
    type=Path,
    default=RAW_DATA_DIR,
    help="Base output directory (default: data/raw/)",
)
parser.add_argument(
    "--simulator",
    type=str,
    choices=["ms", "msms", "msprime", "all"],
    default="all",
    help="Which simulator(s) to run (default: all)",
)
parser.add_argument(
    "--reps",
    type=int,
    default=1,
    help="Number of repetitions (default: 1)"
)

args = parser.parse_args()

config_dir = args.config_dir
outdir = args.outdir

if not config_dir.exists():
    sys.exit(f"Error: Config directory not found: {config_dir}")

# Wenn keine configs angegeben -> alle yaml im Ordner
if args.configs:
    config_files = [config_dir / name for name in args.configs]
else:
    config_files = sorted(config_dir.glob("*.yaml"))

if not config_files:
    sys.exit(f"No config files found in {config_dir}")

simulators = SIMULATORS if args.simulator=="all" else [args.simulator]

for config_path in config_files:
    print(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    config_outdir = outdir / config_path.stem
    config_outdir.mkdir(parents=True, exist_ok=True)

    print(f"Running simulations for {config_path.stem} -> {config_outdir}")
    run_all_simulations(
        config_path=str(config_path),
        config=cfg,
        output_dir=config_outdir,
        simulators=simulators,
        reps=args.reps
    )


