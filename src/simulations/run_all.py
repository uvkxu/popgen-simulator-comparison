import argparse
import sys
import yaml
import os
import itertools

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from snakemake_vars import CONFIG_DIR, CMD_DIR, RAW_DATA_DIR, FILE_FORMATS
from pathlib import Path
from ms_runner import run_ms_simulation, get_effective_population_size
from msms_runner import run_msms_simulation, run_msms_ms_simulation
from msprime_runner import run_msprime_simulation, run_mspms_simulation

SIMULATORS = {"ms", "mspms", "msms"}
SIM_FUNCTIONS = {
    "ms": run_ms_simulation,
    "mspms": run_mspms_simulation,
    "msms": run_msms_simulation,
    "msms_ms":  run_msms_ms_simulation
}


def run_all_simulations(config_path, output_dir, simulators=SIMULATORS, reps=1, rec_reps=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_run_dir = Path(output_dir)
    base_run_dir.mkdir(parents=True, exist_ok=True)

    if config.get("recombination_rate", 0) != 0 and rec_reps:
        reps = rec_reps

    total_sample_size = sum(p.get("sample_size", 0) for p in config.get("populations", [])) \
                        if "populations" in config else int(config["sample_size"])

    sample_structure = [
        p.get("sample_size", 0)
        for p in config.get("populations", [])
        if p.get("sample_size", 0) > 0
    ]

    simulator_outputs = {
        sim: FILE_FORMATS["raw"].format(SIM=sim)
        for sim in simulators
    }
    metadata = {
                "simulators": simulators,
                "sequence_length": config.get("sequence_length"),
                "population_size": get_effective_population_size(config),
                "num_replicates": config.get("num_replicates"),
                "overall_samples": total_sample_size,
                "sample_structure": sample_structure,
                "mutation_rate": config.get("mutation_rate", None),
                "recombination_rate": config.get("recombination_rate", None),
                "config": str(Path(config_path).name),
                "reps": reps,
                "simulator_outputs": simulator_outputs
            }
    with open(base_run_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, sort_keys=False)

    for i in range(reps):
        rep_dir = base_run_dir if reps == 1 else base_run_dir / f"rep{i+1}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        for sim in simulators:
            if sim not in SIM_FUNCTIONS:
                print(f"⚠️ Simulator {sim} unknown, skipping")
                continue
            output_file = rep_dir / FILE_FORMATS["raw"].format(SIM=sim)
            SIM_FUNCTIONS[sim](config=config, output_file=output_file)

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
    nargs="*",
    type=str,
    help="None, one or more config filenames to run (default: all *.yaml in config-dir)",
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
    choices=["ms", "msms", "mspms", "all"],
    default="all",
    help="Which simulator(s) to run (default: all)",
)
parser.add_argument(
    "--reps",
    type=int,
    default=1,
    help="Number of repetitions (default: 1)"
)
parser.add_argument(
    "--rec_reps",
    type=int,
    help="Number of repetitions to perform specifically for configurations that include recombination"
)

args = parser.parse_args()
simulators = SIMULATORS if args.simulator=="all" else [args.simulator]
outdir = args.outdir

if args.configs:
    config_dir = args.config_dir

    if not config_dir.exists():
        sys.exit(f"Error: Config directory not found: {config_dir}")

    config_files = [config_dir / name for name in args.configs]

    if not config_files:
        sys.exit(f"No config files found in {config_dir}")

    for config_path in config_files:
        print(config_path)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        config_outdir = outdir / config_path.stem
        config_outdir.mkdir(parents=True, exist_ok=True)

        print(f"Running simulations for {config_path.stem} -> {config_outdir}")
        run_all_simulations(
            config_path=str(config_path),
            output_dir=config_outdir,
            simulators=simulators,
            reps=args.reps,
            rec_reps=args.rec_reps
        )
else:
    sys.exit("❌ Please provide either --configs or --cmd_variables")




