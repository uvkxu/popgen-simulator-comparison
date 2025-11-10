import argparse
import sys
import yaml
import os
import itertools

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from snakemake_vars import CONFIG_DIR, CMD_DIR, RAW_DATA_DIR, SIMULATORS, FILE_FORMATS
from pathlib import Path
from ms_runner import run_ms_simulation
from msms_runner import run_msms_simulation, run_msms_ms_simulation
from msprime_runner import run_msprime_simulation, run_mspms_simulation

SIM_FUNCTIONS = {
    "ms": run_ms_simulation,
    "msprime": run_mspms_simulation,
    "msms": run_msms_simulation,
    "msms_ms":  run_msms_ms_simulation
}

def parse_variable_file(filepath):
    """Reads the variable file and returns a dict of parameter lists."""
    variables = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, values = line.split(":")
            key = key.strip()
            # Split by comma and try to convert to int/float if possible
            vals = [v.strip() for v in values.split(",")]
            def convert(v):
                try:
                    return int(v) if "." not in v else float(v)
                except ValueError:
                    return v
            variables[key] = [convert(v) for v in vals]
    return variables


def build_cmd_from_variables(var_combo, use_recombination=True):
    """
    Builds an ms/msms/msprime command string from one variable combination.

    Expected keys in var_combo:
    - sample_sizes
    - num_replicates
    - rates
    - sequence_lengths
    """
    # Get required parameters, with fallbacks
    sample_size = var_combo.get("sample_sizes")
    num_reps = var_combo.get("num_replicates", 1)
    rate = var_combo.get("rates")
    seq_len = var_combo.get("sequence_lengths")

    # Base command
    cmd = f"{sample_size} {num_reps} -t {rate} -T"

    # Add recombination part
    if use_recombination:
        cmd += f" -r {rate} {seq_len}"

    return cmd


def run_variable_combinations(variable_file, output_dir, simulators=SIMULATORS):
    """Reads all variable combinations and runs simulations for each of them.
    For each parameter set, runs both a recombining and a non-recombining version.
    """
    vars_dict = parse_variable_file(variable_file)
    keys = list(vars_dict.keys())
    combinations = list(itertools.product(*vars_dict.values()))

    print(f"🧩 Found {len(combinations)} combinations in {variable_file}")
    print(f"Simulators: {simulators}")

    # Ensure simulator is always a list
    if isinstance(simulators, str):
        simulators = [simulators]

    for combo in combinations:
        combo_dict = dict(zip(keys, combo))

        # Two variants: with and without recombination
        for use_recombination in [True, False]:
            cmd = build_cmd_from_variables(combo_dict, use_recombination=use_recombination)
            suffix = "with_rec" if use_recombination else "no_rec"
            combo_str = "_".join(f"{k}-{v}" for k, v in combo_dict.items()) + f"_{suffix}"

            combo_dir = Path(output_dir) / combo_str
            combo_dir.mkdir(parents=True, exist_ok=True)

            for sim in simulators:
                output_file = combo_dir / f"{sim}.txt"
                
                if output_file.exists():
                    print(f"⏭️  Skipping {sim} ({suffix}) — already exists: {output_file}")
                    continue

                print(f"🚀 Running {sim} with {combo_dict} ({suffix})")
                SIM_FUNCTIONS[sim](command=cmd, output_file=str(output_file))


def run_all_simulations(config_path, config, output_dir, simulators=SIMULATORS, reps=1, rec_reps=None):
    base_run_dir = Path(output_dir)
    base_run_dir.mkdir(parents=True, exist_ok=True)

    num_reps = rec_reps if rec_reps and config.get("recombination_rate", 0) != 0 else reps

    for i in range(num_reps):
        rep_dir = base_run_dir if num_reps == 1 else base_run_dir / f"rep{i+1}"
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
    "--cmddir",
    type=Path,
    default=CMD_DIR
)
parser.add_argument(
    "--configs",
    nargs="*",
    type=str,
    help="None, one or more config filenames to run (default: all *.yaml in config-dir)",
)
parser.add_argument(
    "--cmd_variables",
    type=str,
    help="Path to a file containing command variables"
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
            config=cfg,
            output_dir=config_outdir,
            simulators=simulators,
            reps=args.reps,
            rec_reps=args.rec_reps
        )
elif args.cmd_variables:
    variables_file = args.cmddir / args.cmd_variables 
    if not variables_file.exists():
        sys.exit(f"❌ Variable file not found: {variables_file}")

    run_variable_combinations(
        variable_file=variables_file,
        output_dir=outdir,
        simulators=simulators
    )
else:
    sys.exit("❌ Please provide either --configs or --cmd_variables")




