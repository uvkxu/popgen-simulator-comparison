import demes
import msprime
import argparse
import sys
import yaml
import os
import itertools
import subprocess
import math
from pathlib import Path
from msprime import BinaryMutationModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from snakemake_vars import CONFIG_DIR, CMD_DIR, RAW_DATA_DIR, SIMULATORS, FILE_FORMATS, DEFAULT_MS_PATH, DEFAULT_MSMS_PATH

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
            vals = [v.strip() for v in values.split(",")]
            def convert(v):
                try:
                    return int(v) if "." not in v else float(v)
                except ValueError:
                    return v
            variables[key] = [convert(v) for v in vals]
    return variables

def get_sample_sizes(config, sample_size=None):
    samples = []
    for pop in config.get("demes"):
        epochs_time = {epoch.get("end_time", 0) for epoch in pop.get("epochs")}
        if any(t == 0 for t in epochs_time):
            s = sample_size if sample_size else pop.get("sample_size")
            if s is None:
                raise ValueError(f"Population '{pop.get('name')}' has no sample_size defined.")
            samples.append(s)
        else:
            samples.append(0)
    return samples

def run_command_demes(graph, sim, output_file, samples, nreps, sim_path=None, precision=None):
    """Run a single simulator and write its raw output to a .txt file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ms/msms/mspms
    if sim in ("ms", "msms", "mspms"):
        cmd = demes.to_ms(graph, sequence_length=graph.sequence_length,
                          samples=samples, N0=graph.population_size)

        if sim in ("ms", "msms"):
            if not sim_path or not os.path.exists(sim_path):
                raise FileNotFoundError(f"{sim} executable not found at {sim_path}.")
            prefix = sim_path
        else:
            prefix = "mspms"

        cmd = f"{prefix} {sum(samples)} {nreps} {cmd} -T"

        if precision:
            if sim in ("ms", "mspms"):
                cmd += f" -p {precision}"
            elif sim == "msms":
                format_str = "0" + (("." + "0"*precision))
                cmd += f' -oFP {format_str}'

        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{sim} failed:\n{result.stderr}")

        with open(output_file, "w") as f:
            f.write(result.stdout)

        # Return relative path from combo folder
        return str(output_file.relative_to(output_file.parents[1]))

    # msprime
    elif sim == "msprime":
        msprime_samples = []
        for i, deme in enumerate(graph.demes):
            if samples[i] > 0:
                msprime_samples.append(
                    msprime.SampleSet(num_samples=samples[i], population=deme.name, time=0, ploidy=1)
                )

        demography = msprime.Demography.from_demes(graph)
        params = {"samples": msprime_samples, "demography": demography, "num_replicates": nreps}
        if graph.recombination_rate:
            params["recombination_rate"] = graph.recombination_rate
        if graph.sequence_length:
            params["sequence_length"] = graph.sequence_length

        replicates = msprime.sim_ancestry(**params)
        lines = []

        for rep_index, ts in enumerate(replicates, start=1):
            mts = msprime.sim_mutations(ts, rate=graph.mutation_rate, model=BinaryMutationModel())
            lines.append(f"// Replicate {rep_index}")
            for tree in ts.trees():
                lines.append(tree.as_newick() + ";")
            lines.append("")
            lines.append(f"segsites: {mts.num_sites}")
            if mts.num_sites > 0:
                positions = " ".join(f"{int(site.position)}" for site in mts.sites())
                lines.append(f"positions: {positions}")
                for row in mts.genotype_matrix().T:
                    lines.append("".join(map(str, row)))
            lines.append("")

        with open(output_file, "w") as f:
            f.write("\n".join(lines))

        return str(output_file.relative_to(output_file.parents[1]))

    else:
        raise ValueError(f"Unsupported simulator: {sim}")


def run_all_simulations(config_path, output_dir, ms_path, msms_path, nreps, usePrecision, simulators=SIMULATORS, reps=1, rec_reps=None, variable_file=None, ms_mod_path = None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    def has_only_one_pop():
        demes = config.get("demes", [])
        return demes is None or len(config.get("demes", [])) == 1

    base_run_dir = Path(output_dir)
    base_run_dir.mkdir(parents=True, exist_ok=True)
    if config.get("recombination_rate", 0) != 0 and rec_reps:
        reps = rec_reps

    # Variable combinations
    if variable_file:
        vars_dict = parse_variable_file(variable_file)
        if has_only_one_pop():
            vars_dict.pop("population_sizes", None)
        keys = list(vars_dict.keys())
        combinations = itertools.product(*vars_dict.values())

        for values in combinations:
            combo = dict(zip(keys, values))
            combo_dir = base_run_dir / "_".join(f"{k}-{v}" for k, v in combo.items())
            combo_dir.mkdir(parents=True, exist_ok=True)

            

            # Kopie des Graphen ggf. überschreiben
            graph = demes.load(config_path)
            graph.sequence_length = combo["sequence_lengths"]
            if "population_sizes" in combo:
                graph.population_size = combo["population_sizes"]

            usePrecision = bool(combo.get("usePrecision", usePrecision))
            precision = None

            if usePrecision:
                precision = math.ceil(math.log10(graph.sequence_length))
                precision = precision if precision > 0 else 1

            samples = get_sample_sizes(config, combo["sample_sizes"])

            simulator_outputs = {}
            for i in range(reps):
                rep_dir = combo_dir / f"rep{i+1}"
                rep_dir.mkdir(parents=True, exist_ok=True)

                for sim in simulators:
                    output_file = rep_dir / f"{sim}_output.txt"
                    sim_path = None
                    if sim == "ms":
                        sim_path = ms_mod_path if ms_mod_path and usePrecision else ms_path
                    elif sim == "msms":
                        sim_path = msms_path

                    rel_path = run_command_demes(
                        graph=graph,
                        sim=sim,
                        output_file=output_file,
                        nreps=nreps,
                        samples=samples,
                        sim_path=sim_path,
                        precision=precision
                    )
                    simulator_outputs[sim] = str(output_file.name)

            # metadata.yaml nur einmal pro combo
            metadata = {
                "simulators": simulators,
                "sequence_length": graph.sequence_length,
                "population_size": graph.population_size,
                "num_replicates": nreps,
                "overall_samples": sum(samples),
                "sample_structure": [s for s in samples if s > 0],
                "mutation_rate": graph.mutation_rate,
                "recombination_rate": graph.recombination_rate,
                "config": str(config_path.name),
                "reps": reps,
                "usePrecision": usePrecision,
                "simulator_outputs": simulator_outputs
            }
            with open(combo_dir / "metadata.yaml", "w") as f:
                yaml.dump(metadata, f, sort_keys=False)

    else:
        # Kein variable_file
        graph = demes.load(config_path)
        samples = get_sample_sizes(config)
        simulator_outputs = {}
        for i in range(reps):
            rep_dir = base_run_dir if reps == 1 else base_run_dir / f"rep{i+1}"
            rep_dir.mkdir(parents=True, exist_ok=True)

            for sim in simulators:
                output_file = rep_dir / f"{sim}_output.txt"
                rel_path = run_command_demes(
                    graph=graph,
                    sim=sim,
                    output_file=output_file,
                    nreps=nreps,
                    samples=samples,
                    sim_path=ms_path if sim=="ms" else msms_path if sim=="msms" else None,
                    usePrecision= usePrecision
                )
                simulator_outputs[sim] = str(output_file.name)

        # metadata.yaml im base_run_dir
        metadata = {
            "simulators": simulators,
            "sequence_length": graph.sequence_length,
            "population_size": graph.population_size,
            "num_replicates": nreps,
            "overall_samples": sum(samples),
            "sample_structure": [s for s in samples if s > 0],
            "mutation_rate": graph.mutation_rate,
            "recombination_rate": graph.recombination_rate,
            "config": str(config_path.name),
            "reps": reps,
            "usePrecision": usePrecision,
            "simulator_outputs": simulator_outputs
        }
        with open(base_run_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f, sort_keys=False)

    print(f"✅ Simulations finished successfully! Results saved in: {base_run_dir}")


parser = argparse.ArgumentParser(
    description="Run simulations with one or more config files."
)

parser.add_argument(
    "--ms_path",
    type=Path,
    default=DEFAULT_MS_PATH,
    help=f"Path to ms executable (default: {DEFAULT_MS_PATH})",
)

parser.add_argument(
    "--ms_mod_path",
    type=Path,
    default=None,
    help=f"Path to ms-mod executable (default: {None})",
)

parser.add_argument(
    "--msms_path",
    type=Path,
    default=DEFAULT_MSMS_PATH,
    help=f"Path to msms executable (default: {DEFAULT_MSMS_PATH})",
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
    help="Number of times to run the simulation command (external repetitions)"
)
parser.add_argument(
    "--nreps",
    type=int,
    default=100,
    help="Number of replicates per simulation in the ms/msms command (internal nreps)"
)
parser.add_argument(
    "--rec_reps",
    type=int,
    help="Number of repetitions to perform specifically for configurations that include recombination"
)
parser.add_argument(
    "--usePrecision",
    type=bool,
    default=True,
    help="Whether to automatically adjust the number of decimal places in the output based on the sequence length"
)

args = parser.parse_args()
simulators = SIMULATORS if args.simulator=="all" else [args.simulator]
outdir = args.outdir

if args.configs:
    config_dir = args.config_dir
    variable_file = args.cmddir / args.cmd_variables

    if not config_dir.exists():
        sys.exit(f"Error: Config directory not found: {config_dir}")

    config_files = [config_dir / name for name in args.configs]

    if not config_files:
        sys.exit(f"No config files found in {config_dir}")

    for config_path in config_files:
        config_outdir = outdir / config_path.stem
        config_outdir.mkdir(parents=True, exist_ok=True)

        print(f"Running simulations for {config_path.stem} -> {config_outdir}")
        run_all_simulations(
            config_path=config_path,
            output_dir=config_outdir,
            ms_path=args.ms_path,
            msms_path=args.msms_path,
            ms_mod_path=args.ms_mod_path,
            simulators=simulators,
            reps=args.reps,
            rec_reps=args.rec_reps,
            nreps=args.nreps,
            variable_file=variable_file,
            usePrecision=args.usePrecision
        )
else:
    sys.exit("❌ Please provide --configs")