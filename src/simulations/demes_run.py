import demes
import msprime
import argparse
import sys
import yaml
import os
import subprocess
import math
from pathlib import Path
from msprime import BinaryMutationModel
import stdpopsim
import json
import re
import time
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from snakemake_vars import CONFIG_DIR, CMD_DIR, RAW_DATA_DIR, SIMULATORS, FILE_FORMATS, \
 DEFAULT_MS_PATH, DEFAULT_MSMS_PATH, DEFAULT_MS_MOD_PATH, DEFAULT_SLIM_PATH, PRECISION, \
 estimate_mem_mb
from utils.io_helpers import log_step


def estimate_msms_heap_gb(
    config,
    combo,
    safety_factor=0.9,
    min_gb=1,
    max_gb=200
):

    # Ask the global estimator
    mem_mb = estimate_mem_mb(
        config=config,
        combo=combo,
        func="simulate",
        simulator="msms"
    )

    # Convert MB → GB and apply safety margin
    heap_gb = math.floor(mem_mb / 1024 * safety_factor)

    # Enforce bounds
    heap_gb = max(min_gb, heap_gb)
    heap_gb = min(max_gb, heap_gb)

    return heap_gb


def get_sample_sizes(config, sample_size=None):
    samples = []
    for pop in config.get("demes"):
        epochs_time = {epoch.get("end_time", 0) for epoch in pop.get("epochs")}
        if any(t == 0 for t in epochs_time):
            s = sample_size if sample_size else pop.get("sample_size")
            if s is None:
                raise ValueError(f"Population '{pop.get('name')}' no sample_size defined.")
            samples.append(s)
        else:
            samples.append(0)
    return samples


def to_ms_format(ts, rep_index):
    lines = []
    lines.append(f"// Replicate {rep_index}")
    for tree in ts.trees():
        lines.append(tree.as_newick() + ";")
    lines.append("")
    lines.append(f"segsites: {ts.num_sites}")
    if ts.num_sites > 0:
        positions = " ".join(f"{int(site.position)}" for site in ts.sites())
        lines.append(f"positions: {positions}")
        for row in ts.genotype_matrix().T:
            lines.append("".join(map(str, row)))
    lines.append("")

    return lines
    

def run_slim_stdpopsim(graph, model_id, species_name, samples, nreps, output_file, slim_path, scaling_factor=1):
    if species_name and model_id:
        species = stdpopsim.get_species(species_name)
        model = species.get_demographic_model(model_id)
    else:
        model = stdpopsim.DemographicModel(
            id="custom_demes_model",
            description="Custom demes model",
            long_description="A custom demes model loaded from a demes graph.",
            model=msprime.Demography.from_demes(graph),
        )

    if graph.recombination_rate:
        contig = stdpopsim.Contig.basic_contig(
                length=graph.sequence_length,
                mutation_rate=graph.mutation_rate,
                recombination_rate=graph.recombination_rate,
                ploidy=1
            )
    else:
        contig = stdpopsim.Contig.basic_contig(
                length=graph.sequence_length,
                mutation_rate=graph.mutation_rate,
                ploidy=1
            )

    msprime_samples = []

    for i, deme in enumerate(graph.demes):
        if samples[i] > 0:
            msprime_samples.append(
                msprime.SampleSet(num_samples=samples[i], population=deme.name, time=0, ploidy=1)
            )

    engine = stdpopsim.get_engine("slim")
    lines = []

    for rep_index in range(1, nreps + 1):
        start_time = time.time()
        ts = engine.simulate(
            demographic_model=model,
            contig=contig,
            samples=msprime_samples,
            slim_path=slim_path,
            slim_scaling_factor=scaling_factor
        ) 
        end_time = time.time()
        elapsed = end_time - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60

        lines.extend([f"# SLiM simulation time: {hours}h {minutes}m {seconds:.2f}s"])
        lines.extend(to_ms_format(ts, rep_index))
    
    with open(output_file, "w") as f:
            f.write("\n".join(lines))


def run_command_demes(config_path, combo, sim, output_file, nreps, sim_path=None, log_file=None):
    """Run a single simulator and write its raw output to a .txt file."""

    if output_file.exists():
        print(f"Output file already exists, skipping: {output_file}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    graph = demes.load(config_path)
    graph.sequence_length = combo["sequence_lengths"]
    if "population_sizes" in combo:
        graph.population_size = combo["population_sizes"]

    precision = None
    usePrecision = bool(combo["usePrecision"])
    if usePrecision:
        precision = PRECISION

    recombination_rate = float(combo.get("recombination_rates", 0))
    if recombination_rate > 0:
        graph.recombination_rate = recombination_rate

    samples = get_sample_sizes(config, combo["sample_sizes"])

    if sim in ("ms", "msms", "mspms"):
        cmd = demes.to_ms(graph, sequence_length=graph.sequence_length,
                          samples=samples, N0=graph.population_size)

        if sim == "ms":
            if not sim_path or not os.path.exists(sim_path):
                raise FileNotFoundError(f"{sim} executable not found at {sim_path}.")
            prefix = sim_path

        elif sim == "msms":
            if not sim_path or not os.path.exists(sim_path):
                raise FileNotFoundError(f"{sim} executable not found at {sim_path}.")
            heap_gb = estimate_msms_heap_gb(config=config_path, combo=combo)
            prefix = f"java -Xmx{heap_gb}G -jar {sim_path}"
    
        else:
            prefix = "mspms"

        cmd = f"{prefix} {sum(samples)} {nreps} {cmd} -T"
        # print(cmd)

        if precision:
            if sim in ("ms", "mspms"):
                cmd += f" -p {precision}"
            elif sim == "msms":
                format_str = "0" + (("." + "0"*precision))
                cmd += f' -oFP {format_str}'

        start_time = time.time()

        result = subprocess.run(cmd.split(), capture_output=True, text=True)

        end_time = time.time()
        elapsed = end_time - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60

        with open(output_file, "w") as f:
            f.write(f"# Simulation time: {hours}h {minutes}m {seconds:.2f}s\n")
            f.write(result.stdout)

        if result.returncode != 0 or result.stderr:
            if log_file:
                err_info = result.stderr.strip()
                log_step(log_file, "simulation", json.dumps(combo), sim, 0, 1, info=err_info)
            else:
                raise RuntimeError(f"Simulation with {sim} failed: {result.stderr}")
        elif log_file:
            log_step(log_file, "simulation", json.dumps(combo), sim, 1, 0)

    # msprime
    elif sim == "msprime":
        msprime_samples = []
        for i, deme in enumerate(graph.demes):
            if samples[i] > 0:
                msprime_samples.append(
                    msprime.SampleSet(num_samples=samples[i], population=deme.name, time=0, ploidy=1)
                )

        demography = msprime.Demography.from_demes(graph)
        params = {
            "samples": msprime_samples, 
            "demography": demography, 
            "num_replicates": nreps}
        if graph.recombination_rate:
            params["recombination_rate"] = graph.recombination_rate
        if graph.sequence_length:
            params["sequence_length"] = graph.sequence_length

        try:
            start_time = time.time()
            replicates = msprime.sim_ancestry(**params)
            lines = []

            for rep_index, ts in enumerate(replicates, start=1):
                mts = msprime.sim_mutations(ts, rate=graph.mutation_rate, model=BinaryMutationModel())
                lines.extend(to_ms_format(mts, rep_index))
            
            end_time = time.time()
            elapsed = end_time - start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = elapsed % 60

            with open(output_file, "w") as f:
                f.write(f"# MSprime simulation time: {hours}h {minutes}m {seconds:.2f}s\n")
                f.write("\n".join(lines))

            # Log success
            if log_file:
                log_step(log_file, "simulation", json.dumps(combo), sim, success_inc=1, failed_inc=0)

        except Exception as e:
            # Log failure
            if log_file:
                log_step(log_file, "simulation", json.dumps(combo), sim, success_inc=0, failed_inc=1, 
                        info=str(e))
            raise


    else:
        raise ValueError(f"Unsupported simulator: {sim}")


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
    default=DEFAULT_MS_MOD_PATH,
    help=f"Path to ms-mod executable (default: {None})",
)

parser.add_argument(
    "--msms_path",
    type=Path,
    default=DEFAULT_MSMS_PATH,
    help=f"Path to msms executable (default: {DEFAULT_MSMS_PATH})",
)

parser.add_argument(
    "--slim_path",
    type=Path,
    default=DEFAULT_SLIM_PATH,
    help=f"Path to slim executable (default: {DEFAULT_SLIM_PATH})"
)

parser.add_argument(
    "--output_file",
    type=Path,
    help="Base output directory (default: data/raw/)",
    required=True
)
parser.add_argument(
    "--simulator",
    type=str,
    choices=SIMULATORS,
    help="Which simulator(s) to run (default: all)",
    required=True
)
parser.add_argument(
    "--nreps",
    type=int,
    default=2,
    help="Number of replicates per simulation in the ms/msms command (internal nreps)"
)
parser.add_argument(
    "--config_path",
    type=Path,
    required=True
)

parser.add_argument(
    "--combo",
    type=str,
    required=True
)

parser.add_argument(
    "--log_file",
    type=Path,
    default=None,
    help="Path to master log file to update (default: None)"
)

parser.add_argument(
    "--copy_file",
    type=Path,
    default=None,
    help="If raw file is just wished to be copied, put it here"
)

args = parser.parse_args()

sim = args.simulator
combo = json.loads(args.combo)
output_file = args.output_file
output_file.parent.mkdir(parents=True, exist_ok=True)
config_path = args.config_path
nreps = args.nreps
log_file = args.log_file
copy_file = args.copy_file

if copy_file is not None and copy_file.exists():
    shutil.copy(copy_file, output_file)
    print(f"Copied existing file {copy_file} -> {output_file}")
    sys.exit(0)  

ms_path = args.ms_path
ms_mod_path = args.ms_mod_path
msms_path = args.msms_path
slim_path = args.slim_path

sim_path = None

if sim=="ms":
    if bool(combo["usePrecision"]):
        sim_path = ms_mod_path
    else:
        sim_path = ms_path
elif sim=="msms":
    sim_path = msms_path
elif sim=="slim" or sim.startswith("slim"):
    sim_path = slim_path

run_command_demes(
    config_path=config_path, 
    combo=combo, 
    sim=sim, 
    output_file=output_file, 
    nreps=nreps, 
    sim_path=sim_path,
    log_file=log_file
)

