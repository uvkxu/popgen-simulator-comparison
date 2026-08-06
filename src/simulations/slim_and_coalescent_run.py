import msprime
import pyslim
import os
import argparse
import json

import msprime
import pyslim
import tskit
import subprocess
import os
from pathlib import Path
import numpy as np
import sys
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from snakemake_vars import DEFAULT_SLIM_PATH

def run_slim_and_coalescent(
    combo,
    output_file,
    nreps,
    config_path,
    SLIM_PATH,
    coalescent_samples=1000,
    slim_generations=100,
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------

    if "population_sizes" in combo:
        N = combo["population_sizes"]
    else:
        N = config["population_size"]

    L = combo["sequence_lengths"]

    if "mutation_rate" in config:
        mu = combo["mutation_rate"]
    else:
        mu = 0

    workdir = output_file.parent

    with open(output_file, "w") as ms_out:

        for rep in range(nreps):

            init_file = workdir / f"neutral_init_{rep}.trees"
            tree_file = workdir / f"neutral_output_{rep}.trees"
            slim_file = workdir / f"reload_{rep}.slim"

            # --------------------------------------------------
            # msprime burn-in
            # --------------------------------------------------

            demography = msprime.Demography()
            demography.add_population(initial_size=N)

            ts = msprime.sim_ancestry(
                samples=coalescent_samples,
                demography=demography,
                sequence_length=L,
                recombination_rate=0.0
            )

            ts = pyslim.annotate(
                ts,
                model_type="WF",
                tick=1,
                stage="late",
            )

            ts = msprime.sim_mutations(
                ts,
                rate=mu,
                model=msprime.SLiMMutationModel(type=1)
            )

            ts.dump(init_file)

            # --------------------------------------------------
            # create SLiM script
            # --------------------------------------------------

            slim_script = f"""
                initialize()
                {{
                    initializeSLiMModelType("WF");
                    initializeTreeSeq(timeUnit="generations");

                    initializeMutationRate({mu});

                    initializeMutationType("m1", 0.5, "f", 0.0);

                    initializeGenomicElementType("g1", m1, 1.0);
                    initializeGenomicElement(g1, 0, L-1);

                    initializeRecombinationRate(0.0);
                }}

                1 late()
                {{
                    sim.readFromPopulationFile("{init_file}");
                }}

                {slim_generations} late()
                {{
                    sim.treeSeqOutput("{tree_file}");
                    sim.simulationFinished();
                }}
                """

            slim_file.write_text(slim_script)

            # --------------------------------------------------
            # run SLiM
            # --------------------------------------------------

            subprocess.run(
                [
                    str(SLIM_PATH),
                    "-s",
                    str(1000 + rep),
                    "-d",
                    f"L={L}",
                    str(slim_file),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # --------------------------------------------------
            # load output tree sequence
            # --------------------------------------------------

            ts = tskit.load(tree_file)

            # --------------------------------------------------
            # sample 16 haplotypes
            # --------------------------------------------------

            rng = np.random.default_rng()

            sample_nodes = ts.samples()

            chosen_nodes = rng.choice(
                sample_nodes,
                size=16,
                replace=False,
            )

            ts = ts.simplify(samples=chosen_nodes)

            # --------------------------------------------------
            # write ms replicate
            # --------------------------------------------------

            ms_out.write("//\n")

            for tree in ts.trees():
                ms_out.write(tree.as_newick() + ";\n")

            ms_out.write("\n")

            ms_out.write(f"segsites: {ts.num_sites}\n")

            if ts.num_sites > 0:

                positions = " ".join(
                    str(int(site.position))
                    for site in ts.sites()
                )

                ms_out.write(f"positions: {positions}\n")

                G = ts.genotype_matrix()

                for hap in G.T:
                    seq = "".join(
                        "1" if x > 0 else "0"
                        for x in hap
                    )
                    ms_out.write(seq + "\n")

            ms_out.write("\n")

            # --------------------------------------------------
            # cleanup
            # --------------------------------------------------

            init_file.unlink(missing_ok=True)
            tree_file.unlink(missing_ok=True)
            slim_file.unlink(missing_ok=True)


parser = argparse.ArgumentParser(
    description="Run simulations with one or more config files."
)

parser.add_argument(
    "--nreps",
    type=int,
    default=2,
    help="Number of replicates per simulation in the ms/msms command (internal nreps)"
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
    "--combo",
    type=str,
    required=True
)

parser.add_argument(
    "--config_path",
    type=Path,
    required=True
)

parser.add_argument(
    "--coalescent_samples",
    type=int,
    default=1000,
    help="Number of haplotypes to sample for coalescent analysis"
)

parser.add_argument(
    "--slim_generations",
    type=int,
    default=100,
    help="Number of generations to run SLiM for"
)

args = parser.parse_args()

combo = json.loads(args.combo)
output_file = args.output_file
output_file.parent.mkdir(parents=True, exist_ok=True)
config_path = args.config_path
nreps = args.nreps
SLIM_PATH = args.slim_path
coalescent_samples = args.coalescent_samples
slim_generations = args.slim_generations

if "one_population" not in config_path.name:
    raise ValueError(f"This script can only run the 'one_population' config, but got {config}.")

run_slim_and_coalescent(
    combo=combo,
    output_file=output_file,
    nreps=nreps,
    config_path=config_path,
    SLIM_PATH=SLIM_PATH,
    coalescent_samples=coalescent_samples,
    slim_generations=slim_generations
)