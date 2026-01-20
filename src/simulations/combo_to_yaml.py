import json
from pathlib import Path
import argparse
import yaml

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

parser = argparse.ArgumentParser(
    description=(
        "Create a metadata YAML file for population genetic simulations "
        "based on a demographic configuration and a parameter combination."
    )
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
    "--output_file",
    type=Path,
    help="Base output directory (default: data/raw/)",
    required=True
)

parser.add_argument(
    "--nreps",
    type=int,
    default=2,
    help="Number of replicates per simulation in the ms/msms command (internal nreps)"
)

args = parser.parse_args()
combo = json.loads(args.combo)
output_file = args.output_file
config_path = args.config_path
nreps = args.nreps

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

samples = get_sample_sizes(config, combo["sample_sizes"])

population_size = combo.get("population_sizes", None)

metadata = {
    "sequence_length": combo["sequence_lengths"],
    "population_size": population_size,
    "num_replicates": nreps,
    "overall_samples": sum(samples),
    "sample_structure": [s for s in samples if s > 0],
    "mutation_rate": config["mutation_rate"],
    "recombination_rate": float(combo.get("recombination_rates", 0)),
    "config": str(config_path.name),
    "usePrecision": bool(combo["usePrecision"])
}

output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w") as f:
    yaml.dump(metadata, f, sort_keys=False)