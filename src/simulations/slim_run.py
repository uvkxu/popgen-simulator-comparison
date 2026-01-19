import stdpopsim
import demes
print(demes.__file__)
print(demes.__version__)
import msprime
from pathlib import Path
import yaml
from stdpopsim.slim_engine import _SLiMEngine
import os
import numpy as np

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


def to_ms_HOT(recombination_map, N0):
    sequence_length = recombination_map.sequence_length
    print(sequence_length)
    mean_rate = recombination_map.mean_rate

    left = recombination_map.left
    right = recombination_map.right
    rate = recombination_map.rate

    spans = list(zip(left, right, rate))

    rho = 4 * N0 * mean_rate * (sequence_length - 1)

    cmd = f" -r {rho:.10f} {int(sequence_length)}"
    cmd += f" -v {len(spans)}"

    for le, ri, r in spans:
        if np.isnan(r):
            scale = 0
        else:
            scale = r / mean_rate
        cmd += f" {int(le+1)} {int(ri)} {scale:.10f}"

    return cmd


all_configs = {
        "HomSap": [ "AmericanAdmixture_4B18", "OutOfAfrica_3G09", "PapuansOutOfAfrica_10J19", 
                    "AncientEurasia_9K19", "Zigzag_1S14",
                    # "OutOfAfricaExtendedNeandertalAdmixturePulse_3I21"
                    ],
        "DroMel": ["OutOfAfrica_2L06"],
        "CanFam": ["EarlyWolfAdmixture_6F14"],
        "AnaPla": ["MallardBlackDuck_2L19"],
        "PonAbe": ["TwoSpecies_2L11"],
        "GorGor": ["GorillaGhost_5P23"] }

model_to_filename = {
    "AmericanAdmixture_4B18": "browning_america.yaml",
    "OutOfAfrica_3G09": "gutenkunst_ooa.yaml",
    "PapuansOutOfAfrica_10J19": "jacobs_papuans.yaml",
    "AncientEurasia_9K19": "kamm_westerneurasia.yaml",
    "OutOfAfrica_2L06": "li_african3epoch.yaml",
    "Zigzag_1S14": "zigzag.yaml",
    # "OutOfAfricaExtendedNeandertalAdmixturePulse_3I21": "iasi_ooa.yaml",
    "EarlyWolfAdmixture_6F14": "freedman_wolf.yaml",
    "MallardBlackDuck_2L19": "lavretsky_ducks.yaml",
    "TwoSpecies_2L11": "locke_orangutan.yaml",
    "GorillaGhost_5P23": "pawar_gorilla.yaml"
}

species = stdpopsim.get_species("AnaPla")
# gm = species.get_genetic_map("HapMapII_GRCh38")
# chrom = species.genome.get_chromosome("chr21")

model = species.get_demographic_model("MallardBlackDuck_2L19")

config_path = "/home/kuchvict/master-thesis/stdpopsim-config/li_african3epoch.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

graph = demes.load(config_path)
graph.population_size = 71560
samples = get_sample_sizes(config, 4)
msprime_samples = []
nreps = 5
# contig = species.get_contig("chr21", genetic_map="HapMapII_GRCh37")
# recombination_map = contig.recombination_map

recombination_map = msprime.RateMap(
    position=[0, 4000, 9000, 11000, 20000, 30000],
    rate=[0, 1e-8, 2.3e-10, 8.5e-9, 3.7e-8],
)

graph.sequence_length = recombination_map.sequence_length

for i, deme in enumerate(graph.demes):
    if samples[i] > 0:
        msprime_samples.append(
            msprime.SampleSet(num_samples=samples[i], population=deme.name, time=0, ploidy=1)
        )

commands_dir = Path("/home/kuchvict/master-thesis/commands")
cmd = demes.to_ms(graph, sequence_length=graph.sequence_length,
                          samples=samples, N0=graph.population_size)
cmd += to_ms_HOT(recombination_map, graph.population_size)

file = Path(commands_dir / "ms_cmd.txt")
with open(file, "w") as f:
    f.write(cmd)

final_cmd = f"ms {sum(samples)} {nreps} -f {file}"




'''
engine = stdpopsim.get_engine("msprime")
ts = engine.simulate(
    demographic_model=model,
    contig=contig,
    samples=msprime_samples,
    # slim_path=Path("./bin/slim"),
)

print(f"Number of trees: {ts.num_trees}")
'''

replicate = msprime.sim_ancestry(
    samples=msprime_samples,
    demography=msprime.Demography.from_demes(graph),
    recombination_rate=recombination_map,
    num_replicates=1
)
num = 0

for ts in replicate:
    num += 1

print(f"Number of trees: {num}")


'''
dirname = "stdpopsim-config"
os.makedirs(dirname, exist_ok=True)

for species_name, model_ids in all_configs.items():
    species = stdpopsim.get_species(species_name)  

    for m_id in model_ids:
        model = species.get_demographic_model(m_id)
        filename = model_to_filename[m_id]
        path = os.path.join(dirname, filename)

        graph = model.model.to_demes()
        if not os.path.exists(path):
            #TODO: write the id and the species into that :D
            demes.dump(
                graph=graph,
                filename=path,
                format='yaml',
                simplified=True
            )
        else:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
                #TODO: write model_id (here: m_id) and species_name into config
            
            # Ensure metadata exists and is an object
            if "metadata" not in config or config["metadata"] is None:
                config["metadata"] = {}

            # Write values into metadata
            config["metadata"]["model_id"] = m_id
            config["metadata"]["species"] = species_name

            # Save updated YAML back to disk
            with open(path, "w") as f:
                yaml.safe_dump(config, f, sort_keys=False)
        
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        contig = stdpopsim.Contig.basic_contig(
            length=1e6,
            mutation_rate=config.get("mutation_rate"),
            recombination_rate=0,
            ploidy=1
        )

        samples = get_sample_sizes(config, 4)
        msprime_samples = []
        for i, deme in enumerate(graph.demes):
            if samples[i] > 0:
                msprime_samples.append(
                    msprime.SampleSet(num_samples=samples[i], population=deme.name, time=0, ploidy=1)
                )

        engine = stdpopsim.get_engine("slim")
        ts = engine.simulate(
            demographic_model=model,
            contig=contig,
            samples=msprime_samples,
            slim_path=Path("./bin/slim"),
        )
        
        print(f"Number of trees: {ts.num_trees}")

        for tree in ts.trees():
            print(tree.as_newick())
        
        print(ts.genotype_matrix().T)
'''

# Bigggg TODO: see if there are no severe differences in my own specification and the ones directly given by stdpopsim
# What to mention in master thesis: slim is a bitch and doesnt like when there are ancestral thinggs, so pops that doesnt exist at time zero