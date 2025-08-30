import os
import msprime
from checker import initial_population_activity

def create_msprime_demography(config):
    dem = msprime.Demography()

    if "populations" in config:
        for i, pop in enumerate(config.get("populations", [])):
            name = pop.get("name")

            dem.add_population(
                name=name,
                initial_size=pop["initial_size"],
                growth_rate=pop.get("growth_rate", 0.0)
            )

        if "migration_rates" in config:
            for mig in config["migration_rates"]:
                dem.set_migration_rate(
                    source=mig["source"],
                    dest=mig["dest"],
                    rate=mig["rate"]
                )

        

    for event in config.get("demography_events", []):
        t = event["time"]
        type = event["type"]
        if type == "size_change":
            pops = event["population"]
            #TODO: Checke ob die Größe aller Populations oder nur die Größe aller inaktiven Populationen geändert wird
            if pops == "all":
                dem.add_population_parameters_change(time=t, initial_size=event["size"])
            else:
                dem.add_population_parameters_change(time=t, initial_size=event["size"], population=pops)
        elif type == "growth_rate_change":
            if event["population"] == "all":
                #TODO: Checke ob die Größe aller Populations oder nur die Größe aller inaktiven Populationen geändert wird
                dem.add_population_parameters_change(time=t, growth_rate=event["rate"])
            else:
                dem.add_population_parameters_change(time=t, growth_rate=event["rate"], population=event["population"])
        elif type == "migration_rate_change":
            dem.add_migration_rate_change(time=t,
                source=event["source"],
                dest=event["dest"],
                rate=event["rate"]
            )
        elif type == "population_split":
            dem.add_population_split(time=t, derived=event["derived"], ancestral=event["ancestral"])
        elif type == "population_admixture":
            #TODO: Problem - ancestral populations sind zu Beginn nicht inaktiv und migrieren schon bei t=0
            dem.add_admixture(time=t, derived=event["derived"], ancestral=event["ancestral"], proportions=event["proportions"])

    print(dem.debug())

    return dem

def run_msprime_simulation(config):
    demography = create_msprime_demography(config)

    samples = []
    for pop in config.get("populations", []):
        if "sample_size" in pop:
            samples.append(
                msprime.SampleSet(
                    num_samples=pop["sample_size"],
                    population=pop["name"],
                    time=0
                )
            )

    ts = msprime.sim_ancestry(
        samples=samples,
        recombination_rate=config["recombination_rate"],
        sequence_length=config["sequence_length"],
        demography=demography,
        ploidy=1,
        random_seed=42
    )
    mts = msprime.sim_mutations(ts, rate=config["mutation_rate"], random_seed=42)

    output_file = config.get("msprime_output_file", "results/msprime/output.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        f.write(f"//\nsegsites: {ts.num_sites}\n")
        if ts.num_sites > 0:
            positions = " ".join(f"{site.position / ts.sequence_length:.6f}" for site in ts.sites())
            f.write(f"positions: {positions}\n")
            gt_matrix = ts.genotype_matrix().T
            for row in gt_matrix:
                f.write("".join(map(str, row)) + "\n")
