import os
import msprime

def create_msprime_demography(config):
    dem = msprime.Demography()

    if "populations" in config:
        for i, pop in enumerate(config.get("populations", [])):
            pop_id = pop["id"]
            assert i == pop_id
            name = pop.get("name", f"pop{pop_id}")

            dem.add_population(
                name=name,
                initial_size=pop["initial_size"],
                growth_rate=pop.get("growth_rate", 0.0)
            )

        if "migration_matrix" in config:
            for i, row in enumerate(config["migration_matrix"]):
                for j, rate in enumerate(row):
                    if i != j:
                        dem.set_migration_rate(i, j, rate)
        elif "migration_rates" in config:
            for mig in config["migration_rates"]:
                dem.set_migration_rate(
                    int(mig["source"]),
                    int(mig["dest"]),
                    mig["rate"]
                )

    for event in config.get("demography_events", []):
        t = event["time"]
        typ = event["type"]
        if typ == "size_change":
            pops = event["population"]
            if pops == "all":
                dem.add_population_parameters_change(time=t, initial_size=event["size"])
            else:
                dem.add_population_parameters_change(time=t, initial_size=event["size"], population=pops)
        elif typ == "growth_rate_change":
            if event["population"] == "all":
                dem.add_population_parameters_change(time=t, growth_rate=event["rate"])
            else:
                dem.add_population_parameters_change(time=t, growth_rate=event["rate"], population=event["population"])
        elif typ == "migration_rate_change":
            dem.add_migration_rate_change(
                source=event["source"],
                dest=event["dest"],
                rate=event["rate"],
                time=event["time"]
            )

    return dem

def run_msprime_simulation(config):
    demography = create_msprime_demography(config)

    samples = []
    for pop in config.get("populations", []):
        samples.extend([msprime.SampleSet(num_samples=pop["sample_size"], population=pop["id"], time=0)])

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
