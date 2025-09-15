import os
import msprime
from ms_runner import get_initial_population_activity

def update_activity(activity, derived=None, ancestral=None):
    derived = derived if isinstance(derived, list) else [derived]
    ancestral = ancestral if isinstance(ancestral, list) else [ancestral]

    if ancestral:
        for a in ancestral:
            activity[a] = True
    if derived:
        for d in derived:
            activity[d] = False
    
    return activity


def set_initial_migration_rates(demography, config, activity):
    explicit_migrations = set()
    populations = config.get("populations", [])
    
    for mig in config.get("migration_rates", []):
        demography.set_migration_rate(source=mig["source"], dest=mig["dest"], rate=mig["rate"])
        explicit_migrations.add((mig["dest"], mig["source"]))

    default_rate = config.get("default_migration_rate", 0.0)
    if default_rate != 0:
        for dest in (p["name"] for p in populations):
            for src in (p["name"] for p in populations):
                if dest != src and (dest, src) not in explicit_migrations:
                    demography.set_migration_rate(source=src, dest=dest, rate=default_rate)



def handle_size_change(demography, event, activity, populations):
    time = event["time"]
    if event.get("population") in (None, "all"):
        for pop in (p["name"] for p in populations):
            if activity[pop]:
                demography.add_population_parameters_change(time=time, initial_size=event["size"], population=pop)
    else:
        demography.add_population_parameters_change(
            time=time, initial_size=event["size"], population=event["population"]
        )


def handle_growth_rate_change(demography, event, activity, populations):
    time = event["time"]
    if event.get("population") in (None, "all"):
        for pop in (p["name"] for p in populations):
            if activity[pop]:
                demography.add_population_parameters_change(time=time, growth_rate=event["rate"], population=pop)
    else:
        demography.add_population_parameters_change(
            time=time, growth_rate=event["rate"], population=event["population"]
        )


def handle_migration_rate_change(demography, event, activity, populations):
    time = event["time"]
    if event.get("population") in (None, "all"):
        for dst in (p["name"] for p in populations if activity[p["name"]]):
            for src in (p["name"] for p in populations if activity[p["name"]]):
                if dst != src:
                    demography.set_migration_rate(source=src, dest=dst, rate=event["rate"])
    else:
        demography.add_migration_rate_change(
            time=time, source=event["source"], dest=event["dest"], rate=event["rate"]
        )


def set_ancestral_growth_rate_at_split(demography, config, time, population):
    default_growth_rate = config.get("default_growth_rate", 0)
    populations = config.get("populations", [])

    growth_rate = [pop.get("growth_rate", default_growth_rate) for pop in populations if pop["name"] == population][0]

    if growth_rate != 0:
        demography.add_population_parameters_change(time=time, growth_rate=growth_rate, population=population)


def handle_population_split(demography, event, activity, config):
    demography.add_population_split(
        time=event["time"], derived=event["derived"], ancestral=event["ancestral"]
    )
    set_ancestral_growth_rate_at_split(demography, config, event["time"], event["ancestral"])

    return update_activity(activity, derived=event["derived"], ancestral=["ancestral"])


def handle_population_admixture(demography, event, activity):
    demography.add_admixture(
        time=event["time"],
        derived=event["derived"],
        ancestral=event["ancestral"],
        proportions=event["proportions"],
    )
    return update_activity(activity, derived=event["derived"])


def handle_population_pulse_migration(demography, event):
    demography.add_mass_migration(
        time=event["time"],
        dest=event["dest"],
        source=event["source"],
        proportion=event["proportion"]
    )


def create_msprime_demography(config):
    demography = msprime.Demography()
    populations = config.get("populations", [])
    population_size = config["population_size"]
    default_growth_rate = config.get("default_growth_rate", 0.0)
    default_migration_rate = config.get("default_migration_rate", 0.0)

    activity = get_initial_population_activity(config)

    # Add populations
    for pop in populations:
        name = pop["name"]
        size = pop.get("initial_size", population_size)
        growth = pop.get("growth_rate", default_growth_rate) if activity[name] else 0.0
        demography.add_population(name=name, initial_size=size, growth_rate=growth)
    
    set_initial_migration_rates(demography, config, activity)

    # Event handlers mapping
    handlers = {
        "size_change":                  lambda ev: handle_size_change(demography, ev, activity, populations),
        "growth_rate_change":           lambda ev: handle_growth_rate_change(demography, ev, activity, populations),
        "migration_rate_change":        lambda ev: handle_migration_rate_change(demography, ev, activity, populations),
        "population_split":             lambda ev: handle_population_split(demography, ev, activity, config),
        "population_admixture":         lambda ev: handle_population_admixture(demography, ev, activity),
        "population_pulse_migration":   lambda ev: handle_population_pulse_migration(demography, ev),
    }

    # Apply demographic events
    for event in config.get("demography_events", []):
        event_type = event["type"]

        if event_type not in handlers:
            raise ValueError(f"Unsupported event type: {event_type}")

        handler = handlers[event_type]

        # Handler may or may not update `activity`
        result = handler(event)
        if result is not None:
            activity = result

    # print(demography.debug())

    return demography

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
    
    params = {
        "samples": samples,
        "sequence_length": config["sequence_length"],
        "demography": demography,
        "num_replicates": config["num_replicates"],
        "ploidy": 1,
        "random_seed": 42,
    }

    for key in ("recombination_rate", "gene_conversion_rate", "gene_conversion_tract_length"):
        if key in config:
            params[key] = config[key]

    replicates = msprime.sim_ancestry(**params)

    output_file = config.get("msprime_output_file", "results/msprime/output.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for rep_index, ts in enumerate(replicates, start=1):
            mts = msprime.sim_mutations(ts, rate=config["mutation_rate"], random_seed=42)

            f.write(f"// Replicate {rep_index}\n")

            # Write trees in Newick format
            if config.get("save_trees", True):
                for tree in ts.trees():
                    f.write(tree.as_newick() + ";\n")
                f.write("\n")  # optional blank line between trees and sequences

            # Write sequences (ms-style)
            f.write(f"segsites: {mts.num_sites}\n")
            if mts.num_sites > 0:
                positions = " ".join(f"{site.position / mts.sequence_length:.6f}" for site in mts.sites())
                f.write(f"positions: {positions}\n")
                gt_matrix = mts.genotype_matrix().T
                for row in gt_matrix:
                    f.write("".join(map(str, row)) + "\n")
            
            f.write("\n")

