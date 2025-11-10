import os
import msprime
from ms_runner import get_initial_population_activity, get_effective_population_size, \
                                    get_rate_multiplier, build_ms_command
import subprocess


def update_activity(activity, derived=None, ancestral=None):
    """
    Update the activity status of populations.

    - Ancestral populations become active (True).
    - Derived populations become inactive (False), e.g., after splits or admixture.

    Note: This function tracks whether a population is currently active so that 
    other demographic events (e.g., migration rate change, size change) are applied 
    only to active populations. 

    In msprime, populations can be in three states: INACTIVE → ACTIVE → PREVIOUSLY_ACTIVE.  
    Here, INACTIVE and PREVIOUSLY_ACTIVE are treated as equivalent, since we only 
    need to distinguish between active and non-active populations.
    """
    if ancestral:
        ancestral = ancestral if isinstance(ancestral, list) else [ancestral]
        for a in ancestral:
            activity[a] = True
    if derived:
        derived = derived if isinstance(derived, list) else [derived]
        for d in derived:
            activity[d] = False


def set_initial_migration_rates(demography, config, activity):
    """
    Initialize migration rates in the msprime Demography object.
    
    - Explicit migration rates are applied first.
    - Default migration rates are applied only between all active populations
      where no explicit rate exists.
    """
    explicit_migrations = set()
    populations = config.get("populations", [])
    
    # Apply explicitly defined migration rates to all active populations
    for mig in config.get("migration_rates", []):
        dest, source = mig["dest"], mig["source"]

        if activity[dest] and activity[source]:
            demography.set_migration_rate(source=mig["source"], dest=mig["dest"], rate=mig["rate"])
            explicit_migrations.add((mig["dest"], mig["source"]))

    default_rate = config.get("default_migration_rate", 0.0)
    if default_rate != 0:
        # Apply default rate to all active population pairs not already specified
        active_populations = [p["name"] for p in populations if activity[p["name"]]]
        for dest in active_populations:
            for src in active_populations:
                if dest != src and (dest, src) not in explicit_migrations:
                    demography.set_migration_rate(source=src, dest=dest, rate=default_rate)


def handle_size_change(demography, event, activity, populations):
    """
    Handle population size change events.

    - If `population` is 'all' or None, the size change is applied to all active populations 
    (or to the single population in a one-population model).
    - Otherwise, the size change is applied only to the specified population.
    """
    time = event["time"]
    if event.get("population") in (None, "all"):
        if not populations: 
            # Case: Single population model
            demography.add_population_parameters_change(time=time, initial_size=event["size"])
        else: 
            # Case: Mulitple population model - apply to all active populations only
            active_populations = [p["name"] for p in populations if activity[p["name"]]]
            for pop in active_populations:
                if activity[pop]:
                    demography.add_population_parameters_change(time=time, initial_size=event["size"], population=pop)
    else:
        demography.add_population_parameters_change(
            time=time, initial_size=event["size"], population=event["population"]
        )


def handle_growth_rate_change(demography, event, activity, populations):
    """
    Handle growth rate change events.
    
    - Similar to population size changes, growth rates can be applied to all active
      populations or a specific population.
    """
    time = event["time"]
    if event.get("population") in (None, "all"):
        if not populations: 
            # Case: Single population model
            demography.add_population_parameters_change(time=time, growth_rate=event["rate"])
        else:
            # Case: Mulitple population model - apply to all active populations only
            active_populations = [p["name"] for p in populations if activity[p["name"]]]
            for pop in active_populations:
                demography.add_population_parameters_change(time=time, growth_rate=event["rate"], population=pop)
    else:
        demography.add_population_parameters_change(
            time=time, growth_rate=event["rate"], population=event["population"]
        )


def handle_migration_rate_change(demography, event, activity, populations):
    """
    Handle migration rate change events.
    
    - Can apply globally to all active population pairs or to specific source-destination pair.
    - Ignores migration from/to inactive populations.
    """
    time = event["time"]
    if "dest" not in event or "source" not in event:
        if "dest" in event or "source" in event:
            raise ValueError("Event must define either both 'dest' and 'source' or neither of them.")

        active_populations = [p["name"] for p in populations if activity[p["name"]]]
        for dst in active_populations:
            for src in active_populations:
                if dst != src:
                    demography.add_migration_rate_change(
                        time=time, source=src, dest=dst, rate=event["rate"]
                    )
    else:
        demography.add_migration_rate_change(
            time=time, source=event["source"], dest=event["dest"], rate=event["rate"]
        )


def set_ancestral_growth_rate_at_split(demography, config, time, population):
    """
    Restore the growth rate for an ancestral population after a split event.
    
    - Uses the population-specific growth rate if defined,
      otherwise uses the default growth rate.
    """
    default_growth_rate = config.get("default_growth_rate", 0)
    populations = config.get("populations", [])

    growth_rate = [pop.get("growth_rate", default_growth_rate) for pop in populations if pop["name"] == population][0]

    demography.add_population_parameters_change(time=time, growth_rate=growth_rate, population=population)


def set_ancestral_migration_rates_at_split(demography, config, activity, time, population):
    for mig in config.get("migration_rates", []):
        dest, source = mig["dest"], mig["source"]

        if (dest == population or source == population) and activity[dest] and activity[source]:
            demography.add_migration_rate_change(
                        time=time, source=source, dest=dest, rate=mig["rate"]
                    )


def handle_population_split(demography, event, activity, config):
    """
    Handle a population split event.
    
    - Derived populations become inactive.
    - Ancestral population becomes active.
    - Growth rate of ancestral population is restored.
    """
    demography.add_population_split(
        time=event["time"], derived=event["derived"], ancestral=event["ancestral"]
    )

    update_activity(activity, derived=event["derived"], ancestral=event["ancestral"])

    set_ancestral_growth_rate_at_split(demography, config, event["time"], event["ancestral"])
    set_ancestral_migration_rates_at_split(demography, config, activity, event["time"], event["ancestral"])


def handle_population_admixture(demography, event, activity):
    """
    Handle an admixture event.
    
    - Derived population becomes inactive.
    - Lineages are distributed among ancestral populations according to specified proportions.
    """
    demography.add_admixture(
        time=event["time"],
        derived=event["derived"],
        ancestral=event["ancestral"],
        proportions=event["proportions"],
    )
    
    update_activity(activity, derived=event["derived"])


def handle_population_pulse_migration(demography, event):
    """
    Handle a pulse migration (mass migration) event.
    
    - Transfers a proportion of lineages from source to destination population at a given time.
    """
    demography.add_mass_migration(
        time=event["time"],
        dest=event["dest"],
        source=event["source"],
        proportion=event["proportion"]
    )


def create_msprime_demography(config):
    """
    Construct an msprime Demography object from a configuration dictionary.
    
    - Adds populations, sets initial growth and migration rates.
    - Applies all demographic events in chronological order.
    """
    demography = msprime.Demography()
    populations = config.get("populations", [])
    population_size = get_effective_population_size(config)
    default_growth_rate = config.get("default_growth_rate", 0.0)
    default_migration_rate = config.get("default_migration_rate", 0.0)

    activity = get_initial_population_activity(config)

    # Add populations with initial size and growth rate
    if "populations" in config:
        for pop in populations:
            name = pop["name"]
            size = pop.get("initial_size", population_size)
            growth = pop.get("growth_rate", default_growth_rate) if activity[name] else 0.0
            demography.add_population(name=name, initial_size=size, growth_rate=growth)

        # Set up initial migration rates
        set_initial_migration_rates(demography, config, activity)
    else:
        demography.add_population(initial_size=population_size, growth_rate=default_growth_rate)
    
    
    # Map event types to handler functions
    handlers = {
        "size_change":                  lambda ev: handle_size_change(demography, ev, activity, populations),
        "growth_rate_change":           lambda ev: handle_growth_rate_change(demography, ev, activity, populations),
        "migration_rate_change":        lambda ev: handle_migration_rate_change(demography, ev, activity, populations),
        "population_split":             lambda ev: handle_population_split(demography, ev, activity, config),
        "population_admixture":         lambda ev: handle_population_admixture(demography, ev, activity),
        "population_pulse_migration":   lambda ev: handle_population_pulse_migration(demography, ev),
    }

    # Apply demographic events in order
    for event in config.get("demography_events", []):
        event_type = event["type"]

        if event_type not in handlers:
            raise ValueError(f"Unsupported event type: {event_type}")

        handler = handlers[event_type]
        handler(event)

    # print(demography.debug())

    return demography


def run_msprime_simulation(config, output_file=None):
    """
    Run a full msprime simulation based on the given configuration.
    
    - Creates populations and applies demographic events.
    - Generates ancestry and mutations.
    - Saves output in Newick tree format and ms-style sequence data.
    """
    demography = create_msprime_demography(config)

    samples = []
    if "populations" in config:
        for pop in config.get("populations"):
            if "sample_size" in pop:
                samples.append(
                    msprime.SampleSet(
                        num_samples=pop["sample_size"],
                        population=pop["name"],
                        time=0
                    )
                )
    else:
        samples.append(
            msprime.SampleSet(
                        num_samples=config["sample_size"],
                        population=0,
                        time=0
                    ))
    
    # Build parameters dictionary for msprime simulation
    params = {
        "samples": samples,
        "sequence_length": config["sequence_length"],
        "demography": demography,
        "num_replicates": config["num_replicates"],
        "ploidy": 1,
        "discrete_genome": True
    }

    # Include optional recombination and gene conversion parameters
    for key in ("recombination_rate", "gene_conversion_rate", "gene_conversion_tract_length"):
        if key in config:
            params[key] = config[key]

    replicates = msprime.sim_ancestry(**params)

    if output_file is None:
        output_file = config.get("ms_output_file", "results/msprime/output.txt")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for rep_index, ts in enumerate(replicates, start=1):
            mts = msprime.sim_mutations(ts, rate=config["mutation_rate"], model=msprime.BinaryMutationModel(), 
                discrete_genome=True)

            f.write(f"// Replicate {rep_index}\n")

            # Write trees in Newick format
            if config.get("save_trees", True):
                for tree in ts.trees():
                    f.write(tree.as_newick() + ";\n")
                f.write("\n")  # optional blank line between trees and sequences

            # Write ms-style sequence data
            f.write(f"segsites: {mts.num_sites}\n")
            if mts.num_sites > 0:
                positions = " ".join(f"{site.position / mts.sequence_length:.6f}" for site in mts.sites())
                f.write(f"positions: {positions}\n")
                gt_matrix = mts.genotype_matrix().T
                for row in gt_matrix:
                    f.write("".join(map(str, row)) + "\n")
            
            f.write("\n")


def build_msprime_command(config):
    ms_cmd = build_ms_command(config)
    parts = ms_cmd.removeprefix("ms").split()

    msms_cmd = "mspms " + " ".join(parts)
    return msms_cmd


def run_mspms_simulation(config=None, command=None, output_file=None):
    if config is not None:
        command = build_msprime_command(config)
    elif command is not None:
        # Falls der übergebene Command nicht mit "mspms" anfängt → füge es hinzu
        stripped = command.strip()
        if not (stripped.startswith("mspms") or stripped.startswith("python -m mspms")):
            command = f"mspms {stripped}"
    else:
        raise ValueError("Either 'config' or 'command' must be provided.")

    if output_file is None:
        if config:
            output_file = config.get("msprime_output_file", "results/msprime/output.txt")
        else:
            output_file = "results/msprime/output.txt"

    print(f"🚀 Running msprime: {command}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    cmd_parts = command.split()

    result = subprocess.run(cmd_parts, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"msprime failed:\n{result.stderr}")

    with open(output_file, "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)
    
    return output_file