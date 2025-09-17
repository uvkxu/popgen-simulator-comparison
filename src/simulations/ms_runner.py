import os
import subprocess


# ----------------------------------
# Recombination & Gene conversion
# ----------------------------------

def set_recombination(config):
    # Construct the recombination part of the ms command.
    # Syntax in ms:
    #   -r ρ L
    # where
    #   ρ = 4 * N_e * r * L
    #   r = per-site recombination probability between adjacent sites
    #   N_e = effective population size
    #   L = sequence length, number of sites

    population_size = config["population_size"]
    seq_length = config["sequence_length"]

    # Compute population-scaled recombination rate (ρ)
    rho = 4 * population_size * float(config.get("recombination_rate", 0)) * seq_length

    # Only add recombination if rho > 0 or if gene conversion is specified
    if rho > 0 or has_gene_conversion(config):
        return f" -r {rho:.8g} {seq_length}"
    return ""


def has_gene_conversion(config):
    # Consistency check: either both parameters are set, or none
    if ("gene_conversion_tract_length" in config) ^ ("gene_conversion_rate" in config):
        raise ValueError("Either both 'gene_conversion_tract_length' and 'gene_conversion_rate' must be specified, or neither.")
    
    return "gene_conversion_tract_length" in config


def set_gene_conversion(config):
    # Construct the gene conversion part of the ms command.
    # Syntax in ms:
    #   -c f λ
    # where
    #   f = g/r   if recombination is defined
    #       (g = per-site gene conversion initiation probability,
    #        r = per-site recombination probability between adjacent sites)
    #   f = 4 * N_e * g   if recombination is not defined
    #   λ = mean gene conversion tract length

    population_size = config["population_size"]
    recomb_rate = float(config.get("recombination_rate", 0))

    if has_gene_conversion(config):
        gen_rate = float(config["gene_conversion_rate"])
        gen_tract_length = int(config["gene_conversion_tract_length"])

        # Compute f depending on whether recombination is defined:
        # - If r > 0: f = g / r
        # - If r = 0: f = 4 * N_e * g
        f = gen_rate / recomb_rate if recomb_rate else 4 * population_size * gen_rate
        return f" -c {f:.8g} {gen_tract_length}"
    return ""


# ----------------------------------
# Initial migration & growth
# ----------------------------------

def set_initial_migration_rates(config, activity, pop_index, num_populations):
    # Construct the migration part of the ms command.
    # Syntax in ms:
    #   -I npop n1 n2 . . . [4*N_e*m]
    #   -m i j M_ij     migration rate from population j to i
    #
    # Notes:
    # 1. ms expects migration rates scaled by 1 / (4 * N_e).
    #    - M_ij = 4 * N_e * m_ij
    #    - m_ij = fraction of population i replaced by migrants from j per generation
    #    - N_e  = effective population size
    #
    # 2. Default migration rate after -I switch:
    #    - If 'default_migration_rate' is specified, ms will evenly distribute it
    #      across all non-diagonal population pairs. To achieve a per-pair migration
    #      fraction m, the input must be scaled by (num_populations - 1):
    #        rate = 4 * N_e * m * (num_populations - 1)
    #
    # 3. Individual migration rates (-m) can override defaults for specific population pairs.
    #    - Here, the rate is set directly: M_ij = 4 * N_e * m_ij
    #    - No additional scaling by (num_populations - 1) is needed.

    factor = 1 / (4 * config["population_size"])
    migration_cmd = ""

    # Add global default migration rate (if specified).
    if "default_migration_rate" in config:
        rate = config["default_migration_rate"] * factor * (num_populations - 1)
        migration_cmd += f" {rate:.8g}"

    # Add individual migration rates (only between active populations).
    for mig in config.get("migration_rates", []):
        dest, source = mig["dest"], mig["source"]
        if activity[dest] and activity[source]:
            rate = mig["rate"] * factor
            migration_cmd += f" -m {pop_index[dest]} {pop_index[source]} {rate:.8g}"

    return migration_cmd


def set_initial_growth_rates(config, activity, pop_index):
    """
    Set up growth rate parameters for ms.

    Returns
    -------
    growth_rate_cmd : str
        The ms command string part defining growth rates.
    growth_rate_map : dict
        A mapping from population name to its current growth rate.
        This map is important because in ms, after a size change event,
        the growth rate of the affected population(s) is automatically
        reset to 0. By returning and tracking growth_rate_map, we can
        later restore the intended growth rates when reconstructing
        the demography.
    """
    # Construct the growth part of the ms command.
    # Syntax in ms:
    #   -G α        global growth rate for all populations
    #   -g i α      growth rate for population i
    #
    # where
    #   α = growth rate scaled by 1 / (4*N_e)

    factor = 1 / (4 * config["population_size"])
    default_growth_rate = config.get("default_growth_rate", 0.0) * factor
    growth_rate_cmd = ""
    growth_rate_map = {}

    # Apply default global growth rate (if non-zero).
    if default_growth_rate != 0:
        growth_rate_cmd = f" -G {default_growth_rate:.8g}"

    # Case: single population (no explicit population list in config).
    if "populations" not in config:
        return growth_rate_cmd, {"single": default_growth_rate} 

    # Case: multiple populations with individual growth rates.
    active_populations = [p for p in config["populations"] if activity[p["name"]]]
    for pop in active_populations:
        pop_name = pop["name"]
        if "growth_rate" in pop:
            rate = pop["growth_rate"] * factor
            growth_rate_map[pop_name] = rate
            growth_rate_cmd += f" -g {pop_index[pop_name]} {rate:.8g}"
        elif default_growth_rate != 0:
            growth_rate_map[pop_name] = default_growth_rate

    return growth_rate_cmd, growth_rate_map


# ----------------------------------
# Population initialization helpers
# ----------------------------------

def set_initial_population_sizes(config, activity, pop_index):
    # Construct population size modifiers for subpopulations.
    # Syntax in ms:
    #   -n i x
    #
    # where
    #   i = population index
    #   x = relative size (N_i / N_e)
    #   and N_e = effective population_size

    population_size = config["population_size"]
    subpop_sizes = ""

    active_populations = [p for p in config["populations"] if activity[p["name"]]]
    for pop in active_populations:
        pop_name = pop["name"]
        if "initial_size" in pop and pop["initial_size"] != population_size:
            subpop_sizes += f" -n {pop_index[pop_name]} {int(pop['initial_size']) / population_size}"
    
    return subpop_sizes


def set_population_sample_sizes(config, activity, pop_index):
    # Construct the sample sizes for each population.
    # Syntax in ms:
    #   ms nsam nrep -I npop n1 n2 ... [migration options]
    #
    # where
    #   nsam = total number of samples (sum of n_i)
    #   npop = number of populations
    #   n_i  = sample size from population i
    #
    # Note: Only active populations can contribute samples.

    sample_sizes = ""

    for pop in config.get("populations", []):
        pop_name = pop["name"]
        sample_size = pop.get("sample_size", 0)
        
        if not activity[pop_name]:
            # Inactive populations cannot provide samples.
            if sample_size != 0:
                raise ValueError("You can't sample from an inactive population")
            continue
        
        sample_sizes += f" {sample_size}"

    return sample_sizes


def get_initial_population_activity(config):
    # Initialize all populations as active by default
    activity = {p: True for p in (pop["name"] for pop in config.get("populations", []))}
    
    # Mark ancestral populations of population splits as inactive
    for event in config.get("demography_events", []):
       if event["type"] == "population_split":
            activity[event["ancestral"]] = False
    
    return activity


def get_initial_population_indices(config, activity):
    # Assign sequential indices to active populations
    pop_index = {}
    num_populations = 0

    active_populations = [p["name"] for p in config.get("populations", []) if activity[p["name"]]]
    for pop in active_populations:
        num_populations += 1
        pop_index[pop] = num_populations

    return num_populations, pop_index


# -------------------
# Event handlers
# -------------------
# Note: In the following functions, all times (t) are scaled to coalescent units
#       by dividing by 4 * N_e (effective population size). Similarly, all rates 
#       (migration, growth) are scaled by 1 / (4 * N_e) when constructing 
#       the ms command strings.


def check_events_sorted(config):
    # Verify that demography events are sorted by time
    # ms requires events to be processed in chronological order
    events = config.get("demography_events", [])
    times = [event["time"] for event in events]
    
    return times == sorted(times)


def handle_size_change_event(event, config, activity, pop_index, growth_rate_map):
    # Handle events that change population sizes
    # ms syntax:
    #   -eN t x       # global population size change at time t
    #   -en t i x     # change size of population i at time t
    # Here x is the scale:
    #   x = N_i / N_e (relative to effective population size)
    #
    # Note: Growth rates must be restored after changing population sizes,
    # since ms resets growth rates to zero whenever a size change occurs.
    size_change_cmd = ""

    population_size = config["population_size"]
    factor = 1/ (4 * population_size)
    t = event["time"] * factor

    if event.get("population") in (None, "all"):
        # Global size change
        size_change_cmd += f" -eN {t:.8g} {event['size']/population_size}"
        # Restore growth rates after size change
        size_change_cmd += restore_growth_rates(config, activity, pop_index, growth_rate_map, t)
    else:
        # Population-specific size change
        pop = event["population"]
        if not activity[pop]:
            raise ValueError("Can't change size of inactive population")
        size_change_cmd += f" -en {t:.8g} {pop_index[pop]} {event['size']/population_size}"
        # Restore growth rate after size change
        size_change_cmd += restore_growth_rates(config, activity, pop_index, growth_rate_map, t, pop)

    return size_change_cmd


def handle_migration_rate_change_event(event, config, activity, pop_index):
    # Handle events that change migration rates
    # ms syntax:
    #   -em t i j M_ij   # set migration rate from j to i at time t
    # Migration rates are scaled: M_ij = 4 * N_e * m_ij
    migration_change_cmd = ""

    factor = 1/ (4 * config["population_size"])
    t = event["time"] * factor
    rate = event["rate"] * factor

    # Case: global migration rate change (all active population pairs)
    if "dest" not in event or "source" not in event:
        if "dest" in event or "source" in event:
            raise ValueError("Event must define either both 'dest' and 'source' or neither of them.")

        # Collect all populations that are currently active
        active_populations = [p["name"] for p in config.get("populations", []) if activity[p["name"]]]

        # Apply the migration rate to every pair of active populations
        for dest in active_populations:
            for source in active_populations:
                if dest != source:
                    migration_change_cmd += f" -em {t:.8g} {pop_index[dest]} {pop_index[source]} {rate:.8g}"
    else:
        # Case: migration rate change between two specific populations
        dest = event["dest"]
        source = event["source"]
        # Both the source and destination populations must be active
        if not (activity[dest] and activity[source]):
            raise ValueError("Inactive population in migration event")
            
        migration_change_cmd += f" -em {t:.8g} {pop_index[dest]} {pop_index[source]} {rate:.8g}"

    return migration_change_cmd


def handle_growth_rate_change_event(event, config, activity, pop_index, growth_rate_map):
    # Handle events that change population growth rates
    # ms syntax:
    #   -eg t i r     # set growth rate of population i to r at time t
    #   -eG t r       # set growth rate globally (for single-population models)
    factor = 1/ (4 * config["population_size"])
    t = event["time"] * factor
    rate = event["rate"] * factor

    # Apply growth rate globally if no population is specified
    if event.get("population") in (None, "all"):
        # Case single population model:
        if "populations" not in config:
            growth_rate_map["single"] = rate
            return f" -eG {t:.8g} {rate:.8g}"

        # Case multiple population model: apply to all active populations
        growth_rate_cmd = ""
        active_populations = [p["name"] for p in config["populations"] if activity[p["name"]]]
        for pop in active_populations:
            growth_rate_map[pop] = rate
            growth_rate_cmd += f" -eg {t:.8g} {pop_index[pop]} {rate:.8g}"
        
        return growth_rate_cmd
            
    # Apply growth rate to a single population
    pop = event["population"]
    growth_rate_map[pop] = rate

    # The population needs to be active at time t in order to change the growth rate
    if not activity[pop]:
        raise ValueError("Can't change growth rate of inactive population")

    return f" -eg {t:.8g} {pop_index[pop]} {rate:.8g}"


def handle_population_split_event(event, config, activity, pop_index):
    population_split_cmd = ""
    factor = 1/ (4 * config["population_size"])
    t = event["time"] * factor

    derived = event["derived"]
    ancestral = event["ancestral"]
    
    # Consistency checks:
    # - Ancestral should be inactive before the split
    # - Derived populations should be active before the split
    if activity[ancestral]:
        raise ValueError("Ancestral population should initially be inactive")
    elif any(not activity[d] for d in derived):
        raise ValueError("Derived populations should initially be active")
    
    # Mark all derived populations as inactive
    for d in derived:
        activity[d] = False

    # Disable migration into derived populations except the first one,
    # since the ancestral population will reuse the index of the first derived
    # migrations might be set to zero via function set_ancestral_migration_rates_at_split
    population_split_cmd += disable_migration_to_derived(config, activity, pop_index, t, derived[1:])

    # Mark the ancestral population as active
    activity[ancestral] = True

    # Reassign the ancestral population to take over the index of the first derived
    pop_index[ancestral] = pop_index[derived[0]]

    # Set the population size of the ancestral population
    # TODO: Ask Stefanos if size adjustment should happen before or after the split
    population_split_cmd += set_ancestral_size_at_split(config, pop_index, t, ancestral)

    # Join all remaining derived populations into the ancestral one
    for i in range(1, len(derived)):
        population_split_cmd += f" -ej {t:.8g} {pop_index[derived[i]]} {pop_index[ancestral]}"
    
    # Remove all derived populations from index mapping
    for d in derived:
        pop_index.pop(d)

    # Set growth rate and migration rates for the ancestral population
    population_split_cmd += set_ancestral_growth_rate_at_split(config, pop_index, t, ancestral)
    population_split_cmd += set_ancestral_migration_rates_at_split(config, activity, pop_index, t, ancestral)

    return population_split_cmd


def handle_population_admixture_event(event, config, activity, pop_index, num_populations):
    population_admixture_cmd = ""

    factor = 1/ (4 * config["population_size"])
    t = event["time"] * factor

    derived = event["derived"]
    ancestral = event["ancestral"]
    proportions = event["proportions"]

    # Sanity checks
    if len(ancestral) != len(proportions):
        raise ValueError("Number of proportions must match the number of ancestral populations")
    if not sum(proportions) == 1:
        raise ValueError("Proportions should sum to 1")
    if not activity[derived]:
        raise ValueError("Derived population must be active")
    if any(not activity[a] for a in ancestral):
        raise ValueError("All ancestral populations must be active")

    # For all but the last ancestral population:
    for i in range(0, len(ancestral) - 1):
        num_populations += 1
        # Conditional proportion for this step (adjusted for remaining lineages)
        p = proportions[i] / sum(proportions[i:])

        # Step 1: Split lineages from the derived population
        # -es t i q   -> at time t, a fraction q of lineages remain in the derived population,
        #                and the remaining fraction (1-q) are moved to a temporary subpopulation
        population_admixture_cmd += f" -es {t:.8g} {pop_index[derived]} {1 - p}"

        # Step 2: Immediately merge temporary subpopulation into the corresponding ancestral population
        # -ej t i j   -> move all lineages from subpopulation i to ancestral j at time t
        population_admixture_cmd += f" -ej {t:.8g} {num_populations} {pop_index[ancestral[i]]}"

    # Last ancestral population receives all remaining lineages from the derived population
    population_admixture_cmd += f" -ej {t:.8g} {pop_index[derived]} {pop_index[ancestral[-1]]}"

    # Disable any future migrations to the derived population and mark it as inactive
    population_admixture_cmd += disable_migration_to_derived(config, activity, pop_index, t, derived)
    activity[derived] = False
    pop_index.pop(derived)

    return population_admixture_cmd, num_populations


def handle_populatin_pulse_migration(event, config, activity, pop_index, num_populations):
    factor = 1 / (4 * config["population_size"])
    t = event["time"] * factor

    dest = event["dest"]
    source = event["source"]

    # Ensure both populations involved in the pulse migration are active
    if not activity[dest]:
        raise ValueError("Destination population must be active")
    if not activity[source]:
        raise ValueError("Source population must be active")

    # Increment the temporary population index for ms
    num_populations += 1
    proportion = event["proportion"]

    # Step 1: Split source population lineages
    # -es t i q   -> at time t, fraction q of lineages remain in the source population,
    #                and the remaining fraction (1-q) move to a temporary subpopulation
    pulse_migration_cmd = f" -es {t:.8g} {pop_index[source]} {1 - proportion}"

    # Step 2: Immediately merge the temporary subpopulation into the destination
    # -ej t i j   -> at time t, all lineages in temporary population i move to destination j
    pulse_migration_cmd += f" -ej {t:.8g} {num_populations} {pop_index[dest]}"

    return pulse_migration_cmd, num_populations


# ----------------------------------
# Helper function for event handlers
# ----------------------------------


def set_ancestral_size_at_split(config, pop_index, time, population):    
    # Set the ancestral population size at the moment of split
    population_size = config["population_size"]
    populations = config.get("populations", [])

    initial_size = [pop.get("initial_size", population_size) for pop in populations if pop["name"] == population][0]

    return f" -en {time:.8g} {pop_index[population]} {initial_size/ population_size}"


def set_ancestral_growth_rate_at_split(config, pop_index, time, population):
    # Set the growth rate of the ancestral population at split time
    default_growth_rate = config.get("default_growth_rate", 0)
    populations = config.get("populations", [])

    growth_rate = [pop.get("growth_rate", default_growth_rate) for pop in populations if pop["name"] == population][0]

    factor = 1 / (4 * config["population_size"])
    adjusted_growth_rate = growth_rate * factor

    return f" -eg {time:.8g} {pop_index[population]} {adjusted_growth_rate:.8g}"


def set_ancestral_migration_rates_at_split(config, activity, pop_index, time, population):
    # Configure migration rates for the ancestral population at split
    factor = 1 / (4 * config["population_size"])  # scaling factor
    migration_cmd = ""
    handled_migrations = set()

    # Handle explicitly defined migration rates involving the ancestral population
    for mig in config.get("migration_rates", []):
        dest, source = mig["dest"], mig["source"]

        if (dest == population or source == population) and activity[dest] and activity[source]:
            rate = mig["rate"] * factor
            migration_cmd += f" -em {time} {pop_index[dest]} {pop_index[source]} {rate:.8g}"

            handled_migrations.add((dest, source))
    
    # TODO: Ask Stefanos if it makes more sense to set them to the default migration rate
    # Set all other migrations involving the ancestral population to zero
    for pop in (p["name"] for p in config["populations"]):
        if activity[pop] and pop != population:
            if (pop, population) not in handled_migrations:
                migration_cmd += f" -em {time} {pop_index[pop]} {pop_index[population]} {0}"
            if (population, pop) not in handled_migrations:
                migration_cmd += f" -em {time} {pop_index[population]} {pop_index[pop]} {0}"
    
    return migration_cmd


def disable_migration_to_derived(config, activity, pop_index, time, populations):
    # Disable migration into derived populations at split
    # The migration out of the derived populations is already set to zero by ms
    migration_cmd = ""
    populations = populations if isinstance(populations, list) else [populations]
    
    for dest in populations:
        for source in (p["name"] for p in config["populations"] if activity[p["name"]]):
            if source != dest:
                migration_cmd += f" -em {time} {pop_index[dest]} {pop_index[source]} {0}"
    
    return migration_cmd


def restore_growth_rates(config, activity, pop_index, growth_rate_map, time, population=None):
    # Restore growth rates after size change events
    if "populations" in config:
        populations = (
            [p["name"] for p in config.get("populations", []) if activity[p["name"]]]
            if population is None
            else [population]
        )
        growth_rate_cmd = ""

        for pop in populations:
            rate = growth_rate_map.get(pop, 0.0)

            if rate != 0:
                growth_rate_cmd += f" -eg {time:.8g} {pop_index[pop]} {rate:.8g}"
        
        return growth_rate_cmd
    elif growth_rate_map["single"] != 0:  
        # only one population in the whole model
        return f" -eG {time:.8g} {growth_rate_map['single']:.8g}"

    return ""


def build_ms_command(config):
    total_sample_size = sum(p.get("sample_size", 0) for p in config.get("populations", [])) \
                        if "populations" in config else int(config["sample_size"])
    population_size = int(config["population_size"])
    num_replicates = int(config.get("num_replicates", 1))
    mut_rate = float(config["mutation_rate"])
    seq_length = int(config["sequence_length"])

    num_populations = 0
    activity = pop_index = {}

    # Mutation
    theta = 4 * population_size * mut_rate * seq_length
    cmd = f"ms {total_sample_size} {num_replicates} -t {theta:.8g}"

    # Tree output
    if config.get("save_trees", True):
        cmd += " -T"

    # Recombination
    cmd += set_recombination(config)

    # Gene conversion
    cmd += set_gene_conversion(config)

    # Initialization of subpopulations
    if "populations" in config:
        migration_rate_map = {}
        activity = get_initial_population_activity(config)
        num_populations, pop_index = get_initial_population_indices(config, activity)

        cmd += f" -I {num_populations}"
        cmd += set_population_sample_sizes(config, activity, pop_index)
        cmd += set_initial_migration_rates(config, activity, pop_index, num_populations)
        cmd += set_initial_population_sizes(config, activity, pop_index)

    growth_rates_cmd, growth_rate_map = set_initial_growth_rates(config, activity, pop_index)
    cmd += growth_rates_cmd

    # Event handeling
    if not check_events_sorted(config):
        raise ValueError("Demography events are not sorted by time!")

    handlers = {
        "size_change":                  lambda ev: (handle_size_change_event(ev, config, activity, pop_index, growth_rate_map), num_populations),
        "migration_rate_change":        lambda ev: (handle_migration_rate_change_event(ev, config, activity, pop_index), num_populations),
        "growth_rate_change":           lambda ev: (handle_growth_rate_change_event(ev, config, activity, pop_index, growth_rate_map), num_populations),
        "population_split":             lambda ev: (handle_population_split_event(ev, config, activity, pop_index), num_populations),
        "population_admixture":         lambda ev: handle_population_admixture_event(ev, config, activity, pop_index, num_populations),
        "population_pulse_migration":   lambda ev: handle_populatin_pulse_migration(ev, config, activity, pop_index, num_populations)
    }

    for event in config.get("demography_events", []):
        event_type = event["type"]
        if event_type not in handlers:
            raise ValueError(f"Unsupported event type: {event_type}")

        cmd_part, num_populations = handlers[event_type](event)
        cmd += cmd_part

    return cmd


def run_ms_simulation(config):
    ms_path = config.get("ms_path")
    if not ms_path or not os.path.exists(ms_path):
        raise FileNotFoundError(f"ms executable not found at {ms_path}. Please check the path in config.")

    command = build_ms_command(config)
    output_file = config.get("ms_output_file", "results/ms/output.txt")

    print(f"🚀 Running ms: {command}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    cmd_parts = command.split()
    cmd_parts[0] = ms_path

    result = subprocess.run(cmd_parts, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ms failed:\n{result.stderr}")

    with open(output_file, "w") as f:
        f.write(result.stdout)