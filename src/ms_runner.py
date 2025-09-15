import os
import subprocess


# ----------------------------------
# Recombination & Gene conversion
# ----------------------------------

def set_recombination(config):
    # Return the recombination part of the ms command
    population_size = config["population_size"]
    seq_length = config["sequence_length"]

    rho = 4 * population_size * float(config.get("recombination_rate", 0)) * seq_length

    if rho > 0 or has_gene_conversion(config):
        return f" -r {rho:.8g} {seq_length}"
    return ""


def has_gene_conversion(config):
    # XOR check: either both or none must be specified
    if ("gene_conversion_tract_length" in config) ^ ("gene_conversion_rate" in config):
        raise ValueError("Either both 'gene_conversion_tract_length' and 'gene_conversion_rate' must be specified, or neither.")
    
    return "gene_conversion_tract_length" in config


def set_gene_conversion(config):
    # Return the gene conversion part of the ms command
    population_size = config["population_size"]
    seq_length = config["sequence_length"]
    recomb_rate = float(config.get("recombination_rate", 0)) * seq_length

    if has_gene_conversion(config):
        gen_rate = float(config["gene_conversion_rate"])
        gen_tract_length = int(config["gene_conversion_tract_length"])
        f = gen_rate / recomb_rate if recomb_rate else 4 * population_size * gen_rate
        return f" -c {f:.8g} {gen_tract_length}"
    return ""


# ----------------------------------
# Initial migration & growth
# ----------------------------------


def set_initial_migration_rates(config, activity, pop_index):
    factor = 1 / (4 * config["population_size"])  # scaling factor for rates
    migration_cmd = ""

    if "default_migration_rate" in config:
        rate = config["default_migration_rate"] * factor
        migration_cmd += f" {rate:.8g}"

    for mig in config.get("migration_rates", []):
        dest, source = mig["dest"], mig["source"]
        if activity[dest] and activity[source]:
            rate = mig["rate"] * factor
            migration_cmd += f" -m {pop_index[dest]} {pop_index[source]} {rate:.8g}"

    return migration_cmd


def set_initial_growth_rates(config, activity, pop_index):
    factor = 1 / (4 * config["population_size"])  # scaling factor for rates
    default_growth_rate = config.get("default_growth_rate", 0.0) * factor
    growth_rate_cmd = ""
    growth_rate_map = {}

    if default_growth_rate != 0:
        growth_rate_cmd = f" -G {default_growth_rate:.8g}"

    if "populations" not in config:
        return growth_rate_cmd, {"single": default_growth_rate} 

    # Initial growth rates
    for pop in config["populations"]:
        pop_name = pop["name"]
        if activity[pop_name]:
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
    population_size = config["population_size"]
    subpop_sizes = ""

    for pop in config.get("populations", []):
        pop_name = pop["name"]
        if activity[pop_name] and "initial_size" in pop and pop['initial_size'] != population_size:
                subpop_sizes += f" -n {pop_index[pop_name]} {int(pop['initial_size']) / population_size}"
    
    return subpop_sizes


def set_population_sample_sizes(config, activity, pop_index):
    sample_sizes = ""

    for pop in config.get("populations", []):
        pop_name = pop["name"]

        sample_size = pop.get("sample_size", 0)
        
        if not activity[pop_name]:
            if sample_size != 0:
                raise ValueError("You can't sample from an inactive population")
            continue
        
        sample_sizes += f" {sample_size}"

    return sample_sizes


def get_initial_population_activity(config):
    pop_names = [pop["name"] for pop in config.get("populations", [])]
    activity = {pid: True for pid in pop_names}  # Default: aktiv
    
    for event in config.get("demography_events", []):
       if event["type"] == "population_split":
            activity[event["ancestral"]] = False
    
    return activity


def get_initial_population_indices(config, activity):
    pop_index = {}
    num_populations = 0

    for pop in config.get("populations", []):
        pop_name = pop["name"]

        if not activity[pop_name]:
            continue
        
        num_populations += 1
        pop_index[pop_name] = num_populations

    return num_populations, pop_index


# ----------------------------------
# Event handlers & helpers
# ----------------------------------

def check_events_sorted(config):
    events = config.get("demography_events", [])
    times = [event["time"] for event in events]
    
    return times == sorted(times)


def set_ancestral_size_at_split(config, pop_index, time, population):    
    population_size = config["population_size"]
    populations = config.get("populations", [])

    initial_size = [pop.get("initial_size", population_size) for pop in populations if pop["name"] == population][0]

    return f" -en {time:.8g} {pop_index[population]} {initial_size/ population_size}"


def set_ancestral_growth_rate_at_split(config, pop_index, time, population):
    default_growth_rate = config.get("default_growth_rate", 0)
    populations = config.get("populations", [])

    growth_rate = [pop.get("growth_rate", default_growth_rate) for pop in populations if pop["name"] == population][0]

    factor = 1 / (4 * config["population_size"])
    adjusted_growth_rate = growth_rate * factor

    if adjusted_growth_rate != 0:
        return f" -eg {time:.8g} {pop_index[population]} {adjusted_growth_rate:.8g}"
    return ""


def set_ancestral_migration_rates_at_split(config, activity, pop_index, time, population):
    factor = 1 / (4 * config["population_size"])  # scaling factor for rates
    migration_cmd = ""
    handled_migrations = set()

    for mig in config.get("migration_rates", []):
        dest, source = mig["dest"], mig["source"]

        if (dest == population or source == population) and activity[dest] and activity[source]:
            rate = mig["rate"] * factor
            migration_cmd += f" -em {time} {pop_index[dest]} {pop_index[source]} {rate:.8g}"

            handled_migrations.add((dest, source))
    
    for pop in (p["name"] for p in config["populations"]):
        if activity[pop] and pop != population:
            if (pop, population) not in handled_migrations:
                migration_cmd += f" -em {time} {pop_index[pop]} {pop_index[population]} {0}"
            if (population, pop) not in handled_migrations:
                migration_cmd += f" -em {time} {pop_index[population]} {pop_index[pop]} {0}"
    
    return migration_cmd



def restore_growth_rates(config, activity, pop_index, growth_rate_map, time, population=None):
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
    elif growth_rate_map["single"] != 0: # there only exits one population in the whole model
        return f" -eG {time:.8g} {growth_rate_map["single"]:.8g}"

    return ""


def handle_size_change_event(event, config, activity, pop_index, growth_rate_map):
    size_change_cmd = ""

    population_size = config["population_size"]
    factor = 1/ (4 * population_size)
    t = event["time"] * factor

    if event.get("population") in (None, "all"):
        size_change_cmd += f" -eN {t:.8g} {event['size']/population_size}"
        size_change_cmd += restore_growth_rates(config, activity, pop_index, growth_rate_map, t)
    else:
        pop = event["population"]
        if not activity[pop]:
            raise ValueError("Can't change size of inactive population")
        size_change_cmd += f" -en {t:.8g} {pop_index[pop]} {event['size']/population_size}"
        size_change_cmd += restore_growth_rates(config, activity, pop_index, growth_rate_map, t, pop)

    return size_change_cmd


def handle_migration_rate_change_event(event, config, activity, pop_index):
    migration_change_cmd = ""

    factor = 1/ (4 * config["population_size"])
    t = event["time"] * factor
    populations = config.get("populations", [])

    if "dest" not in event or "source" not in event:
        if "dest" in event or "source" in event:
            raise ValueError("Event must define either both 'dest' and 'source' or neither of them.")

        rate = event["rate"] * factor
        for dest_pop in populations:
            dest = dest_pop["name"]
            if not activity[dest]:
                continue
            for source_pop in populations:
                source = source_pop["name"]
                if dest == source or not activity[source]:
                    continue
                migration_change_cmd += f" -em {t:.8g} {pop_index[dest]} {pop_index[source]} {rate:.8g}"
    else:
        dest = event["dest"]
        source = event["source"]
        if not (activity[dest] and activity[source]):
            raise ValueError("Inactive population in migration event")
        rate = event["rate"] * factor
        migration_change_cmd += f" -em {t:.8g} {pop_index[dest]} {pop_index[source]} {rate:.8g}"

    return migration_change_cmd


def handle_growth_rate_change_event(event, config, activity, pop_index, growth_rate_map):
    factor = 1/ (4 * config["population_size"])
    t = event["time"] * factor
    rate = event["rate"] * factor

    if event.get("population") in (None, "all"):
        if "populations" not in config:
            growth_rate_map["single"] = rate
            return f" -eG {t:.8g} {rate:.8g}"

        growth_rate_cmd = ""

        for pop in (p["name"] for p in config["populations"] if activity[p["name"]]):
            growth_rate_map[pop] = rate
            growth_rate_cmd += f" -eg {t:.8g} {pop_index[pop]} {rate:.8g}"
        
        return growth_rate_cmd
            
    
    pop = event["population"]
    growth_rate_map[pop] = rate

    if not activity[pop]:
        raise ValueError("Can't change growth rate of inactive population")

    return f" -eg {t:.8g} {pop_index[pop]} {rate:.8g}"


def handle_population_split_event(event, config, activity, pop_index):
    population_split_cmd = ""

    factor = 1/ (4 * config["population_size"])
    t = event["time"] * factor

    derived = event["derived"]
    ancestral = event["ancestral"]
    
    if activity[ancestral]:
        raise ValueError("Ancestral population should initially be inactive")
    elif any(not activity[d] for d in derived):
        raise ValueError("Derived populations should initially be active")

    pop_index[ancestral] = pop_index[derived[0]]

    # Setting the population size of the ancestral population

    # TODO: Ask Stefanos if it is important to change the size before or after split
    population_split_cmd += set_ancestral_size_at_split(config, pop_index, t, ancestral)

    for i in range(1, len(derived)):
        population_split_cmd += f" -ej {t:.8g} {pop_index[derived[i]]} {pop_index[ancestral]}"

    # Mark the ancestral population as active
    activity[ancestral] = True
    # Mark all derived populations as inactive
    for d in derived:
        activity[d] = False
        pop_index.pop(d)

    # Setting the growth rate of the ancestral population
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

    if not len(ancestral) == len(proportions):
        raise ValueError("The number of proportions must match the number of ancestral populations")
    
    if not sum(proportions) == 1:
        raise ValueError("Proportions should add up to one")

    if not activity[derived]:
        raise ValueError("Derived population should be active")

    if any(not activity[a] for a in ancestral):
        raise ValueError("Ancestral populations should all be active")

    for i in range(0, len(ancestral)-1):
        num_populations += 1
        proportion = proportions[i]/sum(proportions[i:])

        population_admixture_cmd += f" -es {t} {pop_index[derived]} {1 - proportion}"
        population_admixture_cmd += f" -ej {t} {num_populations} {pop_index[ancestral[i]]}"
    
    population_admixture_cmd += f" -ej {t} {pop_index[derived]} {pop_index[ancestral[-1]]}"

    activity[derived] = False
    pop_index.pop(derived)

    return population_admixture_cmd, num_populations


def handle_populatin_pulse_migration(event, config, activity, pop_index, num_populations):
    factor = 1/ (4 * config["population_size"])
    t = event["time"] * factor

    dest = event["dest"]
    source = event["source"]

    if not activity[dest]:
        raise ValueError("Destination population must be active")

    if not activity[source]:
        raise ValueError("Source population must be active")

    num_populations += 1
    proportion = event["proportion"]

    pulse_migration_cmd = f" -es {t} {pop_index[source]} {1 - proportion}"
    pulse_migration_cmd += f" -ej {t} {num_populations} {pop_index[dest]}"

    return pulse_migration_cmd, num_populations


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
        cmd += set_initial_migration_rates(config, activity, pop_index)
        cmd += set_initial_population_sizes(config, activity, pop_index)

    growth_rates_cmd, growth_rate_map = set_initial_growth_rates(config, activity, pop_index)
    cmd += growth_rates_cmd

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