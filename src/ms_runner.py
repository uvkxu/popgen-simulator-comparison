import os
import subprocess

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

'''
def restore_migration_rates(config, activity, pop_index, time):
    factor = 1 / (4 * config["population_size"])  # scaling factor for rates
    migration_cmd = ""
    changed_migrations = set()

    for event in reversed(config.get("demography_events", [])):
        if event["type"] != "migration_rate_change" or event["time"] >= time:
            continue

        if "dest" not in event or "source" not in event:
            rate = event["rate"] /(config["population_size"] - 1)
            return f" -eM {time:.8g} {rate:.8g}" + migration_cmd

        dest, source = event["dest"], event["source"]
        if activity[dest] and activity[source] and (dest, source) not in changed_migrations:
            # Apply the event-specific migration rate
            rate = event["rate"] * factor
            migration_cmd += f" -em {time:.8g} {pop_index[dest]} {pop_index[source]} {rate:.8g}"
            changed_migrations.add((dest, source))

    for mig in config.get("migration_rates", []):
        dest, source = mig["dest"], mig["source"]
        if activity[dest] and activity[source] and (dest, source) not in changed_migrations:
            rate = mig["rate"] * factor
            migration_cmd += f" -em {time:.8g} {pop_index[dest]} {pop_index[source]} {rate:.8g}"

    return f" -eM {time:.8g} {0}" + migration_cmd
'''

def set_initial_growth_rates(config, activity, pop_index):
    factor = 1 / (4 * config["population_size"])  # scaling factor for rates
    growth_rate_cmd = ""

    if "default_growth_rate" in config:
        rate = config["default_growth_rate"] * factor
        growth_rate_cmd = f" -G {rate:.8g}"

    # Initial growth rates
    for pop in config.get("populations", []):
        pop_name = pop["name"]
        if "growth_rate" not in pop or not activity[pop_name]:
            continue

        rate = pop["growth_rate"] * factor
        growth_rate_cmd += f" -g {pop_index[pop_name]} {rate:.8g}"

    return growth_rate_cmd


def restore_growth_rates(config, activity, pop_index, time, population=None):
    factor = 1 / (4 * config["population_size"])  # scaling factor for rates

    if population:
        assert activity[population]
        for event in reversed(config.get("demography_events", [])):
            event_time = event["time"] * factor
            if event["type"] != "growth_rate_change" or event_time >= time:
                continue

            if "population" not in event or event["population"] == "all" or event["population"] == population:
                rate = event["rate"] * factor
                return f" -eg {time:.8g} {rate:.8g}"

        all_growth_rates = {pop["name"]: pop.get("growth_rate", 0) for pop in config.get("populations")}
        rate = all_growth_rates[population]
        rate = (rate if rate != 0 else config.get("default_growth_rate", 0)) * factor
        
        if rate != 0:
            return f" -eg {time:.8g} {pop_index[population]} {rate:.8g}"
        return ""

    growth_rate_cmd = ""
    changed_growth_rates = set()  # track populations already updated at given time

    # Handle past demography events before the given time
    for event in reversed(config.get("demography_events", [])):
        event_time = event["time"] * factor
        if event["type"] != "growth_rate_change" or event_time >= time:
            continue

        # Global growth rate change for all populations
        if "population" not in event or event["population"] == "all":
            rate = event["rate"] * factor
            return f" -eG {time:.8g} {rate:.8g}" + growth_rate_cmd # global change overrides everything else

        # Population-specific growth rate change
        pop = event["population"]
        if activity[pop] and pop not in changed_growth_rates:
            rate = event["rate"] * factor
            growth_rate_cmd += f" -eg {time:.8g} {pop_index[pop]} {rate:.8g}"
            changed_growth_rates.add(pop)

    if "default_growth_rate" in config:
        rate = config["default_growth_rate"] * factor
        growth_rate_cmd = f" -eG {time:.8g} {rate:.8g}" + growth_rate_cmd

    # Initial growth rates (or those not overridden by events)
    for pop in config.get("populations", []):
        pop_name = pop["name"]
        if "growth_rate" not in pop or not activity[pop_name] or pop_name in changed_growth_rates:
            continue

        rate = pop["growth_rate"] * factor
        growth_rate_cmd += f" -eg {time:.8g} {pop_index[pop_name]} {rate:.8g}"

    return growth_rate_cmd

def set_initial_population_sizes(config, activity, pop_index):
    population_size = config["population_size"]
    subpop_sizes = ""

    for pop in config.get("populations", []):
        pop_name = pop["name"]
        if activity[pop_name] and "initial_size" in pop and pop['initial_size'] != population_size:
                subpop_sizes += f" -n {pop_index[pop_name]} {int(pop['initial_size']) / population_size}"
    
    return subpop_sizes


def set_ancestral_size_at_split(config, activity, pop_index, time, population):
    assert activity[population], f"Can't change the size of an inactive population: {population}"
    for event in reversed(config.get("demography_events", [])):
        event_time = event["time"]/ (4 * config["population_size"])
        if event["type"] != "size_change" or event_time >= time:
            continue

        if "population" not in event or event["population"] == "all":
            return ""
        # No need to check event["population"] == population here,
        # because a pop_size change is only triggered when a subpopulation
        # transitions from inactive to active.
    
    population_size = config["population_size"]
    all_pop_sizes = {pop["name"]: pop.get("initial_size", population_size) for pop in config.get("populations")}
    initial_size = all_pop_sizes[population]

    if initial_size != population_size:
        return f" -en {time:.8g} {pop_index[population]} {initial_size/ population_size}"
    return ""


def set_population_sample_sizes(config, activity, pop_index):
    sample_sizes = ""

    for pop in config.get("populations", []):
        pop_name = pop["name"]

        sample_size = pop.get("sample_size", 0) 
        assert (not activity[pop_name] and sample_size == 0) or activity[pop_name], "You can't sample from an inactive population"
        
        if not activity[pop_name]:
            continue
        
        sample_sizes += f" {sample_size}"

    return sample_sizes


def get_initial_population_activity(config):
    pop_names = [pop["name"] for pop in config.get("populations", [])]
    activity = {pid: True for pid in pop_names}  # Default: aktiv
    
    for event in config.get("demography_events", []):
        type = event["type"]
        
        if type == "population_split":
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


def check_events_sorted(config):
    events = config.get("demography_events", [])
    times = [event["time"] for event in events]
    
    return times == sorted(times)


def build_ms_command(config):

    ms_path = config.get("ms_path", "ms")
    total_sample_size = sum(p.get("sample_size", 0) for p in config.get("populations", [])) \
                        if "populations" in config else int(config["sample_size"])
    populations = config.get("populations", [])
    num_populations = 0
    population_size = int(config["population_size"])
    num_replicates = int(config.get("num_replicates", 1))
    mut_rate = float(config["mutation_rate"])
    seq_length = int(config["sequence_length"])

    theta = 4 * population_size * mut_rate * seq_length
    cmd = f"ms {total_sample_size} {num_replicates} -t {theta:.8g}"

    # Tree output
    if config.get("save_trees", True):
        cmd += " -T"

    recomb_rate = float(config.get("recombination_rate", 0)) * seq_length
    rho = 4 * population_size * recomb_rate

    has_gene_conv = "gene_conversion_tract_length" in config and "gene_conversion_rate" in config

    if has_gene_conv or rho > 0:
        cmd += f" -r {rho:.8g} {seq_length}"

    if has_gene_conv:
        gen_rate = float(config["gene_conversion_rate"])
        gen_tract_length = float(config["gene_conversion_tract_length"])
        f = gen_rate / recomb_rate if recomb_rate else 4 * population_size * gen_rate
        cmd += f" -c {f:.8g} {gen_tract_length}"

    if "populations" in config:
        activity = get_initial_population_activity(config)
        num_populations, pop_index = get_initial_population_indices(config, activity)

        cmd += f" -I {num_populations}"
        cmd += set_population_sample_sizes(config, activity, pop_index)
        cmd += set_initial_migration_rates(config, activity, pop_index)
        cmd += set_initial_population_sizes(config, activity, pop_index)
        cmd += set_initial_growth_rates(config, activity, pop_index)

    if not check_events_sorted(config):
        raise ValueError("Demography events are not sorted by time!")

    factor = 1/ (4 * population_size)

    for event in config.get("demography_events", []):
        t = event["time"] * factor
        type = event["type"]

        if type == "size_change":
            if "population" not in event or event["population"] == "all":
                cmd += f" -eN {t:.8g} {int(event['size']) / population_size}"
                cmd += restore_growth_rates(config, activity, pop_index, t)
            else:
                assert activity[event['population']], "You can't change the size of an inactive population"

                cmd += f" -en {t:.8g} {pop_index[event['population']]} {int(event['size']) / population_size}"
                cmd += restore_growth_rates(config, activity, pop_index, t, event['population'])

        elif type == "migration_rate_change":
            if "dest" not in event or "source" not in event:
                if "dest" in event or "source" in event:
                        raise ValueError("Event must define either both 'dest' and 'source' or neither of them.")
                
                rate = event["rate"] * factor
                for pop in populations:
                    dest = pop["name"]
                    for pop in populations:
                        source = pop["name"]
                        if dest == source or not activity[dest] or not activity[source]:
                            continue
                
                    cmd += f" -em {t:.8g} {pop_index[dest]} {pop_index[source]} {rate:.8g}"

            else:
                dest = event['dest']
                source = event['source']
                if not (activity[dest] and activity[source]):
                    raise ValueError("You can't migrate to or from an inactive population")

                rate = float(event['rate']) * factor
                cmd += f" -em {t:.8g} {pop_index[dest]} {pop_index[source]} {rate:.8g}"

        elif type == "growth_rate_change":
            if "population" not in event or event["population"] == "all":
                cmd += f" -eG {t:.8g} {float(event['rate']) / (4 * population_size)}"
            else:
                pop = event['population']
                if not activity[pop]:
                    raise ValueError("You cant change the growth rate of an inactive population")

                cmd += f" -eg {t:.8g} {pop_index[event['population']]} {float(event['rate']) / (4 * population_size)}"
        elif type == "population_split":
            derived = event["derived"]
            ancestral = event["ancestral"]
            
            assert not activity[ancestral], "Ancestral population should initially be inactive"
            assert all(activity[d] for d in derived), "Derived populations should initially be active"

            pop_index[ancestral] = pop_index[derived[0]]
            pop_index.pop(derived[0])

            # Mark the ancestral population as active
            activity[ancestral] = True
            # Mark all derived populations as inactive
            for d in derived:
                activity[d] = False

            # Setting the population size of the ancestral population
            cmd += set_ancestral_size_at_split(config, activity, pop_index, t, ancestral)

            for i in range(1, len(derived)):
                cmd += f" -ej {t:.8g} {pop_index[derived[i]]} {pop_index[ancestral]}"

            # Setting the growth rate of the ancestral population
            cmd += restore_growth_rates(config, activity, pop_index, t, ancestral)

            # Reinitialize migration rates according to the current activity configuration
            # cmd += restore_migration_rates(config, activity, pop_index, t)

        elif type == "population_admixture":
            derived = event["derived"]
            ancestral = event["ancestral"]
            proportions = event["proportions"]

            if not len(ancestral) == len(proportions):
                raise ValueError("The number of proportions must match the number of ancestral populations")
            
            if not sum(proportions) == 1:
                raise ValueError("Proportions should add up to one")

            assert activity[derived], "Derived population should initially be active"
            assert all(activity[a] for a in ancestral), f"Ancestral populations should be active"

            #TODO: Check whether the ancestral populations need to be active prior to the admixture event

            activity[derived] = False

            for i in range(0, len(ancestral)):
                num_populations += 1

                cmd += f" -es {t} {pop_index[derived]} {1 - proportions[i]}"
                cmd += f" -ej {t} {num_populations} {pop_index[ancestral[i]]}"

            # Reinitialize migration rates according to the current activity configuration
            # cmd += restore_migration_rates(config, activity, pop_index, t)

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
