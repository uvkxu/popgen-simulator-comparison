import os
import subprocess
from checker import initial_population_activity, check_events_sorted

def set_migration_rates(config, activity, pop_index, time=0):
    """
    Builds a migration command string for ms based on the configuration, 
    active populations, population indices, and the specified time.

    Args:
        config (dict): Simulation configuration containing migration rates and demography events.
        activity (dict): Dictionary mapping population names to boolean (active or not).
        pop_index (dict): Mapping of population names to ms population indices.
        time (int, optional): Time at which the migration rate applies (default: 0).

    Returns:
        str: The ms migration command string.
    """
    factor = 1 / (4 * config["population_size"])  # scaling factor for rates
    migration_cmd = ""
    changed_migrations = set()  # Keep track of migrations already set

    # Handle migration rate changes from demography events before the given time
    if time != 0:
        for event in reversed(config.get("demography_events", [])):
            if event["type"] != "migration_rate_change" or event["time"] >= time:
                continue

            dest, source = event["dest"], event["source"]
            if activity[dest] and activity[source] and (dest, source) not in changed_migrations:
                # Apply the event-specific migration rate
                rate = event["rate"] * factor
                migration_cmd += f" -em {time} {pop_index[dest]} {pop_index[source]} {rate:.8g}"
                changed_migrations.add((dest, source))

    # Handle base migration rates (skip those already overridden by events
    for mig in config.get("migration_rates", []):
        dest, source = mig["dest"], mig["source"]
        if activity[dest] and activity[source] and (dest, source) not in changed_migrations:
            rate = mig["rate"] * factor
            flag = "-m" if time == 0 else f"-em {time}"
            migration_cmd += f" {flag} {pop_index[dest]} {pop_index[source]} {rate:.8g}"

    return migration_cmd


def set_growth_rates(config, activity, pop_index, time=0, population=None):
    """
    Build ms command string for setting population growth rates.

    Args:
        config (dict): Simulation configuration containing populations and demography events.
        activity (dict): Dictionary mapping population names to boolean (whether active or not).
        pop_index (dict): Mapping of population names to ms population indices.
        time (int, optional): Time at which the growth rate applies (default: 0).

    Returns:
        str: Growth rate command string for ms.
    """
    factor = 1 / (4 * config["population_size"])  # scaling factor for rates

    if population:
        for event in reversed(config.get("demography_events", [])):
            if event["type"] != "growth_rate_change" or event["time"] >= time:
                continue

            if "population" not in event or event["population"] == "all" or event["population"] == population:
                rate = event["rate"] * factor
                return f" -eg {time} {rate:.8g}"

        all_growth_rates = {pop["name"]: pop.get("growth_rate", 0) for pop in config.get("populations")}
        rate = all_growth_rates[population]
        rate = rate if rate != 0 else config.get("default_growth_rate", 0)
        
        if rate != 0:
            flag = "-g" if time == 0 else f"-eg {time}"
            return f" {flag} {pop_index[population]} {rate}"
        return ""

    growth_rate_cmd = ""
    changed_growth_rates = set()  # track populations already updated at given time

    # Handle past demography events before the given time
    if time != 0:
        for event in reversed(config.get("demography_events", [])):
            if event["type"] != "growth_rate_change" or event["time"] >= time:
                continue

            # Global growth rate change for all populations
            if "population" not in event or event["population"] == "all"
                rate = event["rate"] * factor
                growth_rate_cmd = f" -eG {time} {rate:.8g}" + growth_rate_cmd
                return growth_rate_cmd  # global change overrides everything else

            # Population-specific growth rate change
            pop = event["population"]
            if activity.get(pop, False) and pop not in changed_growth_rates:
                rate = event["rate"] * factor
                growth_rate_cmd += f" -eg {time} {pop_index[pop]} {rate:.8g}"
                changed_growth_rates.add(pop)

    if "default_growth_rate" in config:
        rate = config["default_growth_rate"] * factor
        flag = "-G" if time == 0 else f"-eG {time}"
        growth_rate_cmd = f" {flag} {rate:.8g}" + growth_rate_cmd

    # Initial growth rates (or those not overridden by events)
    for pop in config.get("populations", []):
        pop_name = pop["name"]
        if "growth_rate" not in pop or not activity[pop_name] or pop_name in changed_growth_rates:
            continue

        rate = pop["growth_rate"]
        flag = "-g" if time == 0 else f"-eg {time}"
        growth_rate_cmd += f" {flag} {pop_index[pop_name]} {rate:.8g}"

    return growth_rate_cmd


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

    # Recombination
    recomb_rate = float(config.get("recombination_rate", 0)) * seq_length
    rho = 4 * population_size * recomb_rate
    cmd += f" -r {rho:.8g} {seq_length}"

    # Gene conversion
    if "gene_conversion_tract_length" in config and "gene_conversion_rate" in config:
        gen_rate = float(config["gene_conversion_rate"])
        f = gen_rate/ recomb_rate if recomb_rate != 0 else 4 * population_size * gen_rate
        gen_tract_length = float(config["gene_conversion_tract_length"])
        cmd += f" -c {f:.8g} {gen_tract_length}"

    # TODO: Uniform migration rate

    activity = initial_population_activity(config)

    pop_index = {}

    if "populations" in config:
        sample_sizes = ""
        subpop_sizes = ""
        growth_rates = ""

        for pop in populations:
            pop_name = pop["name"]

            sample_size = pop.get("sample_size", 0) 
            assert (not activity[pop_name] and sample_size == 0) or activity[pop_name], "You can't sample from an inactive population"
            
            if not activity[pop_name]:
                continue
            
            num_populations += 1
            pop_index[pop_name] = num_populations
            sample_sizes += f" {sample_size}"

            if "initial_size" in pop and pop['initial_size'] != population_size:
                subpop_sizes += f" -n {pop_index[pop_name]} {int(pop['initial_size']) / population_size}"
        cmd += f" -I {num_populations}" + sample_sizes + subpop_sizes
        cmd += set_growth_rates(config, activity, pop_index)

        cmd += set_migration_rates(config, activity, pop_index)

    assert check_events_sorted(config), "Demography events are not sorted by time!"

    for event in config.get("demography_events", []):
        t = event["time"] / (4 * population_size)
        type = event["type"]
        if type == "size_change":
            if event["population"] == "all":
                cmd += f" -eN {t} {int(event['size']) / population_size}"
                cmd += set_growth_rates(config, activity, pop_index, t)
            else:
                assert activity[event['population']], "You cant change the size of an inactive population"

                cmd += f" -en {t} {pop_index[event['population']]} {int(event['size']) / population_size}"
                cmd += set_growth_rates(config, activity, pop_index, t, event['population'])
        elif type == "migration_rate_change":
            dest = event['dest']
            source = event['source']
            assert activity[dest] and activity[source], "You can't migrate to or from an inactive population"

            cmd += f" -em {t} {pop_index[dest]} {pop_index[source]} {float(event['rate']) / (4 * population_size)}"
        elif type == "growth_rate_change":
            if event["population"] == "all":
                cmd += f" -eG {t} {float(event['rate']) / (4 * population_size)}"
            else:
                assert activity[event['population']], "You cant change the growth rate of an inactive population"

                cmd += f" -eg {t} {pop_index[event['population']]} {float(event['rate']) / (4 * population_size)}"
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
            subpop_sizes = {pop["name"]: pop.get("initial_size", population_size) for pop in config.get("populations")}
            cmd += f" -en {t} {pop_index[ancestral]} {subpop_sizes.get(ancestral)/ population_size}"

            for i in range(1, len(derived)):
                cmd += f" -ej {t} {pop_index[derived[i]]} {pop_index[ancestral]}"

            # Setting the growth rate of the ancestral population
            growth_rates = {pop["name"]: pop.get("growth_rate", 0) for pop in config.get("populations")}
            cmd += f" -eg {t} {pop_index[ancestral]} {growth_rates.get(ancestral)/ (4 * population_size)}"

            # Reset all migration rates to 0 at time t
            cmd += f" -eM {t} {0}"
            # Reinitialize migration rates according to the current activity configuration
            cmd += set_migration_rates(config, activity, pop_index, t)

        elif type == "population_admixture":
            pass  # TODO: implement later

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
