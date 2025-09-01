import os
import subprocess
from checker import initial_population_activity, check_events_sorted, get_populations

def set_migration_rates(config, activity, pop_index, population_size, time=0):
    mig_cmd = ""
    
    for mig in config["migration_rates"]:
        dest = mig['dest']
        source = mig['source']
        if not (activity[dest] and activity[source]):
            continue

        mig_cmd_part = f" {pop_index[dest]} {pop_index[source]} {mig['rate'] / (4 * population_size)}"
        if time == 0:
            mig_cmd += f" -m" + mig_cmd_part
        else:
            mig_cmd += f" -em {time}" + mig_cmd_part

    return mig_cmd

def build_ms_command(config):

    ms_path = config.get("ms_path", "ms")
    total_sample_size = sum(p.get("sample_size", 0) for p in config.get("populations", [])) \
                        if "populations" in config else int(config["sample_size"])
    populations = get_populations(config)
    num_populations = len(populations)
    population_size = int(config["population_size"])
    num_replicates = int(config.get("num_replicates", 1))
    mut_rate = float(config["mutation_rate"])
    seq_length = int(config["sequence_length"])

    theta = 4 * population_size * mut_rate * seq_length
    cmd = f"ms {total_sample_size} {num_replicates} -t {theta}"

    #Tree output
    if config.get("save_trees", True):
        cmd += " -T"

    #Recombination
    recomb_rate = float(config.get("recombination_rate", 0)) * seq_length
    rho = 4 * population_size * recomb_rate
    cmd += f" -r {rho} {seq_length}"

    #Gene conversion
    if "gene_conversion_tract_length" in config and "gene_conversion_rate" in config:
        gen_rate = float(config["gene_conversion_rate"])
        f = gen_rate/ recomb_rate if recomb_rate != 0 else 4 * population_size * gen_rate
        gen_tract_length = float(config["gene_conversion_tract_length"])
        cmd += f" -c {f} {gen_tract_length}"

    activity = initial_population_activity(config)

    pop_index = {}

    if "populations" in config:
        cmd += f" -I {num_populations}"
        subpop_sizes = ""
        growth_rates = ""

        for i, pop in enumerate(populations):
            pop_name = pop["name"]
            pop_index[pop_name] = i+1

            sample_size = pop.get("sample_size", 0) 
            assert (not activity[pop_name] and sample_size == 0) or activity[pop_name], "You can't sample from an inactive population"
            
            cmd += f" {sample_size}"

            subpop_sizes += f" -n {pop_index[pop_name]} {int(pop['initial_size']) / population_size}"
            if "growth_rate" in pop and activity[pop_name]:
                growth_rates += f" -g {pop_index[pop_name]} {float(pop['growth_rate']) / (4 * population_size)}"
        cmd += subpop_sizes + growth_rates

        cmd += set_migration_rates(config, activity, pop_index, population_size)

    assert check_events_sorted(config), "Demography events are not sorted by time!"

    for event in config.get("demography_events", []):
        t = event["time"] / (4 * population_size)
        type = event["type"]
        if type == "size_change":
            if event["population"] == "all":
                cmd += f" -eN {t} {int(event['size']) / population_size}"
                #TODO: Vorsicht bei den populationen die nicht aktiv sind... was dann?

            else:
                assert activity[event['population']], "You cant change the size of an inactive population"

                cmd += f" -en {t} {pop_index[event['population']]} {int(event['size']) / population_size}"
        elif type == "migration_rate_change":
            dest = event['dest']
            source = event['source']
            assert activity[dest] and activity[source], "You can't migrate to or from an inactive population"

            cmd += f" -em {t} {pop_index[dest]} {pop_index[source]} {float(event['rate']) / (4 * population_size)}"
        elif type == "growth_rate_change":
            if event["population"] == "all":
                cmd += f" -eG {t} {float(event['rate']) / (4 * population_size)}"
                for pop in populations:
                    pop_name = pop["name"]
                    if not activity[pop_name]:
                        cmd += f" -eg {t} {pop_index[pop_name]} {0}"
            else:
                assert activity[event['population']], "You cant change the growth rate of an inactive population"

                cmd += f" -eg {t} {pop_index[event['population']]} {float(event['rate']) / (4 * population_size)}"
        elif type == "population_split":
            derived = event["derived"]
            ancestral = event["ancestral"]
            
            assert len(derived) == 2, "Only two derived populations are allowed"
            assert not activity[ancestral], "Ancestral population should initially be inactive"
            assert all(activity[d] for d in derived), "Derived populations should initially be active"

            pop_index[ancestral] = pop_index[derived[0]]
            pop_index.pop(derived[0])

            # Mark the ancestral population as active
            activity[ancestral] = True
            # Mark all derived populations as inactive
            for d in derived:
                activity[d] = False

            cmd += f" -ej {t} {pop_index[derived[1]]} {pop_index[ancestral]}"

            # Setting the growth rate of the ancestral population
            growth_rates = {pop["name"]: pop.get("growth_rate", 0) for pop in config.get("populations")}
            cmd += f" -eg {t} {pop_index[ancestral]} {growth_rates.get(ancestral)/ (4 * population_size)}"

            # Reset all migration rates to 0 at time t
            cmd += f" -eM {t} {0}"
            # Reinitialize migration rates according to the current activity configuration
            cmd += set_migration_rates(config, activity, pop_index, population_size, t)

            #TODO: Schaue dir nochmal den Lebenszyklus an!!!

        elif type == "population_admixture":
            #Problem: Admixture is very different from msprime!!
            derived = event["derived"]
            ancestral = event["ancestral"]
            proportions = event["proportions"]
            assert len(ancestral) == 2, "Only two ancestral populations are allowed"
            assert len(proportions) == 2, "Proportions should be two values"
            assert sum(proportions) == 1, "Proportions should add up to one"
            assert activity[derived], "Derived population should initially be active"
            assert all(not activity[a] for a in ancestral), f"Ancestral populations should initially be inactive"

            pop_index[ancestral[0]] = pop_index[derived]
            pop_index.pop(derived)
            num_populations += 1
            pop_index[ancestral[1]] = num_populations

            activity[derived] = False
            for a in ancestral:
                activity[a] = True

            cmd += f" -es {t} {pop_index[ancestral[0]]} {proportions[0]}"

            # Setting the growth rates of the ancestral populations
            growth_rates = {pop["name"]: pop.get("growth_rate", 0) for pop in config.get("populations")}
            for a in ancestral:
                cmd += f" -eg {t} {pop_index[a]} {growth_rates.get(a)/ (4 * population_size)}"
            
            # Reset all migration rates to 0 at time t
            cmd += f" -eM {t} {0}"
            # Reinitialize migration rates according to the current activity configuration
            cmd += set_migration_rates(config, activity, pop_index, population_size, t)

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
