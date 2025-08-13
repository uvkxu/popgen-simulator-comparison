import os
import subprocess

def build_ms_command(config):
    ms_path = config.get("ms_path", "ms")
    total_sample_size = sum(p["sample_size"] for p in config.get("populations", [])) \
                        if "populations" in config else int(config["sample_size"])
    population_size = int(config["population_size"])
    num_replicates = int(config.get("num_replicates", 1))
    mut_rate = float(config["mutation_rate"])
    seq_length = int(config["sequence_length"])

    theta = 4 * population_size * mut_rate * seq_length
    cmd = f"{ms_path} {total_sample_size} {num_replicates} -t {theta}"

    if "recombination_rate" in config:
        recomb_rate = float(config["recombination_rate"])
        rho = 4 * population_size * recomb_rate * seq_length
        cmd += f" -r {rho} {seq_length}"

    if "populations" in config:
        cmd += f" -I {len(config['populations'])}"
        subpop_sizes = ""
        growth_rates = ""
        for i, pop in enumerate(config["populations"]):
            cmd += f" {pop['sample_size']}"
            subpop_sizes += f" -n {i+1} {pop['initial_size'] / population_size}"
            if "growth_rate" in pop:
                growth_rates += f" -g {i+1} {pop['growth_rate'] / (4 * population_size)}"
        cmd += subpop_sizes + growth_rates

        if "migration_matrix" in config:
            for i, row in enumerate(config["migration_matrix"]):
                for j, rate in enumerate(row):
                    if i != j and rate > 0:
                        cmd += f" -m {i+1} {j+1} {rate / (4 * population_size)}"
        elif "migration_rates" in config:
            for mig in config["migration_rates"]:
                cmd += f" -m {int(mig['source'])+1} {int(mig['dest'])+1} {mig['rate'] / (4 * population_size)}"

    for event in config.get("demography_events", []):
        t = event["time"] / (4 * population_size)
        if event["type"] == "size_change":
            if event["population"] == "all":
                cmd += f" -eN {t} {event['size'] / population_size}"
            else:
                cmd += f" -en {t} {event['population']+1} {event['size'] / population_size}"
        elif event["type"] == "migration_rate_change":
            cmd += f" -em {t} {int(event['source'])+1} {int(event['dest'])+1} {event['rate'] / (4 * population_size)}"
        elif event["type"] == "growth_rate_change":
            if event["population"] == "all":
                cmd += f" -eG {t} {event['rate'] / (4 * population_size)}"
            else:
                cmd += f" -eg {t} {event['population']+1} {event['rate'] / (4 * population_size)}"

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
