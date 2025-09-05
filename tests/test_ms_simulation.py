import pytest
import re
from ms_runner import build_ms_command, set_migration_rates
from test_configs import make_config_single_pop, make_config_two_pops_with_migration, make_config_with_gene_conversion, make_population_split_event_config

def extract(cmd, flag):
    """
    Extract numbers following a given flag from a command string.

    Parameters
    ----------
    cmd : str
        The command string (e.g., 'ms ... -m 0 1 0.01 -m 1 2 0.02').
    flag : str
        The flag to look for (e.g., '-t', '-m').
    group_size : int or None, optional
        If specified, groups results into chunks of this size.
        Example: group_size=3 for '-m' (dest, source, rate).

    Returns
    -------
    list
        A list of floats, or a list of lists of floats if group_size is given
        or if the flag appears multiple times.
    """
    number_pattern = r"(?:\d+\.\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?"
    pattern = rf"{re.escape(flag)}((?:\s+{number_pattern})+)"
    matches = re.findall(pattern, cmd)
    if not matches:
        return []
    
    results = []
    for match in matches:
        numbers = [float(x) for x in match.strip().split()]
        results.append(numbers if len(numbers) > 1 else numbers[0])

    # Flatten if only one match
    if len(results) == 1:
        return results[0]
    return results

@pytest.mark.parametrize("config,expected", [
    (make_config_two_pops_with_migration(), "ms 10 1"), 
    (make_config_single_pop(), "ms 15 3"),
    (make_config_with_gene_conversion(), "ms 20 5"),
    (make_population_split_event_config(), "ms 35 20")
])
def test_total_sample_size_and_replicates(config, expected):
    cmd = build_ms_command(config)
    assert cmd.startswith(expected), "Command does not start with expected prefix.\n"


@pytest.mark.parametrize("config,expected_theta", [
    (make_config_two_pops_with_migration(), 0.04), 
    (make_config_single_pop(), 0.4),
    (make_config_with_gene_conversion(), 0.8),
    (make_population_split_event_config(), 0.08)
])
def test_mutation_flag(config, expected_theta):
    cmd = build_ms_command(config)
    found_theta = extract(cmd, "-t")
    assert found_theta == expected_theta, f"Mutation rate (-t) mismatch: expected {expected_theta}, got {found_theta}.\n"


@pytest.mark.parametrize("config,expected_rho,expected_nsites", [
    (make_config_two_pops_with_migration(), 40, 1000),
    (make_config_single_pop(), 0.04, 500),
    (make_config_with_gene_conversion(), 0, 100)
])
def test_recombination_flag(config, expected_rho, expected_nsites):
    cmd = build_ms_command(config)
    found_rho, found_nsites = extract(cmd, "-r")
    assert found_rho == expected_rho, f"Recombination rate (-r) mismatch: expected {expected_rho}, got {found_rho}.\n"
    assert found_nsites == expected_nsites, f"Number of sites (-r) mismatch: expected {expected_nsites}, got {found_nsites}.\n"


@pytest.mark.parametrize("config,expected_rate,expected_tract_length",[
    (make_config_single_pop(), [], []),
    (make_config_with_gene_conversion(), 0.4, 1),
    (make_config_two_pops_with_migration(), 0.0003, 3)
])
def test_gene_conversion_flag(config, expected_rate, expected_tract_length):
    cmd = build_ms_command(config)
    found = extract(cmd, "-c")

    if not found:
        found_rate, found_tract_length = [], []
    else:
        found_rate, found_tract_length = found

    assert found_rate == expected_rate, f"Gene conversion rate mismatch: expected {expected_rate}, got {found_rate}"
    assert found_tract_length == expected_tract_length, f"Gene conversion tract length mismatch: expected {expected_tract_length}, got {found_tract_length}"



@pytest.mark.parametrize("config,tree_flag_exists", [
    (make_config_two_pops_with_migration(), False),
    (make_config_single_pop(), True),
    (make_config_with_gene_conversion(), True)
])
def test_tree_flag(config, tree_flag_exists):
    cmd = build_ms_command(config)
    if tree_flag_exists:
        assert "-T" in cmd, "Expected tree flag (-T) to be present in command, but it was missing.\n"
    else:
        assert "-T" not in cmd, "Expected tree flag (-T) to be absent in command, but it was found.\n"


@pytest.mark.parametrize("config,expected_num_pops_and_samples,expected_sizes,expected_growth_rates", [
    (make_config_two_pops_with_migration(), [2, 5, 5], [2, 0.99], [1, 0.00001]),
    (make_config_single_pop(), [], [], []),
    (make_population_split_event_config(), [4, 5, 5, 10, 15], [[1, 0.2], [2, 0.2], [4, 0.4]], [[2, 3e-6], [4, 3.5e-7]]),
])
def test_population_initialization(config, expected_num_pops_and_samples, expected_sizes, expected_growth_rates):
    cmd = build_ms_command(config)
    found_num_pops_and_sample = extract(cmd, "-I")
    found_sizes = extract(cmd, "-n")
    found_growth_rates = extract(cmd, "-g")

    assert found_num_pops_and_sample == expected_num_pops_and_samples, \
        f"Population structure (-I) mismatch. Expected {expected_num_pops_and_samples}, got {found_num_pops_and_sample}"

    assert found_sizes == expected_sizes, \
        f"Initial sizes (-n) mismatch. Expected {expected_sizes}, got {found_sizes}"

    assert found_growth_rates == expected_growth_rates, \
        f"Growth rates (-g) mismatch. Expected {expected_growth_rates}, got {found_growth_rates}"


@pytest.mark.parametrize("config,activity,pop_index,time,expected_migration", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 0, [[2, 1, 2.5e-06], [1, 2, 2.5e-06]]),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 99, [[2, 1, 2.5e-06], [1, 2, 2.5e-06]]),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 100, [[2, 1, 2.5e-06], [1, 2, 2.5e-06]]),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 101, [[2, 1, 5e-05], [1, 2, 2.5e-06]]),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, 0, []),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, 80, []),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, 100, []),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, 105, []),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, 0, []),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, 17, []),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, 100, []),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, 300, []),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, 0, []),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, 50, []),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, 100, []),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, 140, []),
    (make_config_single_pop(), {}, {}, 0, []),
])
def test_set_migration_rates(config, activity, pop_index, time, expected_migration):
    mig_cmd = set_migration_rates(config, activity, pop_index, time)
    if time == 0:
        found_migration = extract(mig_cmd, "-m")
    else:
        found_migration = extract(mig_cmd, f"-em {time}")
    assert found_migration == expected_migration, f"Migration rates mismatch at time {time} (expected {expected_migration}, got {found_migration})"


@pytest.mark.parametrize("config,time,expected_migration", [
    (make_config_two_pops_with_migration(), 0.025, [2, 1, 5e-5]),
    (make_population_split_event_config(), 0.0025, [1, 2, 2.5e-6]),
    (make_population_split_event_config(), 0.009, [[4, 1, 5e-6], [1, 4, 4.5e-6]])
])
def test_migration_rate_change(config, time, expected_migration):
    cmd = build_ms_command(config)
    found_migration = extract(cmd, f"-em {time}")

    assert found_migration == expected_migration,  \
        f"Migration rates mismatch at time {time} (expected {expected_migration}, got {found_migration})"


@pytest.mark.parametrize("config,time,expected_growth_rate_change", [
    (make_config_two_pops_with_migration(), 0.05, [2, 7.5e-7])
])
def test_growth_rate_change(config, time, expected_growth_rate_change):
    cmd = build_ms_command(config)
    found_growth_rate_change = extract(cmd, f"-eg {time}")

    assert found_growth_rate_change == expected_growth_rate_change, \
        f"Growth rate change mismatch at time {time} (expected {expected_growth_rate_change}, got {found_growth_rate_change})"


@pytest.mark.parametrize("config,time,expected_size_change", [
    (make_config_two_pops_with_migration(), 0.055, [1, 0.9])
])
def test_size_change(config, time, expected_size_change):
    cmd = build_ms_command(config)
    found_size_change = extract(cmd, f"-en {time}")

    assert found_size_change == expected_size_change, \
        f"Size change mismatch at time {time} (expected {expected_pop_size}, got {found_pop_size})"



@pytest.mark.parametrize("config,time,expected_splits,expected_pop_size,expected_growth_rate,expected_migration", [
    (make_population_split_event_config(), 0.005, [[2, 1],[3, 1]], [1, 1.0], [1, 5e-8], [[4, 1, 4e-7],[1, 4, 1e-6]])
])
def test_population_split_event(config, time, expected_splits, expected_pop_size, expected_growth_rate, expected_migration):
    cmd = build_ms_command(config)
    found_splits = extract(cmd, f"-ej {time}")
    found_pop_size = extract(cmd, f"-en {time}")
    found_growth_rate = extract(cmd, f"-eg {time}")
    found_migration = extract(cmd, f"-em {time}")

    assert found_splits == expected_splits, \
        f"Splits mismatch: expected {expected_splits}, but got {found_splits}"

    assert found_pop_size == expected_pop_size, \
        f"Wrong ancestral population size at split (expected {expected_pop_size}, got {found_pop_size})"

    assert found_growth_rate == expected_growth_rate, \
        f"Wrong growth rate at split (expected {expected_growth_rate}, got {found_growth_rate})"

    assert f"-eM {time} 0" in cmd, \
        f"Missing reset of migration rates at time {time}"

    assert found_migration == expected_migration, \
        f"Migration rates mismatch at split: expected {expected_migration}, but got {found_migration}"

    assert cmd.find(f"-eM {time} 0") < cmd.find(f"-em {time}"), \
        f"Reset of migration (-eM {time} 0) does not occur before reinitalization (-em {time})"


'''
def test_admixture_event_in_command():
    config = make_minimal_config()
    config["demography_events"] = [
        {
            "time": 200,
            "type": "population_admixture",
            "derived": "pop1",
            "ancestral": ["pop2", "pop3"],
            "proportions": [0.25, 0.75]
        }
    ]
    config["populations"].append({"name": "pop3", "sample_size": 0, "initial_size": 1000})
    
    cmd = build_ms_command(config)
    assert "-es" in cmd and "-ej" in cmd, "Admixture not correctly translated into ms flags"
'''
