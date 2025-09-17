import pytest
import re
from simulations.ms_runner import *
from test_configs import *

def extract_by_flag(cmd, flag, expected_length=0):
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
        return [[] for _ in range(expected_length)]
    
    results = []
    for match in matches:
        numbers = [float(x) for x in match.strip().split()]
        results.append(numbers if len(numbers) > 1 else numbers[0])

    # Flatten if only one match
    if len(results) == 1:
        return results[0]
    return results


def extract_default_migration_rate(cmd):
    parts = cmd.strip().split()
    
    # Check if "-I" is present at all
    if "-I" not in parts:
        return 0
    
    i = parts.index("-I")
    num_pops = int(parts[i + 1])
    expected_len = 2 + num_pops  # "-I", num_pops, and the sample sizes
    
    # Take everything starting from "-I"
    subparts = parts[i:]
    
    # Case 1: exactly "-I", num_pops, and sample sizes → no migration rate given
    if len(subparts) == expected_len:
        return 0
    
    # Case 2: one more element → that's the migration rate
    if len(subparts) > expected_len:
        try:
            return float(subparts[expected_len])
        except ValueError:
            # If the element cannot be parsed as a number → assume no migration rate
            return 0
    
    # Anything else is unexpected
    assert False, f"Unexpected format for -I argument: {' '.join(subparts)}"

# ---------------------------------------
# Test of num_replicates and sample size
# ---------------------------------------

@pytest.mark.parametrize("config,expected", [
    (make_config_two_pops_with_migration(), "ms 10 1"), 
    (make_config_single_pop(), "ms 15 3"),
    (make_config_with_gene_conversion(), "ms 20 5"),
    (make_population_split_event_config(), "ms 35 20")
])
def test_total_sample_size_and_replicates(config, expected):
    cmd = build_ms_command(config)
    assert cmd.startswith(expected), "Command does not start with expected prefix.\n"


# -----------------------------
# Test of mutation
# -----------------------------


@pytest.mark.parametrize("config,expected_theta", [
    (make_config_two_pops_with_migration(), 0.04), 
    (make_config_single_pop(), 0.4),
    (make_config_with_gene_conversion(), 0.8),
    (make_population_split_event_config(), 0.08)
])
def test_mutation_flag(config, expected_theta):
    cmd = build_ms_command(config)
    found_theta = extract_by_flag(cmd, "-t")
    assert found_theta == expected_theta, f"Mutation rate (-t) mismatch: expected {expected_theta}, got {found_theta}.\n"


# -----------------------------
# Test of recombination
# -----------------------------


@pytest.mark.parametrize("config,expected_rho,expected_nsites", [
    (make_config_two_pops_with_migration(), 40, 1000),
    (make_config_single_pop(), 0.04, 500),
    (make_config_with_gene_conversion(), 0, 100),
    (make_population_split_event_config(), [], [])
])
def test_set_recombination(config, expected_rho, expected_nsites):
    cmd = set_recombination(config)
    found_rho, found_nsites = extract_by_flag(cmd, "-r", 2)
    
    assert found_rho == expected_rho, f"Recombination rate (-r) mismatch: expected {expected_rho}, got {found_rho}.\n"
    assert found_nsites == expected_nsites, f"Number of sites (-r) mismatch: expected {expected_nsites}, got {found_nsites}.\n"


@pytest.mark.parametrize("config,expect_flag", [
    (make_config_two_pops_with_migration(), True),
    (make_config_single_pop(), True),
    (make_config_with_gene_conversion(), True),
    (make_population_split_event_config(), False)
])
def test_recombination_flag_in_command(config, expect_flag):
    cmd = build_ms_command(config)
    if expect_flag:
        assert "-r" in cmd
    else:
        assert "-r" not in cmd


# -----------------------------
# Test of gene conversion
# -----------------------------


@pytest.mark.parametrize("config,expected_rate,expected_tract_length",[
    (make_config_single_pop(), [], []),
    (make_config_with_gene_conversion(), 0.4, 1),
    (make_config_two_pops_with_migration(), 0.3, 3),
    (make_population_split_event_config(), [], [])
])
def test_set_gene_conversion(config, expected_rate, expected_tract_length):
    cmd = build_ms_command(config)
    found_rate, found_tract_length = extract_by_flag(cmd, "-c", 2)

    assert found_rate == expected_rate, f"Gene conversion rate mismatch: expected {expected_rate}, got {found_rate}"
    assert found_tract_length == expected_tract_length, f"Gene conversion tract length mismatch: expected {expected_tract_length}, got {found_tract_length}"


@pytest.mark.parametrize("config,expect_flag",[
    (make_config_single_pop(), False),
    (make_config_with_gene_conversion(), True),
    (make_config_two_pops_with_migration(), True),
    (make_population_split_event_config(), False)
])
def test_gene_conversion_flag_in_command(config, expect_flag):
    cmd = build_ms_command(config)

    if expect_flag:
        assert "-c" in cmd
    else:
        assert "-c" not in cmd


@pytest.mark.parametrize("config,expect_conversion", [
    (make_config_single_pop(), 0),
    (make_config_with_gene_conversion(), 1),
    (make_config_two_pops_with_migration(), 1),
    (make_population_split_event_config(), 0),
    ({"gene_conversion_rate": 3e-6}, -1),
    ({"gene_conversion_tract_length": 3}, -1)
])
def test_has_gene_conversion(config, expect_conversion):
    if expect_conversion == -1:
        with pytest.raises(ValueError):
            has_gene_conversion(config)
    else:
        found = has_gene_conversion(config)
        if expect_conversion == 1:
            assert found, "Expected gene conversion, but got False"
        elif expect_conversion == 0:
            assert not found, "Expected no gene conversion, but got True"

# -----------------------------
# Test of tree flag
# -----------------------------

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

# -------------------------------------------------
# Test of initialization of activity & indices
# -------------------------------------------------

@pytest.mark.parametrize("config,expected_activity", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}),
    (make_population_split_event_config(), {"pop1": True, "pop2": True, "pop3": True, "pop4": False, "pop5": True}),
    (make_config_single_pop(), {})
]
)
def test_get_initial_population_activity(config, expected_activity):
    assert get_initial_population_activity(config) == expected_activity


@pytest.mark.parametrize("config,activity,expected_indices", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop2": 1}),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1}),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {}),
    (make_population_split_event_config(), {"pop1": True, "pop2": True, "pop3": True, "pop4": False, "pop5": True}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop5": 4}),
    (make_config_single_pop(), {}, {})
]
)
def test_get_initial_population_indices(config, activity, expected_indices):
    found_num_populations, found_indices = get_initial_population_indices(config, activity)
    assert found_indices == expected_indices


@pytest.mark.parametrize("config,activity,expected_num_populations", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, 2),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, 1),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, 1),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, 0),
    (make_population_split_event_config(), {"pop1": True, "pop2": True, "pop3": True, "pop4": False, "pop5": True}, 4),
    (make_config_single_pop(), {}, 0)
]
)
def test_initial_num_populations(config, activity, expected_num_populations):
    found_num_populations, found_indices = get_initial_population_indices(config, activity)
    assert found_num_populations == expected_num_populations


# -------------------------------------------------
# Test of initialization of individual sample sizes
# -------------------------------------------------


@pytest.mark.parametrize("config,activity,pop_index,expected_sample_sizes", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, [5, 5]),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, -1),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, -1),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, -1),
    (make_population_split_event_config(), {"pop1": True, "pop2": True, "pop3": True, "pop4": False, "pop5": True}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop5": 4}, [5, 5, 10, 15]),
    (make_config_single_pop(), {}, {}, []),
])
def test_set_population_sample_sizes(config, activity, pop_index, expected_sample_sizes):
    if expected_sample_sizes == -1:
        with pytest.raises(ValueError):
            set_population_sample_sizes(config, activity, pop_index)
    else:
        sample_sizes_cmd = set_population_sample_sizes(config, activity, pop_index)
        found_sample_sizes = [float(x) for x in sample_sizes_cmd.strip().split()]
        
        assert found_sample_sizes == expected_sample_sizes, f"Expected sample sizes {expected_sample_sizes}, got {found_sample_sizes}"


@pytest.mark.parametrize("config,num_populations,expected_sample_sizes", [
    (make_config_two_pops_with_migration(), 2, [5, 5]),
    (make_population_split_event_config(), 4, [5, 5, 10, 15]),
    (make_config_single_pop(), 0, []),
])
def test_sample_sizes_in_command(config, num_populations, expected_sample_sizes):
    cmd = build_ms_command(config)
    found_sample_sizes = extract_by_flag(cmd, f"-I {num_populations}")[:num_populations]

    assert found_sample_sizes == expected_sample_sizes, f"Expected sample sizes {expected_sample_sizes}, but found {found_sample_sizes} in the command"


# ------------------------------------------
# Test of initialization of migration rates
# ------------------------------------------

@pytest.mark.parametrize("config,activity,pop_index,num_populations,expected_default_rate,expected_migration", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 2, 1.25e-5, [[2, 1, 2.5e-06], [1, 2, 2.5e-06]]),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, 2, 1.25e-5, []),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, 2, 1.25e-5, []),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, 2, 1.25e-5, []),
    (make_config_single_pop(), {}, {}, 0, 0, []),
])
def test_set_initial_migration_rates(config, activity, pop_index, num_populations, expected_default_rate, expected_migration):
    mig_cmd = set_initial_migration_rates(config, activity, pop_index, num_populations)
    parts = mig_cmd.strip().split()
    found_migration = extract_by_flag(mig_cmd, "-m")

    if not parts or parts[0] == "-m":
        assert expected_default_rate == 0, f"Found no uniform migration rate, but expected {expected_default_rate}"
    else:
        found_default_rate = float(parts[0])
        assert found_default_rate == expected_default_rate, f"Uniform migration rate is not set correctly, (expected {expected_default_rate}, got {found_default_rate})"

    assert found_migration == expected_migration, f"Migration rates mismatch (expected {expected_migration}, got {found_migration})"


@pytest.mark.parametrize("config,m_flag_exists", [
    (make_config_two_pops_with_migration(), True),
    (make_config_single_pop(), False),
])
def test_migration_flag_in_command(config, m_flag_exists):
    cmd = build_ms_command(config)

    if m_flag_exists:
        assert "-m" in cmd, "Expected '-m' in command, but it is missing"
    else:
        assert "-m" not in cmd, "Unexpected '-m' in command"


@pytest.mark.parametrize("config,expected_default_rate", [
    (make_config_two_pops_with_migration(), 1.25e-5),
    (make_config_single_pop(), 0),
])
def test_default_migration_rate_in_command(config, expected_default_rate):
    cmd = build_ms_command(config)
    found_default_rate = extract_default_migration_rate(cmd)

    assert found_default_rate == expected_default_rate,\
        f"Default migration rate is not added to the command correctly (expected {expected_default_rate}, got {found_default_rate})"


# ------------------------------------------
# Test of initialization of growth rates
# ------------------------------------------

@pytest.mark.parametrize("config,activity,pop_index,expected_default_rate", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 2.5e-7),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, 2.5e-7),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, 2.5e-7),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, 2.5e-7),
    (make_config_single_pop(), {}, {}, []),
])
def test_set_default_growth_rate(config, activity, pop_index, expected_default_rate):
    growth_cmd, _ = set_initial_growth_rates(config, activity, pop_index)
    found_default_rate = extract_by_flag(growth_cmd, "-G")
    assert found_default_rate == expected_default_rate, f"Expected default growth rate {expected_default_rate}, found {found_default_rate}"


@pytest.mark.parametrize("config,activity,pop_index,expected_individual_growth_rates", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, [1, 1e-05]),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, []),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, [1, 1e-05]),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, []),
    (make_config_single_pop(), {}, {}, []),
])
def test_set_individual_growth_rates(config, activity, pop_index, expected_individual_growth_rates):
    growth_cmd, _ = set_initial_growth_rates(config, activity, pop_index)
    found_individual_growth_rates = extract_by_flag(growth_cmd, "-g")
    assert found_individual_growth_rates == expected_individual_growth_rates, f"Expected individual growth rates {expected_individual_growth_rates}, found {found_individual_growth_rates}"


@pytest.mark.parametrize("config,activity,pop_index,expected_growth_rate_map", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, {"pop1": 1e-05, "pop2": 2.5e-7}),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, {"pop2": 2.5e-7}),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, {"pop1": 1e-05}),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, {}),
    (make_config_single_pop(), {}, {}, {"single": 0}),
])
def test_set_initial_growth_rates_of_map(config, activity, pop_index, expected_growth_rate_map):
    _, found_growth_rate_map = set_initial_growth_rates(config, activity, pop_index)

    assert found_growth_rate_map == expected_growth_rate_map

@pytest.mark.parametrize("config,G_flag_exists", [
    (make_config_two_pops_with_migration(), True),
    (make_config_single_pop(), False),
    (make_config_with_gene_conversion(), True)
])
def test_default_growth_flag_in_command(config, G_flag_exists):
    cmd = build_ms_command(config)

    if G_flag_exists:
        assert "-G" in cmd, "Expected '-G' in command, but it is missing"
    else:
        assert "-G" not in cmd, "Unexpected '-G' in command"


@pytest.mark.parametrize("config,g_flag_exists", [
    (make_config_two_pops_with_migration(), True),
    (make_config_single_pop(), False),
])
def test_single_growth_flag_in_command(config, g_flag_exists):
    cmd = build_ms_command(config)

    if g_flag_exists:
        assert "-g" in cmd, "Expected '-g' in command, but it is missing"
    else:
        assert "-g" not in cmd, "Unexpected '-g' in command"


# ------------------------------------------
# Test of initialization of population sizes
# ------------------------------------------

@pytest.mark.parametrize("config, activity, pop_index, expected_sizes", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, [2, 0.99]),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, [2, 0.99]),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, []),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, []),
    (make_config_single_pop(), {}, {}, []),
])
def test_set_initial_population_sizes(config, activity, pop_index, expected_sizes):
    sizes_cmd = set_initial_population_sizes(config, activity, pop_index)
    found_sizes = extract_by_flag(sizes_cmd, "-n")

    assert found_sizes == expected_sizes, f"Expected sizes {expected_sizes}, got {found_sizes}"


@pytest.mark.parametrize("config,n_flag_exists", [
    (make_config_two_pops_with_migration(), True),
    (make_config_single_pop(), False),
])
def test_sizes_flag_in_command(config, n_flag_exists):
    cmd = build_ms_command(config)

    if n_flag_exists:
        assert "-n" in cmd, "Expected '-n' in command, but it is missing"
    else:
        assert "-n" not in cmd, "Unexpected '-n' in command"


# -----------------------------
# Test of migration rate change
# -----------------------------

@pytest.mark.parametrize("event,config,activity,pop_index,time,expected_migration", [
    ({"type": "migration_rate_change", "time": 100, "source": "pop1", "dest": "pop2", "rate": 0.2}, make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 0.025, [2, 1, 5e-5]),
    ({"type": "migration_rate_change", "time": 180, "rate": 0.09}, make_population_split_event_config(), {"pop1": False, "pop2": False, "pop3": False, "pop4": True, "pop5": True}, {"pop4": 1, "pop5": 4}, 0.009, [[1, 4, 4.5e-6], [4, 1, 4.5e-6]]),
    ({"type": "migration_rate_change", "time": 180, "rate": 0.09}, make_population_split_event_config(), {"pop1": False, "pop2": False, "pop3": False, "pop4": False, "pop5": True}, {}, 0.009, []),
    ({"type": "migration_rate_change", "time": 100, "source": "pop1", "dest": "pop2", "rate": 0.2}, make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {}, 0, -1),
    ({"type": "migration_rate_change", "time": 100, "source": "pop1", "dest": "pop2", "rate": 0.2}, make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {}, 0, -1),
    ({"type": "migration_rate_change", "time": 100, "source": "pop1", "dest": "pop2", "rate": 0.2}, make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {}, 0, -1),
    ({"type": "migration_rate_change", "time": 180, "source": "pop4", "rate": 0.09}, make_config_two_pops_with_migration(), {}, {}, 0, -1),
    ({"type": "migration_rate_change", "time": 180, "dest": "pop4", "rate": 0.09}, make_config_two_pops_with_migration(), {}, {}, 0, -1)
])
def test_handle_migration_rate_change_event(event, config, activity, pop_index, time, expected_migration):
    if expected_migration == -1:
        with pytest.raises(ValueError):
            handle_migration_rate_change_event(event, config, activity, pop_index)
    else:
        cmd = handle_migration_rate_change_event(event, config, activity, pop_index)
        found_migration = extract_by_flag(cmd, f"-em {time}")

        assert found_migration == expected_migration,  \
            f"Migration rates mismatch at time {time} (expected {expected_migration}, got {found_migration})"


@pytest.mark.parametrize("config,migration_change_exists", [
    (make_config_two_pops_with_migration(), True),
    (make_population_split_event_config(), True),
    (make_config_with_gene_conversion(), False)
])
def test_migration_rate_change_flag_in_command(config,migration_change_exists):
    cmd = build_ms_command(config)

    if migration_change_exists:
        assert "-em" in cmd, "Expected migration rate change flag '-em' in the command, but it is missing."
    else:
        assert "-em" not in cmd, "Migration rate change flag '-em' found in the command, but it shouldn't be there."

# -----------------------------
# Test of growth rate change
# -----------------------------

@pytest.mark.parametrize("event,config,activity,pop_index,time,expected_growth_rate_change", [
    ({"type": "growth_rate_change", "time": 200, "population": "pop2", "rate": 0.003}, make_config_two_pops_with_migration(), {"pop2": True}, {"pop2": 2}, 0.05, [2, 7.5e-7]),
    ({"type": "growth_rate_change", "time": 200, "population": "pop2", "rate": 0.003}, make_config_two_pops_with_migration(), {"pop2": False}, {"pop2": 2}, 0.05, -1),
    ({"type": "growth_rate_change", "time": 800, "rate": 0.05}, make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 0.2, [[1, 1.25e-5], [2, 1.25e-5]]),
    ({"type": "growth_rate_change", "time": 800, "rate": 0.05}, make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, 0.2, [1, 1.25e-5]),
    ({"type": "growth_rate_change", "time": 800, "rate": 0.05}, make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, 0.2, [2, 1.25e-5]),
    ({"type": "growth_rate_change", "time": 800, "rate": 0.05}, make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, 0.2, []),
])
def test_handle_growth_rate_change_multiple_population_model(event, config, activity, pop_index, time, expected_growth_rate_change):

    if expected_growth_rate_change == -1:
        with pytest.raises(ValueError):
            handle_growth_rate_change_event(event, config, activity, pop_index, {})
    else:
        growth_change_cmd = handle_growth_rate_change_event(event, config, activity, pop_index, {})
        found_growth_rate_change = extract_by_flag(growth_change_cmd, f"-eg {time}")

        assert found_growth_rate_change == expected_growth_rate_change, \
            f"Growth rate change mismatch at time {time} (expected {expected_growth_rate_change}, got {found_growth_rate_change})"


@pytest.mark.parametrize("event,config,time,expected_growth_rate_change", [
    ({"type": "growth_rate_change", "time": 200, "rate": 0.003}, make_config_single_pop(), 0.025, 3.75e-7),
    ({"type": "growth_rate_change", "time": 1000, "rate": 0.07}, make_config_with_gene_conversion(), 0.025, 1.75e-6)
])
def test_handle_growth_rate_change_single_population_model(event, config, time, expected_growth_rate_change):
    cmd = handle_growth_rate_change_event(event, config, {}, {}, {})
    found_growth_rate_change = extract_by_flag(cmd, f"-eG {time}")

    assert found_growth_rate_change == expected_growth_rate_change, \
        f"Growth rate change mismatch at time {time} (expected {expected_growth_rate_change}, got {found_growth_rate_change})"


@pytest.mark.parametrize("config, time", [
    (make_config_two_pops_with_migration(), 0.05),
    (make_config_two_pops_with_migration(), 0.2),
    (make_population_split_event_with_size_change_config(), 0.004)
])
def test_growth_change_eg_flag_in_command(config, time):
    cmd = build_ms_command(config)

    assert f"-eg {time}" in cmd, f"Expected growth rate change flag '-eg {time}' in the command, but it is missing."


@pytest.mark.parametrize("config, time", [
    (make_config_single_pop(), 0.025),
    (make_config_with_gene_conversion(), 0.025)
])
def test_growth_change_eG_flag_in_command(config, time):
    cmd = build_ms_command(config)

    assert f"-eG {time}" in cmd, f"Expected growth rate change flag '-eG {time}' in the command, but it is missing."

# ----------------------------------------------------------------
# Test size change events & restoration of population growth rates
# ----------------------------------------------------------------

@pytest.mark.parametrize("config,time,growth_rate_map,expected_growth", [
    (make_config_single_pop(), 0, {"single": 0}, []),
    (make_config_single_pop(), 0.025, {"single": 3.75e-7}, 3.75e-7),
    (make_config_single_pop(), 0.5, {"single": 3.75e-7}, 3.75e-7)
])
def test_restore_growth_rates_single_population_model(config, time, growth_rate_map, expected_growth):
    growth_rate_cmd = restore_growth_rates(config, {}, {}, growth_rate_map, time)
    found_growth = extract_by_flag(growth_rate_cmd, f"-eG {time}")

    assert found_growth == expected_growth, f"Growth rate restoration of single population failed (expected {expected_growth}, got {found_growth})"


@pytest.mark.parametrize("config,activity,pop_index,growth_rate_map,time,expected_growth", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, {"pop1": 1e-5, "pop2": 2.5e-7}, 0.0375, [[1, 1e-5], [2, 2.5e-7]]),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, {"pop1": 1e-5, "pop2": 7.5e-7}, 0.075, [[1, 1e-5], [2, 7.5e-7]]),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, {"pop1": 1.25e-5, "pop2": 1.25e-5}, 0.2, [[1, 1.25e-5], [2, 1.25e-5]]),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, {"pop1": 1.25e-5, "pop2": 1.25e-5}, 0.25, [[1, 1.25e-5], [2, 1.25e-5]]),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, {"pop1": 1.25e-5, "pop2": 1.25e-5}, 0.25, [1, 1.25e-5]),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, {"pop1": 1.25e-5, "pop2": 1.25e-5}, 0.25, [2, 1.25e-5]),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, {"pop1": 1.25e-5, "pop2": 1.25e-5}, 0.25, []),
])
def test_restore_growth_rates_multiple_population_model_multiple_target(config, activity, pop_index, growth_rate_map, time, expected_growth):
    growth_rate_cmd = restore_growth_rates(config, activity, pop_index, growth_rate_map, time)
    found_growth = extract_by_flag(growth_rate_cmd, f"-eg {time}")

    assert found_growth == expected_growth, f"Growth rate restoration of individual populations failed (expected {expected_individual_growth}, got {found_individual_growth})"


@pytest.mark.parametrize("config,activity,pop_index,growth_rate_map,time,population,expected_growth", [
    (make_config_two_pops_with_migration(), {"pop1": True}, {"pop1": 1}, {"pop1": 1e-5}, 0.01, "pop1", 1e-5),
    (make_config_two_pops_with_migration(), {"pop2": True}, {"pop2": 2}, {"pop2": 2.5e-7}, 0.01, "pop2", 2.5e-7),
    (make_config_two_pops_with_migration(), {"pop2": True}, {"pop2": 2}, {"pop2": 7.5e-7}, 0.075, "pop2", 7.5e-7),
    (make_config_two_pops_with_migration(), {"pop2": True}, {"pop2": 2}, {"pop2": 1.25e-5}, 0.2, "pop2", 1.25e-5),
    (make_config_two_pops_with_migration(), {"pop2": True}, {"pop2": 2}, {"pop2": 1.25e-5}, 0.25, "pop2", 1.25e-5),
])
def test_restore_growth_rates_multiple_population_model_single_target(config, activity, pop_index, growth_rate_map, time, population, expected_growth):
    growth_rate_cmd = restore_growth_rates(config, activity, pop_index, growth_rate_map, time, population)
    found_growth = extract_by_flag(growth_rate_cmd, f"-eg {time} {pop_index[population]}")

    assert found_growth == expected_growth, f"Growth rate restoration of individual populations failed (expected {expected_growth}, got {found_growth})"


@pytest.mark.parametrize("event,config,activity,pop_index,growth_rate_map,time,expected_size_change", [
        ({"type": "size_change", "time": 220, "population": "pop1", "size": 900}, make_config_two_pops_with_migration(), {"pop1": True}, {"pop1": 1}, {"pop1": 1e-5}, 0.055, [1, 0.9]),
        ({"type": "size_change", "time": 220, "population": "pop1", "size": 900}, make_config_two_pops_with_migration(), {"pop1": True}, {"pop1": 2}, {"pop1": 1e-5}, 0.055, [2, 0.9]),
        ({"type": "size_change", "time": 220, "population": "pop1", "size": 900}, make_config_two_pops_with_migration(), {"pop1": False}, {"pop1": 1}, {"pop1": 1e-5}, 0.055, -1),
    ]
)
def test_handle_size_change_multiple_population_model_single_target(event, config, activity, pop_index, growth_rate_map, time, expected_size_change):

    if expected_size_change == -1:
        with pytest.raises(ValueError):
            handle_size_change_event(event, config, activity, pop_index, growth_rate_map)
    else:
        size_change_cmd = handle_size_change_event(event, config, activity, pop_index, growth_rate_map)
        found_size_change = extract_by_flag(size_change_cmd, f"-en {time}")

        assert found_size_change == expected_size_change, f"Size change of single population mismatch (expected {expected_size_change}, got {found_size_change})"


@pytest.mark.parametrize("config,en_flag_exists", [
        (make_config_two_pops_with_migration(), True),
        (make_config_with_gene_conversion(), False),
        (make_config_single_pop(), False),
    ]
)
def test_size_change_single_population_flag_in_command(config, en_flag_exists):
    cmd = build_ms_command(config)
    
    if en_flag_exists:
        assert "-en" in cmd, "Expected size change flag '-en' in the command, but it is missing."
    else:
        assert "-en" not in cmd, "Size change change flag '-en' found in the command, but it shouldn't be there."


@pytest.mark.parametrize("event,config,activity,pop_index,growth_rate_map,time,expected_size_change", [
        ({"type": "size_change", "time": 500, "size": 1000}, make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, {"pop1": 1e-5, "pop2": 7.5e-7}, 0.125, 1.0),
        ({"type": "size_change", "time": 300, "size": 5000}, make_config_single_pop(), {}, {}, {"single": 3.75e-7}, 0.0375, 2.5),
    ]
)
def test_handle_size_change_event_all_populations(event, config, activity, pop_index, growth_rate_map, time, expected_size_change):
    size_change_cmd = handle_size_change_event(event, config, activity, pop_index, growth_rate_map)
    found_size_change = extract_by_flag(size_change_cmd, f"-eN {time}")

    assert found_size_change == expected_size_change, f"Size change of all populations mismatch (expected {expected_size_change}, got {found_size_change})"


@pytest.mark.parametrize("config,eN_flag_exists", [
        (make_config_two_pops_with_migration(), True),
        (make_population_split_event_config(), False),
        (make_config_with_gene_conversion(), False),
        (make_config_single_pop(), True),
    ]
)
def test_size_change_all_populations_flag_in_command(config, eN_flag_exists):
    cmd = build_ms_command(config)
    
    if eN_flag_exists:
        assert "-eN" in cmd, "Expected size change flag '-eN' in the command, but it is missing."
    else:
        assert "-eN" not in cmd, "Size change change flag '-eN' found in the command, but it shouldn't be there."


@pytest.mark.parametrize("config,time,expected_growth", [
    (make_config_single_pop(), 0.0374, []),
    (make_config_single_pop(), 0.0375, 3.75e-7),
])
def test_single_population_model_growth_rate_after_size_change(config, time, expected_growth):
    cmd = build_ms_command(config)
    found_growth = extract_by_flag(cmd, f"-eG {time}")

    assert found_growth == expected_growth, f"Restoration of growth rate mismatch at time {time} (expected {expected_growth}, got {found_growth})"


@pytest.mark.parametrize(
    "config,time,expected_growth", [
        (make_config_two_pops_with_migration(), 0.125, [[1, 1e-5], [2, 7.5e-7]]),
        (make_config_two_pops_with_migration(), 0.055, [1, 1e-5]),
    ]
)
def test_multiple_population_model_growth_rates_after_size_change(config, time, expected_growth):
    cmd = build_ms_command(config)
    found_growth = extract_by_flag(cmd, f"-eg {time}")

    assert found_growth == expected_growth, \
        f"Restoration of growth rate mismatch at time {time} (expected {expected_growth}, got {found_growth})"

#----------------------------------------
# Test population split event + helpers
#----------------------------------------


@pytest.mark.parametrize("event,config,activity_before,pop_index_before,time,expected_splits",[
        ({"type": "population_split", "time": 100, "derived": ["pop1", "pop2", "pop3"], "ancestral": "pop4"}, make_population_split_event_config(), {"pop1": True, "pop2": True, "pop3": True, "pop4": False, "pop5": True}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop5": 4}, 0.005, [[2, 1], [3, 1]]),
        ({"type": "population_split", "time": 180, "derived": ["pop2", "pop3", "pop4"], "ancestral": "pop5"}, make_population_admixture_event_config(), {"pop1": False, "pop2": True, "pop3": True, "pop4": True, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 0.0045, [[3, 2], [4, 2]]),
        ({"type": "population_split", "time": 100, "derived": ["pop1", "pop2", "pop3"], "ancestral": "pop4"}, make_population_split_event_config(), {"pop1": False, "pop2": True, "pop3": True, "pop4": False, "pop5": True}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop5": 4}, 0, -1),
        ({"type": "population_split", "time": 100, "derived": ["pop1", "pop2", "pop3"], "ancestral": "pop4"}, make_population_split_event_config(), {"pop1": True, "pop2": False, "pop3": True, "pop4": False, "pop5": True}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop5": 4}, 0, -1),
        ({"type": "population_split", "time": 100, "derived": ["pop1", "pop2", "pop3"], "ancestral": "pop4"}, make_population_split_event_config(), {"pop1": True, "pop2": True, "pop3": False, "pop4": False, "pop5": True}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop5": 4}, 0, -1),
        ({"type": "population_split", "time": 100, "derived": ["pop1", "pop2", "pop3"], "ancestral": "pop4"}, make_population_split_event_config(), {"pop1": False, "pop2": False, "pop3": True, "pop4": False, "pop5": True}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop5": 4}, 0, -1),
        ({"type": "population_split", "time": 100, "derived": ["pop1", "pop2", "pop3"], "ancestral": "pop4"}, make_population_split_event_config(), {"pop1": False, "pop2": False, "pop3": False, "pop4": False, "pop5": True}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop5": 4}, 0, -1),
        ({"type": "population_split", "time": 100, "derived": ["pop1", "pop2", "pop3"], "ancestral": "pop4"}, make_population_split_event_config(), {"pop1": True, "pop2": True, "pop3": True, "pop4": True, "pop5": True}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop5": 4}, 0, -1),
    ]
)
def test_handle_population_split_event(event, config, activity_before, pop_index_before, time, expected_splits):
    if expected_splits == -1:
        with pytest.raises(ValueError):
            handle_population_split_event(event, config, activity_before, pop_index_before)
    else:
        population_split_cmd = handle_population_split_event(event, config, activity_before, pop_index_before)
        found_splits = extract_by_flag(population_split_cmd, f"-ej {time}")

        assert found_splits == expected_splits, f"Split mismatch at time {time}: expected {expected_splits}, found {found_splits}"


@pytest.mark.parametrize("config,time",[
        (make_population_split_event_config(), 0.005),
        (make_population_split_event_with_size_change_config(), 0.005),
        (make_population_admixture_event_config(), 0.0045)
    ]
)
def test_population_split_flag_in_command(config, time):
    cmd = build_ms_command(config)

    assert f"-ej {time}" in cmd, f"Population split flag '-ej {time}' missing from command"


@pytest.mark.parametrize("config,pop_index,time,population,expected_size",[
        (make_population_split_event_config(), {"pop4": 1}, 0.005, "pop4", 1.0),
        (make_population_split_event_with_size_change_config(), {"pop4": 1}, 0.005, "pop4", 0.2),
        (make_population_admixture_event_config(), {"pop5": 2}, 0.0045, "pop5", 0.2)
    ]
)
def test_set_ancestral_size_at_split(config, pop_index, time, population, expected_size):
    size_cmd = set_ancestral_size_at_split(config, pop_index, time, population)
    found_size = extract_by_flag(size_cmd, f"-en {time} {pop_index[population]}")

    assert found_size == expected_size, f"Ancestral size mismatch at time {time} for population {population}: expected {expected_size}, found {found_size}"


@pytest.mark.parametrize("config,time,ancestral,expected_size",[
        (make_population_split_event_config(), 0.005, 1, 1.0),
        (make_population_split_event_with_size_change_config(), 0.005, 1, 0.2),
        (make_population_admixture_event_config(), 0.0045, 2, 0.2)
    ]
)
def test_if_ancestral_size_flag_in_command(config, time, ancestral, expected_size):
    cmd = build_ms_command(config)
    found_size = extract_by_flag(cmd, f"-en {time} {ancestral}")

    if expected_size:
        assert found_size == expected_size, f"Ancestral size flag mismatch at time {time} for {ancestral}: expected {expected_size}, found {found_size}"
    else:
        assert f"-en {time} {ancestral}" not in cmd, f"Unexpected ancestral size adjustment found at time {time} for {ancestral}"


@pytest.mark.parametrize("config,pop_index,time,population,expected_growth",[
        (make_population_split_event_config(), {"pop4": 1}, 0.005, "pop4", 5e-8),
        (make_population_split_event_with_size_change_config(), {"pop4": 1}, 0.005, "pop4", 5e-8),
        (make_population_admixture_event_config(), {"pop5": 2}, 0.0045, "pop5", 2e-7)
    ]
)
def test_set_ancestral_growth_rate_at_split(config, pop_index, time, population, expected_growth):
    growth_cmd = set_ancestral_growth_rate_at_split(config, pop_index, time, population)
    found_growth = extract_by_flag(growth_cmd, f"-eg {time} {pop_index[population]}")

    assert found_growth == expected_growth, f"Ancestral growth rate mismatch at time {time} for population {population}: expected {expected_growth}, found {found_growth}"


@pytest.mark.parametrize("config,time,ancestral,expected_growth_rate", [
        (make_population_split_event_config(), 0.005, 1, 5e-8),
        (make_population_split_event_with_size_change_config(), 0.005, 1, 5e-8),
        (make_population_admixture_event_config(), 0.0045, 2, 2e-7)
    ]
)
def test_if_ancestral_growth_rate_flag_in_command(config, time, ancestral, expected_growth_rate):
    cmd = build_ms_command(config)
    found_growth_rate = extract_by_flag(cmd, f"-eg {time} {ancestral}")

    assert found_growth_rate == expected_growth_rate, f"Growth rate flag mismatch at time {time} for {ancestral}: expected {expected_growth_rate}, found {found_growth_rate}"

@pytest.mark.parametrize("config,activity,pop_index,time,population,expected_migrations",[
        (make_population_split_event_config(), {"pop1": False, "pop2": False, "pop3": False, "pop4": True, "pop5": True}, {"pop4": 1, "pop5": 4}, 0.005, "pop4", [[4, 1, 4e-7], [1, 4, 1e-6]]),
        (make_population_split_event_config(), {"pop1": False, "pop2": False, "pop3": False, "pop4": True, "pop5": False}, {"pop4": 1, "pop5": 4}, 0.005, "pop4", []),
        (make_population_split_event_config(), {"pop1": False, "pop2": True, "pop3": False, "pop4": True, "pop5": True}, {"pop2": 2, "pop4": 1, "pop5": 4}, 0.005, "pop4", [[4, 1, 4e-7], [1, 4, 1e-6], [2, 1, 0.0], [1, 2, 0.0]]),
    ]
)
def test_set_ancestral_migration_rates_at_split(config, activity, pop_index, time, population, expected_migrations):
    migration_cmd = set_ancestral_migration_rates_at_split(config, activity, pop_index, time, population)
    found_migrations = extract_by_flag(migration_cmd, f"-em {time}")

    assert found_migrations == expected_migrations, f"Ancestral migration rates mismatch at time {time} for population {population}: expected {expected_migrations}, found {found_migrations}"


@pytest.mark.parametrize("config,time,expected_migration_rates", [
        (make_population_split_event_config(), 0.005, [[4, 1, 4e-7], [1, 4, 1e-6]])
    ]
)
def test_if_ancestral_migration_rates_flag_in_command(config, time, expected_migration_rates):
    cmd = build_ms_command(config)
    found_migration_rates = extract_by_flag(cmd, f"-em {time}")

    for rate in expected_migration_rates:
        assert rate in found_migration_rates, (
            f"Missing migration rate at time {time}: {rate} "
            f"(found: {found_migration_rates})"
        )

@pytest.mark.parametrize("config,activity,pop_index,time,populations,expected_migrations",[
        (make_population_split_event_config(), {"pop1": False, "pop2": False, "pop3": False, "pop4": False, "pop5": True}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop5": 4}, 0.005, ["pop2", "pop3"], [[2, 4, 0], [3, 4, 0]]),
        (make_population_admixture_event_config(), {"pop1": False, "pop2": True, "pop3": True, "pop4": True, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 0.0025, "pop1", [[1, 2, 0], [1, 3, 0], [1, 4, 0]]),
        (make_population_admixture_event_config(), {"pop1": False, "pop2": False, "pop3": False, "pop4": False, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 0.0045, ["pop3", "pop4"], [])
    ]
)
def test_disable_migration_to_derived(config, activity, pop_index, time, populations, expected_migrations):
    migration_cmd = disable_migration_to_derived(config, activity, pop_index, time, populations)
    found_migrations = extract_by_flag(migration_cmd, f"-em {time}")

    assert found_migrations == expected_migrations


@pytest.mark.parametrize("config,time,expected_migration_rates", [
        (make_population_split_event_config(), 0.005, [[2, 4, 0], [3, 4, 0]]),
        (make_population_admixture_event_config(), 0.0025, [[1, 2, 0], [1, 3, 0], [1, 4, 0]]),
        (make_population_admixture_event_config(), 0.0045, [])
    ]
)
def test_if_disabled_migration_rates_flag_in_command(config, time, expected_migration_rates):
    cmd = build_ms_command(config)
    found_migration_rates = extract_by_flag(cmd, f"-em {time}")

    for rate in expected_migration_rates:
        assert rate in found_migration_rates, (
            f"Missing migration rate at time {time}: {rate} "
            f"(found: {found_migration_rates})"
        )

#------------------------------------
# Test population admixture event 
#------------------------------------


@pytest.mark.parametrize("event,config,activity_before,pop_index_before,num_populations_before,num_populations_after",[
    ({"type": "population_admixture", "time": 100, "derived": "pop1", "ancestral": ["pop2", "pop3", "pop4"], "proportions": [0.5, 0.3, 0.2]}, make_population_admixture_event_config(),{"pop1": True, "pop2": True, "pop3": True, "pop4": True, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 4, 6),
    ({"type": "population_admixture", "time": 100, "derived": "pop1", "ancestral": ["pop2", "pop3", "pop4"], "proportions": [0.8, 0.3, 0.2]}, make_population_admixture_event_config(),{"pop1": True, "pop2": True, "pop3": True, "pop4": True, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 4, -1),
    ({"type": "population_admixture", "time": 100, "derived": "pop1", "ancestral": ["pop2", "pop3", "pop4"], "proportions": [0.8, 0.2]}, make_population_admixture_event_config(),{"pop1": True, "pop2": True, "pop3": True, "pop4": True, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 4, -1),
    ({"type": "population_admixture", "time": 100, "derived": "pop1", "ancestral": ["pop2", "pop3", "pop4"], "proportions": [0.5, 0.3, 0.2]}, make_population_admixture_event_config(),{"pop1": False, "pop2": True, "pop3": True, "pop4": True, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 4, -1),
    ({"type": "population_admixture", "time": 100, "derived": "pop1", "ancestral": ["pop2", "pop3", "pop4"], "proportions": [0.5, 0.3, 0.2]}, make_population_admixture_event_config(),{"pop1": True, "pop2": False, "pop3": True, "pop4": True, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 4, -1),
    ({"type": "population_admixture", "time": 100, "derived": "pop1", "ancestral": ["pop2", "pop3", "pop4"], "proportions": [0.5, 0.3, 0.2]}, make_population_admixture_event_config(),{"pop1": True, "pop2": True, "pop3": False, "pop4": True, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 4, -1),
    ({"type": "population_admixture", "time": 100, "derived": "pop1", "ancestral": ["pop2", "pop3", "pop4"], "proportions": [0.5, 0.3, 0.2]}, make_population_admixture_event_config(),{"pop1": True, "pop2": True, "pop3": True, "pop4": False, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 4, -1),
    ({"type": "population_admixture", "time": 100, "derived": "pop1", "ancestral": ["pop2", "pop3", "pop4"], "proportions": [0.5, 0.3, 0.2]}, make_population_admixture_event_config(),{"pop1": True, "pop2": False, "pop3": False, "pop4": True, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 4, -1),
    ({"type": "population_admixture", "time": 100, "derived": "pop1", "ancestral": ["pop2", "pop3", "pop4"], "proportions": [0.5, 0.3, 0.2]}, make_population_admixture_event_config(),{"pop1": True, "pop2": False, "pop3": False, "pop4": False, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 4, -1)
])
def test_handle_population_admixture_event_of_num_populations(event, config, activity_before, pop_index_before, num_populations_before, num_populations_after):
    if num_populations_after == -1:
        with pytest.raises(ValueError):
            handle_population_admixture_event(event, config, activity_before, pop_index_before, num_populations_before)
    else:
        _, found_num_populations = handle_population_admixture_event(event, config, activity_before, pop_index_before, num_populations_before)

        assert found_num_populations == num_populations_after, f"Expected num_populations {num_populations_after}, but found {found_num_populations}"


@pytest.mark.parametrize("event,config,activity_before,pop_index_before,num_populations_before,time,derived,expected_proportions",[
    ({"type": "population_admixture", "time": 100, "derived": "pop1", "ancestral": ["pop2", "pop3", "pop4"], "proportions": [0.5, 0.3, 0.2]}, make_population_admixture_event_config(),{"pop1": True, "pop2": True, "pop3": True, "pop4": True, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 4, 0.0025, 1, [0.5, 0.4])
])
def test_handle_population_admixture_event_of_proportions(event, config, activity_before, pop_index_before, num_populations_before, time, derived, expected_proportions):
    population_admixture_cmd, _ = handle_population_admixture_event(event, config, activity_before, pop_index_before, num_populations_before)
    found_proportions = extract_by_flag(population_admixture_cmd, f"-es {time} {derived}")

    assert found_proportions == expected_proportions, f"Expected proportions {expected_proportions}, but found {found_proportions}"


@pytest.mark.parametrize("event,config,activity_before,pop_index_before,num_populations_before,time,expected_splits",[
    ({"type": "population_admixture", "time": 100, "derived": "pop1", "ancestral": ["pop2", "pop3", "pop4"], "proportions": [0.5, 0.3, 0.2]}, make_population_admixture_event_config(), {"pop1": True, "pop2": True, "pop3": True, "pop4": True, "pop5": False}, {"pop1": 1, "pop2": 2, "pop3": 3, "pop4": 4}, 4, 0.0025, [[5, 2], [6, 3], [1, 4]])
])
def test_handle_population_admixture_event_of_splits(event, config, activity_before, pop_index_before, num_populations_before, time, expected_splits):
    population_admixture_cmd, _ = handle_population_admixture_event(event, config, activity_before, pop_index_before, num_populations_before)
    found_splits = extract_by_flag(population_admixture_cmd, f"-ej {time}")

    assert found_splits == expected_splits, f"Expected splits {expected_splits}, but found {found_splits}"


@pytest.mark.parametrize("config,time",[
    (make_population_admixture_event_config(), 0.0025)
])
def test_admixture_event_es_flag_in_command(config, time):
    cmd = build_ms_command(config)
    assert f"-es {time}" in cmd, f"Expected '-es {time}' flag for admixture event, but it was not found in the command"


@pytest.mark.parametrize("config,time",[
    (make_population_admixture_event_config(), 0.0025)
])
def test_admixture_event_ej_flag_in_command(config, time):
    cmd = build_ms_command(config)
    assert f"-ej {time}" in cmd, f"Expected '-ej {time}' flag for admixture event, but it was not found in the command"


#------------------------------------
# Test population pulse migration 
#------------------------------------


@pytest.mark.parametrize("event,config,activity,pop_index,num_populations_before,num_populations_after",[
    ({"type": "population_pulse_migration", "time": 100, "dest": "pop1", "source": "pop2", "proportion": 0.5}, make_population_pulse_migration_event_config(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 2, 3),
    ({"type": "population_pulse_migration", "time": 100, "dest": "pop1", "source": "pop2", "proportion": 0.5}, make_population_pulse_migration_event_config(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, 2, -1),
    ({"type": "population_pulse_migration", "time": 100, "dest": "pop1", "source": "pop2", "proportion": 0.5}, make_population_pulse_migration_event_config(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, 2, -1),
    ({"type": "population_pulse_migration", "time": 100, "dest": "pop1", "source": "pop2", "proportion": 0.5}, make_population_pulse_migration_event_config(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, 2, -1),
    ({"type": "population_pulse_migration", "time": 100, "dest": "pop1", "source": "pop2", "proportion": 0.5}, make_population_pulse_migration_event_config(), {"pop1": True, "pop2": True}, {"pop1": 4, "pop2": 3}, 2, 3),
])
def test_handle_populatin_pulse_migration_of_num_populations(event, config, activity, pop_index, num_populations_before, num_populations_after):
    if num_populations_after == -1:
        with pytest.raises(ValueError):
            handle_populatin_pulse_migration(event, config, activity, pop_index, num_populations_before)
    else:
        _, found_num_populations = handle_populatin_pulse_migration(event, config, activity, pop_index, num_populations_before)

        assert found_num_populations == num_populations_after


@pytest.mark.parametrize("event,config,activity,pop_index,time,source,expected_proportion",[
     ({"type": "population_pulse_migration", "time": 100, "dest": "pop1", "source": "pop2", "proportion": 0.5}, make_population_pulse_migration_event_config(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 3.125e-4, "pop2", 0.5),
     ({"type": "population_pulse_migration", "time": 300, "dest": "pop2", "source": "pop1", "proportion": 0.2}, make_population_pulse_migration_event_config(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 9.375e-4, "pop1", 0.8),
])
def test_handle_populatin_pulse_migration_of_proportion(event, config, activity, pop_index, time, source, expected_proportion):
    pulse_migration_cmd, _  = handle_populatin_pulse_migration(event, config, activity, pop_index, 0)
    found_proportion = extract_by_flag(pulse_migration_cmd, f"-es {time} {pop_index[source]}")

    assert found_proportion == expected_proportion


@pytest.mark.parametrize("event,config,activity,pop_index,num_populations,time,expected_split",[
     ({"type": "population_pulse_migration", "time": 100, "dest": "pop1", "source": "pop2", "proportion": 0.5}, make_population_pulse_migration_event_config(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 2, 3.125e-4, [3, 1]),
     ({"type": "population_pulse_migration", "time": 300, "dest": "pop2", "source": "pop1", "proportion": 0.2}, make_population_pulse_migration_event_config(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 3, 9.375e-4, [4, 2]),
])
def test_handle_populatin_pulse_migration_of_splits(event, config, activity, pop_index, num_populations, time, expected_split):
    pulse_migration_cmd, _  = handle_populatin_pulse_migration(event, config, activity, pop_index, num_populations)
    found_split = extract_by_flag(pulse_migration_cmd, f"-ej {time}")

    assert found_split == expected_split


@pytest.mark.parametrize("config,time",[
    (make_population_pulse_migration_event_config(), 3.125e-4),
    (make_population_pulse_migration_event_config(), 9.375e-4),
])
def test_pulse_migration_event_es_flag_in_command(config, time):
    cmd = build_ms_command(config)

    assert f"-es {time}" in cmd, f"Expected '-es {time}' flag for pulse migration event, but it was not found in the command"


@pytest.mark.parametrize("config,time",[
    (make_population_pulse_migration_event_config(), 3.125e-4),
    (make_population_pulse_migration_event_config(), 9.375e-4),
])
def test_pulse_migration_event_ej_flag_in_command(config, time):
    cmd = build_ms_command(config)
    
    assert f"-ej {time}" in cmd, f"Expected '-ej {time}' flag for pulse migration event, but it was not found in the command"