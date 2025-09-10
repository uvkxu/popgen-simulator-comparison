import pytest
import re
from ms_runner import build_ms_command, set_initial_migration_rates, set_initial_growth_rates, \
                        restore_growth_rates, set_initial_population_sizes, set_ancestral_size_at_split, set_population_sample_sizes, \
                        get_initial_population_activity, get_initial_population_indices, handle_size_change_event, \
                        handle_migration_rate_change_event, handle_growth_rate_change_event, handle_population_split_event, \
                        handle_population_admixture_event, set_recombination, has_gene_conversion, set_gene_conversion

from test_configs import make_config_single_pop, make_config_two_pops_with_migration, make_config_with_gene_conversion, make_population_split_event_config

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

#TODO: Test restore_growth_rates
#TODO: Test split
#TODO: Test set_ancestral_size_at_split
#TODO: Test get_initial_population_activity
#TODO: Test get_initial_population_indices
#TODO: Test demography event handlers + if flag is in command

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
    (make_config_two_pops_with_migration(), 0.0003, 3),
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

@pytest.mark.parametrize("config,activity,pop_index,expected_default_rate,expected_migration", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 1.25e-5, [[2, 1, 2.5e-06], [1, 2, 2.5e-06]]),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, 1.25e-5, []),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, 1.25e-5, []),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, 1.25e-5, []),
    (make_config_single_pop(), {}, {}, 0, []),
])
def test_set_initial_migration_rates(config, activity, pop_index, expected_default_rate, expected_migration):
    mig_cmd = set_initial_migration_rates(config, activity, pop_index)
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

@pytest.mark.parametrize("config,activity,pop_index,expected_default_rate,expected_individual_growth_rates", [
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 2.5e-7, [1, 1e-05]),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, 2.5e-7, []),
    (make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, 2.5e-7, [1, 1e-05]),
    (make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, 2.5e-7, []),
    (make_config_single_pop(), {}, {}, [], []),
])
def test_set_initial_growth_rates(config, activity, pop_index, expected_default_rate, expected_individual_growth_rates):
    growth_cmd = set_initial_growth_rates(config, activity, pop_index)
    found_default_rate = extract_by_flag(growth_cmd, "-G")
    found_individual_growth_rates = extract_by_flag(growth_cmd, "-g")

    assert found_default_rate == expected_default_rate, f"Expected default growth rate {expected_default_rate}, found {found_default_rate}"
    assert found_individual_growth_rates == expected_individual_growth_rates, f"Expected individual growth rates {expected_individual_growth_rates}, found {found_individual_growth_rates}"


@pytest.mark.parametrize("config,G_flag_exists", [
    (make_config_two_pops_with_migration(), True),
    (make_config_single_pop(), False),
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
    ({"type": "growth_rate_change", "time": 200, "population": "pop2", "rate": 0.003}, make_config_two_pops_with_migration(), {"pop2": False}, {"pop2": 2}, 0.05, -1)
])
def test_handle_growth_rate_change_event_single_population(event, config, activity, pop_index, time, expected_growth_rate_change):

    if expected_growth_rate_change == -1:
        with pytest.raises(ValueError):
            handle_growth_rate_change_event(event, config, activity, pop_index)
    else:
        growth_change_cmd = handle_growth_rate_change_event(event, config, activity, pop_index)
        found_growth_rate_change = extract_by_flag(growth_change_cmd, f"-eg {time}")

        assert found_growth_rate_change == expected_growth_rate_change, \
            f"Growth rate change mismatch at time {time} (expected {expected_growth_rate_change}, got {found_growth_rate_change})"


@pytest.mark.parametrize("config, eg_flag_exists", [
    (make_config_two_pops_with_migration(), True),
    (make_config_single_pop(), False),
    (make_config_with_gene_conversion(), False)
])
def test_growth_change_single_population_flag_in_command(config, eg_flag_exists):
    cmd = build_ms_command(config)

    if eg_flag_exists:
        assert "-eg" in cmd, "Expected growth rate change flag '-eg' in the command, but it is missing."
    else:
        assert "-eg" not in cmd, "Growth change rate change flag '-eg' found in the command, but it shouldn't be there."


@pytest.mark.parametrize("event,config,activity,pop_index,time,expected_growth_rate_change", [
    ({"type": "growth_rate_change", "time": 200, "rate": 0.003}, make_config_single_pop(), {}, {}, 0.025, 3.75e-7),
    ({"type": "growth_rate_change", "time": 800, "rate": 0.05}, make_config_two_pops_with_migration(), {"pop1": True, "pop2": True}, {"pop1": 1, "pop2": 2}, 0.2, 1.25e-5),
    ({"type": "growth_rate_change", "time": 800, "rate": 0.05}, make_config_two_pops_with_migration(), {"pop1": True, "pop2": False}, {"pop1": 1, "pop2": 2}, 0.2, 1.25e-5),
    ({"type": "growth_rate_change", "time": 800, "rate": 0.05}, make_config_two_pops_with_migration(), {"pop1": False, "pop2": True}, {"pop1": 1, "pop2": 2}, 0.2, 1.25e-5),
    ({"type": "growth_rate_change", "time": 800, "rate": 0.05}, make_config_two_pops_with_migration(), {"pop1": False, "pop2": False}, {"pop1": 1, "pop2": 2}, 0.2, 1.25e-5),
])
def test_handle_growth_rate_change_event_all_populations(event, config, activity, pop_index, time, expected_growth_rate_change):
    cmd = handle_growth_rate_change_event(event, config, activity, pop_index)
    found_growth_rate_change = extract_by_flag(cmd, f"-eG {time}")

    assert found_growth_rate_change == expected_growth_rate_change, \
        f"Growth rate change mismatch at time {time} (expected {expected_growth_rate_change}, got {found_growth_rate_change})"


@pytest.mark.parametrize("config, eG_flag_exists", [
    (make_config_two_pops_with_migration(), True),
    (make_config_single_pop(), True),
    (make_config_with_gene_conversion(), False),
    (make_population_split_event_config(), False)
])
def test_growth_change_all_populations_flag_in_command(config, eG_flag_exists):
    cmd = build_ms_command(config)

    if eG_flag_exists:
        assert "-eG" in cmd, "Expected growth rate change flag '-eG' in the command, but it is missing."
    else:
        assert "-eG" not in cmd, "Growth change rate change flag '-eG' found in the command, but it shouldn't be there."

# -----------------------------
# Test of size change event
# -----------------------------

@pytest.mark.parametrize(
    "config,time,pop_id,expected_size_change", [
        (make_config_two_pops_with_migration(), 0.055, 1, 0.9),
    ]
)
def test_size_change_single_population(config, time, pop_id, expected_size_change):
    cmd = build_ms_command(config)
    found_size_change = extract_by_flag(cmd, f"-en {time} {pop_id}")
    assert found_size_change == expected_size_change, \
        f"Size change mismatch at time {time} (expected {expected_size_change}, got {found_size_change})"


@pytest.mark.parametrize(
    "config,time,pop_id,expected_growth_rate", [
        (make_config_two_pops_with_migration(), 0.055, 1, 1e-5),
    ]
)
def test_single_population_growth_rate_after_size_change(config, time, pop_id, expected_growth_rate):
    cmd = build_ms_command(config)
    found_growth_rate = extract_by_flag(cmd, f"-eg {time} {pop_id}")
    assert found_growth_rate == expected_growth_rate, \
        f"Single population growth rate after size change at time {time} (expected {expected_growth_rate}, got {found_growth_rate})"


@pytest.mark.parametrize(
    "config,time,expected_size_change", [
        (make_config_two_pops_with_migration(), 0.125, 1.0),
    ]
)
def test_size_change_all_populations(config, time, expected_size_change):
    cmd = build_ms_command(config)
    found_size_change = extract_by_flag(cmd, f"-eN {time}")
    assert found_size_change == expected_size_change, \
        f"Size change mismatch at time {time} (expected {expected_size_change}, got {found_size_change})"


@pytest.mark.parametrize(
    "config,time,expected_default_growth_rate", [
        (make_config_two_pops_with_migration(), 0.125, 2.5e-7),
    ]
)
def test_default_growth_rate_for_all_populations_after_size_change(config, time, expected_default_growth_rate):
    cmd = build_ms_command(config)
    found_default_growth_rate = extract_by_flag(cmd, f"-eG {time}")
    assert found_default_growth_rate == expected_default_growth_rate, \
        f"Default growth rate for all populations after size change at time {time} (expected {expected_default_growth_rate}, got {found_default_growth_rate})"


@pytest.mark.parametrize(
    "config,time,expected_single_growth_rates", [
        (make_config_two_pops_with_migration(), 0.125, [[2, 7.5e-7], [1, 1e-5]]),
    ]
)
def test_individual_population_growth_rates_after_size_change(config, time, expected_single_growth_rates):
    cmd = build_ms_command(config)
    found_single_growth_rates = extract_by_flag(cmd, f"-eg {time}")
    assert found_single_growth_rates == expected_single_growth_rates, \
        f"Individual population growth rates after size change at time {time} (expected {expected_single_growth_rates}, got {found_single_growth_rates})"

'''
@pytest.mark.parametrize("config,time,expected_splits,expected_pop_size,expected_growth_rate", [
    (make_population_split_event_config(), 0.005, [[2, 1],[3, 1]], [1, 1.0], [1, 5e-8])
])
def test_population_split_event(config, time, expected_splits, expected_pop_size, expected_growth_rate):
    cmd = build_ms_command(config)
    found_splits = extract_by_flag(cmd, f"-ej {time}")
    found_pop_size = extract_by_flag(cmd, f"-en {time}")
    found_growth_rate = extract_by_flag(cmd, f"-eg {time}")
    found_migration = extract_by_flag(cmd, f"-em {time}")

    assert found_splits == expected_splits, \
        f"Splits mismatch: expected {expected_splits}, but got {found_splits}"

    assert found_pop_size == expected_pop_size, \
        f"Wrong ancestral population size at split (expected {expected_pop_size}, got {found_pop_size})"

    assert found_growth_rate == expected_growth_rate, \
        f"Wrong growth rate at split (expected {expected_growth_rate}, got {found_growth_rate})"
'''

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
