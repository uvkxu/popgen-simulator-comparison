import pytest
import msprime
import numpy as np
from simulations.msprime_runner import *
from test_configs import *

#---------------------------
# Test of update activity
#---------------------------

@pytest.mark.parametrize(
    "initial, derived, ancestral, expected",
    [
        ({"p1": True, "p2": True}, "p1", None, {"p1": False, "p2": True}),
        ({"p1": False, "p2": False}, None, "p2", {"p1": False, "p2": True}),
        ({"p1": False, "p2": True}, ["p1"], ["p2"], {"p1": False, "p2": True}),
    ],
)
def test_update_activity(initial, derived, ancestral, expected):
    update_activity(initial, derived=derived, ancestral=ancestral)
    assert initial == expected


def test_handle_population_split_and_admixture_update_activity():
    demography = msprime.Demography()
    demography.add_population(name="p1", initial_size=100)
    demography.add_population(name="p2", initial_size=100)
    demography.add_population(name="p3", initial_size=100)
    demography.add_population(name="p4", initial_size=100)
    
    config = {"populations": [{"name": "p1"}, {"name": "p2"}, {"name": "p3"}, {"name": "p4"}]}

    activity = {"p1": True, "p2": True, "p3": True, "p4": False}

    # Admixture
    admixture_event = {"time": 100, "derived": "p1", "ancestral": ["p2", "p3"], "proportions": [0.5, 0.5]}
    handle_population_admixture(demography, admixture_event, activity)
    assert activity["p1"] is False 


    # Split: p2 und p3 entstehen aus p4
    split_event = {"time": 200, "derived": ["p2", "p3"], "ancestral": "p4"}

    handle_population_split(demography, split_event, activity, config)
    assert activity["p4"] is True
    assert activity["p2"] is False
    assert activity["p3"] is False

#--------------------------
# Test of event handlers
#--------------------------

@pytest.mark.parametrize(
    "config,epoch,expected", [
        (make_config_two_pops_with_migration(), 3, {"pop1": 900}),
        (make_config_two_pops_with_migration(), 4, {"pop1": 1000, "pop2": 1000}),
        (make_config_single_pop(), 2, {"pop_0": 5000})
    ],
)
def test_handle_size_change(config,epoch,expected):
    demography = create_msprime_demography(config)
    population_at_epoch = demography.debug().epochs[epoch].populations
    start_sizes_at_epoch = {p.name: p.start_size for p in population_at_epoch}

    for name, size in expected.items():
        assert np.isclose(start_sizes_at_epoch[name], size), (
            f"Population {name} expected size {size}, got {start_sizes_at_epoch[name]}"
        )

@pytest.mark.parametrize(
    "config,epoch,expected", [
        (make_config_two_pops_with_migration(), 2, {"pop1": 0.04, "pop2": 0.003}),
        (make_config_two_pops_with_migration(), 5, {"pop1": 0.05, "pop2": 0.05}),
        (make_config_single_pop(), 1, {"pop_0": 0.003})
    ],
)
def test_handle_growth_rate_change_event(config,epoch,expected):
    demography = create_msprime_demography(config)
    population_at_epoch = demography.debug().epochs[epoch].populations

    assert {p.name: p.growth_rate for p in population_at_epoch} == expected


@pytest.mark.parametrize("config,epoch,expected", [
    (make_config_two_pops_with_migration(), 1, [
        [0, 0.2],
        [0.01 , 0]
    ]),
    (make_population_split_event_config(), 1, [
        [0, 0.01, 0, 0, 0],
        [0.05, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]),
    (make_population_split_event_config(), 3, [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.09],
        [0, 0, 0, 0.09, 0],
    ])
])
def test_handle_migration_rate_change_event(config,epoch,expected):
    demography = create_msprime_demography(config)
    debug = demography.debug()

    actual_matrix = np.array(debug.epochs[epoch].migration_matrix)
    expected_matrix = np.array(expected)

    assert np.allclose(actual_matrix, expected_matrix), f"Expected:\n{expected_matrix}\nGot:\n{actual_matrix}"

@pytest.mark.parametrize("config,epoch,expected", [
    (make_population_admixture_event_config(), 1, {"pop1": 2, "pop2": 1, "pop3": 1, "pop4": 1, "pop5": 0}),
    (make_population_admixture_event_config(), 2, {"pop1": 2, "pop2": 2, "pop3": 2, "pop4": 2, "pop5": 1}),
])
def test_handle_admixture_split_event_states(config,epoch,expected):
    demography = create_msprime_demography(config)
    populations_at_epoch = demography.debug().epochs[epoch].populations

    assert {p.name: p.state for p in populations_at_epoch} == expected

@pytest.mark.parametrize("config,epoch,population,expected_growth_rate,expected_size", [
    (make_population_admixture_event_config(), 2, "pop5", 0.008, 2000),
    (make_population_split_event_config(), 2, "pop4", 0.001, 5000),
    (make_population_split_event_with_size_change_config(), 3, "pop4", 0.001, 1000)
])
def test_growth_rate_and_size_of_ancestral_at_split(config, epoch, population, expected_growth_rate, expected_size):
    demography = create_msprime_demography(config)
    population_at_epoch = [p for p in demography.debug().epochs[epoch].populations if p.name == population][0]

    assert np.isclose(expected_growth_rate, population_at_epoch.growth_rate)
    assert expected_size == population_at_epoch.start_size


#-----------------------------------------
# Test of initilization of the populations
#-----------------------------------------

@pytest.mark.parametrize("config,expected", [
    (make_config_two_pops_with_migration(), {"pop1", "pop2"}),
    (make_population_split_event_config(), {"pop1", "pop2", "pop3", "pop4", "pop5"}),
])
def test_create_msprime_demography_initialization_of_names(config, expected):
    demography = create_msprime_demography(config)

    assert set(p.name for p in demography.populations) == expected


@pytest.mark.parametrize("config,expected", [
    (make_config_two_pops_with_migration(), {"pop1": 1000, "pop2": 990}),
    (make_population_split_event_config(), {"pop1": 1000, "pop2": 1000, "pop3": 5000, "pop4": 5000, "pop5": 2000}),
    (make_config_with_gene_conversion(), {"pop_0": 10000})
])
def test_create_msprime_demography_initialization_of_sizes(config, expected):
    demography = create_msprime_demography(config)

    assert {p.name: p.initial_size for p in demography.populations} == expected


@pytest.mark.parametrize("config,expected", [
    (make_config_two_pops_with_migration(), {"pop1": 0.04, "pop2": 0.001}),
    (make_population_split_event_config(), {"pop1": 0.0, "pop2": 0.06, "pop3": 0.0, "pop4": 0.0, "pop5": 0.007}),
    (make_population_split_event_with_size_change_config(), {"pop1": 0.0, "pop2": 0.06, "pop3": 0.0, "pop4": 0.0}),
    (make_config_with_gene_conversion(), {"pop_0": 0.04})
])
def test_create_msprime_demography_initialization_of_growth_rates(config, expected):
    demography = create_msprime_demography(config)

    assert {p.name: p.growth_rate for p in demography.populations} == expected

@pytest.mark.parametrize("config,expected", [
    (make_config_two_pops_with_migration(), [
        [0, 0.01],
        [0.01 , 0]
    ]),
    (make_population_split_event_config(), [
        [0, 0.01, 0, 0, 0],
        [0.01, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]),
])
def test_set_initial_migration_rates(config, expected):
    demography = create_msprime_demography(config)

    actual_matrix = np.array(demography.migration_matrix)
    expected_matrix = np.array(expected)

    assert np.allclose(actual_matrix, expected_matrix), f"Expected:\n{expected_matrix}\nGot:\n{actual_matrix}"
