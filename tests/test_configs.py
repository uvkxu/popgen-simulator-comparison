def make_config_single_pop():
    return {
        "ms_path": "ms",
        "population_size": 2000,
        "sample_size": 15,
        "mutation_rate": 1e-7,
        "sequence_length": 500,
        "num_replicates": 3,
        "save_trees": True,
        "recombination_rate": 1e-8,
        "demography_events": [
            {"type": "growth_rate_change", "time": 200, "rate": 0.003}
        ]
    }

def make_config_with_gene_conversion():
    return {
        "ms_path": "ms",
        "population_size": 10000,
        "sample_size": 20,
        "mutation_rate": 2e-7,
        "sequence_length": 100,
        "num_replicates": 5,
        "gene_conversion_rate": 0.00001,
        "gene_conversion_tract_length": 1
    }

def make_config_two_pops_with_migration():
    return {
        "ms_path": "ms",
        "population_size": 1000,
        "mutation_rate": 1e-8,
        "sequence_length": 1000,
        "num_replicates": 1,
        "save_trees": False,
        "default_migration_rate": 0.05,
        "default_growth_rate": 0.001,
        "recombination_rate": 1e-5,
        "gene_conversion_rate": 3e-6,
        "gene_conversion_tract_length": 3,
        "populations": [
            {"name": "pop1", "sample_size": 5, "initial_size": 1000, "growth_rate": 0.04},
            {"name": "pop2", "sample_size": 5, "initial_size": 990},
        ],
        "migration_rates": [
            {"source": "pop1", "dest": "pop2", "rate": 0.01},
            {"source": "pop2", "dest": "pop1", "rate": 0.01},
        ],
        "demography_events": [
            {"type": "migration_rate_change", "time": 100, "source": "pop1", "dest": "pop2", "rate": 0.2},
            {"type": "growth_rate_change", "time": 200, "population": "pop2", "rate": 0.003},
            {"type": "size_change", "time": 220, "population": "pop1", "size": 900},
            {"type": "size_change", "time": 500, "size": 1000},
            {"type": "growth_rate_change", "time": 800, "rate": 0.05}
        ]
    }

def make_population_split_event_config():
    return {
        "ms_path": "ms",
        "population_size": 5000,
        "mutation_rate": 4e-8,
        "sample_size": 20,
        "sequence_length": 100,
        "num_replicates": 20,
        "populations": [
            {"name": "pop1", "sample_size": 5, "initial_size": 1000},
            {"name": "pop2", "sample_size": 5, "initial_size": 1000, "growth_rate": 0.06},
            {"name": "pop3", "sample_size": 10},
            {"name": "pop4", "growth_rate": 0.001},
            {"name": "pop5", "sample_size": 15, "initial_size": 2000, "growth_rate": 0.007}
        ],
        "migration_rates": [
            {"source": "pop1", "dest": "pop2", "rate": 0.01},
            {"source": "pop2", "dest": "pop1", "rate": 0.01},
            {"source": "pop4", "dest": "pop5", "rate": 0.008},
            {"source": "pop5", "dest": "pop4", "rate": 0.02},
        ],
        "demography_events": [
            {"type": "migration_rate_change", "time": 50, "source": "pop2", "dest": "pop1", "rate": 0.05},
            {"type": "population_split", "time": 100, "derived": ["pop1", "pop2", "pop3"], "ancestral": "pop4"},
            {"type": "migration_rate_change", "time": 180, "source": "pop4", "dest": "pop5", "rate": 0.1},
            {"type": "migration_rate_change", "time": 180, "rate": 0.09},
        ]
        
    }


def make_population_split_event_with_size_change_config():
    return {
        "ms_path": "ms",
        "population_size": 5000,
        "mutation_rate": 4e-8,
        "sample_size": 20,
        "sequence_length": 100,
        "num_replicates": 20,
        "populations": [
            {"name": "pop1", "sample_size": 5, "initial_size": 1000},
            {"name": "pop2", "sample_size": 5, "initial_size": 1000, "growth_rate": 0.06},
            {"name": "pop3", "sample_size": 10},
            {"name": "pop4", "initial_size": 1000, "growth_rate": 0.001}
        ],
        "migration_rates": [
            {"source": "pop1", "dest": "pop2", "rate": 0.01},
            {"source": "pop2", "dest": "pop1", "rate": 0.01}
        ],
        "demography_events": [
            {"type": "size_change", "time": 50, "size": 4000},
            {"type": "growth_rate_change", "time": 80, "rate": 0.03},
            {"type": "population_split", "time": 100, "derived": ["pop1", "pop2", "pop3"], "ancestral": "pop4"},
        ]
        
    }


def make_population_admixture_event_config():
    return {
        "ms_path": "ms",
        "population_size": 10000,
        "mutation_rate": 9e-9,
        "sequence_length": 100,
        "num_replicates": 20,
        "populations": [
            {"name": "pop1", "sample_size": 20, "initial_size": 5000},
            {"name": "pop2", "initial_size": 500, "growth_rate": 0.06},
            {"name": "pop3"},
            {"name": "pop4", "initial_size": 500, "growth_rate": 0.001},
            {"name": "pop5", "initial_size": 2000, "growth_rate": 0.008}
        ],
        "demography_events": [
            {"type": "population_admixture", "time": 100, "derived": "pop1", "ancestral": ["pop2", "pop3", "pop4"], "proportions": [0.5, 0.3, 0.2]},
            {"type": "population_split", "time": 180, "derived": ["pop2", "pop3", "pop4"], "ancestral": "pop5"},
        ]
    }

