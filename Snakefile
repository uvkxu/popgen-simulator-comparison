include: "snakemake_vars.py"
include: "rules/simulations.smk"
include: "rules/preprocessing.smk"

rule all:
    input:
        expand(f"{TREESTATS_DATA_DIR}/{{config}}/msprime_stats.csv", config=all_configs),
        expand(f"{TREESTATS_DATA_DIR}/{{config}}/ms_stats.csv", config=all_configs),
        expand(f"{TREESTATS_DATA_DIR}/{{config}}/msms_stats.csv", config=all_configs)