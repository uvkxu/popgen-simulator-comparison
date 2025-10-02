include: "snakemake_vars.py"
include: "rules/simulations.smk"
include: "rules/preprocessing.smk"
include: "rules/boxplots.smk"

rule all:
    input:
        expand(f"{TREESTATS_DATA_DIR}/{{config}}", config=all_configs)


