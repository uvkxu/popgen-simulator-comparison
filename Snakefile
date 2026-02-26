include: "snakemake_vars.py"
include: "rules/simulations.smk"
include: "rules/preprocessing.smk"
include: "rules/topfeatures.smk"
include: "rules/plots.smk"


rule all:
    input:
        expand(f"{TREEFEATURES_COLLECT_DIR}/{{config}}/simulator_feature_values.csv", config=all_configs),
        expand(f"{ALLELFEATURES_COLLECT_DIR}/{{config}}/simulator_feature_values.csv", config=all_configs)