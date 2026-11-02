include: "snakemake_vars.py"
include: "rules/simulations.smk"
include: "rules/preprocessing.smk"
include: "rules/topfeatures.smk"
include: "rules/optuna.smk"


rule all:
    input:
        ALL_TREE_OPTUNA, ALL_ALLEL_OPTUNA
        