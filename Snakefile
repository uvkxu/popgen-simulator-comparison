include: "snakemake_vars.py"
include: "rules/simulations.smk"
include: "rules/preprocessing.smk"
include: "rules/topfeatures.smk"
include: "rules/plots.smk"


rule all:
    input:
        ALL_TREE_TOPFEATURES, ALL_ALLEL_TOPFEATURES