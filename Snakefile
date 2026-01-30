include: "snakemake_vars.py"
include: "rules/simulations.smk"
include: "rules/preprocessing.smk"
include: "rules/topfeatures.smk"
include: "rules/plots.smk"


rule all:
    input:
        MEAN_SAMPLE_SIZE_OUTPUT_ALLEL, MEAN_SAMPLE_SIZE_OUTPUT_TREE