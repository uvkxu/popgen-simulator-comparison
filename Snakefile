include: "snakemake_vars.py"
include: "rules/simulations.smk"
include: "rules/preprocessing.smk"
include: "rules/topfeatures.smk"
include: "rules/plots.smk"
include: "rules/effective_pop_size.smk"


rule all:
    input:
        ALL_COPY_COMBO_YAM