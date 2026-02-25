include: "snakemake_vars.py"
include: "rules/simulations.smk"
include: "rules/preprocessing.smk"
include: "rules/topfeatures.smk"
include: "rules/plots.smk"


rule all:
    input:
        expand(
            f"{ALLELFEATURES_PLOTS_DIR}" + "/{config}/deltas.csv",
            config=all_configs
        )