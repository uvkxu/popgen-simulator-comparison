include: "snakemake_vars.py"
include: "rules/simulations.smk"
include: "rules/preprocessing.smk"
include: "rules/topfeatures.smk"
include: "rules/plots.smk"


rule all:
    input:
        expand(
            f"{PLOTS_RESULTS_DIR}/pigotrho/{{simulator}}/.done",
            simulator="ms"
        )