include: "snakemake_vars.py"
include: "rules/simulations.smk"
include: "rules/preprocessing.smk"
include: "rules/topfeatures.smk"
include: "rules/plots.smk"
include: "rules/effective_pop_size.smk"


rule all:
    input:
        str(AGG_OVERALL),
        str(AGG_FEAT),
        str(AGG_PASSED),
        str(AGG_INDEX),
        str(PLOT_BAR_OVERALL),
        str(PLOT_LINE_SEQLEN),
        str(PLOT_LINE_RATIO_MEAN),
        str(PLOT_LINE_RATIO_MIN),
        str(PLOT_BAR_MIG),