include: "snakemake_vars.py"
include: "rules/simulations.smk"
include: "rules/preprocessing.smk"
include: "rules/plots.smk"
include: "rules/topfeatures.smk"

rule all:
    input:
        expand(f"{TREE_FEATURES_DIR}/{{config}}/.done", config=all_own_configs)

''' 
    f"{TREE_FEATURES_DIR}/commands/.done"
    f"{ALLELSTATS_DATA_DIR}/commands/.done"
    f"{RAW_DATA_DIR}/{CMD_SUBDIR}/.done"
    expand(f"{TREESTATS_BOXPLOTS_DIR}/{{config}}/.done", config=all_configs)
    expand(f"{TREE_FEATURES_DIR}/{{config}}/.done", config=all_configs)
    expand(f"{RAW_DATA_DIR}/{{config}}/.done", config=all_configs)
    expand(f"{ALLELSTATS_TOPFEATURES_DIR}/{{config}}/.done", config=all_configs)
    expand(f"{TREESTATS_TOPFEATURES_DIR}/{{config}}/.done", config=all_configs)
    expand(f"results/top_features/treestats_zero/{{config}}/.done", config=all_configs)
'''




