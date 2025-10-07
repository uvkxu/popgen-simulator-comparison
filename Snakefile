include: "snakemake_vars.py"
include: "rules/simulations.smk"
include: "rules/preprocessing.smk"
include: "rules/boxplots.smk"
include: "rules/sanitycheck.smk"
include: "rules/topfeatures.smk"

rule all:
    input:
        *expand(f"{TOPFEATURES_RESULTS_DIR}/{{config}}", config=all_configs),
        f"{TOPFEATURES_RESULTS_DIR}/all"
'''
    input:
        expand(f"{TREESTATS_DATA_DIR}/{{config}}", config=all_configs)
'''

'''
    
'''




