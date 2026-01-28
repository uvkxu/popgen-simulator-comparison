include: "../snakemake_vars.py"

rule compute_treeplots_single_config:
    input:
        directory(f"{RAW_DATA_DIR}/{{config}}")
    output:
        directory(f"{TREEPLOTS_RESULTS_DIR}/{{config}}")
    conda:
        "../envs/treestats-env.yaml"
    shell:
        """
        Rscript src/preprocessing/get_treeplots.R {RAW_DATA_DIR}/{wildcards.config} {TREEPLOTS_RESULTS_DIR}/{wildcards.config}
        """

rule compute_treeplots_every_config:
    input:
        expand(f"{TREEPLOTS_RESULTS_DIR}/{{config}}", config=all_configs)
